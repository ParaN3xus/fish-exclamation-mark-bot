use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use opencv::core::{self, CV_8UC3, Rect, Scalar};
use opencv::imgproc;
use opencv::prelude::*;
use tracing::{error, info, warn};

use crate::audio::{AudioEngine, AudioMatchMask};
use crate::config::AppConfig;
use crate::control::VrchatClicker;
use crate::filter::ObservationFilter;
use crate::policy::{FishingObservation, TimeOptimalBangBangPolicy};
use crate::types::{BBox, BotState, DetectCommand, DetectPacket, FramePacket, Kp, TrackState};
use crate::vision::{
    YoloOrt, clip_box, detect_bright_fish_strategy, mat_bgra_from_bytes, roi_from_outer_and_kp,
};

#[derive(Clone)]
struct YoloJob {
    seq: u64,
    w: i32,
    h: i32,
    bgr: Vec<u8>,
    conf: f32,
}

#[derive(Clone)]
struct YoloOut {
    seq: u64,
    det: Option<crate::types::OuterDet>,
}

fn mat_bgr_from_bytes(w: i32, h: i32, bgr: &[u8]) -> Result<Mat> {
    let mut m = Mat::new_rows_cols_with_default(h, w, CV_8UC3, Scalar::default())?;
    m.data_bytes_mut()?.copy_from_slice(bgr);
    Ok(m)
}

fn safe_click_once(clicker: &mut VrchatClicker) {
    if let Err(e) = clicker.click_once() {
        error!(error = ?e, "control click_once failed");
    }
}

fn safe_set_press(clicker: &mut VrchatClicker, press: bool) {
    if let Err(e) = clicker.set_press(press) {
        error!(error = ?e, press, "control set_press failed");
    }
}

fn safe_poll_focus(clicker: &mut VrchatClicker) {
    if let Err(e) = clicker.poll_focus() {
        error!(error = ?e, "control poll_focus failed");
    }
}

fn safe_shake_head(clicker: &mut VrchatClicker) {
    if let Err(e) = clicker.shake_head() {
        error!(error = ?e, "control shake_head failed");
    }
}

fn state_name(state: BotState) -> &'static str {
    match state {
        BotState::Stopped => "Stopped",
        BotState::WaitingFish => "WaitingFish",
        BotState::BiteOrError => "BiteOrError",
        BotState::Fishing => "Fishing",
        BotState::CollectFish => "CollectFish",
        BotState::ReleaseLine => "ReleaseLine",
    }
}

fn log_transition(
    from: BotState,
    to: BotState,
    reason: &str,
    status_text: &mut String,
    state_entered_at: &mut Option<Instant>,
) {
    info!(
        from = state_name(from),
        to = state_name(to),
        reason,
        "state transition"
    );
    *status_text = format!("{} -> {} ({reason})", state_name(from), state_name(to));
    *state_entered_at = None;
}

fn centered_square_box(w: i32, h: i32) -> BBox {
    let side = w.min(h).max(1);
    let x = (w - side) / 2;
    let y = (h - side) / 2;
    BBox {
        x,
        y,
        w: side,
        h: side,
    }
}

fn bright_endpoints_along_axis(br: BBox, axis_top: Kp, axis_bot: Kp) -> (Kp, Kp) {
    let cx = (br.x + br.w / 2) as f32;
    let cy = (br.y + br.h / 2) as f32;
    let vx = axis_bot.x - axis_top.x;
    let vy = axis_bot.y - axis_top.y;
    let norm = (vx * vx + vy * vy).sqrt().max(1e-6);
    let ux = vx / norm;
    let uy = vy / norm;
    let half = (br.h.max(1) as f32) * 0.5;
    let p0 = Kp {
        x: cx - ux * half,
        y: cy - uy * half,
    };
    let p1 = Kp {
        x: cx + ux * half,
        y: cy + uy * half,
    };
    (p0, p1)
}

fn extend_bright_to_expected_height(
    mut br: BBox,
    roi: BBox,
    frame_h: i32,
    expected_h: i32,
    roi_edge_eps_px: i32,
    extend_min_visible_ratio: f32,
    extend_min_width_ratio: f32,
) -> BBox {
    if expected_h <= br.h {
        return br;
    }

    let missing = expected_h - br.h;
    let br_top = br.y;
    let br_bot = br.y + br.h;
    let roi_top = roi.y;
    let roi_bot = roi.y + roi.h;
    let touch_top = br_top <= roi_top + roi_edge_eps_px;
    let touch_bot = br_bot >= roi_bot - roi_edge_eps_px;

    // Guard against extending edge noise:
    // only extend when we already have a reasonably wide and reasonably visible bar fragment.
    let min_visible_h = (expected_h as f32 * extend_min_visible_ratio).round() as i32;
    let min_width = (roi.w as f32 * extend_min_width_ratio).round() as i32;
    let looks_like_bar_fragment = br.h >= min_visible_h.max(1) && br.w >= min_width.max(1);
    if !looks_like_bar_fragment {
        return br;
    }

    if touch_top && !touch_bot {
        br.y -= missing;
        br.h = expected_h;
    } else if touch_bot && !touch_top {
        br.h = expected_h;
    } else if touch_top && touch_bot {
        // Both sides clipped: split extension to keep center stable.
        let up = missing / 2;
        br.y -= up;
        br.h = expected_h;
    }

    if br.y < 0 {
        br.y = 0;
    }
    let max_h = (frame_h - br.y).max(1);
    br.h = br.h.clamp(1, max_h);
    br
}

fn init_track_from_outer(
    det_b: BBox,
    det_top: Kp,
    det_bot: Kp,
    gray: &Mat,
    w: i32,
    h: i32,
) -> Result<Option<(TrackState, Option<BBox>, Option<BBox>)>> {
    let roi = clip_box(
        roi_from_outer_and_kp(det_b, det_top, det_bot, w, h, 2),
        w,
        h,
    );
    let roi_mat = Mat::roi(gray, Rect::new(roi.x, roi.y, roi.w, roi.h))?.try_clone()?;

    let outer_local = BBox {
        x: det_b.x - roi.x,
        y: det_b.y - roi.y,
        w: det_b.w,
        h: det_b.h,
    };
    let top_local = Kp {
        x: det_top.x - roi.x as f32,
        y: det_top.y - roi.y as f32,
    };
    let bot_local = Kp {
        x: det_bot.x - roi.x as f32,
        y: det_bot.y - roi.y as f32,
    };

    let (br, fish, ptop, pbot, spec) = detect_bright_fish_strategy(
        &roi_mat,
        outer_local,
        top_local,
        bot_local,
        None,
        None,
        None,
    )?;

    let br_abs = br.map(|b| BBox {
        x: b.x + roi.x,
        y: b.y + roi.y,
        ..b
    });
    let fish_abs = fish.map(|f| BBox {
        x: f.x + roi.x,
        y: f.y + roi.y,
        ..f
    });

    let bright_h_norm = br
        .map(|b| ((b.h as f32 / roi.h as f32) * 168.0).round() as i32)
        .unwrap_or(66);

    let state = TrackState {
        roi,
        outer_rel: outer_local,
        top_rel: top_local,
        bot_rel: bot_local,
        bright_h_norm,
        fish_spec: spec,
        proc_top: ptop,
        proc_bot: pbot,
    };

    Ok(Some((state, br_abs, fish_abs)))
}

pub fn run_detect(
    cfg: Arc<AppConfig>,
    rx: Receiver<FramePacket>,
    tx: Sender<DetectPacket>,
    rx_cmd: Receiver<DetectCommand>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let exe = std::env::current_exe()?;
    let exe_dir = exe
        .parent()
        .map(std::path::Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("failed to resolve exe directory"))?;
    let model_path = cfg.model_path(&exe_dir);
    info!(
        model = %model_path.display(),
        imgsz = cfg.yolo.imgsz,
        class_id = cfg.yolo.class_id,
        "initializing yolo/ort worker"
    );
    let model_path_s = model_path.to_string_lossy().into_owned();
    let (tx_yolo_job, rx_yolo_job) = bounded::<YoloJob>(1);
    let (tx_yolo_out, rx_yolo_out) = bounded::<YoloOut>(1);
    let (tx_yolo_init, rx_yolo_init) = bounded::<Result<()>>(1);
    let yolo_stop = stop.clone();
    let yolo_model = model_path_s.clone();
    let yolo_imgsz = cfg.yolo.imgsz;
    let yolo_class_id = cfg.yolo.class_id;
    let yolo_handle = thread::spawn(move || {
        let mut yolo = match YoloOrt::new(&yolo_model, yolo_imgsz, yolo_class_id) {
            Ok(v) => {
                let _ = tx_yolo_init.send(Ok(()));
                v
            }
            Err(e) => {
                let _ = tx_yolo_init.send(Err(e));
                return;
            }
        };
        while !yolo_stop.load(Ordering::Relaxed) {
            let job = match rx_yolo_job.recv_timeout(Duration::from_millis(50)) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let det = (|| -> Result<Option<crate::types::OuterDet>> {
                let frame = mat_bgr_from_bytes(job.w, job.h, &job.bgr)?;
                yolo.detect_outer(&frame, job.conf)
            })()
            .unwrap_or(None);
            let _ = tx_yolo_out.try_send(YoloOut { seq: job.seq, det });
        }
    });
    rx_yolo_init.recv()??;
    info!("yolo/ort worker ready");

    let mut audio = match AudioEngine::new(cfg.as_ref(), &exe_dir) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!(error = ?e, "audio init failed; continue without audio");
            None
        }
    };

    let mut clicker = match VrchatClicker::new(cfg.as_ref()) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!(error = ?e, "control init failed; continue without control");
            None
        }
    };
    let mut bot_state = if cfg.startup.start_stopped {
        BotState::Stopped
    } else {
        BotState::WaitingFish
    };
    let mut track_state: Option<TrackState> = None;
    let mut policy: Option<TimeOptimalBangBangPolicy> = None;
    let mut obs_filter: Option<ObservationFilter> = None;
    let mut first_det_at: Option<Instant> = None;
    let mut state_entered_at: Option<Instant> = None;
    let mut last_obs_tick: Option<Instant> = None;
    let mut last_fishing_yolo_check_at: Option<Instant> = None;
    let mut last_fishing_detect_at: Option<Instant> = None;
    let mut press = false;
    let mut state_machine_enabled = !cfg.startup.start_stopped;
    let mut yolo_submit_seq: u64 = 0;
    let mut yolo_latest_seq: u64 = 0;
    let mut yolo_latest_det: Option<crate::types::OuterDet> = None;
    let mut bite_or_error_last_yolo_seq: u64 = 0;
    let mut fishing_periodic_pending: bool = false;
    let mut fishing_periodic_last_eval_seq: u64 = 0;
    let mut fishing_periodic_miss_once: bool = false;
    let mut fishing_periodic_retry_after: Option<Instant> = None;

    let detect_fps_limit = cfg.state_machine.fishing_detect_fps_limit.max(1.0);
    let detect_interval = Duration::from_secs_f32(1.0 / detect_fps_limit);
    let detect_sleep_interval = Duration::from_millis(cfg.loop_timing.detect_sleep_ms);
    let pipeline_min_interval = detect_interval.max(detect_sleep_interval);
    let mut last_det_tick: Option<Instant> = None;
    let mut last_cap_tick: Option<Instant> = None;
    let mut last_pipeline_tick = Instant::now() - pipeline_min_interval;
    let boot = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        let pkt = match rx.recv_timeout(Duration::from_millis(cfg.loop_timing.detect_recv_timeout_ms)) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let now = Instant::now();
        if now.duration_since(last_pipeline_tick) < pipeline_min_interval {
            continue;
        }
        last_pipeline_tick = now;

        let fps_cap = if let Some(prev_cap_tick) = last_cap_tick {
            1.0f32
                / pkt.captured_at
                    .saturating_duration_since(prev_cap_tick)
                    .as_secs_f32()
                    .max(1e-6)
        } else {
            0.0
        };
        last_cap_tick = Some(pkt.captured_at);

        let fps_det = if let Some(prev_det_tick) = last_det_tick {
            1.0f32
                / now
                    .saturating_duration_since(prev_det_tick)
                    .as_secs_f32()
                    .max(1e-6)
        } else {
            0.0
        };
        last_det_tick = Some(now);

        let audio_mask = match bot_state {
            BotState::WaitingFish => AudioMatchMask {
                bite: true,
                success: false,
                fail: false,
                collected: false,
            },
            BotState::Fishing => AudioMatchMask {
                bite: false,
                success: true,
                fail: true,
                collected: true,
            },
            _ => AudioMatchMask::none(),
        };
        let audio_events = if let Some(audio) = audio.as_mut() {
            audio.poll_with_mask(boot.elapsed().as_millis() as u64, audio_mask)
        } else {
            crate::audio::AudioEvents {
                bite_hit: false,
                success_hit: false,
                fail_hit: false,
                collected_hit: false,
                bite_similarity: 0.0,
                success_similarity: 0.0,
                fail_similarity: 0.0,
                collected_similarity: 0.0,
            }
        };
        if audio_events.bite_hit {
            info!(sound = "bite", sim = audio_events.bite_similarity, "audio hit");
        }
        if audio_events.success_hit {
            info!(
                sound = "success",
                sim = audio_events.success_similarity,
                "audio hit"
            );
        }
        if audio_events.fail_hit {
            info!(sound = "fail", sim = audio_events.fail_similarity, "audio hit");
        }
        if audio_events.collected_hit {
            info!(
                sound = "collected",
                sim = audio_events.collected_similarity,
                "audio hit"
            );
        }

        if let Some(clicker) = clicker.as_mut() {
            safe_poll_focus(clicker);
        }

        let bgra = mat_bgra_from_bytes(pkt.w, pkt.h, &pkt.bgra)?;
        let mut bgr = Mat::default();
        imgproc::cvt_color(
            &bgra,
            &mut bgr,
            imgproc::COLOR_BGRA2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let should_submit_yolo = match bot_state {
            BotState::BiteOrError => true,
            BotState::Fishing => {
                fishing_periodic_pending && yolo_latest_seq == fishing_periodic_last_eval_seq
            }
            _ => false,
        };
        if should_submit_yolo {
            yolo_submit_seq = yolo_submit_seq.wrapping_add(1);
            let yolo_job = YoloJob {
                seq: yolo_submit_seq,
                w: pkt.w,
                h: pkt.h,
                bgr: bgr.data_bytes()?.to_vec(),
                conf: cfg.yolo.conf,
            };
            let _ = tx_yolo_job.try_send(yolo_job);
        }
        while let Ok(out) = rx_yolo_out.try_recv() {
            yolo_latest_seq = out.seq;
            yolo_latest_det = out.det;
        }

        let yolo_square_draw: Option<BBox> = Some(centered_square_box(pkt.w, pkt.h));
        let mut outer_draw: Option<BBox> = None;
        let mut yolo_top_draw: Option<Kp> = None;
        let mut yolo_bot_draw: Option<Kp> = None;
        let mut bright_p0_draw: Option<Kp> = None;
        let mut bright_p1_draw: Option<Kp> = None;
        let mut fish_p_draw: Option<Kp> = None;
        let mut policy_fish_center: Option<f32> = None;
        let mut policy_player_center: Option<f32> = None;
        let mut policy_progress: Option<f32> = None;
        let mut policy_target_half: Option<f32> = None;
        let mut status_text: String;

        while let Ok(cmd) = rx_cmd.try_recv() {
            match cmd {
                DetectCommand::Reset => {
                    let from = bot_state;
                    bot_state = BotState::WaitingFish;
                    first_det_at = None;
                    state_entered_at = None;
                    track_state = None;
                    policy = None;
                    obs_filter = None;
                    last_obs_tick = None;
                    last_fishing_detect_at = None;
                    fishing_periodic_pending = false;
                    fishing_periodic_miss_once = false;
                    fishing_periodic_retry_after = None;
                    if press {
                        if let Some(clicker) = clicker.as_mut() {
                            safe_set_press(clicker, false);
                        }
                        press = false;
                    }
                    info!(
                        from = state_name(from),
                        to = state_name(bot_state),
                        "manual reset"
                    );
                }
                DetectCommand::ForceFishComes => {
                    let from = bot_state;
                    bot_state = BotState::BiteOrError;
                    first_det_at = None;
                    state_entered_at = None;
                    policy = None;
                    obs_filter = None;
                    last_obs_tick = None;
                    last_fishing_detect_at = None;
                    fishing_periodic_pending = false;
                    fishing_periodic_miss_once = false;
                    fishing_periodic_retry_after = None;
                    if press {
                        if let Some(clicker) = clicker.as_mut() {
                            safe_set_press(clicker, false);
                        }
                        press = false;
                    }
                    info!(
                        from = state_name(from),
                        to = state_name(bot_state),
                        "manual force fish comes"
                    );
                }
                DetectCommand::ToggleStateMachine => {
                    state_machine_enabled = !state_machine_enabled;
                    if !state_machine_enabled {
                        first_det_at = None;
                        state_entered_at = None;
                        track_state = None;
                        policy = None;
                        obs_filter = None;
                        last_obs_tick = None;
                        last_fishing_detect_at = None;
                        fishing_periodic_pending = false;
                        fishing_periodic_miss_once = false;
                        fishing_periodic_retry_after = None;
                        bot_state = BotState::Stopped;
                        if press {
                            if let Some(clicker) = clicker.as_mut() {
                                safe_set_press(clicker, false);
                            }
                            press = false;
                        }
                    } else if bot_state == BotState::Stopped {
                        bot_state = BotState::WaitingFish;
                        state_entered_at = None;
                    }
                    info!(
                        enabled = state_machine_enabled,
                        state = state_name(bot_state),
                        "state machine toggled"
                    );
                }
            }
        }

        if state_machine_enabled {
            match bot_state {
                BotState::WaitingFish => {
                    status_text = "WaitingFish".to_string();
                    if state_entered_at.is_none() {
                        state_entered_at = Some(now);
                        info!("entered WaitingFish");
                    }
                    first_det_at = None;
                    track_state = None;
                    policy = None;
                    obs_filter = None;
                    last_obs_tick = None;
                    last_fishing_yolo_check_at = None;
                    last_fishing_detect_at = None;
                    fishing_periodic_pending = false;
                    fishing_periodic_miss_once = false;
                    fishing_periodic_retry_after = None;
                    if press {
                        if let Some(clicker) = clicker.as_mut() {
                            safe_set_press(clicker, false);
                        }
                        press = false;
                    }

                    if audio_events.bite_hit {
                        let from = bot_state;
                        bot_state = BotState::BiteOrError;
                        log_transition(
                            from,
                            bot_state,
                            "bite audio detected",
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    } else if now
                        .duration_since(state_entered_at.unwrap_or(now))
                        .as_millis()
                        >= u128::from(cfg.state_machine.wait_bite_timeout_ms)
                    {
                        if let Some(clicker) = clicker.as_mut() {
                            info!("waiting timeout: shake head before transition");
                            safe_shake_head(clicker);
                        }
                        let from = bot_state;
                        bot_state = BotState::BiteOrError;
                        log_transition(
                            from,
                            bot_state,
                            "WaitingFish timeout",
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    }
                }
                BotState::BiteOrError => {
                    status_text = "BiteOrError".to_string();
                    if state_entered_at.is_none() {
                        state_entered_at = Some(now);
                        info!("entered BiteOrError");
                        first_det_at = None;
                        last_fishing_yolo_check_at = None;
                        last_fishing_detect_at = None;
                        fishing_periodic_pending = false;
                        fishing_periodic_miss_once = false;
                        fishing_periodic_retry_after = None;
                        bite_or_error_last_yolo_seq = yolo_latest_seq;
                        if press {
                            if let Some(clicker) = clicker.as_mut() {
                                safe_set_press(clicker, false);
                            }
                            press = false;
                        }
                        if let Some(clicker) = clicker.as_mut() {
                            info!("bite-or-error action: click once");
                            safe_click_once(clicker);
                        }
                    }
                    if yolo_latest_seq != bite_or_error_last_yolo_seq {
                        bite_or_error_last_yolo_seq = yolo_latest_seq;
                        if let Some(o) = yolo_latest_det {
                        info!(
                            conf = o.conf,
                            x = o.b.x,
                            y = o.b.y,
                            w = o.b.w,
                            h = o.b.h,
                            "yolo detection success in BiteOrError"
                        );
                        outer_draw = Some(o.b);
                        yolo_top_draw = Some(o.top);
                        yolo_bot_draw = Some(o.bot);
                        let in_first_window = now
                            .duration_since(state_entered_at.unwrap_or(now))
                            .as_millis()
                            < u128::from(cfg.state_machine.reel_first_det_timeout_ms);
                        if first_det_at.is_none() && in_first_window {
                            info!("BiteOrError first yolo detection acquired in timeout window");
                            first_det_at = Some(now);
                        } else if now.duration_since(first_det_at.unwrap_or(now)).as_millis()
                            >= u128::from(cfg.state_machine.second_detect_delay_ms)
                        {
                            let mut gray = Mat::default();
                            imgproc::cvt_color(
                                &bgr,
                                &mut gray,
                                imgproc::COLOR_BGR2GRAY,
                                0,
                                core::AlgorithmHint::ALGO_HINT_DEFAULT,
                            )?;
                            if let Some((st, br, fish)) =
                                init_track_from_outer(o.b, o.top, o.bot, &gray, pkt.w, pkt.h)?
                            {
                                info!(
                                    roi_x = st.roi.x,
                                    roi_y = st.roi.y,
                                    roi_w = st.roi.w,
                                    roi_h = st.roi.h,
                                    "BiteOrError second yolo detection success; tracking initialized"
                                );
                                track_state = Some(st);
                                if let Some(b) = br {
                                    let (p0, p1) = bright_endpoints_along_axis(b, o.top, o.bot);
                                    bright_p0_draw = Some(p0);
                                    bright_p1_draw = Some(p1);
                                }
                                if let Some(f) = fish {
                                    fish_p_draw = Some(Kp {
                                        x: (f.x + f.w / 2) as f32,
                                        y: (f.y + f.h / 2) as f32,
                                    });
                                }
                                policy = None;
                                obs_filter = None;
                                last_obs_tick = None;
                                bot_state = BotState::Fishing;
                                log_transition(
                                    BotState::BiteOrError,
                                    bot_state,
                                    "tracking initialized",
                                    &mut status_text,
                                    &mut state_entered_at,
                                );
                            }
                            first_det_at = Some(now);
                        }
                    }
                    }

                    if first_det_at.is_none()
                        && now
                            .duration_since(state_entered_at.unwrap_or(now))
                            .as_millis()
                            >= u128::from(cfg.state_machine.reel_first_det_timeout_ms)
                    {
                        let from = bot_state;
                        bot_state = BotState::WaitingFish;
                        log_transition(
                            from,
                            bot_state,
                            "BiteOrError first-detect timeout",
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    }
                }
                BotState::Fishing => {
                    status_text = "Fishing".to_string();
                    if state_entered_at.is_none() {
                        state_entered_at = Some(now);
                        info!("entered Fishing");
                        last_fishing_yolo_check_at = None;
                        last_fishing_detect_at = None;
                        fishing_periodic_pending = false;
                        fishing_periodic_miss_once = false;
                        fishing_periodic_retry_after = None;
                    }

                    let should_check_yolo = if fishing_periodic_miss_once {
                        match fishing_periodic_retry_after {
                            Some(t) => now >= t,
                            None => true,
                        }
                    } else {
                        match last_fishing_yolo_check_at {
                            None => true,
                            Some(t) => {
                                now.duration_since(t).as_millis()
                                    >= u128::from(cfg.state_machine.fishing_yolo_check_ms)
                            }
                        }
                    };
                    if should_check_yolo {
                        last_fishing_yolo_check_at = Some(now);
                        if fishing_periodic_miss_once {
                            fishing_periodic_retry_after = None;
                        }
                        fishing_periodic_pending = true;
                    }
                    if fishing_periodic_pending && yolo_latest_seq != fishing_periodic_last_eval_seq {
                        fishing_periodic_last_eval_seq = yolo_latest_seq;
                        fishing_periodic_pending = false;
                        if yolo_latest_det.is_none() {
                            if !fishing_periodic_miss_once {
                                fishing_periodic_miss_once = true;
                                // Retry once after 0.1s to avoid same-frame false miss exits.
                                fishing_periodic_retry_after =
                                    Some(now + Duration::from_millis(100));
                                fishing_periodic_pending = false;
                            } else {
                                fishing_periodic_miss_once = false;
                                fishing_periodic_retry_after = None;
                                if press {
                                    if let Some(clicker) = clicker.as_mut() {
                                        safe_set_press(clicker, false);
                                    }
                                    press = false;
                                }
                                bot_state = BotState::CollectFish;
                                log_transition(
                                    BotState::Fishing,
                                    bot_state,
                                    "fishing periodic yolo miss x2",
                                    &mut status_text,
                                    &mut state_entered_at,
                                );
                                continue;
                            }
                        } else {
                            fishing_periodic_miss_once = false;
                            fishing_periodic_retry_after = None;
                        }
                    }
                    let detect_fps = cfg.state_machine.fishing_detect_fps_limit.max(1.0);
                    let detect_interval = Duration::from_secs_f32(1.0 / detect_fps);
                    let allow_fishing_detect = match last_fishing_detect_at {
                        None => true,
                        Some(t) => now.duration_since(t) >= detect_interval,
                    };
                    if allow_fishing_detect {
                        last_fishing_detect_at = Some(now);
                    }

                    if allow_fishing_detect {
                        if let Some(st) = track_state.as_mut() {
                        let mut gray = Mat::default();
                        imgproc::cvt_color(
                            &bgr,
                            &mut gray,
                            imgproc::COLOR_BGR2GRAY,
                            0,
                            core::AlgorithmHint::ALGO_HINT_DEFAULT,
                        )?;
                        let roi = clip_box(st.roi, pkt.w, pkt.h);
                        let roi_mat =
                            Mat::roi(&gray, Rect::new(roi.x, roi.y, roi.w, roi.h))?.try_clone()?;
                        let (br, fish, _pt, _pb, _spec) = detect_bright_fish_strategy(
                            &roi_mat,
                            st.outer_rel,
                            st.top_rel,
                            st.bot_rel,
                            Some(st.bright_h_norm.max(8)),
                            st.fish_spec.as_deref(),
                            Some((st.proc_top, st.proc_bot)),
                        )?;

                        yolo_top_draw = Some(Kp {
                            x: st.top_rel.x + roi.x as f32,
                            y: st.top_rel.y + roi.y as f32,
                        });
                        yolo_bot_draw = Some(Kp {
                            x: st.bot_rel.x + roi.x as f32,
                            y: st.bot_rel.y + roi.y as f32,
                        });

                        let mut br_abs = br.map(|b| BBox {
                            x: b.x + roi.x,
                            y: b.y + roi.y,
                            ..b
                        });
                        let fish_abs = fish.map(|f| BBox {
                            x: f.x + roi.x,
                            y: f.y + roi.y,
                            ..f
                        });
                        outer_draw = Some(roi);

                        if let (Some(mut br), Some(fish)) = (br_abs, fish_abs) {
                            let dt = if let Some(last_tick) = last_obs_tick {
                                now.duration_since(last_tick).as_secs_f32().max(1e-3)
                            } else {
                                1.0 / 60.0
                            };
                            last_obs_tick = Some(now);

                            let expected_h_px = ((st.bright_h_norm.max(8) as f32 / 168.0)
                                * roi.h as f32)
                                .round() as i32;
                            br = extend_bright_to_expected_height(
                                br,
                                roi,
                                pkt.h,
                                expected_h_px.max(1),
                                cfg.vision.roi_edge_eps_px,
                                cfg.vision.extend_min_visible_ratio,
                                cfg.vision.extend_min_width_ratio,
                            );
                            br_abs = Some(br);

                            // Align with src/gym/fishing_env.py: y=0 at bottom, y=1 at top.
                            let fish_center_top01 = (((fish.y + fish.h / 2) - roi.y) as f32
                                / roi.h as f32)
                                .clamp(0.0, 1.0);
                            let player_center_top01 =
                                (((br.y + br.h / 2) - roi.y) as f32 / roi.h as f32).clamp(0.0, 1.0);
                            let fish_center = 1.0 - fish_center_top01;
                            let player_center = 1.0 - player_center_top01;
                            let player_target_half_size =
                                (st.bright_h_norm.max(8) as f32 / 168.0 * 0.5).clamp(0.0, 0.5);

                            let obs_raw = FishingObservation {
                                fish_center,
                                player_center,
                                dt,
                                player_target_half_size,
                            };

                            let filtered = if let Some(f) = obs_filter.as_mut() {
                                f.apply(obs_raw)
                            } else {
                                let mut f = ObservationFilter::new(cfg.filter.clone());
                                let init = f.reset(obs_raw);
                                obs_filter = Some(f);
                                init
                            };

                            let action = {
                                let p = policy.get_or_insert_with(|| {
                                    TimeOptimalBangBangPolicy::from_config(&cfg.policy)
                                });
                                p.act(filtered)
                            };

                            policy_fish_center = Some(filtered.fish_center);
                            policy_player_center = Some(filtered.player_center);
                            policy_target_half = Some(filtered.player_target_half_size);
                            let overlap_like = 1.0
                                - ((filtered.fish_center - filtered.player_center).abs()
                                    / filtered.player_target_half_size.max(1e-4))
                                .clamp(0.0, 1.0);
                            policy_progress = Some(overlap_like);

                            let next_press = action == 1;
                            if next_press != press {
                                if let Some(clicker) = clicker.as_mut() {
                                    safe_set_press(clicker, next_press);
                                }
                                press = next_press;
                            }
                        }

                        if let Some(b) = br_abs {
                            if let (Some(tp), Some(bp)) = (yolo_top_draw, yolo_bot_draw) {
                                let (p0, p1) = bright_endpoints_along_axis(b, tp, bp);
                                bright_p0_draw = Some(p0);
                                bright_p1_draw = Some(p1);
                            }
                        }
                        if let Some(f) = fish_abs {
                            fish_p_draw = Some(Kp {
                                x: (f.x + f.w / 2) as f32,
                                y: (f.y + f.h / 2) as f32,
                            });
                        }
                        } else {
                            let from = bot_state;
                            bot_state = BotState::CollectFish;
                            log_transition(
                                from,
                                bot_state,
                                "lost track",
                                &mut status_text,
                                &mut state_entered_at,
                            );
                        }
                    } else if track_state.is_none() {
                        let from = bot_state;
                        bot_state = BotState::CollectFish;
                        log_transition(
                            from,
                            bot_state,
                            "lost track",
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    }

                    if audio_events.success_hit {
                        if press {
                            if let Some(clicker) = clicker.as_mut() {
                                safe_set_press(clicker, false);
                            }
                            press = false;
                        }
                        bot_state = BotState::CollectFish;
                        log_transition(
                            BotState::Fishing,
                            bot_state,
                            "success audio detected",
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    } else if audio_events.collected_hit || audio_events.fail_hit {
                        if press {
                            if let Some(clicker) = clicker.as_mut() {
                                safe_set_press(clicker, false);
                            }
                            press = false;
                        }
                        bot_state = BotState::ReleaseLine;
                        let reason = if audio_events.collected_hit {
                            "collected audio detected"
                        } else {
                            "fail audio detected"
                        };
                        log_transition(
                            BotState::Fishing,
                            bot_state,
                            reason,
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    } else if now
                        .duration_since(state_entered_at.unwrap_or(now))
                        .as_millis()
                        >= u128::from(cfg.state_machine.fishing_timeout_ms)
                    {
                        bot_state = BotState::BiteOrError;
                        log_transition(
                            BotState::Fishing,
                            bot_state,
                            "Fishing timeout",
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    }
                }
                BotState::CollectFish => {
                    status_text = "CollectFish".to_string();
                    if state_entered_at.is_none() {
                        state_entered_at = Some(now);
                        info!("entered CollectFish");
                    }
                    last_fishing_yolo_check_at = None;
                    last_fishing_detect_at = None;
                    fishing_periodic_pending = false;
                    fishing_periodic_miss_once = false;
                    fishing_periodic_retry_after = None;

                    if now
                        .duration_since(state_entered_at.unwrap_or(now))
                        .as_millis()
                        >= u128::from(cfg.state_machine.result_action_wait_ms)
                    {
                        if let Some(clicker) = clicker.as_mut() {
                            info!("collect action: click once");
                            safe_click_once(clicker);
                        }
                        bot_state = BotState::ReleaseLine;
                        log_transition(
                            BotState::CollectFish,
                            bot_state,
                            "collect action finished",
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    }
                }
                BotState::ReleaseLine => {
                    status_text = "ReleaseLine".to_string();
                    if state_entered_at.is_none() {
                        state_entered_at = Some(now);
                        info!("entered ReleaseLine");
                    }
                    last_fishing_yolo_check_at = None;
                    last_fishing_detect_at = None;
                    fishing_periodic_pending = false;
                    fishing_periodic_miss_once = false;
                    fishing_periodic_retry_after = None;
                    if now
                        .duration_since(state_entered_at.unwrap_or(now))
                        .as_millis()
                        >= u128::from(cfg.state_machine.result_action_wait_ms)
                    {
                        if let Some(clicker) = clicker.as_mut() {
                            info!("release action: click once");
                            safe_click_once(clicker);
                        }
                        bot_state = BotState::WaitingFish;
                        log_transition(
                            BotState::ReleaseLine,
                            bot_state,
                            "release action finished",
                            &mut status_text,
                            &mut state_entered_at,
                        );
                    }
                }
                BotState::Stopped => {
                    last_fishing_yolo_check_at = None;
                    last_fishing_detect_at = None;
                    fishing_periodic_pending = false;
                    fishing_periodic_miss_once = false;
                    fishing_periodic_retry_after = None;
                }
            }
        }

        if let Some(o) = outer_draw {
            imgproc::rectangle(
                &mut bgr,
                Rect::new(o.x, o.y, o.w.max(1), o.h.max(1)),
                Scalar::new(255.0, 0.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }

        if let Some(sq) = yolo_square_draw {
            imgproc::rectangle(
                &mut bgr,
                Rect::new(sq.x, sq.y, sq.w.max(1), sq.h.max(1)),
                Scalar::new(255.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }

        if let (Some(p0), Some(p1)) = (bright_p0_draw, bright_p1_draw) {
            let a = opencv::core::Point::new(p0.x.round() as i32, p0.y.round() as i32);
            let b = opencv::core::Point::new(p1.x.round() as i32, p1.y.round() as i32);
            imgproc::line(
                &mut bgr,
                a,
                b,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::circle(
                &mut bgr,
                a,
                4,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::circle(
                &mut bgr,
                b,
                4,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }

        if let Some(fp) = fish_p_draw {
            imgproc::circle(
                &mut bgr,
                opencv::core::Point::new(fp.x.round() as i32, fp.y.round() as i32),
                5,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }

        if let Some(kp) = yolo_top_draw {
            imgproc::circle(
                &mut bgr,
                opencv::core::Point::new(kp.x.round() as i32, kp.y.round() as i32),
                5,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }
        if let Some(kp) = yolo_bot_draw {
            imgproc::circle(
                &mut bgr,
                opencv::core::Point::new(kp.x.round() as i32, kp.y.round() as i32),
                5,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_8,
                0,
            )?;
        }

        let cap_to_policy_ms = Instant::now().duration_since(pkt.captured_at).as_secs_f32() * 1000.0;

        let bytes = bgr.data_bytes()?.to_vec();
        let _ = tx.try_send(DetectPacket {
            t_sec: boot.elapsed().as_secs_f64(),
            w: pkt.w,
            h: pkt.h,
            bgr: bytes,
            state: bot_state,
            press,
            state_machine_enabled,
            bite_similarity: audio_events.bite_similarity,
            success_similarity: audio_events.success_similarity,
            fail_similarity: audio_events.fail_similarity,
            collected_similarity: audio_events.collected_similarity,
            bite_hit: audio_events.bite_hit,
            success_hit: audio_events.success_hit,
            fail_hit: audio_events.fail_hit,
            collected_hit: audio_events.collected_hit,
            policy_fish_center,
            policy_player_center,
            policy_progress,
            policy_target_half,
            fps_cap,
            fps_det,
            cap_to_policy_ms,
        });

    }

    if press {
        if let Some(clicker) = clicker.as_mut() {
            safe_set_press(clicker, false);
        }
    }
    let _ = yolo_handle.join();

    Ok(())
}



