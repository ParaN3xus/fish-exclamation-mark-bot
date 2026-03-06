use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotBounds, PlotPoint, PlotPoints, Text, VLine};
use tracing::{info, warn};
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    HOT_KEY_MODIFIERS, MOD_NOREPEAT, RegisterHotKey, UnregisterHotKey, VK_F1, VK_F2, VK_F3,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetMessageW, MSG, PostThreadMessageW, TranslateMessage, WM_HOTKEY, WM_QUIT,
};

use crate::config::AppConfig;
use crate::types::{BotState, DetectCommand, DetectPacket};

const AUDIO_WINDOW_SEC: f64 = 8.0;
const POLICY_WINDOW_SEC: f64 = 8.0;
const MAX_HISTORY_SEC: f64 = 60.0;
const PLOT_PAN_STEP_SEC: f64 = 1.0;
const REPAINT_MS: u64 = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DebugTab {
    Vision,
    Audio,
    Policy,
}

struct AudioSample {
    t: f64,
    bite: f64,
    success: f64,
    fail: f64,
    collected: f64,
}

struct AudioEvent {
    t: f64,
    label: &'static str,
    sim: f64,
}

#[derive(Clone, Copy)]
struct PolicySample {
    t: f64,
    fish: f64,
    player: f64,
    target_half: f64,
}

struct PreviewApp {
    rx: Receiver<DetectPacket>,
    tx_cmd: Sender<DetectCommand>,
    stop: Arc<AtomicBool>,
    texture: Option<egui::TextureHandle>,
    status_text: String,
    frame_size: Option<[usize; 2]>,
    detection_enabled: bool,
    tab: DebugTab,
    audio_history: VecDeque<AudioSample>,
    audio_events: VecDeque<AudioEvent>,
    policy_history: VecDeque<PolicySample>,
    last_policy_sample: Option<PolicySample>,
    latest_t: f64,
    audio_follow_latest: bool,
    audio_view_end_t: f64,
    policy_follow_latest: bool,
    policy_view_end_t: f64,
    th_bite: f64,
    th_success: f64,
    th_fail: f64,
    th_collected: f64,
    default_target_half: f64,
}

impl PreviewApp {
    fn new(
        rx: Receiver<DetectPacket>,
        tx_cmd: Sender<DetectCommand>,
        stop: Arc<AtomicBool>,
        cfg: Arc<AppConfig>,
    ) -> Self {
        Self {
            rx,
            tx_cmd,
            stop,
            texture: None,
            status_text: "starting...".to_string(),
            frame_size: None,
            detection_enabled: true,
            tab: DebugTab::Vision,
            audio_history: VecDeque::new(),
            audio_events: VecDeque::new(),
            policy_history: VecDeque::new(),
            last_policy_sample: None,
            latest_t: 0.0,
            audio_follow_latest: true,
            audio_view_end_t: AUDIO_WINDOW_SEC,
            policy_follow_latest: true,
            policy_view_end_t: POLICY_WINDOW_SEC,
            th_bite: cfg.audio.bite_threshold as f64,
            th_success: cfg.audio.success_threshold as f64,
            th_fail: cfg.audio.fail_threshold as f64,
            th_collected: cfg.audio.collected_threshold as f64,
            default_target_half: cfg.policy.fish_target_half_size as f64,
        }
    }

    fn send_cmd(&mut self, cmd: DetectCommand) {
        let _ = self.tx_cmd.try_send(cmd);
    }

    fn trim_history(&mut self) {
        let min_t = self.latest_t - MAX_HISTORY_SEC;
        while self
            .audio_history
            .front()
            .map(|x| x.t < min_t)
            .unwrap_or(false)
        {
            let _ = self.audio_history.pop_front();
        }
        while self
            .audio_events
            .front()
            .map(|x| x.t < min_t)
            .unwrap_or(false)
        {
            let _ = self.audio_events.pop_front();
        }
        while self
            .policy_history
            .front()
            .map(|x| x.t < min_t)
            .unwrap_or(false)
        {
            let _ = self.policy_history.pop_front();
        }
    }

    fn ingest_latest_packet(&mut self, ctx: &egui::Context) {
        let mut latest = None;
        while let Ok(pkt) = self.rx.try_recv() {
            latest = Some(pkt);
        }

        let Some(pkt) = latest else {
            return;
        };

        self.latest_t = pkt.t_sec;
        self.status_text = format!(
            "state={:?} press={} cap_fps={:.1} det_fps={:.1} bite_sim={:.2} success_sim={:.2} fail_sim={:.2} collected_sim={:.2}",
            pkt.state,
            pkt.press as i32,
            pkt.fps_cap,
            pkt.fps_det,
            pkt.bite_similarity,
            pkt.success_similarity,
            pkt.fail_similarity,
            pkt.collected_similarity,
        );
        self.detection_enabled = pkt.state_machine_enabled;

        self.audio_history.push_back(AudioSample {
            t: pkt.t_sec,
            bite: pkt.bite_similarity as f64,
            success: pkt.success_similarity as f64,
            fail: pkt.fail_similarity as f64,
            collected: pkt.collected_similarity as f64,
        });

        if pkt.bite_hit {
            self.audio_events.push_back(AudioEvent {
                t: pkt.t_sec,
                label: "bite",
                sim: pkt.bite_similarity as f64,
            });
        }
        if pkt.success_hit {
            self.audio_events.push_back(AudioEvent {
                t: pkt.t_sec,
                label: "success",
                sim: pkt.success_similarity as f64,
            });
        }
        if pkt.fail_hit {
            self.audio_events.push_back(AudioEvent {
                t: pkt.t_sec,
                label: "fail",
                sim: pkt.fail_similarity as f64,
            });
        }
        if pkt.collected_hit {
            self.audio_events.push_back(AudioEvent {
                t: pkt.t_sec,
                label: "collected",
                sim: pkt.collected_similarity as f64,
            });
        }

        let mut touched = false;
        let mut cur = self.last_policy_sample.unwrap_or(PolicySample {
            t: pkt.t_sec,
            fish: 0.5,
            player: 0.5,
            target_half: self.default_target_half,
        });
        if let Some(v) = pkt.policy_fish_center {
            cur.fish = v as f64;
            touched = true;
        }
        if let Some(v) = pkt.policy_player_center {
            cur.player = v as f64;
            touched = true;
        }
        if let Some(v) = pkt.policy_target_half {
            cur.target_half = v as f64;
            touched = true;
        }
        let _ = pkt.policy_progress;
        if touched {
            cur.t = pkt.t_sec;
            self.last_policy_sample = Some(cur);
            self.policy_history.push_back(cur);
        } else if pkt.state == BotState::Fishing {
            if let Some(mut prev) = self.last_policy_sample {
                prev.t = pkt.t_sec;
                self.policy_history.push_back(prev);
            }
        }

        self.trim_history();

        let image = bgr_to_color_image(pkt.w as usize, pkt.h as usize, &pkt.bgr);
        self.frame_size = Some([pkt.w as usize, pkt.h as usize]);

        if let Some(tex) = &mut self.texture {
            tex.set(image, egui::TextureOptions::LINEAR);
        } else {
            self.texture = Some(ctx.load_texture("preview", image, egui::TextureOptions::LINEAR));
        }
    }

    fn draw_tab_selector(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, DebugTab::Vision, "Vision Debug");
            ui.selectable_value(&mut self.tab, DebugTab::Audio, "Audio Debug");
            ui.selectable_value(&mut self.tab, DebugTab::Policy, "Policy Debug");
        });
    }

    fn draw_visual_tab(&mut self, ui: &mut egui::Ui) {
        if let (Some(tex), Some([w, h])) = (&self.texture, self.frame_size) {
            let avail = ui.available_size();
            let src = egui::vec2(w as f32, h as f32);
            let scale = (avail.x / src.x).min(avail.y / src.y).max(0.01);
            let draw_size = src * scale;
            ui.vertical_centered(|ui| {
                ui.image((tex.id(), draw_size));
            });
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Waiting for frames...");
            });
        }
    }

    fn draw_audio_tab(&mut self, ui: &mut egui::Ui) {
        let latest_x_max = self.latest_t.max(AUDIO_WINDOW_SEC);
        if self.audio_follow_latest {
            self.audio_view_end_t = latest_x_max;
        } else {
            self.audio_view_end_t = self.audio_view_end_t.clamp(AUDIO_WINDOW_SEC, latest_x_max);
        }
        let x_max = self.audio_view_end_t;
        let x_min = (x_max - AUDIO_WINDOW_SEC).max(0.0);

        ui.horizontal(|ui| {
            if ui.button("<< 1s").clicked() {
                self.audio_follow_latest = false;
                self.audio_view_end_t =
                    (self.audio_view_end_t - PLOT_PAN_STEP_SEC).max(AUDIO_WINDOW_SEC);
            }
            if ui.button("1s >>").clicked() {
                self.audio_follow_latest = false;
                self.audio_view_end_t =
                    (self.audio_view_end_t + PLOT_PAN_STEP_SEC).min(latest_x_max);
            }
            if ui.button("Sync Latest").clicked() {
                self.audio_follow_latest = true;
                self.audio_view_end_t = latest_x_max;
            }
            ui.label(if self.audio_follow_latest {
                "mode: follow latest"
            } else {
                "mode: manual"
            });
        });

        let bite_pts = PlotPoints::from_iter(self.audio_history.iter().map(|s| [s.t, s.bite]));
        let success_pts =
            PlotPoints::from_iter(self.audio_history.iter().map(|s| [s.t, s.success]));
        let fail_pts = PlotPoints::from_iter(self.audio_history.iter().map(|s| [s.t, s.fail]));
        let collected_pts =
            PlotPoints::from_iter(self.audio_history.iter().map(|s| [s.t, s.collected]));
        let bite_color = egui::Color32::from_rgb(220, 60, 60);
        let success_color = egui::Color32::from_rgb(40, 170, 80);
        let fail_color = egui::Color32::from_rgb(230, 145, 30);
        let collected_color = egui::Color32::from_rgb(60, 120, 230);

        let plot_resp = Plot::new("audio_debug_plot")
            .legend(Legend::default())
            .allow_drag([true, false])
            .allow_scroll([true, false])
            .allow_zoom(false)
            .height((ui.available_height() - 28.0).max(120.0))
            .show(ui, |plot_ui| {
                plot_ui.set_plot_bounds(PlotBounds::from_min_max([x_min, 0.0], [x_max, 1.02]));

                plot_ui.line(Line::new("bite", bite_pts).width(2.4).color(bite_color));
                plot_ui.line(
                    Line::new("success", success_pts)
                        .width(2.4)
                        .color(success_color),
                );
                plot_ui.line(Line::new("fail", fail_pts).width(2.4).color(fail_color));
                plot_ui.line(
                    Line::new("collected", collected_pts)
                        .width(2.4)
                        .color(collected_color),
                );

                plot_ui.hline(
                    HLine::new("th_bite", self.th_bite)
                        .width(2.0)
                        .color(bite_color)
                        .style(LineStyle::Dashed { length: 8.0 }),
                );
                plot_ui.hline(
                    HLine::new("th_success", self.th_success)
                        .width(2.0)
                        .color(success_color)
                        .style(LineStyle::Dashed { length: 8.0 }),
                );
                plot_ui.hline(
                    HLine::new("th_fail", self.th_fail)
                        .width(2.0)
                        .color(fail_color)
                        .style(LineStyle::Dashed { length: 8.0 }),
                );
                plot_ui.hline(
                    HLine::new("th_collected", self.th_collected)
                        .width(2.0)
                        .color(collected_color)
                        .style(LineStyle::Dashed { length: 8.0 }),
                );

                for ev in self.audio_events.iter().filter(|e| e.t >= x_min) {
                    let ev_color = match ev.label {
                        "bite" => bite_color,
                        "success" => success_color,
                        "fail" => fail_color,
                        "collected" => collected_color,
                        _ => egui::Color32::GRAY,
                    };
                    plot_ui.vline(
                        VLine::new("", ev.t)
                            .color(ev_color)
                            .width(1.5)
                            .allow_hover(false),
                    );
                    let txt = format!("{} {:.2}", ev.label, ev.sim);
                    plot_ui.text(Text::new(
                        "",
                        PlotPoint::new(ev.t, 0.98),
                        txt,
                    ));
                }
            });

        let b = plot_resp.transform.bounds();
        let new_end = b.max()[0].max(AUDIO_WINDOW_SEC).min(latest_x_max);
        if (new_end - x_max).abs() > 1e-4 {
            self.audio_follow_latest = false;
            self.audio_view_end_t = new_end;
        }
    }

    fn draw_policy_tab(&mut self, ui: &mut egui::Ui) {
        let latest_x_max = self.latest_t.max(POLICY_WINDOW_SEC);
        if self.policy_follow_latest {
            self.policy_view_end_t = latest_x_max;
        } else {
            self.policy_view_end_t = self.policy_view_end_t.clamp(POLICY_WINDOW_SEC, latest_x_max);
        }
        let x_max = self.policy_view_end_t;
        let x_min = (x_max - POLICY_WINDOW_SEC).max(0.0);

        ui.horizontal(|ui| {
            if ui.button("<< 1s").clicked() {
                self.policy_follow_latest = false;
                self.policy_view_end_t =
                    (self.policy_view_end_t - PLOT_PAN_STEP_SEC).max(POLICY_WINDOW_SEC);
            }
            if ui.button("1s >>").clicked() {
                self.policy_follow_latest = false;
                self.policy_view_end_t =
                    (self.policy_view_end_t + PLOT_PAN_STEP_SEC).min(latest_x_max);
            }
            if ui.button("Sync Latest").clicked() {
                self.policy_follow_latest = true;
                self.policy_view_end_t = latest_x_max;
            }
            ui.label(if self.policy_follow_latest {
                "mode: follow latest"
            } else {
                "mode: manual"
            });
        });

        let fish_pts = PlotPoints::from_iter(self.policy_history.iter().map(|s| [s.t, s.fish]));
        let player_pts = PlotPoints::from_iter(self.policy_history.iter().map(|s| [s.t, s.player]));
        let fish_color = egui::Color32::from_rgb(220, 60, 60);
        let player_color = egui::Color32::from_rgb(50, 120, 235);

        let right_w = 80.0f32;
        let gap = 8.0f32;
        let full = ui.available_size();
        let (full_rect, _) = ui.allocate_exact_size(full, egui::Sense::hover());
        let left_w = (full_rect.width() - right_w - gap).max(120.0);

        let left_rect = egui::Rect::from_min_max(
            full_rect.min,
            egui::pos2(full_rect.min.x + left_w, full_rect.max.y),
        );
        let right_rect = egui::Rect::from_min_max(
            egui::pos2(left_rect.max.x + gap, full_rect.min.y),
            full_rect.max,
        );

        let mut left_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(left_rect)
                .layout(egui::Layout::top_down(egui::Align::Min)),
        );
        let plot_resp = Plot::new("policy_debug_plot")
            .legend(Legend::default())
            .allow_drag([true, false])
            .allow_scroll([true, false])
            .allow_zoom(false)
            .height(left_rect.height())
            .show(&mut left_ui, |plot_ui| {
                plot_ui.set_plot_bounds(PlotBounds::from_min_max([x_min, 0.0], [x_max, 1.02]));
                plot_ui.line(Line::new("fish", fish_pts).color(fish_color));
                plot_ui.line(Line::new("player", player_pts).color(player_color));
            });
        let b = plot_resp.transform.bounds();
        let new_end = b.max()[0].max(POLICY_WINDOW_SEC).min(latest_x_max);
        if (new_end - x_max).abs() > 1e-4 {
            self.policy_follow_latest = false;
            self.policy_view_end_t = new_end;
        }

        let painter = ui.painter_at(right_rect);
        let strip_margin_x = 4.0;
        let strip_margin_top = 0.0;
        let strip_margin_bottom = 31.0;
        let strip = egui::Rect::from_min_max(
            egui::pos2(
                right_rect.left() + strip_margin_x,
                right_rect.top() + strip_margin_top,
            ),
            egui::pos2(
                right_rect.right() - strip_margin_x,
                right_rect.bottom() - strip_margin_bottom,
            ),
        );
        painter.rect_filled(strip, 4.0, egui::Color32::WHITE);
        painter.rect_stroke(
            strip,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::GRAY),
            egui::StrokeKind::Middle,
        );

        if let Some(last) = self.policy_history.back() {
            let y_of =
                |v: f64| -> f32 { strip.bottom() - (v.clamp(0.0, 1.0) as f32) * strip.height() };
            let fish_y = y_of(last.fish);
            let player_y = y_of(last.player);
            let player_half = (last.target_half as f32).clamp(0.01, 0.45) * strip.height();

            let player_rect = egui::Rect::from_min_max(
                egui::pos2(strip.left() + 2.0, player_y - player_half),
                egui::pos2(strip.right() - 2.0, player_y + player_half),
            );
            painter.rect_filled(
                player_rect,
                2.0,
                egui::Color32::from_rgba_unmultiplied(
                    player_color.r(),
                    player_color.g(),
                    player_color.b(),
                    100,
                ),
            );

            let fish_half = 0.05f32 * strip.height();
            let fish_rect = egui::Rect::from_min_max(
                egui::pos2(strip.left() + 3.0, fish_y - fish_half),
                egui::pos2(strip.right() - 3.0, fish_y + fish_half),
            );
            painter.rect_filled(
                fish_rect,
                2.0,
                egui::Color32::from_rgba_unmultiplied(
                    fish_color.r(),
                    fish_color.g(),
                    fish_color.b(),
                    90,
                ),
            );
            painter.circle_filled(
                egui::pos2(strip.center().x, fish_y),
                3.5,
                fish_color,
            );
            painter.circle_filled(
                egui::pos2(strip.center().x, player_y),
                3.5,
                player_color,
            );
        }
    }
}

impl eframe::App for PreviewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.stop.load(Ordering::Relaxed) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        self.ingest_latest_packet(ctx);

        if ctx.input(|i| i.key_pressed(egui::Key::Q)) {
            self.stop.store(true, Ordering::Relaxed);
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.label(&self.status_text);
                ui.horizontal(|ui| {
                    if ui.button("Reset (F1)").clicked() {
                        self.send_cmd(DetectCommand::Reset);
                    }
                    if ui.button("Force Comes (F2)").clicked() {
                        self.send_cmd(DetectCommand::ForceFishComes);
                    }
                    let stop_start_label = if self.detection_enabled {
                        "Stop (F3)"
                    } else {
                        "Start (F3)"
                    };
                    if ui.button(stop_start_label).clicked() {
                        self.send_cmd(DetectCommand::ToggleStateMachine);
                    }
                    if ui.button("Quit").clicked() {
                        self.stop.store(true, Ordering::Relaxed);
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                self.draw_tab_selector(ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            DebugTab::Vision => self.draw_visual_tab(ui),
            DebugTab::Audio => self.draw_audio_tab(ui),
            DebugTab::Policy => self.draw_policy_tab(ui),
        });

        ctx.request_repaint_after(Duration::from_millis(REPAINT_MS));
    }
}

impl Drop for PreviewApp {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

struct GlobalHotkeys {
    stop: Arc<AtomicBool>,
    thread_id: Arc<AtomicU32>,
    handle: Option<thread::JoinHandle<()>>,
}

impl GlobalHotkeys {
    fn start(tx_cmd: Sender<DetectCommand>, stop: Arc<AtomicBool>) -> Self {
        let thread_id = Arc::new(AtomicU32::new(0));
        let thread_id_c = thread_id.clone();
        let stop_c = stop.clone();
        let handle = thread::spawn(move || {
            let tid = unsafe { GetCurrentThreadId() };
            thread_id_c.store(tid, Ordering::Relaxed);

            let id_reset = 1i32;
            let id_force = 2i32;
            let id_start = 3i32;

            let mods = HOT_KEY_MODIFIERS(MOD_NOREPEAT.0);
            let ok1 = unsafe { RegisterHotKey(None, id_reset, mods, VK_F1.0 as u32) }.is_ok();
            let ok2 = unsafe { RegisterHotKey(None, id_force, mods, VK_F2.0 as u32) }.is_ok();
            let ok3 = unsafe { RegisterHotKey(None, id_start, mods, VK_F3.0 as u32) }.is_ok();
            if ok1 && ok2 && ok3 {
                info!("global hotkeys registered: F1=Reset, F2=Force Comes, F3=Start/Stop");
            } else {
                warn!(
                    f1_ok = ok1,
                    f2_ok = ok2,
                    f3_ok = ok3,
                    "global hotkeys registration failed/partial (key may be occupied by other app)"
                );
            }

            while !stop_c.load(Ordering::Relaxed) {
                let mut msg = MSG::default();
                let ret = unsafe { GetMessageW(&mut msg, None, 0, 0) };
                if ret.0 <= 0 {
                    break;
                }
                if msg.message == WM_HOTKEY {
                    match msg.wParam.0 as i32 {
                        1 => {
                            info!("hotkey F1 -> Reset");
                            let _ = tx_cmd.try_send(DetectCommand::Reset);
                        }
                        2 => {
                            info!("hotkey F2 -> ForceFishComes");
                            let _ = tx_cmd.try_send(DetectCommand::ForceFishComes);
                        }
                        3 => {
                            info!("hotkey F3 -> ToggleStateMachine");
                            let _ = tx_cmd.try_send(DetectCommand::ToggleStateMachine);
                        }
                        _ => {}
                    }
                }
                unsafe {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }

            let _ = unsafe { UnregisterHotKey(None, id_reset) };
            let _ = unsafe { UnregisterHotKey(None, id_force) };
            let _ = unsafe { UnregisterHotKey(None, id_start) };
        });

        Self {
            stop,
            thread_id,
            handle: Some(handle),
        }
    }
}

impl Drop for GlobalHotkeys {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let tid = self.thread_id.load(Ordering::Relaxed);
        if tid != 0 {
            let _ = unsafe { PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0)) };
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn bgr_to_color_image(w: usize, h: usize, bgr: &[u8]) -> egui::ColorImage {
    let mut rgba = vec![0u8; w * h * 4];
    for (src, dst) in bgr.chunks_exact(3).zip(rgba.chunks_exact_mut(4)) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = 255;
    }
    egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba)
}

pub fn run_ui(
    rx: Receiver<DetectPacket>,
    tx_cmd: Sender<DetectCommand>,
    stop: Arc<AtomicBool>,
    cfg: Arc<AppConfig>,
) -> Result<()> {
    let _hotkeys = GlobalHotkeys::start(tx_cmd.clone(), stop.clone());

    let window_title = format!("FISH! Bot v{}", env!("CARGO_PKG_VERSION"));
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(window_title.clone())
            .with_inner_size([800.0, 500.0]),
        ..Default::default()
    };

    let app_stop = stop.clone();
    eframe::run_native(
        &window_title,
        native_options,
        Box::new(move |_cc| {
            Ok(Box::new(PreviewApp::new(
                rx,
                tx_cmd,
                app_stop.clone(),
                cfg.clone(),
            )))
        }),
    )
    .map_err(|e| anyhow!("egui ui failed: {e}"))?;

    Ok(())
}
