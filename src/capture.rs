use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Sender, TrySendError};
use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::{GraphicsCaptureApi, InternalCaptureControl};
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};
use windows_capture::window::Window;
use tracing::{info, warn};

use crate::config::AppConfig;
use crate::types::FramePacket;
use crate::vrc_window::{clear_target_hwnd, set_target_hwnd};

#[derive(Clone)]
struct CaptureFlags {
    tx: Sender<FramePacket>,
    stop: Arc<AtomicBool>,
    min_interval: Duration,
    min_interval_us: Arc<AtomicU64>,
    restart_generation: Arc<AtomicU64>,
    session_generation: u64,
}

struct CaptureWorker {
    tx: Sender<FramePacket>,
    stop: Arc<AtomicBool>,
    min_interval_us: Arc<AtomicU64>,
    restart_generation: Arc<AtomicU64>,
    session_generation: u64,
    last_sent: Instant,
    scratch: Vec<u8>,
}

pub fn capture_min_interval(cfg: &AppConfig) -> Duration {
    let capture_interval = Duration::from_millis(cfg.loop_timing.capture_sleep_ms);
    let detect_fps = cfg.state_machine.fishing_detect_fps_limit.max(1.0);
    let detect_interval = Duration::from_secs_f32(1.0 / detect_fps);
    capture_interval.max(detect_interval)
}

impl GraphicsCaptureApiHandler for CaptureWorker {
    type Flags = CaptureFlags;
    type Error = anyhow::Error;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            tx: ctx.flags.tx,
            stop: ctx.flags.stop,
            min_interval_us: ctx.flags.min_interval_us,
            restart_generation: ctx.flags.restart_generation,
            session_generation: ctx.flags.session_generation,
            last_sent: Instant::now() - ctx.flags.min_interval,
            scratch: Vec::new(),
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        if self.restart_generation.load(Ordering::Relaxed) != self.session_generation {
            let _ = capture_control.stop();
            return Ok(());
        }

        if self.stop.load(Ordering::Relaxed) {
            let _ = capture_control.stop();
            return Ok(());
        }

        let min_interval =
            Duration::from_micros(self.min_interval_us.load(Ordering::Relaxed).max(1));
        if self.last_sent.elapsed() < min_interval {
            return Ok(());
        }
        self.last_sent = Instant::now();

        let mut buf = frame.buffer()?;
        let w_u = buf.width() as usize;
        let h_u = buf.height() as usize;
        if w_u == 0 || h_u == 0 {
            return Ok(());
        }
        let row_pitch = buf.row_pitch() as usize;
        let Some(row_bytes) = w_u.checked_mul(4) else {
            warn!(w = w_u, "drop frame: row size overflow");
            return Ok(());
        };
        if row_pitch < row_bytes {
            warn!(row_pitch, row_bytes, "drop frame: invalid row_pitch");
            return Ok(());
        }
        let Some(packed_len) = row_bytes.checked_mul(h_u) else {
            warn!(w = w_u, h = h_u, "drop frame: packed size overflow");
            return Ok(());
        };
        let Some(raw_needed) = row_pitch.checked_mul(h_u) else {
            warn!(row_pitch, h = h_u, "drop frame: raw size overflow");
            return Ok(());
        };

        let raw = buf.as_raw_buffer();
        if raw.len() < raw_needed {
            warn!(
                got = raw.len(),
                need = raw_needed,
                w = w_u,
                h = h_u,
                row_pitch,
                "drop frame: short raw buffer during resize"
            );
            return Ok(());
        }

        if self.scratch.len() != packed_len {
            self.scratch.resize(packed_len, 0);
        }
        if row_pitch == row_bytes {
            self.scratch[..packed_len].copy_from_slice(&raw[..packed_len]);
        } else {
            for y in 0..h_u {
                let src_off = y * row_pitch;
                let dst_off = y * row_bytes;
                self.scratch[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&raw[src_off..src_off + row_bytes]);
            }
        }

        let w = w_u as i32;
        let h = h_u as i32;
        let pkt = FramePacket {
            w,
            h,
            bgra: std::mem::take(&mut self.scratch),
            captured_at: Instant::now(),
        };
        match self.tx.try_send(pkt) {
            Ok(()) => {}
            Err(TrySendError::Full(pkt)) | Err(TrySendError::Disconnected(pkt)) => {
                // Recover the owned buffer so next frame can reuse allocation.
                self.scratch = pkt.bgra;
            }
        }
        Ok(())
    }
}

pub fn run_capture(
    _cfg: Arc<AppConfig>,
    tx: Sender<FramePacket>,
    stop: Arc<AtomicBool>,
    min_interval_us: Arc<AtomicU64>,
    restart_generation: Arc<AtomicU64>,
) -> Result<()> {
    let mut last_no_window_log = Instant::now() - Duration::from_secs(10);

    while !stop.load(Ordering::Relaxed) {
        let window = match Window::from_name("VRChat") {
            Ok(w) => w,
            Err(_) => {
                clear_target_hwnd();
                if last_no_window_log.elapsed() >= Duration::from_secs(2) {
                    warn!("VRChat window not found");
                    last_no_window_log = Instant::now();
                }
                thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        let hwnd = window.as_raw_hwnd();
        set_target_hwnd(hwnd);
        info!(hwnd = ?hwnd, "capture attached to VRChat window");

        let flags = CaptureFlags {
            tx: tx.clone(),
            stop: stop.clone(),
            min_interval: Duration::from_micros(min_interval_us.load(Ordering::Relaxed).max(1)),
            min_interval_us: min_interval_us.clone(),
            restart_generation: restart_generation.clone(),
            session_generation: restart_generation.load(Ordering::Relaxed),
        };

        let min_update_interval = if GraphicsCaptureApi::is_minimum_update_interval_supported().unwrap_or(false) {
            MinimumUpdateIntervalSettings::Custom(flags.min_interval)
        } else {
            warn!(
                min_interval = ?flags.min_interval,
                "GraphicsCaptureSession.MinUpdateInterval unsupported on this Windows version; falling back to default"
            );
            MinimumUpdateIntervalSettings::Default
        };

        let settings = Settings::new(
            window,
            CursorCaptureSettings::Default,
            DrawBorderSettings::Default,
            SecondaryWindowSettings::Default,
            min_update_interval,
            DirtyRegionSettings::Default,
            ColorFormat::Bgra8,
            flags,
        );

        if let Err(e) = CaptureWorker::start(settings) {
            if !stop.load(Ordering::Relaxed) {
                warn!(error = ?e, "capture session ended, retrying");
                thread::sleep(Duration::from_millis(120));
            }
        }
    }

    Ok(())
}

