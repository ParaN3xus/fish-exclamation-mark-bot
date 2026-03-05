use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::Sender;
use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
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
}

struct CaptureWorker {
    tx: Sender<FramePacket>,
    stop: Arc<AtomicBool>,
    min_interval: Duration,
    last_sent: Instant,
    scratch: Vec<u8>,
}

impl GraphicsCaptureApiHandler for CaptureWorker {
    type Flags = CaptureFlags;
    type Error = anyhow::Error;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            tx: ctx.flags.tx,
            stop: ctx.flags.stop,
            min_interval: ctx.flags.min_interval,
            last_sent: Instant::now() - ctx.flags.min_interval,
            scratch: Vec::new(),
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        if self.stop.load(Ordering::Relaxed) {
            let _ = capture_control.stop();
            return Ok(());
        }

        if self.last_sent.elapsed() < self.min_interval {
            return Ok(());
        }
        self.last_sent = Instant::now();

        let w = frame.width() as i32;
        let h = frame.height() as i32;
        let buf = frame.buffer()?;
        let raw = buf.as_nopadding_buffer(&mut self.scratch);
        let _ = self.tx.try_send(FramePacket {
            w,
            h,
            bgra: raw.to_vec(),
        });
        Ok(())
    }
}

pub fn run_capture(cfg: Arc<AppConfig>, tx: Sender<FramePacket>, stop: Arc<AtomicBool>) -> Result<()> {
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
            min_interval: Duration::from_millis(cfg.loop_timing.capture_sleep_ms),
        };

        let settings = Settings::new(
            window,
            CursorCaptureSettings::Default,
            DrawBorderSettings::Default,
            SecondaryWindowSettings::Default,
            MinimumUpdateIntervalSettings::Default,
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

