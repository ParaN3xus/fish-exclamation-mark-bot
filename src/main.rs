use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use anyhow::Result;
use crossbeam_channel::bounded;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

mod audio;
mod capture;
mod config;
mod control;
mod detect;
mod filter;
mod policy;
mod types;
mod ui;
mod vision;
mod vrc_window;

use capture::run_capture;
use config::load_or_create_config;
use detect::run_detect;
use types::{DetectCommand, DetectPacket, FramePacket};
use ui::run_ui;
use vision::init_ort_runtime;

fn main() -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    let loaded = load_or_create_config()?;
    let cfg_path = loaded.path.clone();
    let cfg = Arc::new(loaded.app);
    info!(path = %loaded.path.display(), "config loaded");
    info!("starting bot");
    init_ort_runtime()?;
    let stop = Arc::new(AtomicBool::new(false));
    let (tx_cap, rx_cap) = bounded::<FramePacket>(1);
    let (tx_det, rx_det) = bounded::<DetectPacket>(1);
    let (tx_det_buf, rx_det_buf) = bounded::<Vec<u8>>(1);
    let (tx_cmd, rx_cmd) = bounded::<DetectCommand>(8);

    let t1_stop = stop.clone();
    let t1_cfg = cfg.clone();
    let t1 = thread::spawn(move || {
        if let Err(e) = run_capture(t1_cfg, tx_cap, t1_stop.clone()) {
            error!(error = ?e, "capture thread error");
            t1_stop.store(true, Ordering::Relaxed);
        }
    });

    let t2_stop = stop.clone();
    let t2_cfg = cfg.clone();
    let t2 = thread::spawn(move || {
        if let Err(e) = run_detect(t2_cfg, rx_cap, tx_det, rx_det_buf, rx_cmd, t2_stop.clone()) {
            error!(error = ?e, "detect thread error");
            t2_stop.store(true, Ordering::Relaxed);
        }
    });

    let ui_res = run_ui(rx_det, tx_det_buf, tx_cmd, stop.clone(), cfg, cfg_path);
    stop.store(true, Ordering::Relaxed);

    let _ = t1.join();
    let _ = t2.join();
    ui_res
}

