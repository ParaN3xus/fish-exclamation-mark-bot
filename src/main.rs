use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::bounded;
use notify::{RecursiveMode, Watcher};
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

use capture::{capture_min_interval, run_capture};
use config::load_or_create_config;
use detect::run_detect;
use types::{DetectCommand, DetectPacket, FramePacket};
use ui::run_ui;
use vision::init_ort_runtime;

fn start_config_watcher(
    cfg_path: std::path::PathBuf,
    tx_cmd: crossbeam_channel::Sender<DetectCommand>,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let cfg_name = match cfg_path.file_name().map(|v| v.to_owned()) {
            Some(v) => v,
            None => return,
        };
        let watch_dir = cfg_path
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| std::path::PathBuf::from("."));

        let (tx_evt, rx_evt) = mpsc::channel();
        let mut watcher = match notify::recommended_watcher(move |res| {
            let _ = tx_evt.send(res);
        }) {
            Ok(w) => w,
            Err(e) => {
                error!(error = ?e, "config watcher init failed");
                return;
            }
        };

        if let Err(e) = watcher.watch(&watch_dir, RecursiveMode::NonRecursive) {
            error!(error = ?e, path = %watch_dir.display(), "config watcher start failed");
            return;
        }
        info!(path = %cfg_path.display(), "config watcher started");

        let mut last_signal = Instant::now() - Duration::from_millis(300);
        while !stop.load(Ordering::Relaxed) {
            let evt = match rx_evt.recv_timeout(Duration::from_millis(200)) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let Ok(evt) = evt else { continue };
            if !evt.paths.iter().any(|p| p.file_name() == Some(&cfg_name)) {
                continue;
            }
            if last_signal.elapsed() < Duration::from_millis(300) {
                continue;
            }
            last_signal = Instant::now();
            let _ = tx_cmd.try_send(DetectCommand::ReloadConfig);
        }
    })
}

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
    let cap_min_interval_us = Arc::new(AtomicU64::new(
        capture_min_interval(cfg.as_ref()).as_micros() as u64,
    ));
    let cap_restart_generation = Arc::new(AtomicU64::new(0));
    let (tx_cap, rx_cap) = bounded::<FramePacket>(1);
    let (tx_det, rx_det) = bounded::<DetectPacket>(1);
    let (tx_det_buf, rx_det_buf) = bounded::<Vec<u8>>(1);
    let (tx_cmd, rx_cmd) = bounded::<DetectCommand>(8);

    let t1_stop = stop.clone();
    let t1_cfg = cfg.clone();
    let t1_cap_min_interval_us = cap_min_interval_us.clone();
    let t1_cap_restart_generation = cap_restart_generation.clone();
    let t1 = thread::spawn(move || {
        if let Err(e) = run_capture(
            t1_cfg,
            tx_cap,
            t1_stop.clone(),
            t1_cap_min_interval_us,
            t1_cap_restart_generation,
        ) {
            error!(error = ?e, "capture thread error");
            t1_stop.store(true, Ordering::Relaxed);
        }
    });

    let t2_stop = stop.clone();
    let t2_cfg = cfg.clone();
    let t2_cfg_path = cfg_path.clone();
    let t2 = thread::spawn(move || {
        if let Err(e) = run_detect(
            t2_cfg,
            t2_cfg_path,
            cap_min_interval_us.clone(),
            cap_restart_generation.clone(),
            rx_cap,
            tx_det,
            rx_det_buf,
            rx_cmd,
            t2_stop.clone(),
        ) {
            error!(error = ?e, "detect thread error");
            t2_stop.store(true, Ordering::Relaxed);
        }
    });

    let watcher = start_config_watcher(cfg_path.clone(), tx_cmd.clone(), stop.clone());

    let ui_res = run_ui(rx_det, tx_det_buf, tx_cmd, stop.clone(), cfg, cfg_path);
    stop.store(true, Ordering::Relaxed);

    let _ = t1.join();
    let _ = t2.join();
    let _ = watcher.join();
    ui_res
}

