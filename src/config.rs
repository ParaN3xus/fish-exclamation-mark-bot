use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::filter::FilterConfig;
use crate::policy::PolicyConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub startup: StartupConfig,
    pub paths: PathsConfig,
    pub yolo: YoloConfig,
    pub loop_timing: LoopTimingConfig,
    pub state_machine: StateMachineConfig,
    pub vision: VisionConfig,
    pub audio: AudioConfig,
    pub control: ControlConfig,
    pub filter: FilterConfig,
    pub policy: PolicyConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            startup: StartupConfig::default(),
            paths: PathsConfig::default(),
            yolo: YoloConfig::default(),
            loop_timing: LoopTimingConfig::default(),
            state_machine: StateMachineConfig::default(),
            vision: VisionConfig::default(),
            audio: AudioConfig::default(),
            control: ControlConfig::default(),
            filter: FilterConfig::default(),
            policy: PolicyConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn model_path(&self, exe_dir: &Path) -> PathBuf {
        let p = PathBuf::from(&self.paths.model_path);
        if p.is_absolute() {
            return p;
        }
        let assets_prefix = PathBuf::from(&self.paths.assets_dir);
        if p.starts_with(&assets_prefix) {
            return resolve_from_exe_path(exe_dir, &p);
        }
        self.asset_path(exe_dir, &self.paths.model_path)
    }

    pub fn bite_template_path(&self, exe_dir: &Path) -> PathBuf {
        self.asset_path(exe_dir, &self.paths.bite_template)
    }

    pub fn success_template_path(&self, exe_dir: &Path) -> PathBuf {
        self.asset_path(exe_dir, &self.paths.success_template)
    }

    pub fn fail_template_path(&self, exe_dir: &Path) -> PathBuf {
        self.asset_path(exe_dir, &self.paths.fail_template)
    }

    pub fn collected_template_path(&self, exe_dir: &Path) -> PathBuf {
        self.asset_path(exe_dir, &self.paths.collected_template)
    }

    fn asset_path(&self, exe_dir: &Path, file_name: &str) -> PathBuf {
        let rel = PathBuf::from(&self.paths.assets_dir).join(file_name);
        resolve_from_exe_path(exe_dir, &rel)
    }
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub path: PathBuf,
    pub app: AppConfig,
}

pub fn load_or_create_config() -> Result<LoadedConfig> {
    let exe = std::env::current_exe().context("failed to get current_exe")?;
    let exe_dir = exe
        .parent()
        .map(Path::to_path_buf)
        .context("failed to resolve exe directory")?;
    let cfg_path = exe_dir.join("config.toml");

    if !cfg_path.exists() {
        let cfg = AppConfig::default();
        let text = toml::to_string_pretty(&cfg).context("serialize default config")?;
        fs::write(&cfg_path, text).with_context(|| format!("write {}", cfg_path.display()))?;
        return Ok(LoadedConfig {
            path: cfg_path,
            app: cfg,
        });
    }

    let raw =
        fs::read_to_string(&cfg_path).with_context(|| format!("read {}", cfg_path.display()))?;
    let cfg: AppConfig = toml::from_str(&raw).with_context(|| {
        format!(
            "parse {} as toml; please fix file format",
            cfg_path.display()
        )
    })?;

    Ok(LoadedConfig {
        path: cfg_path,
        app: cfg,
    })
}

fn resolve_from_exe_path(exe_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        exe_dir.join(path)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StartupConfig {
    pub start_stopped: bool,
    pub ui_window_width: f32,
    pub ui_window_height: f32,
}

impl Default for StartupConfig {
    fn default() -> Self {
        Self {
            start_stopped: false,
            ui_window_width: 800.0,
            ui_window_height: 500.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PathsConfig {
    pub model_path: String,
    pub assets_dir: String,
    pub bite_template: String,
    pub success_template: String,
    pub fail_template: String,
    pub collected_template: String,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            model_path: "best.onnx".to_string(),
            assets_dir: "assets".to_string(),
            bite_template: "bite.ogg".to_string(),
            success_template: "success.ogg".to_string(),
            fail_template: "fail.ogg".to_string(),
            collected_template: "collected.ogg".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct YoloConfig {
    pub class_id: i32,
    pub conf: f32,
    pub imgsz: i32,
}

impl Default for YoloConfig {
    fn default() -> Self {
        Self {
            class_id: 0,
            conf: 0.15,
            imgsz: 640,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoopTimingConfig {
    pub capture_sleep_ms: u64,
    pub detect_sleep_ms: u64,
    pub detect_recv_timeout_ms: u64,
}

impl Default for LoopTimingConfig {
    fn default() -> Self {
        Self {
            capture_sleep_ms: 4,
            detect_sleep_ms: 1,
            detect_recv_timeout_ms: 200,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StateMachineConfig {
    pub second_detect_delay_ms: u64,
    pub wait_bite_timeout_ms: u64,
    pub reel_first_det_timeout_ms: u64,
    pub fishing_timeout_ms: u64,
    pub fishing_yolo_check_ms: u64,
    pub fishing_detect_fps_limit: f32,
    pub result_action_wait_ms: u64,
}

impl Default for StateMachineConfig {
    fn default() -> Self {
        Self {
            second_detect_delay_ms: 160,
            wait_bite_timeout_ms: 34_000,
            reel_first_det_timeout_ms: 2_000,
            fishing_timeout_ms: 60_000,
            fishing_yolo_check_ms: 500,
            fishing_detect_fps_limit: 60.0,
            result_action_wait_ms: 1_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub roi_edge_eps_px: i32,
    pub extend_min_visible_ratio: f32,
    pub extend_min_width_ratio: f32,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            roi_edge_eps_px: 2,
            extend_min_visible_ratio: 0.45,
            extend_min_width_ratio: 0.35,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub enable_global_audio_listener: bool,
    pub fft_size: usize,
    pub hop: usize,
    pub bar_count: usize,
    pub live_seconds: f32,
    pub poll_ms: u64,
    pub bite_threshold: f32,
    pub success_threshold: f32,
    pub fail_threshold: f32,
    pub collected_threshold: f32,
    pub min_energy: f32,
    pub trigger_cooldown_ms: u64,
    pub template_prefix_seconds: f32,
    pub loudness_target_rms: f32,
    pub loudness_gain_min: f32,
    pub loudness_gain_max: f32,
    pub trim_peak_ratio: f32,
    pub trim_floor: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enable_global_audio_listener: false,
            fft_size: 2048,
            hop: 512,
            bar_count: 48,
            live_seconds: 0.3,
            poll_ms: 20,
            bite_threshold: 0.65,
            success_threshold: 0.70,
            fail_threshold: 0.65,
            collected_threshold: 0.8,
            min_energy: 0.00007,
            trigger_cooldown_ms: 450,
            template_prefix_seconds: 0.2,
            loudness_target_rms: 0.10,
            loudness_gain_min: 0.35,
            loudness_gain_max: 6.0,
            trim_peak_ratio: 0.06,
            trim_floor: 0.002,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ControlConfig {
    pub osc_target_host: String,
    pub osc_target_port: u16,
    pub click_hold_ms: u64,
    pub jump_press_time_s: f32,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            osc_target_host: "127.0.0.1".to_string(),
            osc_target_port: 9000,
            click_hold_ms: 50,
            jump_press_time_s: 0.01,
        }
    }
}
