use serde::{Deserialize, Serialize};

use crate::policy::FishingObservation;

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn clamp01(v: f32) -> f32 {
    clamp(v, 0.0, 1.0)
}

fn alpha_from_tau(dt: f32, tau_sec: f32) -> f32 {
    let dt = dt.max(1e-4);
    let tau = tau_sec.max(1e-5);
    1.0 - (-dt / tau).exp()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FilterConfig {
    pub fish_tau_sec: f32,
    pub player_tau_sec: f32,
    pub target_half_tau_sec: f32,
    pub max_fish_step_per_sec: f32,
    pub max_player_step_per_sec: f32,
    pub max_target_half_step_per_sec: f32,
    pub outlier_jump_base: f32,
    pub outlier_jump_per_sec: f32,
    pub edge_zone: f32,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            fish_tau_sec: 0.020,
            player_tau_sec: 0.024,
            target_half_tau_sec: 0.120,
            max_fish_step_per_sec: 6.0,
            max_player_step_per_sec: 3.3,
            max_target_half_step_per_sec: 0.35,
            outlier_jump_base: 0.06,
            outlier_jump_per_sec: 2.4,
            edge_zone: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ObservationFilter {
    cfg: FilterConfig,
    last: Option<FishingObservation>,
}

impl ObservationFilter {
    pub fn new(cfg: FilterConfig) -> Self {
        Self { cfg, last: None }
    }

    pub fn reset(&mut self, obs: FishingObservation) -> FishingObservation {
        let out = FishingObservation {
            fish_center: clamp01(obs.fish_center),
            player_center: clamp01(obs.player_center),
            dt: obs.dt.max(1e-4),
            player_target_half_size: clamp(obs.player_target_half_size, 0.0, 0.5),
        };
        self.last = Some(out);
        out
    }

    pub fn apply(&mut self, obs: FishingObservation) -> FishingObservation {
        let dt = obs.dt.max(1e-4);
        let fish_raw = clamp01(obs.fish_center);
        let player_raw = clamp01(obs.player_center);
        let target_half_raw = clamp(obs.player_target_half_size, 0.0, 0.5);

        let Some(prev) = self.last else {
            return self.reset(obs);
        };

        let fish_candidate = self.reject_edge_outlier(prev.fish_center, fish_raw, dt);
        let player_candidate = self.reject_edge_outlier(prev.player_center, player_raw, dt);

        let fish_f = self.filter_pos_channel(
            prev.fish_center,
            fish_candidate,
            dt,
            self.cfg.fish_tau_sec,
            self.cfg.max_fish_step_per_sec,
        );
        let player_f = self.filter_pos_channel(
            prev.player_center,
            player_candidate,
            dt,
            self.cfg.player_tau_sec,
            self.cfg.max_player_step_per_sec,
        );
        let target_half_f =
            self.filter_target_half(prev.player_target_half_size, target_half_raw, dt);

        let out = FishingObservation {
            fish_center: fish_f,
            player_center: player_f,
            dt,
            player_target_half_size: target_half_f,
        };
        self.last = Some(out);
        out
    }

    fn reject_edge_outlier(&self, prev: f32, raw: f32, dt: f32) -> f32 {
        let edge = self.cfg.edge_zone.clamp(0.0, 0.49);
        let near_edge = raw <= edge || raw >= 1.0 - edge;
        let prev_mid = prev > edge && prev < 1.0 - edge;
        let jump = (raw - prev).abs();
        let jump_threshold =
            self.cfg.outlier_jump_base.max(0.0) + self.cfg.outlier_jump_per_sec.max(0.0) * dt;

        if jump >= jump_threshold && near_edge && prev_mid {
            prev
        } else {
            raw
        }
    }

    fn filter_pos_channel(
        &self,
        prev: f32,
        raw: f32,
        dt: f32,
        tau_sec: f32,
        max_step_per_sec: f32,
    ) -> f32 {
        let max_step = max_step_per_sec.max(0.0) * dt;
        let limited = prev + (raw - prev).clamp(-max_step, max_step);
        let a = alpha_from_tau(dt, tau_sec).clamp(0.0, 1.0);
        clamp01(prev + a * (limited - prev))
    }

    fn filter_target_half(&self, prev: f32, raw: f32, dt: f32) -> f32 {
        let max_step = self.cfg.max_target_half_step_per_sec.max(0.0) * dt;
        let limited = prev + (raw - prev).clamp(-max_step, max_step);
        let a = alpha_from_tau(dt, self.cfg.target_half_tau_sec).clamp(0.0, 1.0);
        clamp(prev + a * (limited - prev), 0.0, 0.5)
    }
}
