use serde::{Deserialize, Serialize};

use crate::policy::FishingObservation;

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn clamp01(v: f32) -> f32 {
    clamp(v, 0.0, 1.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FilterConfig {
    pub filter_alpha: f32,
    pub target_half_alpha: f32,
    pub max_pos_step_per_sec: f32,
    pub max_target_half_step_per_sec: f32,
    pub outlier_jump_threshold: f32,
    pub edge_zone: f32,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            filter_alpha: 0.35,
            target_half_alpha: 0.25,
            max_pos_step_per_sec: 2.8,
            max_target_half_step_per_sec: 1.2,
            outlier_jump_threshold: 0.45,
            edge_zone: 0.08,
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
        let fish = clamp01(obs.fish_center);
        let player = clamp01(obs.player_center);
        let target_half = clamp(obs.player_target_half_size, 0.0, 0.5);

        let Some(prev) = self.last else {
            return self.reset(obs);
        };

        let fish_f = self.filter_pos(prev.fish_center, fish, dt);
        let player_f = self.filter_pos(prev.player_center, player, dt);
        let target_half_f = self.filter_target_half(prev.player_target_half_size, target_half, dt);

        let out = FishingObservation {
            fish_center: fish_f,
            player_center: player_f,
            dt,
            player_target_half_size: target_half_f,
        };
        self.last = Some(out);
        out
    }

    fn filter_pos(&self, prev: f32, raw: f32, dt: f32) -> f32 {
        let mut candidate = raw;
        let jump = (candidate - prev).abs();
        let edge = self.cfg.edge_zone.clamp(0.0, 0.49);
        let near_edge = candidate <= edge || candidate >= 1.0 - edge;
        let prev_mid = prev > edge && prev < 1.0 - edge;

        if self.cfg.outlier_jump_threshold > 0.0
            && jump >= self.cfg.outlier_jump_threshold
            && near_edge
            && prev_mid
        {
            candidate = prev;
        }

        let max_step = self.cfg.max_pos_step_per_sec.max(0.0) * dt;
        let limited = prev + (candidate - prev).clamp(-max_step, max_step);
        let a = self.cfg.filter_alpha.clamp(0.0, 1.0);
        clamp01(prev + a * (limited - prev))
    }

    fn filter_target_half(&self, prev: f32, raw: f32, dt: f32) -> f32 {
        let max_step = self.cfg.max_target_half_step_per_sec.max(0.0) * dt;
        let limited = prev + (raw - prev).clamp(-max_step, max_step);
        let a = self.cfg.target_half_alpha.clamp(0.0, 1.0);
        clamp(prev + a * (limited - prev), 0.0, 0.5)
    }
}
