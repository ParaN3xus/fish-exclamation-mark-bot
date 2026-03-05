use serde::{Deserialize, Serialize};

use crate::policy::{FishingObservation, TimeOptimalBangBangPolicy};

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn clamp01(v: f32) -> f32 {
    clamp(v, 0.0, 1.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObservationCompletionConfig {
    pub player_speed: f32,
    pub gravity: f32,
    pub player_meas_gain: f32,
    pub fish_alpha: f32,
    pub fish_beta: f32,
    pub target_half_ema_alpha: f32,
}

impl Default for ObservationCompletionConfig {
    fn default() -> Self {
        Self {
            player_speed: 3.75,
            gravity: 1.25,
            player_meas_gain: 0.45,
            fish_alpha: 0.62,
            fish_beta: 0.09,
            target_half_ema_alpha: 0.08,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplerConfig {
    pub policy_fps: f32,
    pub filter_alpha: f32,
    pub target_half_alpha: f32,
    pub max_pos_step_per_sec: f32,
    pub max_target_half_step_per_sec: f32,
    pub outlier_jump_threshold: f32,
    pub edge_zone: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            policy_fps: 60.0,
            filter_alpha: 0.35,
            target_half_alpha: 0.25,
            max_pos_step_per_sec: 2.8,
            max_target_half_step_per_sec: 1.2,
            outlier_jump_threshold: 0.45,
            edge_zone: 0.08,
        }
    }
}

#[derive(Debug)]
struct ObservationFrameCompletion {
    player_speed: f32,
    gravity: f32,
    player_meas_gain: f32,
    fish_alpha: f32,
    fish_beta: f32,
    target_half_ema_alpha: f32,

    player_x: f32,
    player_v: f32,
    fish_x: f32,
    fish_v: f32,
    target_half: f32,
    initialized: bool,
}

impl Default for ObservationFrameCompletion {
    fn default() -> Self {
        Self::from_config(&ObservationCompletionConfig::default())
    }
}

impl ObservationFrameCompletion {
    fn from_config(cfg: &ObservationCompletionConfig) -> Self {
        Self {
            player_speed: cfg.player_speed,
            gravity: cfg.gravity,
            player_meas_gain: cfg.player_meas_gain,
            fish_alpha: cfg.fish_alpha,
            fish_beta: cfg.fish_beta,
            target_half_ema_alpha: cfg.target_half_ema_alpha,
            player_x: 0.5,
            player_v: 0.0,
            fish_x: 0.5,
            fish_v: 0.0,
            target_half: 0.0,
            initialized: false,
        }
    }

    fn reset(&mut self, obs: FishingObservation) -> FishingObservation {
        self.player_x = clamp01(obs.player_center);
        self.player_v = 0.0;
        self.fish_x = clamp01(obs.fish_center);
        self.fish_v = 0.0;
        self.target_half = clamp(obs.player_target_half_size, 0.0, 0.5);
        self.initialized = true;
        FishingObservation {
            fish_center: self.fish_x,
            player_center: self.player_x,
            dt: obs.dt.max(1e-6),
            player_target_half_size: self.target_half,
        }
    }

    fn predict_player(&self, dt: f32, action: i32) -> (f32, f32) {
        let mut v = self.player_v - self.gravity * dt;
        if action == 1 {
            v += self.player_speed * dt;
        }
        let x = self.player_x + v * dt;
        if x <= 0.0 {
            return (0.0, -0.3 * v);
        }
        if x >= 1.0 {
            return (1.0, -0.3 * v);
        }
        (x, v)
    }

    fn step(&mut self, raw_obs: FishingObservation, action: i32) -> FishingObservation {
        if !self.initialized {
            return self.reset(raw_obs);
        }

        let dt = raw_obs.dt.max(1e-6);

        self.target_half = clamp(
            (1.0 - self.target_half_ema_alpha) * self.target_half
                + self.target_half_ema_alpha * clamp(raw_obs.player_target_half_size, 0.0, 0.5),
            0.0,
            0.5,
        );

        let (player_pred_x, player_pred_v) = self.predict_player(dt, action);
        let player_innov = clamp01(raw_obs.player_center) - player_pred_x;
        self.player_x = clamp01(player_pred_x + self.player_meas_gain * player_innov);
        self.player_v = player_pred_v + (self.player_x - player_pred_x) / dt;

        let fish_pred_x = self.fish_x + self.fish_v * dt;
        let fish_innov = clamp01(raw_obs.fish_center) - fish_pred_x;
        self.fish_x = clamp01(fish_pred_x + self.fish_alpha * fish_innov);
        self.fish_v = clamp(self.fish_v + (self.fish_beta * fish_innov) / dt, -6.0, 6.0);

        FishingObservation {
            fish_center: self.fish_x,
            player_center: self.player_x,
            dt,
            player_target_half_size: self.target_half,
        }
    }
}

#[derive(Debug)]
pub struct PolicyActionSampler {
    policy: TimeOptimalBangBangPolicy,
    policy_dt: f32,
    elapsed_since_decision: f32,
    action: i32,
    initialized: bool,
    buffer: Vec<FishingObservation>,
    completion: ObservationFrameCompletion,
    filter_cfg: SamplerConfig,
    filtered_last: Option<FishingObservation>,
}

impl PolicyActionSampler {
    pub fn new_with_completion(
        policy: TimeOptimalBangBangPolicy,
        sampler_cfg: SamplerConfig,
        completion_cfg: ObservationCompletionConfig,
    ) -> Self {
        let fps = sampler_cfg.policy_fps.max(1.0);
        Self {
            policy,
            policy_dt: 1.0 / fps,
            elapsed_since_decision: 0.0,
            action: 0,
            initialized: false,
            buffer: Vec::new(),
            completion: ObservationFrameCompletion::from_config(&completion_cfg),
            filter_cfg: sampler_cfg,
            filtered_last: None,
        }
    }

    fn with_dt(obs: FishingObservation, dt: f32) -> FishingObservation {
        FishingObservation {
            fish_center: obs.fish_center,
            player_center: obs.player_center,
            dt: dt.max(1e-6),
            player_target_half_size: obs.player_target_half_size,
        }
    }

    pub fn reset(&mut self, initial_obs: FishingObservation) -> i32 {
        self.policy.reset();
        self.elapsed_since_decision = 0.0;
        self.buffer.clear();
        let filtered = self.filter_observation(initial_obs);
        let completed = self.completion.reset(filtered);
        let first_dt = filtered.dt.max(self.policy_dt);
        self.action = self.policy.act(Self::with_dt(completed, first_dt));
        self.initialized = true;
        self.action
    }

    pub fn observe(&mut self, obs: FishingObservation) -> i32 {
        if !self.initialized {
            return self.reset(obs);
        }

        let filtered = self.filter_observation(obs);
        let env_dt = filtered.dt.max(1e-6);
        let completed = self.completion.step(filtered, self.action);
        self.buffer.push(completed);
        self.elapsed_since_decision += env_dt;

        let effective_policy_dt = self.policy_dt.max(env_dt);
        if self.elapsed_since_decision + 1e-12 < effective_policy_dt {
            return self.action;
        }

        while self.elapsed_since_decision + 1e-12 >= effective_policy_dt {
            if self.buffer.is_empty() {
                break;
            }
            let mut next_action = self.action;
            for frame_obs in &self.buffer {
                next_action = self.policy.act(*frame_obs);
            }
            self.action = next_action;
            self.buffer.clear();
            self.elapsed_since_decision =
                (self.elapsed_since_decision - effective_policy_dt).max(0.0);
        }

        self.action
    }

    pub fn last_filtered_observation(&self) -> Option<FishingObservation> {
        self.filtered_last
    }

    fn filter_observation(&mut self, obs: FishingObservation) -> FishingObservation {
        let dt = obs.dt.max(1e-4);
        let fish = clamp01(obs.fish_center);
        let player = clamp01(obs.player_center);
        let target_half = clamp(obs.player_target_half_size, 0.0, 0.5);

        let Some(prev) = self.filtered_last else {
            let init = FishingObservation {
                fish_center: fish,
                player_center: player,
                dt,
                player_target_half_size: target_half,
            };
            self.filtered_last = Some(init);
            return init;
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
        self.filtered_last = Some(out);
        out
    }

    fn filter_pos(&self, prev: f32, raw: f32, dt: f32) -> f32 {
        let mut candidate = raw;
        let jump = (candidate - prev).abs();
        let edge = self.filter_cfg.edge_zone.clamp(0.0, 0.49);
        let near_edge = candidate <= edge || candidate >= 1.0 - edge;
        let prev_mid = prev > edge && prev < 1.0 - edge;

        if self.filter_cfg.outlier_jump_threshold > 0.0
            && jump >= self.filter_cfg.outlier_jump_threshold
            && near_edge
            && prev_mid
        {
            candidate = prev;
        }

        let max_step = self.filter_cfg.max_pos_step_per_sec.max(0.0) * dt;
        let limited = prev + (candidate - prev).clamp(-max_step, max_step);
        let a = self.filter_cfg.filter_alpha.clamp(0.0, 1.0);
        clamp01(prev + a * (limited - prev))
    }

    fn filter_target_half(&self, prev: f32, raw: f32, dt: f32) -> f32 {
        let max_step = self.filter_cfg.max_target_half_step_per_sec.max(0.0) * dt;
        let limited = prev + (raw - prev).clamp(-max_step, max_step);
        let a = self.filter_cfg.target_half_alpha.clamp(0.0, 1.0);
        clamp(prev + a * (limited - prev), 0.0, 0.5)
    }
}
