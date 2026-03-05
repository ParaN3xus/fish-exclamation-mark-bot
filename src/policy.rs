use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct FishingObservation {
    pub fish_center: f32,
    pub player_center: f32,
    pub dt: f32,
    pub player_target_half_size: f32,
}

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn clamp01(v: f32) -> f32 {
    clamp(v, 0.0, 1.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PolicyConfig {
    pub player_speed: f32,
    pub gravity: f32,
    pub equipment_strength: i32,
    pub equipment_expertise: i32,
    pub initial_difficulty_estimate: f32,
    pub difficulty_up_blend: f32,
    pub difficulty_down_blend: f32,
    pub robust_horizon_base_seconds: f32,
    pub robust_horizon_danger_seconds: f32,
    pub robust_horizon_min_steps: usize,
    pub robust_horizon_max_steps: usize,
    pub robust_risk_base: f32,
    pub robust_risk_danger_gain: f32,
    pub robust_risk_dnorm_gain: f32,
    pub robust_risk_max: f32,
    pub danger_progress_ref: f32,
    pub target_bias_base: f32,
    pub target_bias_danger_gain: f32,
    pub recovery_band_base: f32,
    pub recovery_band_danger_gain: f32,
    pub safe_band_base: f32,
    pub safe_band_danger_gain: f32,
    pub one_step_override_danger_threshold: f32,
    pub robust_eval_difficulty_threshold: f32,
    pub robust_eval_danger_threshold: f32,
    pub robust_inner_target_bias_base: f32,
    pub robust_inner_target_bias_danger_gain: f32,
    pub observed_threshold_scale: f32,
    pub observed_threshold_clip_max: f32,
    pub bar_height: f32,
    pub fish_target_hitbox_size: f32,
    pub easy_target_size: f32,
    pub hard_target_size: f32,
    pub easy_direction_time: f32,
    pub hard_direction_time: f32,
    pub easy_fish_smooth_time: f32,
    pub hard_fish_smooth_time: f32,
    pub easy_catch_speed: f32,
    pub hard_catch_speed: f32,
    pub easy_lose_speed: f32,
    pub hard_lose_speed: f32,
    pub lose_speed_escalation_rate: f32,
    pub easy_max_lose_speed_multiplier: f32,
    pub hard_max_lose_speed_multiplier: f32,
    pub fish_target_half_size: f32,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            player_speed: 3.75,
            gravity: 1.25,
            equipment_strength: 0,
            equipment_expertise: 0,
            initial_difficulty_estimate: 6.2653747,
            difficulty_up_blend: 0.3154392,
            difficulty_down_blend: 0.0,
            robust_horizon_base_seconds: 0.2704534,
            robust_horizon_danger_seconds: 0.40064702,
            robust_horizon_min_steps: 13,
            robust_horizon_max_steps: 18,
            robust_risk_base: 0.64286464,
            robust_risk_danger_gain: 0.025346905,
            robust_risk_dnorm_gain: 0.11611094,
            robust_risk_max: 0.7921877,
            danger_progress_ref: 0.36775503,
            target_bias_base: 0.037549336,
            target_bias_danger_gain: 0.035614316,
            recovery_band_base: 3.0121188,
            recovery_band_danger_gain: 0.605336,
            safe_band_base: 0.81094545,
            safe_band_danger_gain: 0.10027581,
            one_step_override_danger_threshold: 0.06664068,
            robust_eval_difficulty_threshold: 5.974913,
            robust_eval_danger_threshold: 0.146039,
            robust_inner_target_bias_base: 0.028590992,
            robust_inner_target_bias_danger_gain: 0.08816427,
            observed_threshold_scale: 0.8483099,
            observed_threshold_clip_max: 0.258311,
            bar_height: 2.8,
            fish_target_hitbox_size: 0.1,
            easy_target_size: 1.2,
            hard_target_size: 0.7,
            easy_direction_time: 0.5,
            hard_direction_time: 0.4,
            easy_fish_smooth_time: 1.0,
            hard_fish_smooth_time: 0.19,
            easy_catch_speed: 0.2,
            hard_catch_speed: 0.06,
            easy_lose_speed: 0.1,
            hard_lose_speed: 0.15,
            lose_speed_escalation_rate: 0.1,
            easy_max_lose_speed_multiplier: 1.0,
            hard_max_lose_speed_multiplier: 3.0,
            fish_target_half_size: 0.1 / (2.8 * 2.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TimeOptimalBangBangPolicy {
    player_speed: f32,
    gravity: f32,
    vel_epsilon: f32,

    equipment_strength: i32,
    equipment_expertise: i32,

    initial_difficulty_estimate: f32,
    difficulty_up_blend: f32,
    difficulty_down_blend: f32,

    robust_horizon_base_seconds: f32,
    robust_horizon_danger_seconds: f32,
    robust_horizon_min_steps: usize,
    robust_horizon_max_steps: usize,

    robust_risk_base: f32,
    robust_risk_danger_gain: f32,
    robust_risk_dnorm_gain: f32,
    robust_risk_max: f32,

    danger_progress_ref: f32,
    target_bias_base: f32,
    target_bias_danger_gain: f32,
    recovery_band_base: f32,
    recovery_band_danger_gain: f32,
    safe_band_base: f32,
    safe_band_danger_gain: f32,

    one_step_override_danger_threshold: f32,
    robust_eval_difficulty_threshold: f32,
    robust_eval_danger_threshold: f32,
    robust_inner_target_bias_base: f32,
    robust_inner_target_bias_danger_gain: f32,

    observed_threshold_scale: f32,
    observed_threshold_clip_max: f32,

    a_up: f32,
    a_down: f32,

    bar_height: f32,
    fish_target_hitbox_size: f32,
    easy_target_size: f32,
    hard_target_size: f32,
    easy_direction_time: f32,
    hard_direction_time: f32,
    easy_fish_smooth_time: f32,
    hard_fish_smooth_time: f32,
    easy_catch_speed: f32,
    hard_catch_speed: f32,
    easy_lose_speed: f32,
    hard_lose_speed: f32,
    lose_speed_escalation_rate: f32,
    easy_max_lose_speed_multiplier: f32,
    hard_max_lose_speed_multiplier: f32,
    fish_target_half_size: f32,

    last_player_center: Option<f32>,
    last_fish_center: Option<f32>,
    v_est: f32,
    fish_v_est: f32,
    last_action: i32,

    fish_target_est: f32,
    time_since_target_change: f32,
    difficulty_est: f32,

    progress_est: f32,
    elapsed_est: f32,
    have_step_observation: bool,
}

impl TimeOptimalBangBangPolicy {
    pub fn from_config(cfg: &PolicyConfig) -> Self {
        let player_speed = cfg.player_speed;
        let gravity = cfg.gravity;
        let a_up = player_speed - gravity;
        let a_down = gravity;
        let initial_difficulty_estimate = clamp(cfg.initial_difficulty_estimate, 1.0, 9.0);

        Self {
            player_speed,
            gravity,
            vel_epsilon: 1e-6,
            equipment_strength: cfg.equipment_strength,
            equipment_expertise: cfg.equipment_expertise,
            initial_difficulty_estimate,
            difficulty_up_blend: clamp(cfg.difficulty_up_blend, 0.0, 1.0),
            difficulty_down_blend: clamp(cfg.difficulty_down_blend, 0.0, 1.0),
            robust_horizon_base_seconds: cfg.robust_horizon_base_seconds,
            robust_horizon_danger_seconds: cfg.robust_horizon_danger_seconds,
            robust_horizon_min_steps: cfg.robust_horizon_min_steps,
            robust_horizon_max_steps: cfg.robust_horizon_max_steps,
            robust_risk_base: clamp(cfg.robust_risk_base, 0.0, 1.0),
            robust_risk_danger_gain: cfg.robust_risk_danger_gain,
            robust_risk_dnorm_gain: cfg.robust_risk_dnorm_gain,
            robust_risk_max: clamp(cfg.robust_risk_max, 0.0, 0.999_999),
            danger_progress_ref: cfg.danger_progress_ref.max(1e-6),
            target_bias_base: cfg.target_bias_base,
            target_bias_danger_gain: cfg.target_bias_danger_gain,
            recovery_band_base: cfg.recovery_band_base,
            recovery_band_danger_gain: cfg.recovery_band_danger_gain,
            safe_band_base: cfg.safe_band_base,
            safe_band_danger_gain: cfg.safe_band_danger_gain,
            one_step_override_danger_threshold: cfg.one_step_override_danger_threshold,
            robust_eval_difficulty_threshold: cfg.robust_eval_difficulty_threshold,
            robust_eval_danger_threshold: cfg.robust_eval_danger_threshold,
            robust_inner_target_bias_base: cfg.robust_inner_target_bias_base,
            robust_inner_target_bias_danger_gain: cfg.robust_inner_target_bias_danger_gain,
            observed_threshold_scale: cfg.observed_threshold_scale.max(0.0),
            observed_threshold_clip_max: clamp(cfg.observed_threshold_clip_max, 0.017_857, 0.499_999),
            a_up,
            a_down,
            bar_height: cfg.bar_height,
            fish_target_hitbox_size: cfg.fish_target_hitbox_size,
            easy_target_size: cfg.easy_target_size,
            hard_target_size: cfg.hard_target_size,
            easy_direction_time: cfg.easy_direction_time,
            hard_direction_time: cfg.hard_direction_time,
            easy_fish_smooth_time: cfg.easy_fish_smooth_time,
            hard_fish_smooth_time: cfg.hard_fish_smooth_time,
            easy_catch_speed: cfg.easy_catch_speed,
            hard_catch_speed: cfg.hard_catch_speed,
            easy_lose_speed: cfg.easy_lose_speed,
            hard_lose_speed: cfg.hard_lose_speed,
            lose_speed_escalation_rate: cfg.lose_speed_escalation_rate,
            easy_max_lose_speed_multiplier: cfg.easy_max_lose_speed_multiplier,
            hard_max_lose_speed_multiplier: cfg.hard_max_lose_speed_multiplier,
            fish_target_half_size: cfg.fish_target_half_size,
            last_player_center: None,
            last_fish_center: None,
            v_est: 0.0,
            fish_v_est: 0.0,
            last_action: 0,
            fish_target_est: 0.5,
            time_since_target_change: 0.0,
            difficulty_est: initial_difficulty_estimate,
            progress_est: 0.1,
            elapsed_est: 0.0,
            have_step_observation: false,
        }
    }

    pub fn reset(&mut self) {
        self.last_player_center = None;
        self.last_fish_center = None;
        self.v_est = 0.0;
        self.fish_v_est = 0.0;
        self.last_action = 0;
        self.fish_target_est = 0.5;
        self.time_since_target_change = 0.0;
        self.difficulty_est = self.initial_difficulty_estimate;
        self.progress_est = 0.1;
        self.elapsed_est = 0.0;
        self.have_step_observation = false;
    }

    fn difficulty_normalized(&self, difficulty: f32) -> f32 {
        (clamp(difficulty, 1.0, 9.0) - 1.0) / 8.0
    }

    fn difficulty_scale(&self, difficulty: f32) -> f32 {
        self.difficulty_normalized(difficulty).powf(1.7)
    }

    fn direction_time(&self, difficulty: f32) -> f32 {
        lerp(
            self.easy_direction_time,
            self.hard_direction_time,
            self.difficulty_normalized(difficulty),
        )
    }

    fn fish_decay_rate(&self, difficulty: f32) -> f32 {
        let base_smooth_time = lerp(
            self.easy_fish_smooth_time,
            self.hard_fish_smooth_time,
            self.difficulty_normalized(difficulty),
        );
        let base_decay_rate = 1.0 / base_smooth_time.max(0.001);
        let clamped_strength = clamp(self.equipment_strength as f32, -100.0, 100.0);
        let strength_effect = (clamped_strength / 100.0) * self.difficulty_scale(difficulty);
        let strength_multiplier = clamp(1.0 - strength_effect, 0.1, 10.0);
        base_decay_rate * strength_multiplier
    }

    fn catch_speed(&self, difficulty: f32) -> f32 {
        lerp(
            self.easy_catch_speed,
            self.hard_catch_speed,
            self.difficulty_normalized(difficulty),
        )
    }

    fn base_lose_speed(&self, difficulty: f32) -> f32 {
        lerp(
            self.easy_lose_speed,
            self.hard_lose_speed,
            self.difficulty_normalized(difficulty),
        )
    }

    fn max_lose_multiplier(&self, difficulty: f32) -> f32 {
        lerp(
            self.easy_max_lose_speed_multiplier,
            self.hard_max_lose_speed_multiplier,
            self.difficulty_normalized(difficulty),
        )
    }

    fn overlap_threshold(&self, difficulty: f32) -> f32 {
        let d_norm = self.difficulty_normalized(difficulty);
        let difficulty_scale = self.difficulty_scale(difficulty);
        let clamped_exp = clamp(self.equipment_expertise as f32, -100.0, 100.0);
        let base_target_size = lerp(self.easy_target_size, self.hard_target_size, d_norm);
        let expertise_effect = (clamped_exp / 100.0) * difficulty_scale;
        let expertise_mul = (1.0 + expertise_effect).max(0.5);
        let player_target_size = base_target_size * expertise_mul;
        (self.fish_target_hitbox_size + player_target_size) / (self.bar_height * 2.0)
    }

    fn effective_overlap_threshold(&self, obs: FishingObservation, difficulty: f32) -> f32 {
        let estimated_threshold = self.overlap_threshold(difficulty);
        let observed_player_half = clamp(obs.player_target_half_size, 0.0, 0.5);
        if observed_player_half <= 0.0 {
            return estimated_threshold;
        }
        let observed_threshold =
            self.fish_target_half_size + self.observed_threshold_scale * observed_player_half;
        clamp(
            observed_threshold,
            self.fish_target_half_size,
            self.observed_threshold_clip_max,
        )
    }

    fn lose_speed_at(&self, elapsed_time: f32, difficulty: f32) -> f32 {
        let grace = clamp01((elapsed_time - 1.0) / 4.0);
        let escalation = (1.0 + elapsed_time * self.lose_speed_escalation_rate)
            .min(self.max_lose_multiplier(difficulty));
        self.base_lose_speed(difficulty) * escalation * grace
    }

    fn progress_delta(&self, catching: bool, elapsed_time: f32, difficulty: f32, dt: f32) -> f32 {
        if catching {
            self.catch_speed(difficulty) * dt
        } else {
            -self.lose_speed_at(elapsed_time, difficulty) * dt
        }
    }

    fn update_progress_estimate(&mut self, obs: FishingObservation) {
        if !self.have_step_observation {
            self.have_step_observation = true;
            return;
        }

        let dt = obs.dt.max(self.vel_epsilon);
        self.elapsed_est += dt;
        let difficulty = self.difficulty_est;
        let threshold = self.effective_overlap_threshold(obs, difficulty);
        let catching = (obs.fish_center - obs.player_center).abs() < threshold;
        let delta = self.progress_delta(catching, self.elapsed_est, difficulty, dt);
        self.progress_est = clamp01(self.progress_est + delta);
    }

    fn estimate_kinematics(&mut self, obs: FishingObservation) -> (f32, f32) {
        if self.last_player_center.is_none() || self.last_fish_center.is_none() {
            self.last_player_center = Some(obs.player_center);
            self.last_fish_center = Some(obs.fish_center);
            self.v_est = 0.0;
            self.fish_v_est = 0.0;
            self.fish_target_est = obs.fish_center;
            self.time_since_target_change = 0.0;
            return (0.0, 0.0);
        }

        let dt = obs.dt.max(self.vel_epsilon);
        let last_player_center = self.last_player_center.unwrap_or(obs.player_center);
        let last_fish_center = self.last_fish_center.unwrap_or(obs.fish_center);
        let raw_player_v = (obs.player_center - last_player_center) / dt;
        let raw_fish_v = (obs.fish_center - last_fish_center) / dt;
        self.v_est = 0.8 * self.v_est + 0.2 * raw_player_v;
        self.fish_v_est = 0.65 * self.fish_v_est + 0.35 * raw_fish_v;

        let difficulty = self.difficulty_est;
        let decay = self.fish_decay_rate(difficulty);
        let alpha = 1.0 - (-decay * dt).exp();
        let mut inferred_target = if alpha > 1e-6 {
            last_fish_center + (obs.fish_center - last_fish_center) / alpha
        } else {
            obs.fish_center
        };
        inferred_target = clamp01(inferred_target);

        let d_norm = self.difficulty_normalized(difficulty);
        let jump_detect_threshold = 0.035 + 0.065 * d_norm;
        let jump_distance = (inferred_target - self.fish_target_est).abs();
        if jump_distance > jump_detect_threshold {
            let observed_interval = dt.max(self.time_since_target_change + dt);
            let interval_norm = clamp01((self.easy_direction_time - observed_interval) / 0.1);
            let jump_norm = clamp01((jump_distance - 0.18) / 0.12);
            let obs_norm = 0.45 * interval_norm + 0.55 * jump_norm;
            let current_norm = self.difficulty_normalized(self.difficulty_est);
            let blended_norm = if obs_norm >= current_norm {
                (1.0 - self.difficulty_up_blend) * current_norm
                    + self.difficulty_up_blend * obs_norm
            } else {
                (1.0 - self.difficulty_down_blend) * current_norm
                    + self.difficulty_down_blend * obs_norm
            };
            self.difficulty_est = clamp(1.0 + 8.0 * blended_norm, 1.0, 9.0);
            self.time_since_target_change = 0.0;
        } else {
            self.time_since_target_change += dt;
        }

        self.fish_target_est = 0.7 * self.fish_target_est + 0.3 * inferred_target;
        self.last_player_center = Some(obs.player_center);
        self.last_fish_center = Some(obs.fish_center);

        (self.v_est, self.fish_v_est)
    }

    fn predict_fish_center(
        &self,
        fish_center: f32,
        difficulty: f32,
        danger: f32,
        fish_velocity: f32,
    ) -> f32 {
        let d_norm = self.difficulty_normalized(difficulty);
        let horizon = 0.10 + 0.14 * d_norm + 0.12 * danger;
        let decay = self.fish_decay_rate(difficulty);
        let alpha_h = 1.0 - (-decay * horizon).exp();

        let direction_time = self.direction_time(difficulty);
        let change_uncertainty = if direction_time <= 1e-6 {
            0.0
        } else {
            clamp01(
                (self.time_since_target_change - 0.72 * direction_time) / (0.35 * direction_time),
            )
        };

        let blended_target = lerp(
            self.fish_target_est,
            0.5 + 0.25 * clamp(fish_velocity, -1.0, 1.0),
            0.55 * change_uncertainty,
        );

        clamp01(fish_center + (blended_target - fish_center) * alpha_h)
    }

    fn max_fish_jump(&self, difficulty: f32) -> f32 {
        lerp(0.18, 0.3, self.difficulty_normalized(difficulty))
    }

    fn step_player(
        &self,
        player_center: f32,
        player_velocity: f32,
        dt: f32,
        action: i32,
    ) -> (f32, f32) {
        let mut v = player_velocity - self.gravity * dt;
        if action == 1 {
            v += self.player_speed * dt;
        }
        let x = player_center + v * dt;
        if x <= 0.0 {
            return (0.0, -0.3 * v);
        }
        if x >= 1.0 {
            return (1.0, -0.3 * v);
        }
        (x, v)
    }

    fn quadratic_roots(a: f32, b: f32, c: f32) -> Option<(f32, f32)> {
        if a.abs() <= 1e-12 {
            if b.abs() <= 1e-12 {
                return None;
            }
            let root = -c / b;
            return Some((root, root));
        }
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let sqrt_disc = disc.sqrt();
        Some(((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)))
    }

    fn min_time_first_action(&self, y0: f32, r0: f32) -> i32 {
        let mut best_time = f32::INFINITY;
        let mut best_action = if y0 + 0.2 * r0 > 0.0 { 1 } else { 0 };

        for (start_action, w1, w2) in [(1, -self.a_up, self.a_down), (0, self.a_down, -self.a_up)] {
            let a = 0.5 * w1 * (1.0 - w1 / w2);
            let b = r0 * (1.0 - w1 / w2);
            let c = y0 - 0.5 * (r0 * r0) / w2;
            let Some((r1, r2)) = Self::quadratic_roots(a, b, c) else {
                continue;
            };

            for t1 in [r1, r2] {
                if t1 < 0.0 {
                    continue;
                }
                let rel1 = r0 + w1 * t1;
                let t2 = -rel1 / w2;
                if t2 < 0.0 {
                    continue;
                }
                let total = t1 + t2;
                if total < best_time {
                    best_time = total;
                    best_action = start_action;
                }
            }
        }

        best_action
    }

    fn evaluate_first_action_robust(
        &self,
        first_action: i32,
        obs: FishingObservation,
        player_velocity: f32,
        danger: f32,
        threshold: f32,
    ) -> f32 {
        let dt = obs.dt.max(self.vel_epsilon);
        let horizon_steps =
            ((self.robust_horizon_base_seconds + self.robust_horizon_danger_seconds * danger) / dt)
                .round()
                .max(self.robust_horizon_min_steps as f32)
                .min(self.robust_horizon_max_steps as f32) as usize;

        let difficulty = self.difficulty_est;
        let d_norm = self.difficulty_normalized(difficulty);
        let direction_time = self.direction_time(difficulty);
        let time_to_next_change = (direction_time - self.time_since_target_change).max(0.0);
        let alpha = 1.0 - (-self.fish_decay_rate(difficulty) * dt).exp();
        let max_jump = self.max_fish_jump(difficulty);

        let low_target = clamp01(clamp(
            0.01,
            obs.fish_center - max_jump,
            obs.fish_center + max_jump,
        ));
        let high_target = clamp01(clamp(
            0.99,
            obs.fish_center - max_jump,
            obs.fish_center + max_jump,
        ));
        let mid_target = 0.5 * (low_target + high_target);

        let scenarios: Vec<f32> = if time_to_next_change > horizon_steps as f32 * dt + 1e-9 {
            vec![self.fish_target_est]
        } else {
            vec![low_target, mid_target, high_target]
        };

        let mut scores = Vec::with_capacity(scenarios.len());
        for scenario_target_after_change in scenarios {
            let mut fish = obs.fish_center;
            let mut fish_target = self.fish_target_est;
            let mut time_since_change = self.time_since_target_change;
            let mut player = obs.player_center;
            let mut player_v = player_velocity;
            let mut progress = self.progress_est;
            let mut elapsed = self.elapsed_est;
            let mut min_progress = progress;
            let mut outside_time = 0.0;
            let mut fish_vel = self.fish_v_est;

            for step_idx in 0..horizon_steps {
                time_since_change += dt;
                if time_since_change >= direction_time {
                    time_since_change -= direction_time;
                    fish_target = scenario_target_after_change;
                }

                let fish_prev = fish;
                fish = clamp01(fish + (fish_target - fish) * alpha);
                fish_vel = (fish - fish_prev) / dt;

                let action = if step_idx == 0 {
                    first_action
                } else {
                    let rel_v = fish_vel - player_v;
                    let target_bias = (self.robust_inner_target_bias_base
                        + self.robust_inner_target_bias_danger_gain * danger)
                        * threshold;
                    let y = (fish - player) - target_bias;
                    self.min_time_first_action(y, rel_v)
                };

                let (next_player, next_v) = self.step_player(player, player_v, dt, action);
                player = next_player;
                player_v = next_v;

                elapsed += dt;
                let catching = (fish - player).abs() < threshold;
                progress += self.progress_delta(catching, elapsed, difficulty, dt);
                progress = clamp01(progress);
                min_progress = min_progress.min(progress);
                if !catching {
                    outside_time += dt;
                }
            }

            let terminal_error = fish - player - (0.10 + 0.06 * danger) * threshold;
            let terminal_rel_v = fish_vel - player_v;
            let score = 2.2 * progress + 1.3 * min_progress
                - 1.2 * outside_time
                - 1.5 * terminal_error * terminal_error
                - 0.08 * terminal_rel_v * terminal_rel_v;
            scores.push(score);
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let worst = scores[0];
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let risk_weight = clamp(
            self.robust_risk_base
                + self.robust_risk_danger_gain * danger
                + self.robust_risk_dnorm_gain * d_norm,
            0.0,
            self.robust_risk_max,
        );

        risk_weight * worst + (1.0 - risk_weight) * mean
    }

    pub fn act(&mut self, obs: FishingObservation) -> i32 {
        self.update_progress_estimate(obs);
        let (player_v, fish_v) = self.estimate_kinematics(obs);

        let dt = obs.dt.max(self.vel_epsilon);
        let difficulty = self.difficulty_est;
        let threshold = self.effective_overlap_threshold(obs, difficulty);
        let error_now = obs.fish_center - obs.player_center;
        let rel_v = fish_v - player_v;

        let danger =
            clamp01((self.danger_progress_ref - self.progress_est) / self.danger_progress_ref);
        let predicted_fish = self.predict_fish_center(obs.fish_center, difficulty, danger, fish_v);
        let target_bias =
            (self.target_bias_base + self.target_bias_danger_gain * danger) * threshold;
        let y0 = (predicted_fish - obs.player_center) - target_bias;

        let recovery_band =
            threshold * (self.recovery_band_base + self.recovery_band_danger_gain * danger);
        if error_now.abs() > recovery_band {
            let action = if error_now + 0.22 * rel_v > 0.0 { 1 } else { 0 };
            self.last_action = action;
            return action;
        }

        if obs.player_center > 0.985 && player_v > 0.0 {
            self.last_action = 0;
            return 0;
        }
        if obs.player_center < 0.015 && player_v < 0.0 {
            self.last_action = 1;
            return 1;
        }

        let mut action = self.min_time_first_action(y0, rel_v);

        let safe_band = threshold * (self.safe_band_base - self.safe_band_danger_gain * danger);
        if error_now.abs() < safe_band && rel_v.abs() < (0.26 + 0.18 * danger) {
            let mut s = y0 + 0.12 * rel_v;
            let hysteresis = (0.03 + 0.03 * danger) * threshold;
            if self.last_action == 1 {
                s -= hysteresis;
            } else {
                s += hysteresis;
            }
            action = if s > 0.0 { 1 } else { 0 };
        }

        if danger > self.one_step_override_danger_threshold {
            let fish_next = clamp01(obs.fish_center + fish_v * dt);
            let (x0, _) = self.step_player(obs.player_center, player_v, dt, 0);
            let (x1, _) = self.step_player(obs.player_center, player_v, dt, 1);
            let out0 = ((fish_next - x0).abs() - threshold).max(0.0);
            let out1 = ((fish_next - x1).abs() - threshold).max(0.0);
            if out0 + 1e-9 < out1 {
                action = 0;
            } else if out1 + 1e-9 < out0 {
                action = 1;
            }
        }

        if difficulty >= self.robust_eval_difficulty_threshold
            || danger > self.robust_eval_danger_threshold
        {
            let score0 = self.evaluate_first_action_robust(0, obs, player_v, danger, threshold);
            let score1 = self.evaluate_first_action_robust(1, obs, player_v, danger, threshold);
            if score0 > score1 + 1e-10 {
                action = 0;
            } else if score1 > score0 + 1e-10 {
                action = 1;
            }
        }

        self.last_action = action;
        action
    }
}

