use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct FishingObservation {
    pub fish_center: f32,
    pub player_center: f32,
    pub dt: f32,
    pub player_target_half_size: f32,
}

fn clamp(value: f32, low: f32, high: f32) -> f32 {
    value.max(low).min(high)
}

fn clamp01(value: f32) -> f32 {
    clamp(value, 0.0, 1.0)
}

fn default_scenario_offsets() -> Vec<f32> {
    vec![-0.374_769, -0.187_385, 0.0, 0.187_385, 0.374_769]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PolicyConfig {
    pub horizon_steps: usize,
    pub scenario_offsets: Vec<f32>,
    pub fish_velocity_lpf: f32,
    pub player_velocity_lpf: f32,
    pub velocity_observe_gain: f32,
    pub velocity_clip: f32,
    pub sim_player_acc_press: f32,
    pub sim_player_acc_release: f32,
    pub sim_player_vel_min: f32,
    pub sim_player_vel_max: f32,
    pub sim_player_bounce: f32,
    pub sim_fish_damping: f32,
    pub sim_fish_offset_gain: f32,
    pub sim_fish_vel_min: f32,
    pub sim_fish_vel_max: f32,
    pub sim_fish_bounce: f32,
    pub proximity_outside_scale: f32,
    pub switch_penalty: f32,
    pub time_penalty_per_step: f32,
    pub score_mean_weight: f32,
    pub score_worst_weight: f32,
    pub fish_target_half_size: f32,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 21,
            scenario_offsets: default_scenario_offsets(),
            fish_velocity_lpf: 0.7,
            player_velocity_lpf: 0.7,
            velocity_observe_gain: 0.3,
            velocity_clip: 4.0,
            sim_player_acc_press: 2.5,
            sim_player_acc_release: -1.25,
            sim_player_vel_min: -4.5,
            sim_player_vel_max: 4.5,
            sim_player_bounce: -0.3,
            sim_fish_damping: 0.92,
            sim_fish_offset_gain: 0.08,
            sim_fish_vel_min: -3.0,
            sim_fish_vel_max: 3.0,
            sim_fish_bounce: -0.5,
            proximity_outside_scale: 2.1,
            switch_penalty: 0.04,
            time_penalty_per_step: 0.01,
            score_mean_weight: 0.7,
            score_worst_weight: 0.3,
            fish_target_half_size: 0.1 / (2.8 * 2.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochasticOutputFeedbackMpcPolicy {
    cfg: PolicyConfig,
    prev_fish: Option<f32>,
    prev_player: Option<f32>,
    fish_velocity_est: f32,
    player_velocity_est: f32,
    last_action: i32,
}

impl StochasticOutputFeedbackMpcPolicy {
    pub fn from_config(cfg: &PolicyConfig) -> Self {
        let mut merged = cfg.clone();
        if merged.horizon_steps == 0 {
            merged.horizon_steps = 1;
        }
        if merged.scenario_offsets.is_empty() {
            merged.scenario_offsets = default_scenario_offsets();
        }
        Self {
            cfg: merged,
            prev_fish: None,
            prev_player: None,
            fish_velocity_est: 0.0,
            player_velocity_est: 0.0,
            last_action: 0,
        }
    }

    fn candidate_sequences(&self) -> Vec<Vec<i32>> {
        let h = self.cfg.horizon_steps;
        let last = self.last_action;
        let flip = 1 - last;
        vec![
            vec![0; h],
            vec![1; h],
            vec![last; h],
            vec![flip; h],
            (0..h).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect(),
            (0..h).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect(),
            (0..h)
                .map(|i| if (i / 2) % 2 == 0 { 0 } else { 1 })
                .collect(),
            (0..h)
                .map(|i| if (i / 2) % 2 == 0 { 1 } else { 0 })
                .collect(),
        ]
    }

    fn score_sequence(
        &self,
        sequence: &[i32],
        obs: FishingObservation,
        fish_velocity_est: f32,
        player_velocity_est: f32,
    ) -> f32 {
        let dt = obs.dt.max(1e-6);
        let target_half = obs.player_target_half_size;
        let mut scenario_scores = Vec::with_capacity(self.cfg.scenario_offsets.len());

        for &offset in &self.cfg.scenario_offsets {
            let mut fish_pos = obs.fish_center;
            let mut player_pos = obs.player_center;
            let mut fish_vel = fish_velocity_est + offset;
            let mut player_vel = player_velocity_est;
            let mut scenario_score = 0.0f32;

            for (step, &action) in sequence.iter().enumerate() {
                let player_acc = if action == 1 {
                    self.cfg.sim_player_acc_press
                } else {
                    self.cfg.sim_player_acc_release
                };
                player_vel = clamp(
                    player_vel + player_acc * dt,
                    self.cfg.sim_player_vel_min,
                    self.cfg.sim_player_vel_max,
                );
                player_pos = clamp01(player_pos + player_vel * dt);
                if player_pos <= 0.0 || player_pos >= 1.0 {
                    player_vel *= self.cfg.sim_player_bounce;
                }

                fish_vel = clamp(
                    self.cfg.sim_fish_damping * fish_vel + self.cfg.sim_fish_offset_gain * offset,
                    self.cfg.sim_fish_vel_min,
                    self.cfg.sim_fish_vel_max,
                );
                fish_pos = clamp01(fish_pos + fish_vel * dt);
                if fish_pos <= 0.0 || fish_pos >= 1.0 {
                    fish_vel *= self.cfg.sim_fish_bounce;
                }

                let distance = (fish_pos - player_pos).abs();
                let in_overlap = distance < target_half;
                let proximity_term = if in_overlap {
                    1.0
                } else {
                    -self.cfg.proximity_outside_scale * distance
                };
                let switch_penalty = if step > 0 && sequence[step] != sequence[step - 1] {
                    self.cfg.switch_penalty
                } else {
                    0.0
                };
                let time_penalty = self.cfg.time_penalty_per_step * step as f32;
                scenario_score += proximity_term - switch_penalty - time_penalty;
            }

            scenario_scores.push(scenario_score);
        }

        if scenario_scores.is_empty() {
            return f32::NEG_INFINITY;
        }
        let score_mean = scenario_scores.iter().sum::<f32>() / scenario_scores.len() as f32;
        let score_worst = scenario_scores
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        self.cfg.score_mean_weight * score_mean + self.cfg.score_worst_weight * score_worst
    }

    pub fn act(&mut self, obs: FishingObservation) -> i32 {
        let dt = obs.dt.max(1e-6);
        if self.prev_fish.is_none() || self.prev_player.is_none() {
            self.prev_fish = Some(obs.fish_center);
            self.prev_player = Some(obs.player_center);
            self.last_action = if obs.fish_center > obs.player_center {
                1
            } else {
                0
            };
            return self.last_action;
        }

        let fish_v_raw = (obs.fish_center - self.prev_fish.unwrap_or(obs.fish_center)) / dt;
        let player_v_raw = (obs.player_center - self.prev_player.unwrap_or(obs.player_center)) / dt;
        let clipped_fish_v = clamp(fish_v_raw, -self.cfg.velocity_clip, self.cfg.velocity_clip);
        let clipped_player_v = clamp(player_v_raw, -self.cfg.velocity_clip, self.cfg.velocity_clip);

        self.fish_velocity_est = self.cfg.fish_velocity_lpf * self.fish_velocity_est
            + self.cfg.velocity_observe_gain * clipped_fish_v;
        self.player_velocity_est = self.cfg.player_velocity_lpf * self.player_velocity_est
            + self.cfg.velocity_observe_gain * clipped_player_v;

        let mut best_action = self.last_action;
        let mut best_score = f32::NEG_INFINITY;
        for sequence in self.candidate_sequences() {
            let score =
                self.score_sequence(&sequence, obs, self.fish_velocity_est, self.player_velocity_est);
            if score > best_score {
                best_score = score;
                best_action = sequence[0];
            }
        }

        self.prev_fish = Some(obs.fish_center);
        self.prev_player = Some(obs.player_center);
        self.last_action = best_action;
        best_action
    }
}

pub type TimeOptimalBangBangPolicy = StochasticOutputFeedbackMpcPolicy;
