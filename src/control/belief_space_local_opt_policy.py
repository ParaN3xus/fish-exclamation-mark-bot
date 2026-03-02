from __future__ import annotations

import math

from src.gym.fishing_env import FishingObservation

from .time_optimal_bangbang_policy import TimeOptimalBangBangPolicy


class BeliefSpaceLocalOptPolicy(TimeOptimalBangBangPolicy):
    """
    Belief-space local optimization policy (non-RL).

    A lightweight output-feedback controller that keeps a compact belief over
    hidden fish target uncertainty and evaluates binary first actions through
    short stochastic rollouts.
    """

    def __init__(
        self,
        player_speed: float = 3.0,
        gravity: float = 1.5,
        vel_epsilon: float = 1e-6,
        equipment_strength: int = 0,
        equipment_expertise: int = 0,
        smoothed_fps: float = 60.0,
        is_vr: bool = False,
        initial_difficulty_estimate: float = 8.36370917321805,
        horizon_base_seconds: float = 0.3544855165700896,
        horizon_danger_seconds: float = 0.07069091316859279,
        q_var_init: float = 0.00045129741931029246,
        q_process_noise_per_second: float = 0.0012271106169625013,
        q_jump_noise_gain: float = 7.352591618196708,
        q_stability_decay: float = 0.9829561330174689,
        uncertainty_threshold_gain: float = 0.18247663879434314,
        risk_worst_weight_base: float = 0.5585009025740797,
        risk_worst_weight_danger_gain: float = 0.7400583630848814,
        target_bias_base: float = 0.2301793526053328,
        target_bias_danger_gain: float = 0.28299317034146937,
        recovery_band_base: float = 2.0170104258319883,
        recovery_band_danger_gain: float = 0.04298606726678378,
        tie_margin: float = 1e-4,
    ) -> None:
        super().__init__(
            player_speed=player_speed,
            gravity=gravity,
            vel_epsilon=vel_epsilon,
            equipment_strength=equipment_strength,
            equipment_expertise=equipment_expertise,
            smoothed_fps=smoothed_fps,
            is_vr=is_vr,
            initial_difficulty_estimate=initial_difficulty_estimate,
        )
        self.name = "belief_space_local_opt"

        self.horizon_base_seconds = max(0.04, horizon_base_seconds)
        self.horizon_danger_seconds = max(0.0, horizon_danger_seconds)
        self.q_var_init = max(1e-6, q_var_init)
        self.q_process_noise_per_second = max(0.0, q_process_noise_per_second)
        self.q_jump_noise_gain = max(0.0, q_jump_noise_gain)
        self.q_stability_decay = self._clamp(q_stability_decay, 0.80, 0.9999)
        self.uncertainty_threshold_gain = max(0.0, uncertainty_threshold_gain)
        self.risk_worst_weight_base = self._clamp(risk_worst_weight_base, 0.0, 1.0)
        self.risk_worst_weight_danger_gain = max(0.0, risk_worst_weight_danger_gain)
        self.target_bias_base = target_bias_base
        self.target_bias_danger_gain = target_bias_danger_gain
        self.recovery_band_base = recovery_band_base
        self.recovery_band_danger_gain = recovery_band_danger_gain
        self.tie_margin = max(0.0, tie_margin)

        self._q_var = self.q_var_init

    def reset(self) -> None:
        super().reset()
        self._q_var = self.q_var_init

    def _update_belief_uncertainty(
        self,
        *,
        dt: float,
        difficulty: float,
        jump_distance: float,
        jump_detect_threshold: float,
    ) -> None:
        # Time propagation
        self._q_var += self.q_process_noise_per_second * dt

        direction_time = self._direction_time(difficulty)
        if direction_time > 1e-6:
            near_change = self._clamp01(
                (self._time_since_target_change - 0.65 * direction_time)
                / (0.35 * direction_time)
            )
            self._q_var += near_change * self.q_process_noise_per_second * dt * 2.0

        if jump_distance > jump_detect_threshold:
            ratio = self._clamp01(
                (jump_distance - jump_detect_threshold)
                / max(1e-6, 0.30 - jump_detect_threshold)
            )
            self._q_var += (
                self.q_jump_noise_gain
                * ratio
                * self.q_process_noise_per_second
            )
        else:
            self._q_var *= self.q_stability_decay

        self._q_var = self._clamp(self._q_var, 1e-7, 0.04)

    def _estimate_kinematics(self, obs: FishingObservation) -> tuple[float, float]:
        if self._last_player_center is None or self._last_fish_center is None:
            self._q_var = self.q_var_init
            return super()._estimate_kinematics(obs)

        dt = max(obs.dt, self.vel_epsilon)
        difficulty_before = self._difficulty_est
        decay = self._fish_decay_rate(difficulty_before)
        alpha = 1.0 - math.exp(-decay * dt)
        if alpha > 1e-6:
            inferred_target = self._last_fish_center + (
                (obs.fish_center - self._last_fish_center) / alpha
            )
        else:
            inferred_target = obs.fish_center
        inferred_target = self._clamp01(inferred_target)

        d_norm = self._difficulty_normalized(difficulty_before)
        jump_detect_threshold = 0.035 + 0.065 * d_norm
        jump_distance = abs(inferred_target - self._fish_target_est)

        pv, fv = super()._estimate_kinematics(obs)
        self._update_belief_uncertainty(
            dt=dt,
            difficulty=self._difficulty_est,
            jump_distance=jump_distance,
            jump_detect_threshold=jump_detect_threshold,
        )
        return pv, fv

    def _belief_scenarios(
        self,
        *,
        fish_center: float,
        difficulty: float,
        horizon_steps: int,
        dt: float,
    ) -> list[tuple[float, float, float]]:
        sigma = math.sqrt(max(self._q_var, 1e-9))
        q_mid = self._fish_target_est
        q_low = self._clamp01(q_mid - sigma)
        q_high = self._clamp01(q_mid + sigma)

        direction_time = self._direction_time(difficulty)
        time_to_change = max(0.0, direction_time - self._time_since_target_change)
        horizon_time = horizon_steps * dt
        jump_prob = self._clamp01(1.0 - time_to_change / max(horizon_time, dt))

        max_jump = self._max_fish_jump(difficulty)
        low_target = self._clamp01(
            self._clamp(0.01, fish_center - max_jump, fish_center + max_jump)
        )
        high_target = self._clamp01(
            self._clamp(0.99, fish_center - max_jump, fish_center + max_jump)
        )

        # (weight, initial_target, post-change target)
        scenarios: list[tuple[float, float, float]] = []
        points = ((0.5, q_mid), (0.25, q_low), (0.25, q_high))
        for point_weight, q0 in points:
            if jump_prob < 1e-6:
                scenarios.append((point_weight, q0, q0))
                continue
            stay_w = point_weight * (1.0 - jump_prob)
            jump_w = point_weight * jump_prob * 0.5
            if stay_w > 0.0:
                scenarios.append((stay_w, q0, q0))
            scenarios.append((jump_w, q0, low_target))
            scenarios.append((jump_w, q0, high_target))

        return scenarios

    def _rollout_score(
        self,
        *,
        first_action: int,
        obs: FishingObservation,
        player_velocity: float,
        danger: float,
        threshold: float,
    ) -> float:
        dt = max(obs.dt, self.vel_epsilon)
        horizon_steps = max(
            8,
            min(
                28,
                int(
                    round(
                        (
                            self.horizon_base_seconds
                            + self.horizon_danger_seconds * danger
                        )
                        / dt
                    )
                ),
            ),
        )
        difficulty = self._difficulty_est
        direction_time = self._direction_time(difficulty)
        decay = self._fish_decay_rate(difficulty)
        alpha = 1.0 - math.exp(-decay * dt)

        scenarios = self._belief_scenarios(
            fish_center=obs.fish_center,
            difficulty=difficulty,
            horizon_steps=horizon_steps,
            dt=dt,
        )

        scores: list[float] = []
        weights: list[float] = []

        for weight, q0, q_after_change in scenarios:
            fish = obs.fish_center
            q = q0
            player = obs.player_center
            player_v_sim = player_velocity
            elapsed = self._elapsed_est
            progress = self._progress_est
            min_progress = progress
            outside_time = 0.0
            fish_v = self._fish_v_est
            local_t = self._time_since_target_change

            for step_idx in range(horizon_steps):
                local_t += dt
                if local_t >= direction_time:
                    local_t -= direction_time
                    q = q_after_change

                fish_prev = fish
                fish = self._clamp01(fish + (q - fish) * alpha)
                fish_v = (fish - fish_prev) / dt

                if step_idx == 0:
                    action = first_action
                else:
                    rel_v = fish_v - player_v_sim
                    target_bias = (
                        self.target_bias_base + self.target_bias_danger_gain * danger
                    ) * threshold
                    y = (fish - player) - target_bias
                    action = self._min_time_first_action(y, rel_v)

                player, player_v_sim = self._step_player(
                    player,
                    player_v_sim,
                    dt,
                    action,
                )

                elapsed += dt
                sigma = math.sqrt(max(self._q_var, 1e-9))
                eff_th = max(0.01, threshold - self.uncertainty_threshold_gain * sigma)
                catching = abs(fish - player) < eff_th
                progress += self._progress_delta(
                    catching=catching,
                    elapsed_time=elapsed,
                    difficulty=difficulty,
                    dt=dt,
                )
                progress = self._clamp01(progress)
                min_progress = min(min_progress, progress)
                if not catching:
                    outside_time += dt

            terminal_error = fish - player - 0.12 * threshold
            terminal_rel_v = fish_v - player_v_sim
            score = (
                2.6 * progress
                + 1.2 * min_progress
                - 1.4 * outside_time
                - 1.5 * terminal_error * terminal_error
                - 0.09 * terminal_rel_v * terminal_rel_v
                - 2.0 * self._q_var
            )
            scores.append(score)
            weights.append(weight)

        if not scores:
            return -1e9

        mean_score = 0.0
        for w, s in zip(weights, scores):
            mean_score += w * s

        worst_score = min(scores)
        risk_w = self._clamp(
            self.risk_worst_weight_base + self.risk_worst_weight_danger_gain * danger,
            0.0,
            0.995,
        )
        return risk_w * worst_score + (1.0 - risk_w) * mean_score

    def act(self, obs: FishingObservation) -> int:
        self._update_progress_estimate(obs)
        player_v, fish_v = self._estimate_kinematics(obs)

        difficulty = self._difficulty_est
        threshold = self._overlap_threshold(difficulty)
        error_now = obs.fish_center - obs.player_center
        rel_v = fish_v - player_v
        danger = self._clamp01((0.48 - self._progress_est) / 0.48)

        # Fast recovery if far from overlap region.
        recovery_band = threshold * (
            self.recovery_band_base + self.recovery_band_danger_gain * danger
        )
        if abs(error_now) > recovery_band:
            action = 1 if (error_now + 0.2 * rel_v) > 0.0 else 0
            self._last_action = action
            return action

        # Boundary protection.
        if obs.player_center > 0.985 and player_v > 0.0:
            self._last_action = 0
            return 0
        if obs.player_center < 0.015 and player_v < 0.0:
            self._last_action = 1
            return 1

        # Nominal local action from minimum-time switching structure.
        predicted_fish = self._predict_fish_center(
            fish_center=obs.fish_center,
            difficulty=difficulty,
            danger=danger,
            fish_velocity=fish_v,
        )
        target_bias = (
            self.target_bias_base + self.target_bias_danger_gain * danger
        ) * threshold
        y0 = (predicted_fish - obs.player_center) - target_bias
        nominal = self._min_time_first_action(y0, rel_v)

        score0 = self._rollout_score(
            first_action=0,
            obs=obs,
            player_velocity=player_v,
            danger=danger,
            threshold=threshold,
        )
        score1 = self._rollout_score(
            first_action=1,
            obs=obs,
            player_velocity=player_v,
            danger=danger,
            threshold=threshold,
        )

        if nominal == 0:
            score0 += self.tie_margin
        else:
            score1 += self.tie_margin

        action = 0 if score0 >= score1 else 1
        self._last_action = action
        return action
