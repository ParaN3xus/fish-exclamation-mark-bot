from __future__ import annotations

from src.gym.fishing_env import FishingObservation

from .time_optimal_bangbang_policy import TimeOptimalBangBangPolicy


class StochasticOutputFeedbackMPCPolicy(TimeOptimalBangBangPolicy):
    """Output-feedback stochastic MPC policy (single-policy, no training)."""

    def __init__(
        self,
        player_speed: float = 3.0,
        gravity: float = 1.5,
        vel_epsilon: float = 1e-6,
        equipment_strength: int = 0,
        equipment_expertise: int = 0,
        smoothed_fps: float = 60.0,
        is_vr: bool = False,
        progress_ref: float = 0.42,
        hard_start: float = 6.0,
        hard_span: float = 3.0,
        danger_hard_gain: float = 0.10,
        conservative_danger_floor: float = 0.55,
        conservative_danger_hard_gain: float = 0.25,
        hard_nominal_threshold: float = 0.80,
        conservative_weight_base: float = 0.20,
        conservative_weight_hard_gain: float = 0.60,
        tie_margin_base: float = 4e-4,
        tie_margin_danger_gain: float = 2e-3,
        tie_margin_hard_reduction: float = 0.75,
    ) -> None:
        super().__init__(
            player_speed=player_speed,
            gravity=gravity,
            vel_epsilon=vel_epsilon,
            equipment_strength=equipment_strength,
            equipment_expertise=equipment_expertise,
            smoothed_fps=smoothed_fps,
            is_vr=is_vr,
        )
        self.name = "stochastic_output_feedback_mpc"

        self.progress_ref = max(1e-6, progress_ref)
        self.hard_start = hard_start
        self.hard_span = max(1e-6, hard_span)
        self.danger_hard_gain = danger_hard_gain
        self.conservative_danger_floor = conservative_danger_floor
        self.conservative_danger_hard_gain = conservative_danger_hard_gain
        self.hard_nominal_threshold = hard_nominal_threshold
        self.conservative_weight_base = conservative_weight_base
        self.conservative_weight_hard_gain = conservative_weight_hard_gain
        self.tie_margin_base = tie_margin_base
        self.tie_margin_danger_gain = tie_margin_danger_gain
        self.tie_margin_hard_reduction = tie_margin_hard_reduction

    def _nominal_action(
        self,
        *,
        obs: FishingObservation,
        threshold: float,
        danger: float,
        player_velocity: float,
        fish_velocity: float,
    ) -> int:
        error_now = obs.fish_center - obs.player_center
        rel_v = fish_velocity - player_velocity

        predicted_fish = self._predict_fish_center(
            fish_center=obs.fish_center,
            difficulty=self._difficulty_est,
            danger=danger,
            fish_velocity=fish_velocity,
        )
        target_bias = (0.07 + 0.12 * danger) * threshold
        y0 = (predicted_fish - obs.player_center) - target_bias

        action = self._min_time_first_action(y0, rel_v)

        recovery_band = threshold * (1.8 + 1.3 * danger)
        if abs(error_now) > recovery_band:
            return 1 if (error_now + 0.22 * rel_v) > 0.0 else 0

        safe_band = threshold * (0.80 - 0.10 * danger)
        if abs(error_now) < safe_band and abs(rel_v) < (0.24 + 0.20 * danger):
            s = y0 + 0.12 * rel_v
            hysteresis = (0.03 + 0.03 * danger) * threshold
            if self._last_action == 1:
                s -= hysteresis
            else:
                s += hysteresis
            action = 1 if s > 0.0 else 0

        return action

    def _boundary_guard(
        self,
        *,
        player_center: float,
        player_velocity: float,
    ) -> int | None:
        if player_center > 0.985 and player_velocity > 0.0:
            return 0
        if player_center < 0.015 and player_velocity < 0.0:
            return 1
        return None

    def _immediate_risk_override(
        self,
        *,
        obs: FishingObservation,
        threshold: float,
        player_velocity: float,
        fish_velocity: float,
        action: int,
        danger: float,
    ) -> int:
        difficulty_est = self._difficulty_est
        if danger <= 0.2 and difficulty_est < 7.0:
            return action

        dt = max(obs.dt, self.vel_epsilon)
        if difficulty_est >= 7.0:
            fish_next = self._predict_fish_center(
                fish_center=obs.fish_center,
                difficulty=difficulty_est,
                danger=max(0.3, danger),
                fish_velocity=fish_velocity,
            )
        else:
            fish_next = self._clamp01(obs.fish_center + fish_velocity * dt)

        x0, _ = self._step_player(obs.player_center, player_velocity, dt, 0)
        x1, _ = self._step_player(obs.player_center, player_velocity, dt, 1)
        out0 = max(0.0, abs(fish_next - x0) - threshold)
        out1 = max(0.0, abs(fish_next - x1) - threshold)

        if out0 + 1e-9 < out1:
            return 0
        if out1 + 1e-9 < out0:
            return 1
        return action

    def act(self, obs: FishingObservation) -> int:
        self._update_progress_estimate(obs)
        player_v, fish_v = self._estimate_kinematics(obs)

        boundary_action = self._boundary_guard(
            player_center=obs.player_center,
            player_velocity=player_v,
        )
        if boundary_action is not None:
            self._last_action = boundary_action
            return boundary_action

        difficulty_est = self._difficulty_est
        threshold = self._overlap_threshold(difficulty_est)
        base_danger = self._clamp01(
            (self.progress_ref - self._progress_est) / self.progress_ref
        )
        hard_factor = self._clamp01((difficulty_est - self.hard_start) / self.hard_span)
        danger = self._clamp01(base_danger + self.danger_hard_gain * hard_factor)

        nominal = self._nominal_action(
            obs=obs,
            threshold=threshold,
            danger=danger,
            player_velocity=player_v,
            fish_velocity=fish_v,
        )

        score0_nom = self._evaluate_first_action_robust(
            first_action=0,
            obs=obs,
            player_velocity=player_v,
            danger=danger,
            threshold=threshold,
        )
        score1_nom = self._evaluate_first_action_robust(
            first_action=1,
            obs=obs,
            player_velocity=player_v,
            danger=danger,
            threshold=threshold,
        )

        conservative_danger = max(
            danger,
            self.conservative_danger_floor
            + self.conservative_danger_hard_gain * hard_factor,
        )
        score0_cons = self._evaluate_first_action_robust(
            first_action=0,
            obs=obs,
            player_velocity=player_v,
            danger=conservative_danger,
            threshold=threshold,
        )
        score1_cons = self._evaluate_first_action_robust(
            first_action=1,
            obs=obs,
            player_velocity=player_v,
            danger=conservative_danger,
            threshold=threshold,
        )

        if hard_factor >= self.hard_nominal_threshold:
            score0 = score0_nom
            score1 = score1_nom
            tie_margin = 0.0
        else:
            conservative_weight = self._clamp01(
                self.conservative_weight_base
                + self.conservative_weight_hard_gain * hard_factor
            )
            score0 = (
                1.0 - conservative_weight
            ) * score0_nom + conservative_weight * score0_cons
            score1 = (
                1.0 - conservative_weight
            ) * score1_nom + conservative_weight * score1_cons
            tie_margin = (
                self.tie_margin_base + self.tie_margin_danger_gain * danger
            ) * (1.0 - self.tie_margin_hard_reduction * hard_factor)

        if nominal == 0:
            score0 += tie_margin
        else:
            score1 += tie_margin

        action = 0 if score0 >= score1 else 1
        action = self._immediate_risk_override(
            obs=obs,
            threshold=threshold,
            player_velocity=player_v,
            fish_velocity=fish_v,
            action=action,
            danger=danger,
        )

        self._last_action = action
        return action
