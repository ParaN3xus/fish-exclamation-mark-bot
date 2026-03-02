from __future__ import annotations

from src.gym.fishing_env import FishingObservation

from .time_optimal_bangbang_policy import TimeOptimalBangBangPolicy


class StochasticOutputFeedbackMPCPolicy(TimeOptimalBangBangPolicy):
    """
    Output-feedback stochastic MPC policy (non-training based).

    Design:
    1) Reconstruct hidden dynamics online from observations.
    2) At each step, solve a short stochastic predictive control problem over two
       candidate first inputs (0/1), where future inputs are closed-loop.
    3) Apply the first control, then repeat at the next observation (receding
       horizon), with success-first risk bias.
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
            difficulty=obs.difficulty,
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
        if danger <= 0.2:
            return action

        dt = max(obs.dt, self.vel_epsilon)
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

        threshold = self._overlap_threshold(obs.difficulty)
        danger = self._clamp01((0.42 - self._progress_est) / 0.42)

        nominal = self._nominal_action(
            obs=obs,
            threshold=threshold,
            danger=danger,
            player_velocity=player_v,
            fish_velocity=fish_v,
        )

        # Stochastic receding-horizon evaluation over first input candidates.
        score0 = self._evaluate_first_action_robust(
            first_action=0,
            obs=obs,
            player_velocity=player_v,
            danger=danger,
            threshold=threshold,
        )
        score1 = self._evaluate_first_action_robust(
            first_action=1,
            obs=obs,
            player_velocity=player_v,
            danger=danger,
            threshold=threshold,
        )

        # Keep nominal preference when scores are nearly identical.
        tie_margin = 5e-4 + 4e-3 * danger
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
