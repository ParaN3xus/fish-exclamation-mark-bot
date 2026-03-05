from __future__ import annotations

from src.gym.fishing_env import FishingObservation

from .policy import Policy


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class TimeOptimalBangBangPolicy(Policy):
    """
    Fast O(1) bang-bang controller:
    - hysteresis to avoid chattering
    - short lookahead feed-forward from relative velocity estimate
    - no dependency on hidden game parameters
    """

    def __init__(
        self,
        *,
        min_deadzone: float = 0.006,
        hysteresis_scale: float = 0.7,
        lookahead_seconds: float = 0.14,
    ) -> None:
        super().__init__(name="bangbang_hysteresis_ff")
        self.min_deadzone = min_deadzone
        self.hysteresis_scale = hysteresis_scale
        self.lookahead_seconds = lookahead_seconds
        self.reset()

    def reset(self) -> None:
        self._prev_fish: float | None = None
        self._prev_player: float | None = None
        self._fish_velocity_est = 0.0
        self._player_velocity_est = 0.0
        self._last_action = 0

    def act(self, obs: FishingObservation) -> int:
        dt = max(obs.dt, 1e-6)

        if self._prev_fish is None or self._prev_player is None:
            self._prev_fish = obs.fish_center
            self._prev_player = obs.player_center
            self._last_action = 1 if obs.fish_center > obs.player_center else 0
            return self._last_action

        fish_v_raw = (obs.fish_center - self._prev_fish) / dt
        player_v_raw = (obs.player_center - self._prev_player) / dt

        # Keep estimates smooth; raw finite differences are noisy.
        self._fish_velocity_est = (
            0.7 * self._fish_velocity_est + 0.3 * _clamp(fish_v_raw, -3.0, 3.0)
        )
        self._player_velocity_est = (
            0.7 * self._player_velocity_est + 0.3 * _clamp(player_v_raw, -3.0, 3.0)
        )

        error = obs.fish_center - obs.player_center
        rel_velocity = self._fish_velocity_est - self._player_velocity_est
        predicted_error = error + rel_velocity * self.lookahead_seconds

        deadzone = max(self.min_deadzone, obs.player_target_half_size * 0.45)
        hysteresis = deadzone * self.hysteresis_scale + 0.002

        action = self._last_action
        if self._last_action == 1:
            if predicted_error < -hysteresis:
                action = 0
        else:
            if predicted_error > hysteresis:
                action = 1

        # Hard safety guards near boundaries.
        if obs.player_center <= 0.01:
            action = 1
        elif obs.player_center >= 0.99:
            action = 0

        self._prev_fish = obs.fish_center
        self._prev_player = obs.player_center
        self._last_action = action
        return action
