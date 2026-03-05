from __future__ import annotations

from src.gym.fishing_env import FishingObservation

from .policy import Policy


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class BeliefSpaceLocalOptPolicy(Policy):
    """
    Non-learning lookup-table policy.
    The table is built from a simplified local-opt rollout model over
    (position error, relative velocity) and queried in O(1) at runtime.
    """

    def __init__(
        self,
        *,
        error_bins: int = 121,
        rel_velocity_bins: int = 121,
        horizon_steps: int = 10,
    ) -> None:
        super().__init__(name="lookup_local_opt")
        self.error_bins = error_bins
        self.rel_velocity_bins = rel_velocity_bins
        self.horizon_steps = horizon_steps
        self.error_low = -0.75
        self.error_high = 0.75
        self.rel_v_low = -3.0
        self.rel_v_high = 3.0
        self._table: list[list[int]] = self._build_policy_table()
        self.reset()

    def reset(self) -> None:
        self._prev_fish: float | None = None
        self._prev_player: float | None = None
        self._fish_velocity_est = 0.0
        self._player_velocity_est = 0.0
        self._last_action = 0

    def _score_first_action(
        self,
        *,
        initial_error: float,
        initial_rel_velocity: float,
        first_action: int,
        assumed_target_half_size: float,
    ) -> float:
        dt = 1.0 / 60.0
        error = initial_error
        rel_velocity = initial_rel_velocity
        score = 0.0
        discount = 1.0

        for step in range(self.horizon_steps):
            action = first_action if step == 0 else (1 if error > 0.0 else 0)
            player_acc = 2.5 if action == 1 else -1.25

            # Fish acceleration is unknown and omitted in this surrogate model.
            rel_velocity = _clamp(rel_velocity - player_acc * dt, -4.0, 4.0)
            error = _clamp(error + rel_velocity * dt, -1.0, 1.0)

            distance = abs(error)
            overlap_bonus = 0.35 if distance < assumed_target_half_size else 0.0
            score += discount * (overlap_bonus - distance)
            discount *= 0.95

        return score

    def _build_policy_table(self) -> list[list[int]]:
        table: list[list[int]] = []
        assumed_target_half_size = 0.16
        for e_idx in range(self.error_bins):
            row: list[int] = []
            e_ratio = e_idx / max(1, self.error_bins - 1)
            error = self.error_low + (self.error_high - self.error_low) * e_ratio
            for rv_idx in range(self.rel_velocity_bins):
                rv_ratio = rv_idx / max(1, self.rel_velocity_bins - 1)
                rel_velocity = self.rel_v_low + (
                    self.rel_v_high - self.rel_v_low
                ) * rv_ratio
                score_release = self._score_first_action(
                    initial_error=error,
                    initial_rel_velocity=rel_velocity,
                    first_action=0,
                    assumed_target_half_size=assumed_target_half_size,
                )
                score_press = self._score_first_action(
                    initial_error=error,
                    initial_rel_velocity=rel_velocity,
                    first_action=1,
                    assumed_target_half_size=assumed_target_half_size,
                )
                row.append(1 if score_press > score_release else 0)
            table.append(row)
        return table

    def _bin_index(self, value: float, low: float, high: float, bins: int) -> int:
        clipped = _clamp(value, low, high)
        normalized = (clipped - low) / max(1e-12, (high - low))
        return int(round(normalized * max(0, bins - 1)))

    def act(self, obs: FishingObservation) -> int:
        dt = max(obs.dt, 1e-6)

        if self._prev_fish is None or self._prev_player is None:
            self._prev_fish = obs.fish_center
            self._prev_player = obs.player_center
            self._last_action = 1 if obs.fish_center > obs.player_center else 0
            return self._last_action

        fish_v_raw = (obs.fish_center - self._prev_fish) / dt
        player_v_raw = (obs.player_center - self._prev_player) / dt
        self._fish_velocity_est = (
            0.75 * self._fish_velocity_est + 0.25 * _clamp(fish_v_raw, -4.0, 4.0)
        )
        self._player_velocity_est = (
            0.75 * self._player_velocity_est + 0.25 * _clamp(player_v_raw, -4.0, 4.0)
        )

        error = obs.fish_center - obs.player_center
        rel_velocity = self._fish_velocity_est - self._player_velocity_est

        e_idx = self._bin_index(error, self.error_low, self.error_high, self.error_bins)
        rv_idx = self._bin_index(
            rel_velocity, self.rel_v_low, self.rel_v_high, self.rel_velocity_bins
        )
        action = self._table[e_idx][rv_idx]

        # Small deadband near center to reduce jitter.
        center_deadband = max(0.004, obs.player_target_half_size * 0.1)
        if abs(error) < center_deadband:
            action = self._last_action

        if obs.player_center <= 0.01:
            action = 1
        elif obs.player_center >= 0.99:
            action = 0

        self._prev_fish = obs.fish_center
        self._prev_player = obs.player_center
        self._last_action = action
        return action
