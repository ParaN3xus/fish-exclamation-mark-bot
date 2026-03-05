from __future__ import annotations

from src.gym.fishing_env import FishingObservation

from .policy import Policy


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


class StochasticOutputFeedbackMPCPolicy(Policy):
    """
    Fixed-budget stochastic MPC with an output-feedback state estimate.
    Uses only observable signals and short-horizon rollout scoring.
    """

    def __init__(
        self,
        *,
        horizon_steps: int = 12,
        scenario_offsets: tuple[float, ...] = (-0.30, -0.12, 0.0, 0.12, 0.30),
    ) -> None:
        super().__init__(name="stochastic_short_mpc")
        self.horizon_steps = horizon_steps
        self.scenario_offsets = scenario_offsets
        self.reset()

    def reset(self) -> None:
        self._prev_fish: float | None = None
        self._prev_player: float | None = None
        self._fish_velocity_est = 0.0
        self._player_velocity_est = 0.0
        self._last_action = 0

    def _candidate_sequences(self) -> list[list[int]]:
        h = self.horizon_steps
        last = self._last_action
        flip = 1 - last
        seqs = [
            [0] * h,
            [1] * h,
            [last] * h,
            [flip] * h,
            [0 if (i % 2 == 0) else 1 for i in range(h)],
            [1 if (i % 2 == 0) else 0 for i in range(h)],
            [0 if ((i // 2) % 2 == 0) else 1 for i in range(h)],
            [1 if ((i // 2) % 2 == 0) else 0 for i in range(h)],
        ]
        return seqs

    def _score_sequence(
        self,
        sequence: list[int],
        *,
        obs: FishingObservation,
        fish_velocity_est: float,
        player_velocity_est: float,
    ) -> float:
        dt = max(obs.dt, 1e-6)
        target_half = obs.player_target_half_size
        scores: list[float] = []

        for offset in self.scenario_offsets:
            fish_pos = obs.fish_center
            player_pos = obs.player_center
            fish_vel = fish_velocity_est + offset
            player_vel = player_velocity_est
            scenario_score = 0.0

            for step, action in enumerate(sequence):
                player_acc = 2.5 if action == 1 else -1.25
                player_vel = _clamp(player_vel + player_acc * dt, -4.5, 4.5)
                player_pos = _clamp01(player_pos + player_vel * dt)
                if player_pos <= 0.0 or player_pos >= 1.0:
                    player_vel *= -0.3

                # Simple fish uncertainty model: bounded drift + damping.
                fish_vel = _clamp(0.92 * fish_vel + 0.08 * offset, -3.0, 3.0)
                fish_pos = _clamp01(fish_pos + fish_vel * dt)
                if fish_pos <= 0.0 or fish_pos >= 1.0:
                    fish_vel *= -0.5

                distance = abs(fish_pos - player_pos)
                in_overlap = distance < target_half
                proximity_term = 1.0 if in_overlap else -2.1 * distance
                switch_penalty = (
                    0.04
                    if (step > 0 and sequence[step] != sequence[step - 1])
                    else 0.0
                )
                time_penalty = 0.01 * step
                scenario_score += proximity_term - switch_penalty - time_penalty

            scores.append(scenario_score)

        score_mean = sum(scores) / max(1, len(scores))
        score_worst = min(scores) if scores else 0.0
        return 0.7 * score_mean + 0.3 * score_worst

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
            0.7 * self._fish_velocity_est + 0.3 * _clamp(fish_v_raw, -4.0, 4.0)
        )
        self._player_velocity_est = (
            0.7 * self._player_velocity_est + 0.3 * _clamp(player_v_raw, -4.0, 4.0)
        )

        best_action = self._last_action
        best_score = float("-inf")
        for sequence in self._candidate_sequences():
            score = self._score_sequence(
                sequence,
                obs=obs,
                fish_velocity_est=self._fish_velocity_est,
                player_velocity_est=self._player_velocity_est,
            )
            if score > best_score:
                best_score = score
                best_action = sequence[0]

        self._prev_fish = obs.fish_center
        self._prev_player = obs.player_center
        self._last_action = best_action
        return best_action
