from __future__ import annotations

from dataclasses import dataclass

from src.gym.fishing_env import FishingObservation


@dataclass(slots=True)
class Policy:
    name: str

    def reset(self) -> None:
        return None

    def act(self, obs: FishingObservation) -> int:
        raise NotImplementedError


class BaselinePolicy(Policy):
    """Simple chase controller for comparison."""

    def __init__(self, deadzone: float = 0.01) -> None:
        super().__init__(name="baseline_chase")
        self.deadzone = deadzone

    def act(self, obs: FishingObservation) -> int:
        error = obs.fish_center - obs.player_center
        if error > self.deadzone:
            return 1
        if error < -self.deadzone:
            return 0
        return 0


class TimeOptimalBangBangPolicy(Policy):
    """
    Time-optimal bang-bang controller for the double-integrator subproblem.

    System from source code:
      x_dot = v
      v_dot in {a_up, a_down}
      a_up = playerSpeed - gravity = 1.5
      a_down = -gravity = -1.5

    For the minimum-time transfer (x, v) -> (x_target, 0)
    with bounded acceleration, Pontryagin's Maximum
    Principle gives bang-bang control with switching by
    stopping distance. For asymmetric accelerations we use
    the direction-dependent braking capability:
      stop_dist = v_along_target^2 / (2 * a_brake)

    Control law used here:
    - accelerate toward target while |e| > v^2/(2a) or moving away from target
    - otherwise brake to reduce terminal speed
    """

    def __init__(
        self, player_speed: float = 3.0, gravity: float = 1.5, vel_epsilon: float = 1e-6
    ) -> None:
        super().__init__(name="time_optimal_bangbang")
        self.player_speed = player_speed
        self.gravity = gravity
        self.vel_epsilon = vel_epsilon

        self.a_up = player_speed - gravity
        self.a_down = -gravity
        if self.a_up <= 0.0:
            raise ValueError(
                "player_speed must be larger than gravity for controllability."
            )
        self._last_player_center: float | None = None
        self._last_dt: float | None = None
        self._v_est = 0.0

    def reset(self) -> None:
        self._last_player_center = None
        self._last_dt = None
        self._v_est = 0.0

    def _estimate_velocity(self, obs: FishingObservation) -> float:
        if (
            self._last_player_center is None
            or self._last_dt is None
            or self._last_dt <= 0.0
        ):
            self._last_player_center = obs.player_center
            self._last_dt = obs.dt
            self._v_est = 0.0
            return 0.0

        raw_v = (obs.player_center - self._last_player_center) / max(
            self._last_dt, self.vel_epsilon
        )
        self._v_est = 0.8 * self._v_est + 0.2 * raw_v
        self._last_player_center = obs.player_center
        self._last_dt = obs.dt
        return self._v_est

    def act(self, obs: FishingObservation) -> int:
        e = obs.fish_center - obs.player_center
        v = self._estimate_velocity(obs)

        if abs(e) < 1e-9 and abs(v) < self.vel_epsilon:
            return 0

        direction = 1.0 if e >= 0.0 else -1.0
        v_along_target = v * direction
        if direction > 0.0:
            accel_toward = self.a_up
            accel_brake = -self.a_down
        else:
            accel_toward = -self.a_down
            accel_brake = self.a_up

        stopping_distance = (v_along_target * v_along_target) / (
            2.0 * max(accel_brake, self.vel_epsilon)
        )

        should_accelerate_toward_target = (v_along_target < 0.0) or (
            abs(e) > stopping_distance
        )
        if should_accelerate_toward_target:
            desired_acc = direction * accel_toward
        else:
            desired_acc = -direction * accel_brake

        # action=1 gives +a_up, action=0 gives a_down
        accel_if_pressed = self.a_up
        accel_if_released = self.a_down
        if abs(accel_if_pressed - desired_acc) <= abs(accel_if_released - desired_acc):
            return 1
        return 0
