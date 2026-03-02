from __future__ import annotations

from src.gym.fishing_env import FishingObservation

from .policy import Policy


class BaselinePolicy(Policy):
    """Simple chase controller for comparison."""

    def __init__(
        self,
        deadzone: float = 0.01,
        equipment_strength: int = 0,
        equipment_expertise: int = 0,
    ) -> None:
        super().__init__(name="baseline_chase")
        self.deadzone = deadzone
        self.equipment_strength = equipment_strength
        self.equipment_expertise = equipment_expertise

    def act(self, obs: FishingObservation) -> int:
        error = obs.fish_center - obs.player_center
        if error > self.deadzone:
            return 1
        if error < -self.deadzone:
            return 0
        return 0
