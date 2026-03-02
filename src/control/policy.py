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
