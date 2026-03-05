"""Control policies for fishing minigame."""

from .baseline_policy import BaselinePolicy
from .policy import Policy

__all__ = [
    "Policy",
    "BaselinePolicy",
]
