from __future__ import annotations

from dataclasses import dataclass
import math
from random import Random


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _inverse_lerp(a: float, b: float, value: float) -> float:
    if a == b:
        return 0.0
    return _clamp01((value - a) / (b - a))


@dataclass(slots=True, frozen=True)
class FishingObservation:
    fish_center: float
    player_center: float
    dt: float
    difficulty: float


@dataclass(slots=True)
class FishingEnvConfig:
    dt: float = 1.0 / 60.0
    difficulty: int = 5
    equipment_strength: int = 0
    equipment_expertise: int = 0
    is_vr: bool = False
    smoothed_fps: float = 60.0
    max_steps: int = 7_200


class FishingEnv:
    """
    Numeric simulator aligned with local/FishingMinigame.cs core loop:
    - fish update (function_4)
    - player update (function_5)
    - progress update (function_6)
    - end condition (function_9)
    - difficulty parameterization (function_11)
    """

    # Constants mirrored from FishingMinigame.cs
    bar_height = 400.0
    fish_target_hitbox_size = 30.0
    player_speed = 3.0
    gravity = 1.5
    vr_target_size_bonus = 0.2
    vr_lose_speed_multiplier = 0.82
    fps_assist_max_benefit_fps = 20.0
    fps_assist_cutoff_fps = 45.0
    fps_assist_max_bonus = 0.2
    fps_fish_speed_min_multiplier = 0.6
    fps_fish_slowdown_start_difficulty = 6
    fps_fish_vr_slowdown_multiplier = 0.82
    fps_jump_size_min_multiplier = 0.55
    fps_direction_time_max_multiplier = 1.35
    easy_target_size = 120.0
    hard_target_size = 40.0
    easy_direction_time = 2.5
    hard_direction_time = 0.6
    easy_fish_smooth_time = 0.3
    hard_fish_smooth_time = 0.05
    easy_catch_speed = 3.0
    hard_catch_speed = 1.2
    easy_lose_speed = 0.8
    hard_lose_speed = 2.5
    lose_speed_escalation_rate = 0.15
    easy_max_lose_speed_multiplier = 1.5
    hard_max_lose_speed_multiplier = 3.0

    def __init__(self, config: FishingEnvConfig | None = None) -> None:
        self.config = config or FishingEnvConfig()
        self.rng = Random()

        # Runtime state (mirrors gameplay variables)
        self.current_difficulty = 5
        self.current_player_target_size = 80.0
        self.current_direction_change_time = 1.5
        self.current_catch_progress_speed = 2.0
        self.current_lose_progress_speed = 1.0
        self.current_max_lose_speed_multiplier = 3.0
        self.current_fish_decay_rate = 3.0

        self.fish_position = 0.5
        self.player_position = 0.5
        self.player_velocity = 0.0
        self.catch_progress = 0.1
        self.fish_direction_timer = 0.0
        self.fish_target_position = 0.5
        self.total_fight_time = 0.0
        self.step_count = 0
        self.terminated = False
        self.success = False

        self._apply_difficulty_parameters(self.config.difficulty)

    @property
    def overlap_threshold(self) -> float:
        assist_amount = _inverse_lerp(self.fps_assist_cutoff_fps, self.fps_assist_max_benefit_fps, self.config.smoothed_fps)
        assisted_target_size = self.current_player_target_size * (1.0 + assist_amount * self.fps_assist_max_bonus)
        return (self.fish_target_hitbox_size + assisted_target_size) / (self.bar_height * 2.0)

    def set_dt(self, dt: float) -> None:
        self.config.dt = max(1e-6, dt)

    def _apply_difficulty_parameters(self, difficulty: int) -> None:
        self.current_difficulty = int(_clamp(float(difficulty), 1.0, 9.0))
        difficulty_normalized = (float(self.current_difficulty) - 1.0) / 8.0
        difficulty_scale = difficulty_normalized**1.7

        base_target_size = _lerp(self.easy_target_size, self.hard_target_size, difficulty_normalized)
        clamped_expertise = int(_clamp(float(self.config.equipment_expertise), -100.0, 100.0))
        expertise_effect = (clamped_expertise / 100.0) * difficulty_scale
        expertise_multiplier = max(0.5, 1.0 + expertise_effect)
        self.current_player_target_size = base_target_size * expertise_multiplier

        self.current_direction_change_time = _lerp(self.easy_direction_time, self.hard_direction_time, difficulty_normalized)
        self.current_catch_progress_speed = _lerp(self.easy_catch_speed, self.hard_catch_speed, difficulty_normalized)
        self.current_lose_progress_speed = _lerp(self.easy_lose_speed, self.hard_lose_speed, difficulty_normalized)
        self.current_max_lose_speed_multiplier = _lerp(
            self.easy_max_lose_speed_multiplier,
            self.hard_max_lose_speed_multiplier,
            difficulty_normalized,
        )

        base_smooth_time = _lerp(self.easy_fish_smooth_time, self.hard_fish_smooth_time, difficulty_normalized)
        base_decay_rate = 1.0 / max(base_smooth_time, 0.001)
        clamped_strength = int(_clamp(float(self.config.equipment_strength), -100.0, 100.0))
        strength_effect = (clamped_strength / 100.0) * difficulty_scale
        strength_multiplier = _clamp(1.0 - strength_effect, 0.1, 10.0)
        self.current_fish_decay_rate = base_decay_rate * strength_multiplier

        if self.config.is_vr:
            self.current_player_target_size = self.current_player_target_size * (1.0 + self.vr_target_size_bonus)
            self.current_lose_progress_speed = self.current_lose_progress_speed * self.vr_lose_speed_multiplier

    def _observation(self) -> FishingObservation:
        return FishingObservation(
            fish_center=self.fish_position,
            player_center=self.player_position,
            dt=self.config.dt,
            difficulty=float(self.current_difficulty),
        )

    def reset(self, *, seed: int | None = None, difficulty: int | None = None) -> tuple[FishingObservation, dict[str, float]]:
        if seed is not None:
            self.rng.seed(seed)
        if difficulty is not None:
            self._apply_difficulty_parameters(difficulty)
        else:
            self._apply_difficulty_parameters(self.config.difficulty)

        self.fish_position = 0.5
        self.player_position = 0.5
        self.player_velocity = 0.0
        self.catch_progress = 0.1
        self.fish_direction_timer = 0.0
        self.fish_target_position = self.rng.uniform(0.3, 0.7)
        self.total_fight_time = 0.0
        self.step_count = 0
        self.terminated = False
        self.success = False
        return self._observation(), self._info()

    def _update_fish(self) -> None:
        dt = self.config.dt
        self.fish_direction_timer += dt

        assist_amount = _inverse_lerp(self.fps_assist_cutoff_fps, self.fps_assist_max_benefit_fps, self.config.smoothed_fps)
        dir_jump_difficulty_factor = 0.0
        if self.current_difficulty >= self.fps_fish_slowdown_start_difficulty:
            dir_jump_difficulty_factor = _clamp01(
                float(self.current_difficulty - self.fps_fish_slowdown_start_difficulty)
                / (9.0 - float(self.fps_fish_slowdown_start_difficulty))
            )

        combined_dir_jump_assist = assist_amount * dir_jump_difficulty_factor
        effective_direction_change_time = self.current_direction_change_time
        if combined_dir_jump_assist > 0.0:
            effective_direction_change_time = effective_direction_change_time * _lerp(
                1.0, self.fps_direction_time_max_multiplier, combined_dir_jump_assist
            )
        effective_direction_change_time = max(1e-6, effective_direction_change_time)

        while self.fish_direction_timer >= effective_direction_change_time:
            self.fish_direction_timer -= effective_direction_change_time
            raw_target = self.rng.uniform(0.01, 0.99)
            difficulty_normalized = (float(self.current_difficulty) - 1.0) / 8.0
            max_jump = _lerp(0.18, 0.3, difficulty_normalized)

            if combined_dir_jump_assist > 0.0:
                target_min_jump = _clamp(self.fps_jump_size_min_multiplier, 0.25, 1.0)
                jump_mul = _lerp(1.0, target_min_jump, combined_dir_jump_assist)
                if self.config.is_vr:
                    jump_mul = jump_mul * _lerp(1.0, self.fps_fish_vr_slowdown_multiplier, assist_amount)
                max_jump = max_jump * _clamp(jump_mul, 0.05, 1.0)

            clamped_target = _clamp(raw_target, self.fish_position - max_jump, self.fish_position + max_jump)
            self.fish_target_position = _clamp01(clamped_target)

        effective_decay_rate = self.current_fish_decay_rate
        difficulty_slow_factor = 0.0
        if self.current_difficulty >= self.fps_fish_slowdown_start_difficulty:
            difficulty_slow_factor = _clamp01(
                float(self.current_difficulty - self.fps_fish_slowdown_start_difficulty)
                / (9.0 - float(self.fps_fish_slowdown_start_difficulty))
            )
        if assist_amount > 0.0 and difficulty_slow_factor > 0.0:
            target_min = _clamp(self.fps_fish_speed_min_multiplier, 0.01, 1.0)
            combined = assist_amount * difficulty_slow_factor
            fps_slow_multiplier = _lerp(1.0, target_min, combined)
            if self.config.is_vr:
                fps_slow_multiplier = fps_slow_multiplier * _lerp(
                    1.0, self.fps_fish_vr_slowdown_multiplier, assist_amount
                )
            effective_decay_rate = effective_decay_rate * fps_slow_multiplier

        alpha = 1.0 - math.exp(-effective_decay_rate * dt)
        self.fish_position = _clamp01(self.fish_position + (self.fish_target_position - self.fish_position) * alpha)

    def _update_player(self, is_input_pressed: bool) -> None:
        dt = self.config.dt
        self.player_velocity = self.player_velocity - self.gravity * dt
        if is_input_pressed:
            self.player_velocity = self.player_velocity + self.player_speed * dt

        self.player_position = self.player_position + self.player_velocity * dt
        self.player_position = _clamp01(self.player_position)
        if self.player_position <= 0.0 or self.player_position >= 1.0:
            self.player_velocity = self.player_velocity * -0.3

    def _update_progress(self) -> None:
        dt = self.config.dt
        distance = abs(self.fish_position - self.player_position)
        catching_fish = distance < self.overlap_threshold
        if catching_fish:
            self.catch_progress += self.current_catch_progress_speed * dt
        else:
            grace_period_multiplier = _clamp01((self.total_fight_time - 1.0) / 4.0)
            escalation_multiplier = 1.0 + self.total_fight_time * self.lose_speed_escalation_rate
            escalation_multiplier = min(escalation_multiplier, self.current_max_lose_speed_multiplier)
            modified_lose_speed = self.current_lose_progress_speed * escalation_multiplier * grace_period_multiplier
            self.catch_progress -= modified_lose_speed * dt
        self.catch_progress = _clamp01(self.catch_progress)

    def _info(self) -> dict[str, float]:
        return {
            "fish_position": self.fish_position,
            "player_position": self.player_position,
            "player_velocity": self.player_velocity,
            "fish_target_position": self.fish_target_position,
            "catch_progress": self.catch_progress,
            "total_fight_time": self.total_fight_time,
            "current_difficulty": float(self.current_difficulty),
            "overlap_threshold": self.overlap_threshold,
            "current_player_target_size": self.current_player_target_size,
        }

    def step(self, action: int | bool) -> tuple[FishingObservation, float, bool, bool, dict[str, float]]:
        if self.terminated:
            raise RuntimeError("Episode already terminated. Call reset() before step().")

        is_input_pressed = bool(action)
        previous_progress = self.catch_progress

        self.total_fight_time += self.config.dt
        self.step_count += 1

        self._update_fish()
        self._update_player(is_input_pressed)
        self._update_progress()

        reward = self.catch_progress - previous_progress
        truncated = self.step_count >= self.config.max_steps

        if self.catch_progress >= 1.0:
            self.terminated = True
            self.success = True
            reward += 1.0
        elif self.catch_progress <= 0.0:
            self.terminated = True
            self.success = False
            reward -= 1.0
        elif truncated:
            self.terminated = True
            self.success = False

        return self._observation(), reward, self.terminated, truncated, self._info()

    def render_ascii(self, width: int = 80) -> str:
        width = max(20, width)
        fish_idx = int(round(self.fish_position * (width - 1)))
        player_idx = int(round(self.player_position * (width - 1)))
        half_window = int(round(self.overlap_threshold * (width - 1)))
        left = max(0, player_idx - half_window)
        right = min(width - 1, player_idx + half_window)

        chars = [" "] * width
        for idx in range(left, right + 1):
            chars[idx] = "="
        chars[player_idx] = "P"
        chars[fish_idx] = "F" if fish_idx != player_idx else "*"
        bar = "".join(chars)
        return (
            f"|{bar}|\n"
            f"time={self.total_fight_time:6.2f}s progress={self.catch_progress:5.3f} "
            f"fish={self.fish_position:5.3f} player={self.player_position:5.3f} vel={self.player_velocity:6.3f}"
        )
