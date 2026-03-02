from __future__ import annotations

import math

from src.gym.fishing_env import FishingObservation

from .policy import Policy


class TimeOptimalBangBangPolicy(Policy):
    """
    Success-first bang-bang strategy grounded in minimum-time control.

    Core idea:
    1) Treat relative motion (fish - player) as an asymmetric double integrator.
    2) At each step, solve the one-switch minimum-time transfer analytically and
       take the first control of the best sequence.
    3) Add robust prediction and progress-risk protection so optimization target is
       not only shortest centering time, but sustained capture probability.
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
        initial_difficulty_estimate: float = 6.0,
    ) -> None:
        super().__init__(name="time_optimal_bangbang")
        self.player_speed = player_speed
        self.gravity = gravity
        self.vel_epsilon = vel_epsilon
        self.equipment_strength = equipment_strength
        self.equipment_expertise = equipment_expertise
        self.smoothed_fps = smoothed_fps
        self.is_vr = is_vr
        self.initial_difficulty_estimate = self._clamp(
            float(initial_difficulty_estimate), 1.0, 9.0
        )

        # Net player accelerations (from env update equation).
        self.a_up = player_speed - gravity
        self.a_down = gravity
        if self.a_up <= 0.0:
            raise ValueError(
                "player_speed must be larger than gravity for controllability."
            )

        # Constants mirrored from env / game source.
        self.bar_height = 2.8
        self.fish_target_hitbox_size = 0.1
        self.easy_target_size = 1.2
        self.hard_target_size = 0.7
        self.easy_direction_time = 0.5
        self.hard_direction_time = 0.4
        self.easy_fish_smooth_time = 1.0
        self.hard_fish_smooth_time = 0.19
        self.easy_catch_speed = 0.2
        self.hard_catch_speed = 0.06
        self.easy_lose_speed = 0.1
        self.hard_lose_speed = 0.15
        self.lose_speed_escalation_rate = 0.1
        self.easy_max_lose_speed_multiplier = 1.0
        self.hard_max_lose_speed_multiplier = 3.0
        self.fps_assist_cutoff_fps = 30.0
        self.fps_assist_max_benefit_fps = 15.0
        self.fps_assist_max_bonus = 0.05
        self.vr_target_size_bonus = 0.04

        self._last_player_center: float | None = None
        self._last_fish_center: float | None = None
        self._v_est = 0.0
        self._fish_v_est = 0.0
        self._last_action = 0

        # Hidden-state reconstruction for fish first-order model.
        self._fish_target_est = 0.5
        self._time_since_target_change = 0.0
        self._difficulty_est = self.initial_difficulty_estimate

        # Progress estimate reconstructed from observations.
        self._progress_est = 0.1
        self._elapsed_est = 0.0
        self._have_step_observation = False

    def reset(self) -> None:
        self._last_player_center = None
        self._last_fish_center = None
        self._v_est = 0.0
        self._fish_v_est = 0.0
        self._last_action = 0

        self._fish_target_est = 0.5
        self._time_since_target_change = 0.0
        self._difficulty_est = self.initial_difficulty_estimate

        self._progress_est = 0.1
        self._elapsed_est = 0.0
        self._have_step_observation = False

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _clamp01(value: float) -> float:
        return TimeOptimalBangBangPolicy._clamp(value, 0.0, 1.0)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _difficulty_normalized(self, difficulty: float) -> float:
        return (self._clamp(difficulty, 1.0, 9.0) - 1.0) / 8.0

    def _difficulty_scale(self, difficulty: float) -> float:
        return self._difficulty_normalized(difficulty) ** 1.7

    def _direction_time(self, difficulty: float) -> float:
        return self._lerp(
            self.easy_direction_time,
            self.hard_direction_time,
            self._difficulty_normalized(difficulty),
        )

    def _fish_decay_rate(self, difficulty: float) -> float:
        base_smooth_time = self._lerp(
            self.easy_fish_smooth_time,
            self.hard_fish_smooth_time,
            self._difficulty_normalized(difficulty),
        )
        base_decay_rate = 1.0 / max(base_smooth_time, 0.001)
        clamped_strength = int(
            self._clamp(float(self.equipment_strength), -100.0, 100.0)
        )
        strength_effect = (clamped_strength / 100.0) * self._difficulty_scale(
            difficulty
        )
        strength_multiplier = self._clamp(1.0 - strength_effect, 0.1, 10.0)
        return base_decay_rate * strength_multiplier

    def _catch_speed(self, difficulty: float) -> float:
        return self._lerp(
            self.easy_catch_speed,
            self.hard_catch_speed,
            self._difficulty_normalized(difficulty),
        )

    def _base_lose_speed(self, difficulty: float) -> float:
        return self._lerp(
            self.easy_lose_speed,
            self.hard_lose_speed,
            self._difficulty_normalized(difficulty),
        )

    def _max_lose_multiplier(self, difficulty: float) -> float:
        return self._lerp(
            self.easy_max_lose_speed_multiplier,
            self.hard_max_lose_speed_multiplier,
            self._difficulty_normalized(difficulty),
        )

    def _overlap_threshold(self, difficulty: float) -> float:
        d_norm = self._difficulty_normalized(difficulty)
        difficulty_scale = self._difficulty_scale(difficulty)

        clamped_exp = int(self._clamp(float(self.equipment_expertise), -100.0, 100.0))
        base_target_size = self._lerp(
            self.easy_target_size,
            self.hard_target_size,
            d_norm,
        )
        expertise_effect = (clamped_exp / 100.0) * difficulty_scale
        expertise_mul = max(0.5, 1.0 + expertise_effect)
        player_target_size = base_target_size * expertise_mul
        if self.is_vr:
            player_target_size = player_target_size * (1.0 + self.vr_target_size_bonus)

        assist_amount = self._clamp01(
            (self.smoothed_fps - self.fps_assist_cutoff_fps)
            / (self.fps_assist_max_benefit_fps - self.fps_assist_cutoff_fps)
        )
        assisted_target_size = player_target_size * (
            1.0 + assist_amount * self.fps_assist_max_bonus
        )
        return (self.fish_target_hitbox_size + assisted_target_size) / (
            self.bar_height * 2.0
        )

    def _lose_speed_at(self, elapsed_time: float, difficulty: float) -> float:
        grace = self._clamp01((elapsed_time - 1.0) / 4.0)
        escalation = min(
            1.0 + elapsed_time * self.lose_speed_escalation_rate,
            self._max_lose_multiplier(difficulty),
        )
        return self._base_lose_speed(difficulty) * escalation * grace

    def _progress_delta(
        self,
        *,
        catching: bool,
        elapsed_time: float,
        difficulty: float,
        dt: float,
    ) -> float:
        if catching:
            return self._catch_speed(difficulty) * dt
        return -self._lose_speed_at(elapsed_time, difficulty) * dt

    def _update_progress_estimate(self, obs: FishingObservation) -> None:
        if not self._have_step_observation:
            self._have_step_observation = True
            return

        dt = max(obs.dt, self.vel_epsilon)
        self._elapsed_est += dt
        difficulty = self._difficulty_est
        threshold = self._overlap_threshold(difficulty)
        catching = abs(obs.fish_center - obs.player_center) < threshold
        delta = self._progress_delta(
            catching=catching,
            elapsed_time=self._elapsed_est,
            difficulty=difficulty,
            dt=dt,
        )
        self._progress_est = self._clamp01(self._progress_est + delta)

    def _estimate_kinematics(self, obs: FishingObservation) -> tuple[float, float]:
        if self._last_player_center is None or self._last_fish_center is None:
            self._last_player_center = obs.player_center
            self._last_fish_center = obs.fish_center
            self._v_est = 0.0
            self._fish_v_est = 0.0
            self._fish_target_est = obs.fish_center
            self._time_since_target_change = 0.0
            return 0.0, 0.0

        dt = max(obs.dt, self.vel_epsilon)
        raw_player_v = (obs.player_center - self._last_player_center) / dt
        raw_fish_v = (obs.fish_center - self._last_fish_center) / dt
        self._v_est = 0.8 * self._v_est + 0.2 * raw_player_v
        self._fish_v_est = 0.65 * self._fish_v_est + 0.35 * raw_fish_v

        difficulty = self._difficulty_est
        decay = self._fish_decay_rate(difficulty)
        alpha = 1.0 - math.exp(-decay * dt)
        if alpha > 1e-6:
            inferred_target = self._last_fish_center + (
                (obs.fish_center - self._last_fish_center) / alpha
            )
        else:
            inferred_target = obs.fish_center
        inferred_target = self._clamp01(inferred_target)

        d_norm = self._difficulty_normalized(difficulty)
        jump_detect_threshold = 0.035 + 0.065 * d_norm
        jump_distance = abs(inferred_target - self._fish_target_est)
        if jump_distance > jump_detect_threshold:
            observed_interval = max(dt, self._time_since_target_change + dt)
            interval_norm = self._clamp01(
                (self.easy_direction_time - observed_interval) / 0.1
            )
            jump_norm = self._clamp01((jump_distance - 0.18) / 0.12)
            obs_norm = 0.45 * interval_norm + 0.55 * jump_norm
            current_norm = self._difficulty_normalized(self._difficulty_est)
            blended_norm = 0.85 * current_norm + 0.15 * obs_norm
            self._difficulty_est = self._clamp(1.0 + 8.0 * blended_norm, 1.0, 9.0)
            self._time_since_target_change = 0.0
        else:
            self._time_since_target_change += dt
        self._fish_target_est = 0.7 * self._fish_target_est + 0.3 * inferred_target

        self._last_player_center = obs.player_center
        self._last_fish_center = obs.fish_center
        return self._v_est, self._fish_v_est

    def _predict_fish_center(
        self,
        *,
        fish_center: float,
        difficulty: float,
        danger: float,
        fish_velocity: float,
    ) -> float:
        d_norm = self._difficulty_normalized(difficulty)
        horizon = 0.10 + 0.14 * d_norm + 0.12 * danger

        decay = self._fish_decay_rate(difficulty)
        alpha_h = 1.0 - math.exp(-decay * horizon)

        direction_time = self._direction_time(difficulty)
        if direction_time <= 1e-6:
            change_uncertainty = 0.0
        else:
            # As we approach expected re-direction time, avoid over-committing.
            change_uncertainty = self._clamp01(
                (self._time_since_target_change - 0.72 * direction_time)
                / (0.35 * direction_time)
            )
        blended_target = self._lerp(
            self._fish_target_est,
            0.5 + 0.25 * self._clamp(fish_velocity, -1.0, 1.0),
            0.55 * change_uncertainty,
        )

        return self._clamp01(fish_center + (blended_target - fish_center) * alpha_h)

    def _max_fish_jump(self, difficulty: float) -> float:
        return self._lerp(0.18, 0.3, self._difficulty_normalized(difficulty))

    def _step_player(
        self,
        player_center: float,
        player_velocity: float,
        dt: float,
        action: int,
    ) -> tuple[float, float]:
        v = player_velocity - self.gravity * dt
        if action == 1:
            v += self.player_speed * dt
        x = player_center + v * dt
        if x <= 0.0:
            return 0.0, -0.3 * v
        if x >= 1.0:
            return 1.0, -0.3 * v
        return x, v

    @staticmethod
    def _quadratic_roots(a: float, b: float, c: float) -> tuple[float, float] | None:
        if abs(a) <= 1e-12:
            if abs(b) <= 1e-12:
                return None
            root = -c / b
            return root, root
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)
        return ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a))

    def _min_time_first_action(self, y0: float, r0: float) -> int:
        # Relative dynamics y_dot=r, r_dot in {+a_down, -a_up}.
        # Evaluate both one-switch sequences analytically.
        best_time = float("inf")
        best_action = 1 if y0 + 0.2 * r0 > 0.0 else 0

        # (start_action, w1, w2)
        # action=1 => r_dot=-a_up ; action=0 => r_dot=+a_down
        sequences = (
            (1, -self.a_up, +self.a_down),
            (0, +self.a_down, -self.a_up),
        )

        for start_action, w1, w2 in sequences:
            # y(t_end)=0 with one switch gives quadratic in t1:
            # A t1^2 + B t1 + C = 0
            a = 0.5 * w1 * (1.0 - w1 / w2)
            b = r0 * (1.0 - w1 / w2)
            c = y0 - 0.5 * (r0 * r0) / w2
            roots = self._quadratic_roots(a, b, c)
            if roots is None:
                continue

            for t1 in roots:
                if t1 < 0.0:
                    continue
                r1 = r0 + w1 * t1
                t2 = -r1 / w2
                if t2 < 0.0:
                    continue
                total_time = t1 + t2
                if total_time < best_time:
                    best_time = total_time
                    best_action = start_action

        return best_action

    def _evaluate_first_action_robust(
        self,
        *,
        first_action: int,
        obs: FishingObservation,
        player_velocity: float,
        danger: float,
        threshold: float,
    ) -> float:
        dt = max(obs.dt, self.vel_epsilon)
        horizon_steps = max(8, min(22, int(round((0.22 + 0.22 * danger) / dt))))
        difficulty = self._difficulty_est
        d_norm = self._difficulty_normalized(difficulty)

        direction_time = self._direction_time(difficulty)
        time_to_next_change = max(0.0, direction_time - self._time_since_target_change)
        alpha = 1.0 - math.exp(-self._fish_decay_rate(difficulty) * dt)
        max_jump = self._max_fish_jump(difficulty)

        low_target = self._clamp01(
            self._clamp(
                0.01,
                obs.fish_center - max_jump,
                obs.fish_center + max_jump,
            )
        )
        high_target = self._clamp01(
            self._clamp(
                0.99,
                obs.fish_center - max_jump,
                obs.fish_center + max_jump,
            )
        )
        mid_target = 0.5 * (low_target + high_target)

        scenarios: tuple[float, ...]
        if time_to_next_change > horizon_steps * dt + 1e-9:
            scenarios = (self._fish_target_est,)
        else:
            scenarios = (low_target, mid_target, high_target)

        scenario_scores: list[float] = []
        for scenario_target_after_change in scenarios:
            fish = obs.fish_center
            fish_target = self._fish_target_est
            time_since_change = self._time_since_target_change
            player = obs.player_center
            player_v_sim = player_velocity
            progress = self._progress_est
            elapsed = self._elapsed_est
            min_progress = progress
            outside_time = 0.0
            fish_vel = self._fish_v_est

            for step_idx in range(horizon_steps):
                time_since_change += dt
                if time_since_change >= direction_time:
                    time_since_change -= direction_time
                    fish_target = scenario_target_after_change

                fish_prev = fish
                fish = self._clamp01(fish + (fish_target - fish) * alpha)
                fish_vel = (fish - fish_prev) / dt

                if step_idx == 0:
                    action = first_action
                else:
                    rel_v = fish_vel - player_v_sim
                    target_bias = (0.08 + 0.12 * danger) * threshold
                    y = (fish - player) - target_bias
                    action = self._min_time_first_action(y, rel_v)

                player, player_v_sim = self._step_player(
                    player,
                    player_v_sim,
                    dt,
                    action,
                )

                elapsed += dt
                catching = abs(fish - player) < threshold
                progress += self._progress_delta(
                    catching=catching,
                    elapsed_time=elapsed,
                    difficulty=difficulty,
                    dt=dt,
                )
                progress = self._clamp01(progress)
                min_progress = min(min_progress, progress)
                if not catching:
                    outside_time += dt

            terminal_error = fish - player - (0.10 + 0.06 * danger) * threshold
            terminal_rel_v = fish_vel - player_v_sim
            scenario_score = (
                2.2 * progress
                + 1.3 * min_progress
                - 1.2 * outside_time
                - 1.5 * terminal_error * terminal_error
                - 0.08 * terminal_rel_v * terminal_rel_v
            )
            scenario_scores.append(scenario_score)

        scenario_scores.sort()
        worst = scenario_scores[0]
        mean = sum(scenario_scores) / float(len(scenario_scores))
        risk_weight = 0.65 + 0.3 * danger + 0.1 * d_norm
        return risk_weight * worst + (1.0 - risk_weight) * mean

    def act(self, obs: FishingObservation) -> int:
        self._update_progress_estimate(obs)
        player_v, fish_v = self._estimate_kinematics(obs)

        dt = max(obs.dt, self.vel_epsilon)
        difficulty = self._difficulty_est
        threshold = self._overlap_threshold(difficulty)
        error_now = obs.fish_center - obs.player_center
        rel_v = fish_v - player_v

        danger = self._clamp01((0.4 - self._progress_est) / 0.4)
        predicted_fish = self._predict_fish_center(
            fish_center=obs.fish_center,
            difficulty=difficulty,
            danger=danger,
            fish_velocity=fish_v,
        )

        # Stay slightly below fish center to absorb downward slips.
        target_bias = (0.08 + 0.10 * danger) * threshold
        y0 = (predicted_fish - obs.player_center) - target_bias

        recovery_band = threshold * (1.8 + 1.2 * danger)
        if abs(error_now) > recovery_band:
            action = 1 if (error_now + 0.22 * rel_v) > 0.0 else 0
            self._last_action = action
            return action

        # Boundary protection to avoid wasting time on wall bounces.
        if obs.player_center > 0.985 and player_v > 0.0:
            self._last_action = 0
            return 0
        if obs.player_center < 0.015 and player_v < 0.0:
            self._last_action = 1
            return 1

        action = self._min_time_first_action(y0, rel_v)

        # In-band regulation with hysteresis reduces chattering and misses.
        safe_band = threshold * (0.78 - 0.12 * danger)
        if abs(error_now) < safe_band and abs(rel_v) < (0.26 + 0.18 * danger):
            s = y0 + 0.12 * rel_v
            hysteresis = (0.03 + 0.03 * danger) * threshold
            if self._last_action == 1:
                s -= hysteresis
            else:
                s += hysteresis
            action = 1 if s > 0.0 else 0

        # One-step risk check: choose action that better preserves overlap when fragile.
        if danger > 0.2:
            fish_next = self._clamp01(obs.fish_center + fish_v * dt)
            x0, _ = self._step_player(obs.player_center, player_v, dt, 0)
            x1, _ = self._step_player(obs.player_center, player_v, dt, 1)
            out0 = max(0.0, abs(fish_next - x0) - threshold)
            out1 = max(0.0, abs(fish_next - x1) - threshold)
            if out0 + 1e-9 < out1:
                action = 0
            elif out1 + 1e-9 < out0:
                action = 1

        if difficulty >= 7.0 or danger > 0.15:
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
            if score0 > score1 + 1e-10:
                action = 0
            elif score1 > score0 + 1e-10:
                action = 1

        self._last_action = action
        return action
