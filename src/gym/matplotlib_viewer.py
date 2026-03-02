from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
import time
from typing import Any

from src.control import Policy
from src.gym.fishing_env import FishingEnv

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from matplotlib.animation import FuncAnimation  # pyright: ignore[reportMissingImports]
from matplotlib.patches import Rectangle  # pyright: ignore[reportMissingImports]


@dataclass(slots=True)
class RenderEpisodeSummary:
    run_index: int
    seed: int
    success: bool
    total_time: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def render_matplotlib_runs(
    env: FishingEnv,
    policy: Policy,
    *,
    base_seed: int,
    difficulty: int,
    runs: int,
    window_seconds: float,
    render_fps: float,
    time_scale: float,
    max_steps_per_frame: int,
) -> list[RenderEpisodeSummary]:
    """
    Render one or multiple episodes in a single matplotlib window.

    Layout:
    - Left: scrolling fish/player trajectory curves
    - Right: one vertical overlay strip (fish + player)
    """

    difficulty = int(_clamp(float(difficulty), 1.0, 9.0))
    window_seconds = max(0.5, window_seconds)
    render_fps = max(1.0, render_fps)
    time_scale = max(1e-6, time_scale)
    max_steps_per_frame = max(1, max_steps_per_frame)
    dt = env.config.dt
    history_maxlen = max(100, int(window_seconds / max(dt, 1e-4)) + 10)

    rng = Random(base_seed)
    summaries: list[RenderEpisodeSummary] = []

    fig = plt.figure(figsize=(12.0, 6.5))
    curve_ax = fig.add_axes((0.07, 0.12, 0.70, 0.80))
    strip_ax = fig.add_axes((0.82, 0.12, 0.12, 0.80))

    curve_ax.set_ylim(0.0, 1.0)
    curve_ax.set_xlabel("time (s)")
    curve_ax.set_ylabel("normalized position")
    curve_ax.grid(True, alpha=0.25)

    strip_ax.set_xlim(0.0, 1.0)
    strip_ax.set_ylim(0.0, 1.0)
    strip_ax.set_xticks([])
    strip_ax.set_yticks([0.0, 0.5, 1.0])
    strip_ax.set_title("Minigame GUI")

    (fish_line,) = curve_ax.plot([], [], color="#1f77b4", lw=1.8, label="fish")
    (player_line,) = curve_ax.plot([], [], color="#ff7f0e", lw=1.8, label="player")
    (progress_line,) = curve_ax.plot([], [], color="#2ca02c", lw=1.8, label="progress")
    progress_text = curve_ax.text(
        0.01, 0.98, "", transform=curve_ax.transAxes, va="top", ha="left"
    )
    curve_ax.legend(loc="upper right")

    strip_bg = Rectangle(
        (0.2, 0.0), 0.6, 1.0, facecolor="#f5f6f8", edgecolor="#333333", lw=1.0
    )
    strip_ax.add_patch(strip_bg)
    fish_hitbox = Rectangle(
        (0.24, 0.0), 0.52, 0.0, facecolor="#3b82f6", edgecolor="none", alpha=0.45
    )
    player_box = Rectangle(
        (0.22, 0.0), 0.56, 0.0, facecolor="#f59e0b", edgecolor="none", alpha=0.45
    )
    strip_ax.add_patch(fish_hitbox)
    strip_ax.add_patch(player_box)
    fish_marker = strip_ax.scatter(
        [0.38], [0.5], s=42, c=["#1d4ed8"], zorder=5, label="fish center"
    )
    player_marker = strip_ax.scatter(
        [0.62], [0.5], s=42, c=["#92400e"], zorder=5, label="player center"
    )
    strip_ax.legend(loc="lower center", fontsize=8, framealpha=0.8)

    times: deque[float] = deque(maxlen=history_maxlen)
    fish_hist: deque[float] = deque(maxlen=history_maxlen)
    player_hist: deque[float] = deque(maxlen=history_maxlen)
    progress_hist: deque[float] = deque(maxlen=history_maxlen)

    run_idx = 0
    current_seed = 0
    current_obs: Any = None
    pause_frames = 0
    max_runs = runs
    last_wall_time = time.perf_counter()
    sim_time_budget = 0.0

    def start_new_episode() -> bool:
        nonlocal run_idx, current_seed, current_obs, last_wall_time, sim_time_budget
        if max_runs > 0 and run_idx >= max_runs:
            return False
        current_seed = rng.randint(0, 2_000_000_000)
        current_obs, _ = env.reset(seed=current_seed, difficulty=difficulty)
        policy.reset()
        times.clear()
        fish_hist.clear()
        player_hist.clear()
        progress_hist.clear()
        times.append(env.total_fight_time)
        fish_hist.append(env.fish_position)
        player_hist.append(env.player_position)
        progress_hist.append(env.catch_progress)
        last_wall_time = time.perf_counter()
        sim_time_budget = 0.0
        run_idx += 1
        return True

    if not start_new_episode():
        return summaries

    def update_right_strips() -> None:
        fish_half = env.fish_target_hitbox_size / (env.bar_height * 2.0)
        player_half = env.current_player_target_size / (env.bar_height * 2.0)

        fish_y0 = _clamp(env.fish_position - fish_half, 0.0, 1.0)
        fish_y1 = _clamp(env.fish_position + fish_half, 0.0, 1.0)
        fish_hitbox.set_y(fish_y0)
        fish_hitbox.set_height(max(0.001, fish_y1 - fish_y0))
        fish_marker.set_offsets([[0.38, env.fish_position]])

        player_y0 = _clamp(env.player_position - player_half, 0.0, 1.0)
        player_y1 = _clamp(env.player_position + player_half, 0.0, 1.0)
        player_box.set_y(player_y0)
        player_box.set_height(max(0.001, player_y1 - player_y0))
        player_marker.set_offsets([[0.62, env.player_position]])

    def update_curve_artists() -> None:
        fish_line.set_data(list(times), list(fish_hist))
        player_line.set_data(list(times), list(player_hist))
        progress_line.set_data(list(times), list(progress_hist))
        t_now = env.total_fight_time
        curve_ax.set_xlim(max(0.0, t_now - window_seconds), max(window_seconds, t_now))

        progress_text.set_text(
            f"policy={policy.name}  "
            f"run={run_idx}/{max_runs if max_runs > 0 else 'inf'}  "
            f"seed={current_seed}\n"
            f"difficulty={difficulty}  progress={env.catch_progress:.3f}  "
            f"time={env.total_fight_time:.3f}s"
        )
        update_right_strips()

    def _finalize_and_maybe_restart() -> bool:
        nonlocal pause_frames
        summaries.append(
            RenderEpisodeSummary(
                run_index=run_idx,
                seed=current_seed,
                success=env.success,
                total_time=env.total_fight_time,
            )
        )
        pause_frames = int(0.6 * render_fps)
        if max_runs > 0 and run_idx >= max_runs:
            return False
        return True

    interval_ms = int(1000.0 / render_fps)

    def _on_close(_: Any) -> None:
        anim.event_source.stop()

    def animate(_frame: int) -> tuple[Any, ...]:
        nonlocal current_obs, pause_frames, last_wall_time, sim_time_budget
        now = time.perf_counter()
        elapsed_wall = max(0.0, now - last_wall_time)
        last_wall_time = now

        if pause_frames > 0:
            pause_frames -= 1
            if pause_frames == 0:
                if not start_new_episode():
                    anim.event_source.stop()
            update_curve_artists()
            return (
                fish_line,
                player_line,
                progress_line,
                progress_text,
                fish_hitbox,
                player_box,
                fish_marker,
                player_marker,
            )

        sim_time_budget += elapsed_wall * time_scale
        step_count = int(sim_time_budget / max(dt, 1e-9))
        if step_count <= 0:
            update_curve_artists()
            return (
                fish_line,
                player_line,
                progress_line,
                progress_text,
                fish_hitbox,
                player_box,
                fish_marker,
                player_marker,
            )
        step_count = min(step_count, max_steps_per_frame)

        done = False
        truncated = False
        for _ in range(step_count):
            action = policy.act(current_obs)
            current_obs, _reward, done, truncated, _info = env.step(action)

            times.append(env.total_fight_time)
            fish_hist.append(env.fish_position)
            player_hist.append(env.player_position)
            progress_hist.append(env.catch_progress)

            sim_time_budget -= dt
            if sim_time_budget < 0.0:
                sim_time_budget = 0.0
            if done or truncated:
                break

        update_curve_artists()

        if done or truncated:
            should_continue = _finalize_and_maybe_restart()
            if not should_continue:
                anim.event_source.stop()
                plt.close(fig)

        return (
            fish_line,
            player_line,
            progress_line,
            progress_text,
            fish_hitbox,
            player_box,
            fish_marker,
            player_marker,
        )

    update_curve_artists()
    fig.canvas.mpl_connect("close_event", _on_close)
    anim = FuncAnimation(
        fig, animate, interval=interval_ms, blit=False, cache_frame_data=False
    )
    plt.show()
    return summaries
