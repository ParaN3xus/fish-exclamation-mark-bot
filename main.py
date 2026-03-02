from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import os
from dataclasses import dataclass

from src.control import (
    BaselinePolicy,
    BeliefSpaceLocalOptPolicy,
    Policy,
    StochasticOutputFeedbackMPCPolicy,
    TimeOptimalBangBangPolicy,
)
from src.gym import FishingEnv, FishingEnvConfig


@dataclass(slots=True)
class EvalResult:
    policy_name: str
    difficulty: int
    episodes: int
    success_rate: float
    avg_success_time: float
    avg_episode_time: float


@dataclass(slots=True, frozen=True)
class EvalTask:
    policy_key: str
    difficulty: int
    seed_base: int
    episode_start: int
    episode_count: int
    dt: float
    max_steps: int
    equipment_strength: int
    equipment_expertise: int
    smoothed_fps: float
    is_vr: bool


@dataclass(slots=True)
class _EvalAccumulator:
    policy_name: str
    episodes: int = 0
    success_count: int = 0
    success_time_sum: float = 0.0
    episode_time_sum: float = 0.0


def parse_difficulties(raw: str) -> list[int]:
    raw = raw.strip()
    if "-" in raw:
        left, right = raw.split("-", maxsplit=1)
        start = int(left)
        end = int(right)
        if start > end:
            start, end = end, start
        return [int(x) for x in range(max(1, start), min(9, end) + 1)]
    result: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1 or value > 9:
            raise ValueError(f"difficulty out of range [1,9]: {value}")
        result.append(value)
    if not result:
        raise ValueError("no difficulty parsed")
    return result


def parse_policy_names(raw: str, available: tuple[str, ...]) -> list[str]:
    normalized = raw.strip().lower()
    if normalized == "all":
        return list(available)

    names: list[str] = []
    seen: set[str] = set()
    for token in normalized.split(","):
        name = token.strip()
        if not name:
            continue
        if name not in available:
            raise ValueError(
                f"unknown policy '{name}', available: {', '.join(available)}"
            )
        if name not in seen:
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError("no policy parsed")
    return names


def resolve_eval_workers(raw: int) -> int:
    if raw < 0:
        raise ValueError("--eval-workers must be >= 0")
    if raw > 0:
        return raw
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)


def build_policy(
    policy_key: str,
    *,
    player_speed: float,
    gravity: float,
    equipment_strength: int,
    equipment_expertise: int,
    smoothed_fps: float,
    is_vr: bool,
) -> Policy:
    if policy_key == "mpc":
        return StochasticOutputFeedbackMPCPolicy(
            player_speed=player_speed,
            gravity=gravity,
            equipment_strength=equipment_strength,
            equipment_expertise=equipment_expertise,
            smoothed_fps=smoothed_fps,
            is_vr=is_vr,
        )
    if policy_key == "bangbang":
        return TimeOptimalBangBangPolicy(
            player_speed=player_speed,
            gravity=gravity,
            equipment_strength=equipment_strength,
            equipment_expertise=equipment_expertise,
            smoothed_fps=smoothed_fps,
            is_vr=is_vr,
        )
    if policy_key == "belief":
        return BeliefSpaceLocalOptPolicy(
            player_speed=player_speed,
            gravity=gravity,
            equipment_strength=equipment_strength,
            equipment_expertise=equipment_expertise,
            smoothed_fps=smoothed_fps,
            is_vr=is_vr,
        )
    if policy_key == "baseline":
        return BaselinePolicy(
            equipment_strength=equipment_strength,
            equipment_expertise=equipment_expertise,
        )
    raise ValueError(f"unsupported policy key: {policy_key}")


def run_episode(
    env: FishingEnv,
    policy: Policy,
    seed: int,
    difficulty: int,
) -> tuple[bool, float]:
    obs, _ = env.reset(seed=seed, difficulty=difficulty)
    policy.reset()

    done = False
    truncated = False
    while not done and not truncated:
        action = policy.act(obs)
        obs, _, done, truncated, _ = env.step(action)

    return env.success, env.total_fight_time


def run_eval_task(
    task: EvalTask,
) -> tuple[str, int, str, int, int, float, float]:
    config = FishingEnvConfig(
        dt=task.dt,
        difficulty=task.difficulty,
        equipment_strength=task.equipment_strength,
        equipment_expertise=task.equipment_expertise,
        is_vr=task.is_vr,
        smoothed_fps=task.smoothed_fps,
        max_steps=task.max_steps,
    )
    env = FishingEnv(config=config)
    policy = build_policy(
        task.policy_key,
        player_speed=env.player_speed,
        gravity=env.gravity,
        equipment_strength=task.equipment_strength,
        equipment_expertise=task.equipment_expertise,
        smoothed_fps=task.smoothed_fps,
        is_vr=task.is_vr,
    )

    success_count = 0
    success_time_sum = 0.0
    episode_time_sum = 0.0
    for ep in range(task.episode_start, task.episode_start + task.episode_count):
        ep_seed = task.seed_base + task.difficulty * 100_000 + ep
        success, total_time = run_episode(
            env=env,
            policy=policy,
            seed=ep_seed,
            difficulty=task.difficulty,
        )
        episode_time_sum += total_time
        if success:
            success_count += 1
            success_time_sum += total_time

    return (
        task.policy_key,
        task.difficulty,
        policy.name,
        task.episode_count,
        success_count,
        success_time_sum,
        episode_time_sum,
    )


def evaluate_policies(
    *,
    policy_keys: list[str],
    difficulties: list[int],
    episodes: int,
    seed: int,
    dt: float,
    max_steps: int,
    equipment_strength: int,
    equipment_expertise: int,
    smoothed_fps: float,
    is_vr: bool,
    eval_workers: int,
) -> dict[str, list[EvalResult]]:
    tasks: list[EvalTask] = []
    accumulators: dict[tuple[str, int], _EvalAccumulator] = {}

    chunk_size = (
        episodes
        if eval_workers <= 1
        else max(1, episodes // (eval_workers * 2))
    )
    for key in policy_keys:
        for difficulty in difficulties:
            policy = build_policy(
                key,
                player_speed=FishingEnv.player_speed,
                gravity=FishingEnv.gravity,
                equipment_strength=equipment_strength,
                equipment_expertise=equipment_expertise,
                smoothed_fps=smoothed_fps,
                is_vr=is_vr,
            )
            accumulators[(key, difficulty)] = _EvalAccumulator(policy_name=policy.name)

            for start in range(0, episodes, chunk_size):
                count = min(chunk_size, episodes - start)
                tasks.append(
                    EvalTask(
                        policy_key=key,
                        difficulty=difficulty,
                        seed_base=seed,
                        episode_start=start,
                        episode_count=count,
                        dt=dt,
                        max_steps=max_steps,
                        equipment_strength=equipment_strength,
                        equipment_expertise=equipment_expertise,
                        smoothed_fps=smoothed_fps,
                        is_vr=is_vr,
                    )
                )

    def consume(
        result: tuple[str, int, str, int, int, float, float],
    ) -> None:
        (
            key,
            difficulty,
            policy_name,
            episode_count,
            success_count,
            success_time_sum,
            episode_time_sum,
        ) = result
        acc = accumulators[(key, difficulty)]
        acc.policy_name = policy_name
        acc.episodes += episode_count
        acc.success_count += success_count
        acc.success_time_sum += success_time_sum
        acc.episode_time_sum += episode_time_sum

    if eval_workers <= 1:
        for task in tasks:
            consume(run_eval_task(task))
    else:
        try:
            with ProcessPoolExecutor(max_workers=eval_workers) as executor:
                for result in executor.map(run_eval_task, tasks):
                    consume(result)
        except (PermissionError, OSError):
            print(
                "process pool unavailable in current environment; "
                "falling back to serial evaluation."
            )
            for task in tasks:
                consume(run_eval_task(task))

    results_by_key: dict[str, list[EvalResult]] = {}
    for key in policy_keys:
        results: list[EvalResult] = []
        for difficulty in difficulties:
            acc = accumulators[(key, difficulty)]
            success_rate = acc.success_count / acc.episodes if acc.episodes > 0 else 0.0
            avg_success_time = (
                acc.success_time_sum / acc.success_count
                if acc.success_count > 0
                else math.nan
            )
            avg_episode_time = (
                acc.episode_time_sum / acc.episodes
                if acc.episodes > 0
                else math.nan
            )
            results.append(
                EvalResult(
                    policy_name=acc.policy_name,
                    difficulty=difficulty,
                    episodes=acc.episodes,
                    success_rate=success_rate,
                    avg_success_time=avg_success_time,
                    avg_episode_time=avg_episode_time,
                )
            )
        results_by_key[key] = results

    return results_by_key


def print_results_table(
    results_by_key: dict[str, list[EvalResult]],
    policy_order: list[str],
    difficulties: list[int],
) -> None:
    print(
        "difficulty | policy                        | success_rate | "
        "avg_success_time(s) | avg_episode_time(s)"
    )
    print(
        "-----------+-------------------------------+--------------+---------------------+-------------------"
    )

    lookup: dict[str, dict[int, EvalResult]] = {
        key: {result.difficulty: result for result in results}
        for key, results in results_by_key.items()
    }

    for difficulty in difficulties:
        for idx, key in enumerate(policy_order):
            result = lookup[key][difficulty]
            success_time = (
                "nan"
                if math.isnan(result.avg_success_time)
                else f"{result.avg_success_time:7.3f}"
            )
            difficulty_text = f"{difficulty:9d}" if idx == 0 else f"{'':9s}"
            print(
                f"{difficulty_text} | {result.policy_name:29s} | "
                f"{result.success_rate:10.3%} | {success_time:>19s} | "
                f"{result.avg_episode_time:17.3f}"
            )
        print(
            "-----------+-------------------------------+--------------+---------------------+-------------------"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fishing minigame gym + control evaluation"
    )
    parser.add_argument(
        "--episodes", type=int, default=200, help="episodes per difficulty"
    )
    parser.add_argument(
        "--difficulties", type=str, default="1-9", help="e.g. 1-9 or 1,3,5,7,9"
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="mpc,bangbang,belief,baseline",
        help=(
            "comma-separated policies or 'all' "
            "(choices: mpc,bangbang,belief,baseline)"
        ),
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=0,
        help="parallel worker processes for evaluation (0=auto, 1=serial)",
    )
    parser.add_argument(
        "--dt", type=float, default=1.0 / 60.0, help="simulation step size"
    )
    parser.add_argument(
        "--max-steps", type=int, default=7_200, help="max steps per episode"
    )
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument(
        "--smoothed-fps",
        type=float,
        default=60.0,
        help="use 60 to disable FPS assist effects",
    )
    parser.add_argument("--equipment-strength", type=int, default=0)
    parser.add_argument("--equipment-expertise", type=int, default=0)
    parser.add_argument("--vr", action="store_true", help="simulate VR modifiers")

    parser.add_argument(
        "--render",
        action="store_true",
        help="render in matplotlib (supports multiple episodes)",
    )
    parser.add_argument(
        "--render-policy",
        choices=["mpc", "bangbang", "belief", "baseline"],
        default="mpc",
    )
    parser.add_argument("--render-difficulty", type=int, default=5)
    parser.add_argument(
        "--render-fps", type=float, default=60.0, help="matplotlib update FPS"
    )
    parser.add_argument(
        "--render-runs",
        type=int,
        default=5,
        help="number of episodes to run in one render session (0=infinite)",
    )
    parser.add_argument(
        "--render-window-seconds",
        type=float,
        default=8.0,
        help="left-side scrolling curve window in seconds",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    difficulties = parse_difficulties(args.difficulties)
    eval_workers = resolve_eval_workers(args.eval_workers)

    available_policies = ("mpc", "bangbang", "belief", "baseline")
    selected_keys = parse_policy_names(args.policies, available_policies)

    if args.render:
        from src.gym.matplotlib_viewer import render_matplotlib_runs

        config = FishingEnvConfig(
            dt=args.dt,
            difficulty=difficulties[0],
            equipment_strength=args.equipment_strength,
            equipment_expertise=args.equipment_expertise,
            is_vr=args.vr,
            smoothed_fps=args.smoothed_fps,
            max_steps=args.max_steps,
        )
        env = FishingEnv(config=config)
        selected_policy = build_policy(
            args.render_policy,
            player_speed=env.player_speed,
            gravity=env.gravity,
            equipment_strength=args.equipment_strength,
            equipment_expertise=args.equipment_expertise,
            smoothed_fps=args.smoothed_fps,
            is_vr=args.vr,
        )
        difficulty = int(max(1, min(9, args.render_difficulty)))
        summaries = render_matplotlib_runs(
            env=env,
            policy=selected_policy,
            base_seed=args.seed,
            difficulty=difficulty,
            runs=max(0, args.render_runs),
            window_seconds=args.render_window_seconds,
            render_fps=args.render_fps,
        )
        if summaries:
            total_runs = len(summaries)
            success_runs = sum(1 for s in summaries if s.success)
            avg_time = sum(s.total_time for s in summaries) / total_runs
            avg_success_time = (
                sum(s.total_time for s in summaries if s.success) / success_runs
                if success_runs > 0
                else math.nan
            )
            avg_success_text = (
                f"{avg_success_time:.3f}s"
                if not math.isnan(avg_success_time)
                else "nan"
            )
            success_rate_text = f"{success_runs / total_runs:.3%}"
            print(
                f"render session done: runs={total_runs} "
                f"success_rate={success_rate_text} "
                f"avg_time={avg_time:.3f}s avg_success_time={avg_success_text}"
            )
            for s in summaries:
                result_label = "SUCCESS" if s.success else "FAIL"
                print(
                    f"run={s.run_index} seed={s.seed} result={result_label} "
                    f"time={s.total_time:.3f}s"
                )
        else:
            print("render session done: no episode executed.")
        return

    results_by_key = evaluate_policies(
        policy_keys=selected_keys,
        difficulties=difficulties,
        episodes=args.episodes,
        seed=args.seed,
        dt=args.dt,
        max_steps=args.max_steps,
        equipment_strength=args.equipment_strength,
        equipment_expertise=args.equipment_expertise,
        smoothed_fps=args.smoothed_fps,
        is_vr=args.vr,
        eval_workers=eval_workers,
    )

    print_results_table(results_by_key, selected_keys, difficulties)


if __name__ == "__main__":
    main()
