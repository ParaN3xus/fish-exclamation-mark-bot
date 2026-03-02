from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

from src.control import BaselinePolicy, Policy, TimeOptimalBangBangPolicy
from src.gym import FishingEnv, FishingEnvConfig


@dataclass(slots=True)
class EvalResult:
    policy_name: str
    difficulty: int
    episodes: int
    success_rate: float
    avg_success_time: float
    avg_episode_time: float


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


def run_episode(env: FishingEnv, policy: Policy, seed: int, difficulty: int, render: bool, render_fps: float) -> tuple[bool, float]:
    obs, _ = env.reset(seed=seed, difficulty=difficulty)
    policy.reset()

    done = False
    truncated = False
    while not done and not truncated:
        action = policy.act(obs)
        obs, _, done, truncated, _ = env.step(action)

        if render:
            print("\x1b[2J\x1b[H", end="")
            print(f"policy={policy.name} difficulty={difficulty}")
            print(env.render_ascii())
            if render_fps > 0:
                time.sleep(1.0 / render_fps)

    return env.success, env.total_fight_time


def evaluate_policy(
    env: FishingEnv,
    policy: Policy,
    difficulties: list[int],
    episodes: int,
    seed: int,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for difficulty in difficulties:
        successes = 0
        success_time_sum = 0.0
        episode_time_sum = 0.0
        for ep in range(episodes):
            ep_seed = seed + difficulty * 100_000 + ep
            success, t = run_episode(env, policy, ep_seed, difficulty, render=False, render_fps=0.0)
            episode_time_sum += t
            if success:
                successes += 1
                success_time_sum += t

        success_rate = successes / episodes if episodes > 0 else 0.0
        avg_success_time = success_time_sum / successes if successes > 0 else math.nan
        avg_episode_time = episode_time_sum / episodes if episodes > 0 else math.nan
        results.append(
            EvalResult(
                policy_name=policy.name,
                difficulty=difficulty,
                episodes=episodes,
                success_rate=success_rate,
                avg_success_time=avg_success_time,
                avg_episode_time=avg_episode_time,
            )
        )
    return results


def print_comparison(
    bangbang_results: list[EvalResult],
    baseline_results: list[EvalResult],
) -> None:
    print("difficulty | policy                | success_rate | avg_success_time(s) | avg_episode_time(s)")
    print("-----------+-----------------------+--------------+---------------------+-------------------")
    lookup_baseline = {r.difficulty: r for r in baseline_results}
    for b in bangbang_results:
        base = lookup_baseline[b.difficulty]
        b_success_time = "nan" if math.isnan(b.avg_success_time) else f"{b.avg_success_time:7.3f}"
        base_success_time = "nan" if math.isnan(base.avg_success_time) else f"{base.avg_success_time:7.3f}"
        print(
            f"{b.difficulty:9d} | {b.policy_name:21s} | {b.success_rate:10.3%} | {b_success_time:>19s} | {b.avg_episode_time:17.3f}"
        )
        print(
            f"{'':9s} | {base.policy_name:21s} | {base.success_rate:10.3%} | {base_success_time:>19s} | {base.avg_episode_time:17.3f}"
        )
        print("-----------+-----------------------+--------------+---------------------+-------------------")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fishing minigame gym + control evaluation")
    parser.add_argument("--episodes", type=int, default=200, help="episodes per difficulty")
    parser.add_argument("--difficulties", type=str, default="1-9", help="e.g. 1-9 or 1,3,5,7,9")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="simulation step size")
    parser.add_argument("--max-steps", type=int, default=7_200, help="max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--smoothed-fps", type=float, default=60.0, help="use 60 to disable FPS assist effects")
    parser.add_argument("--equipment-strength", type=int, default=0)
    parser.add_argument("--equipment-expertise", type=int, default=0)
    parser.add_argument("--vr", action="store_true", help="simulate VR modifiers")

    parser.add_argument("--render", action="store_true", help="render one episode in ASCII")
    parser.add_argument("--render-policy", choices=["bangbang", "baseline"], default="bangbang")
    parser.add_argument("--render-difficulty", type=int, default=5)
    parser.add_argument("--render-fps", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    difficulties = parse_difficulties(args.difficulties)

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

    bangbang = TimeOptimalBangBangPolicy(player_speed=env.player_speed, gravity=env.gravity)
    baseline = BaselinePolicy()

    if args.render:
        selected_policy: Policy = bangbang if args.render_policy == "bangbang" else baseline
        difficulty = int(max(1, min(9, args.render_difficulty)))
        success, t = run_episode(
            env=env,
            policy=selected_policy,
            seed=args.seed,
            difficulty=difficulty,
            render=True,
            render_fps=args.render_fps,
        )
        print("\nresult:", "SUCCESS" if success else "FAIL", f"time={t:.3f}s")
        return

    bangbang_results = evaluate_policy(
        env=env,
        policy=bangbang,
        difficulties=difficulties,
        episodes=args.episodes,
        seed=args.seed,
    )
    baseline_results = evaluate_policy(
        env=env,
        policy=baseline,
        difficulties=difficulties,
        episodes=args.episodes,
        seed=args.seed,
    )

    print_comparison(bangbang_results, baseline_results)


if __name__ == "__main__":
    main()
