from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
import os
from random import Random
from typing import Any

from src.control import (
    BaselinePolicy,
    BeliefSpaceLocalOptPolicy,
    Policy,
    StochasticOutputFeedbackMPCPolicy,
    TimeOptimalBangBangPolicy,
)
from src.gym import FishingEnv, FishingEnvConfig

POLICY_CHOICES: tuple[str, ...] = ("baseline", "bangbang", "belief", "mpc")


@dataclass(slots=True, frozen=True)
class TrialTask:
    policy_key: str
    trial_id: int
    params: dict[str, Any]
    difficulties: tuple[int, ...]
    objective_difficulties: tuple[int, ...]
    episodes: int
    seeds_per_trial: int
    dt: float
    max_steps: int
    equipment_strength: int
    equipment_expertise: int
    is_vr: bool
    seed_base: int


@dataclass(slots=True)
class TrialResult:
    policy_key: str
    trial_id: int
    params: dict[str, Any]
    objective_success_rate: float
    mean_success_rate: float
    avg_success_time_objective: float
    avg_episode_time: float
    success_rate_by_difficulty: dict[int, float]


def parse_difficulties(raw: str) -> list[int]:
    text = raw.strip()
    if "-" in text:
        left, right = text.split("-", maxsplit=1)
        start = int(left)
        end = int(right)
        if start > end:
            start, end = end, start
        return [int(x) for x in range(max(1, start), min(9, end) + 1)]

    values: list[int] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if value < 1 or value > 9:
            raise ValueError(f"difficulty out of range [1,9]: {value}")
        values.append(value)
    if not values:
        raise ValueError("no difficulty parsed")
    return values


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


def resolve_workers(raw: int) -> int:
    if raw < 0:
        raise ValueError("--workers must be >= 0")
    if raw > 0:
        return raw
    return max(1, os.cpu_count() or 1)


def _symmetric_offsets(amplitude: float, count: int) -> tuple[float, ...]:
    if count <= 1:
        return (0.0,)
    center = (count - 1) / 2.0
    scale = max(1e-9, center)
    values = []
    for i in range(count):
        ratio = (i - center) / scale
        values.append(round(amplitude * ratio, 6))
    return tuple(values)


def sample_params(policy_key: str, rng: Random) -> dict[str, Any]:
    if policy_key == "baseline":
        return {"deadzone": round(rng.uniform(0.001, 0.08), 6)}

    if policy_key == "bangbang":
        return {
            "min_deadzone": round(rng.uniform(0.001, 0.04), 6),
            "hysteresis_scale": round(rng.uniform(0.2, 1.4), 6),
            "lookahead_seconds": round(rng.uniform(0.05, 0.26), 6),
        }

    if policy_key == "belief":
        return {
            "error_bins": rng.choice([61, 81, 101, 121, 141]),
            "rel_velocity_bins": rng.choice([61, 81, 101, 121, 141]),
            "horizon_steps": rng.randint(6, 16),
        }

    if policy_key == "mpc":
        scenario_count = rng.choice([3, 5, 7])
        amplitude = rng.uniform(0.1, 0.45)
        return {
            "horizon_steps": rng.randint(8, 22),
            "scenario_offsets": _symmetric_offsets(amplitude, scenario_count),
        }

    raise ValueError(f"unsupported policy key: {policy_key}")


def build_policy(policy_key: str, params: dict[str, Any]) -> Policy:
    if policy_key == "baseline":
        return BaselinePolicy(deadzone=float(params["deadzone"]))
    if policy_key == "bangbang":
        return TimeOptimalBangBangPolicy(
            min_deadzone=float(params["min_deadzone"]),
            hysteresis_scale=float(params["hysteresis_scale"]),
            lookahead_seconds=float(params["lookahead_seconds"]),
        )
    if policy_key == "belief":
        return BeliefSpaceLocalOptPolicy(
            error_bins=int(params["error_bins"]),
            rel_velocity_bins=int(params["rel_velocity_bins"]),
            horizon_steps=int(params["horizon_steps"]),
        )
    if policy_key == "mpc":
        offsets_raw = params["scenario_offsets"]
        offsets = tuple(float(x) for x in offsets_raw)
        return StochasticOutputFeedbackMPCPolicy(
            horizon_steps=int(params["horizon_steps"]),
            scenario_offsets=offsets,
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


def evaluate_trial(task: TrialTask) -> TrialResult:
    policy = build_policy(task.policy_key, task.params)
    config = FishingEnvConfig(
        dt=task.dt,
        difficulty=task.difficulties[0],
        equipment_strength=task.equipment_strength,
        equipment_expertise=task.equipment_expertise,
        is_vr=task.is_vr,
        max_steps=task.max_steps,
    )
    env = FishingEnv(config=config)

    success_counts: dict[int, int] = {d: 0 for d in task.difficulties}
    episode_counts: dict[int, int] = {d: 0 for d in task.difficulties}
    success_time_sums: dict[int, float] = {d: 0.0 for d in task.difficulties}
    total_episode_time = 0.0

    for seed_index in range(task.seeds_per_trial):
        local_seed_base = task.seed_base + seed_index * 10_000_000
        for difficulty in task.difficulties:
            for episode in range(task.episodes):
                episode_seed = local_seed_base + difficulty * 100_000 + episode
                success, episode_time = run_episode(
                    env=env,
                    policy=policy,
                    seed=episode_seed,
                    difficulty=difficulty,
                )
                episode_counts[difficulty] += 1
                total_episode_time += episode_time
                if success:
                    success_counts[difficulty] += 1
                    success_time_sums[difficulty] += episode_time

    success_rate_by_difficulty: dict[int, float] = {}
    for difficulty in task.difficulties:
        count = episode_counts[difficulty]
        success_rate_by_difficulty[difficulty] = (
            success_counts[difficulty] / count if count > 0 else 0.0
        )

    objective_rates = [
        success_rate_by_difficulty[d] for d in task.objective_difficulties
    ]
    objective_success_rate = (
        sum(objective_rates) / len(objective_rates) if objective_rates else 0.0
    )

    mean_success_rate = sum(success_rate_by_difficulty.values()) / max(
        1, len(success_rate_by_difficulty)
    )

    objective_success_count = sum(
        success_counts[d] for d in task.objective_difficulties
    )
    objective_success_time_sum = sum(
        success_time_sums[d] for d in task.objective_difficulties
    )
    avg_success_time_objective = (
        objective_success_time_sum / objective_success_count
        if objective_success_count > 0
        else math.inf
    )

    total_episodes = sum(episode_counts.values())
    avg_episode_time = (
        total_episode_time / total_episodes
        if total_episodes > 0
        else math.inf
    )

    return TrialResult(
        policy_key=task.policy_key,
        trial_id=task.trial_id,
        params=task.params,
        objective_success_rate=objective_success_rate,
        mean_success_rate=mean_success_rate,
        avg_success_time_objective=avg_success_time_objective,
        avg_episode_time=avg_episode_time,
        success_rate_by_difficulty=success_rate_by_difficulty,
    )


def _result_sort_key(result: TrialResult) -> tuple[float, float, float, float, int]:
    return (
        -result.objective_success_rate,
        -result.mean_success_rate,
        result.avg_success_time_objective,
        result.avg_episode_time,
        result.trial_id,
    )


def _format_rate_by_diff(
    success_rate_by_difficulty: dict[int, float], difficulties: tuple[int, ...]
) -> str:
    parts = []
    for difficulty in difficulties:
        rate = success_rate_by_difficulty[difficulty]
        parts.append(f"d{difficulty}:{rate:5.1%}")
    return " ".join(parts)


def _serializable_params(params: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            converted[key] = list(value)
        else:
            converted[key] = value
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel hyperparameter tuning for fishing policies."
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="bangbang,belief,mpc",
        help=(
            "comma-separated policy keys or 'all' "
            f"(choices: {','.join(POLICY_CHOICES)})"
        ),
    )
    parser.add_argument(
        "--trials-per-policy",
        type=int,
        default=120,
        help="number of random-search trials per policy",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="episodes per difficulty per seed in each trial",
    )
    parser.add_argument(
        "--seeds-per-trial",
        type=int,
        default=2,
        help="independent seed sets used for each trial",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        default="1-9",
        help="evaluation difficulties, e.g. 1-9 or 4,5,6,7,8,9",
    )
    parser.add_argument(
        "--objective-difficulties",
        type=str,
        default="4-9",
        help="difficulties used in objective mean success rate",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel workers (0=auto)",
    )
    parser.add_argument("--seed", type=int, default=42, help="base RNG seed")
    parser.add_argument("--top-k", type=int, default=5, help="top trials to print")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--max-steps", type=int, default=7_200)
    parser.add_argument("--equipment-strength", type=int, default=0)
    parser.add_argument("--equipment-expertise", type=int, default=0)
    parser.add_argument("--vr", action="store_true", help="simulate VR modifiers")
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="optional output json path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workers = resolve_workers(args.workers)
    policies = parse_policy_names(args.policies, POLICY_CHOICES)
    difficulties = tuple(parse_difficulties(args.difficulties))
    objective_difficulties = tuple(parse_difficulties(args.objective_difficulties))

    missing = [d for d in objective_difficulties if d not in difficulties]
    if missing:
        raise ValueError(
            "objective difficulties must be a subset of --difficulties, missing: "
            + ",".join(str(d) for d in missing)
        )
    if args.trials_per_policy <= 0:
        raise ValueError("--trials-per-policy must be > 0")
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.seeds_per_trial <= 0:
        raise ValueError("--seeds-per-trial must be > 0")

    tasks: list[TrialTask] = []
    for policy_index, policy_key in enumerate(policies):
        for trial_id in range(args.trials_per_policy):
            trial_seed = args.seed + policy_index * 1_000_000 + trial_id * 7_919
            rng = Random(trial_seed)
            params = sample_params(policy_key, rng)
            tasks.append(
                TrialTask(
                    policy_key=policy_key,
                    trial_id=trial_id,
                    params=params,
                    difficulties=difficulties,
                    objective_difficulties=objective_difficulties,
                    episodes=args.episodes,
                    seeds_per_trial=args.seeds_per_trial,
                    dt=args.dt,
                    max_steps=args.max_steps,
                    equipment_strength=args.equipment_strength,
                    equipment_expertise=args.equipment_expertise,
                    is_vr=args.vr,
                    seed_base=trial_seed * 97 + 11,
                )
            )

    results_by_policy: dict[str, list[TrialResult]] = {p: [] for p in policies}
    total_tasks = len(tasks)
    print(
        "starting tuning:",
        f"policies={','.join(policies)}",
        f"trials_per_policy={args.trials_per_policy}",
        f"total_trials={total_tasks}",
        f"workers={workers}",
        f"objective_difficulties={','.join(str(d) for d in objective_difficulties)}",
    )

    progress_step = max(1, total_tasks // 20)
    completed = 0

    if workers <= 1:
        for task in tasks:
            result = evaluate_trial(task)
            results_by_policy[result.policy_key].append(result)
            completed += 1
            if completed % progress_step == 0 or completed == total_tasks:
                print(f"progress {completed}/{total_tasks}")
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(evaluate_trial, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    results_by_policy[result.policy_key].append(result)
                    completed += 1
                    if completed % progress_step == 0 or completed == total_tasks:
                        print(f"progress {completed}/{total_tasks}")
        except (PermissionError, OSError):
            print(
                "process pool unavailable in current environment; "
                "falling back to serial tuning."
            )
            for task in tasks:
                result = evaluate_trial(task)
                results_by_policy[result.policy_key].append(result)
                completed += 1
                if completed % progress_step == 0 or completed == total_tasks:
                    print(f"progress {completed}/{total_tasks}")

    best_by_policy: dict[str, TrialResult] = {}
    for policy_key in policies:
        results = sorted(results_by_policy[policy_key], key=_result_sort_key)
        if not results:
            continue
        best = results[0]
        best_by_policy[policy_key] = best

        print()
        print(f"policy={policy_key} trials={len(results)}")
        for rank, result in enumerate(results[: max(1, args.top_k)], start=1):
            objective_time_text = (
                f"{result.avg_success_time_objective:.3f}"
                if math.isfinite(result.avg_success_time_objective)
                else "inf"
            )
            diff_text = _format_rate_by_diff(
                result.success_rate_by_difficulty, objective_difficulties
            )
            print(
                f"  #{rank:02d} trial={result.trial_id:04d} "
                f"obj_sr={result.objective_success_rate:7.3%} "
                f"mean_sr={result.mean_success_rate:7.3%} "
                f"obj_t={objective_time_text:>7s}s "
                f"ep_t={result.avg_episode_time:7.3f}s"
            )
            print(f"      obj_diff_rates: {diff_text}")
            print(
                "      params:",
                json.dumps(_serializable_params(result.params), ensure_ascii=False),
            )

    if best_by_policy:
        global_best = min(best_by_policy.values(), key=_result_sort_key)
        print()
        print(
            "best_overall:",
            f"policy={global_best.policy_key}",
            f"trial={global_best.trial_id}",
            f"obj_sr={global_best.objective_success_rate:7.3%}",
            f"mean_sr={global_best.mean_success_rate:7.3%}",
        )
        print(
            "best_overall_params:",
            json.dumps(_serializable_params(global_best.params), ensure_ascii=False),
        )

    if args.out_json:
        output: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "config": {
                "policies": policies,
                "trials_per_policy": args.trials_per_policy,
                "episodes": args.episodes,
                "seeds_per_trial": args.seeds_per_trial,
                "difficulties": difficulties,
                "objective_difficulties": objective_difficulties,
                "workers": workers,
                "seed": args.seed,
                "dt": args.dt,
                "max_steps": args.max_steps,
                "equipment_strength": args.equipment_strength,
                "equipment_expertise": args.equipment_expertise,
                "vr": args.vr,
            },
            "best_by_policy": {
                key: asdict(value) for key, value in best_by_policy.items()
            },
            "all_results": {
                key: [asdict(result) for result in results_by_policy[key]]
                for key in policies
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"wrote json: {args.out_json}")


if __name__ == "__main__":
    main()
