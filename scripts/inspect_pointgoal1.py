"""Inspect the DSRL SafetyPointGoal1 offline dataset.

The file is one sequence of transitions. Episode boundaries are the steps
where terminals or timeouts is true.
"""

from pathlib import Path

import h5py
import numpy as np

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
REQUIRED_KEYS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "costs",
    "terminals",
    "timeouts",
)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        print("file", path.name)
        print("size_mb", round(path.stat().st_size / 1e6, 1))
        print("keys", sorted(handle.keys()))
        data = {}
        for key in REQUIRED_KEYS:
            if key not in handle:
                raise KeyError(f"missing {key}")
            data[key] = handle[key][:]
            print(f"  {key}: shape={data[key].shape} dtype={data[key].dtype}")
        extra = sorted(set(handle.keys()) - set(REQUIRED_KEYS))
        for key in extra:
            value = handle[key][:]
            print(f"  {key}: shape={value.shape} dtype={value.dtype}")
            data[key] = value
    return data


def episode_bounds(terminals: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    done = np.flatnonzero((terminals.reshape(-1) > 0) | (timeouts.reshape(-1) > 0))
    if done.size == 0:
        return np.array([[0, len(terminals)]], dtype=int)
    starts = np.concatenate(([0], done[:-1] + 1))
    ends = done + 1
    return np.column_stack((starts, ends))


def summarize(data: dict[str, np.ndarray]) -> None:
    rewards = data["rewards"].reshape(-1).astype(np.float64)
    costs = data["costs"].reshape(-1).astype(np.float64)
    bounds = episode_bounds(data["terminals"], data["timeouts"])
    lengths = bounds[:, 1] - bounds[:, 0]
    ep_reward = np.array([rewards[start:end].sum() for start, end in bounds])
    ep_cost = np.array([costs[start:end].sum() for start, end in bounds])
    ep_cost_steps = np.array([(costs[start:end] > 0).sum() for start, end in bounds])

    print("\ntransitions", rewards.size)
    print("episodes", len(bounds))
    print("length min/median/max", int(lengths.min()), float(np.median(lengths)), int(lengths.max()))
    print("reward per step min/mean/max", rewards.min(), rewards.mean(), rewards.max())
    print("cost per step min/mean/max", costs.min(), costs.mean(), costs.max())
    print("unique step costs", np.unique(costs)[:12], "count", np.unique(costs).size)
    print("fraction of steps with cost > 0", float((costs > 0).mean()))
    print("terminal steps", int((data["terminals"].reshape(-1) > 0).sum()))
    print("timeout steps", int((data["timeouts"].reshape(-1) > 0).sum()))
    print("episode reward min/p25/median/p75/max")
    print(" ", np.quantile(ep_reward, [0, 0.25, 0.5, 0.75, 1]).round(3))
    print("episode cost min/p25/median/p75/max")
    print(" ", np.quantile(ep_cost, [0, 0.25, 0.5, 0.75, 1]).round(3))
    print("episodes with zero cost", int((ep_cost == 0).sum()))

    reward_hi = np.quantile(ep_reward, 0.75)
    cost_hi = np.quantile(ep_cost, 0.75)
    cost_lo = np.quantile(ep_cost, 0.25)
    high_reward_high_cost = np.flatnonzero((ep_reward >= reward_hi) & (ep_cost >= cost_hi))
    high_reward_low_cost = np.flatnonzero((ep_reward >= reward_hi) & (ep_cost <= cost_lo))
    print("\nhigh-reward threshold (p75)", round(reward_hi, 3))
    print("high-cost threshold (p75)", round(cost_hi, 3))
    print("low-cost threshold (p25)", round(cost_lo, 3))
    print("high reward and high cost", high_reward_high_cost.size)
    print("high reward and low cost", high_reward_low_cost.size)

    print("\nexample high-reward high-cost episodes")
    order = high_reward_high_cost[np.argsort(-ep_cost[high_reward_high_cost])][:5]
    for index in order:
        start, end = bounds[index]
        print(
            f"  episode {index}: length {end - start}, "
            f"reward {ep_reward[index]:.3f}, cost {ep_cost[index]:.3f}, "
            f"costly steps {int(ep_cost_steps[index])}"
        )

    print("\nexample high-reward low-cost episodes")
    order = high_reward_low_cost[np.argsort(-ep_reward[high_reward_low_cost])][:5]
    for index in order:
        start, end = bounds[index]
        print(
            f"  episode {index}: length {end - start}, "
            f"reward {ep_reward[index]:.3f}, cost {ep_cost[index]:.3f}, "
            f"costly steps {int(ep_cost_steps[index])}"
        )

    # Cost in this dataset is 0 or 1, so a "spike" is a run of consecutive costly steps.
    burst_lengths = []
    for start, end in bounds:
        costly = costs[start:end] > 0
        if not costly.any():
            continue
        changes = np.diff(np.concatenate(([0], costly.view(np.int8), [0])))
        burst_lengths.extend((np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).tolist())
    bursts = np.array(burst_lengths) if burst_lengths else np.array([0])
    reward_spike_level = rewards.mean() + 5 * rewards.std()
    reward_spike_steps = int((rewards >= reward_spike_level).sum())
    print("\ncost bursts", bursts.size)
    print("burst length min/median/p75/max", int(bursts.min()), float(np.median(bursts)), float(np.quantile(bursts, 0.75)), int(bursts.max()))
    print("reward spike level (mean + 5 std)", round(float(reward_spike_level), 3))
    print("steps at or above that reward", reward_spike_steps)

    # Show the cost trace of the highest-cost episode so the shape is visible.
    richest = int(np.argmax(ep_cost))
    start, end = bounds[richest]
    trace = costs[start:end]
    positive_idx = np.flatnonzero(trace > 0)
    print("\nhighest-cost episode", richest)
    print("  length", end - start, "reward", round(float(ep_reward[richest]), 3), "cost", round(float(ep_cost[richest]), 3))
    print("  costly steps", positive_idx.size)
    if positive_idx.size:
        preview = positive_idx[:15]
        print("  first costly step indexes and costs", list(zip(preview.tolist(), trace[preview].round(3).tolist())))


if __name__ == "__main__":
    summarize(load_dataset(DATA_PATH))
