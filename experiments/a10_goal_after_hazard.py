"""Test assumption A10: do goals follow hazard stays more often than chance?

Plan fixed before running:
  - Window: a goal within WINDOW steps after the event ends. Events whose
    window would run past the end of the episode are excluded.
  - A: hazard stays vs near misses, all episodes.
  - B: the same, only in episodes that have both (controls for agent type).
  - C: chance rate, the share of all WINDOW-step windows in the same
    episodes that contain a goal.
  - Extra: change in goal closeness from the start to the end of each event.
Decision rule: ratio < 1.2 drop or redefine the flag; > 1.5 keep it;
strong only for high-cost agents means it describes agents, not incidents.

Needs data/processed/steps.parquet and events.parquet (scripts/build_events.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
WINDOW = 30
EPISODE_LENGTH = 1000
BOOTSTRAP_ROUNDS = 2000
rng = np.random.default_rng(0)


def goal_steps_by_episode(events: pd.DataFrame) -> np.ndarray:
    """Boolean array [episode, step], True where a goal was reached."""
    n_episodes = events["episode"].max() + 1
    goals = np.zeros((n_episodes, EPISODE_LENGTH), bool)
    g = events[events["event_type"] == "goal_reached"]
    goals[g["episode"], g["start_step"]] = True
    return goals


def goal_follows(events: pd.DataFrame, goals: np.ndarray) -> pd.DataFrame:
    """For each event, whether a goal is reached within WINDOW steps after it ends."""
    events = events[events["stop_step"] + WINDOW <= EPISODE_LENGTH].copy()
    cumulative = np.concatenate((np.zeros((len(goals), 1), int), goals.cumsum(axis=1)), axis=1)
    ep, stop = events["episode"].to_numpy(), events["stop_step"].to_numpy()
    events["goal_follows"] = cumulative[ep, stop + WINDOW] - cumulative[ep, stop] > 0
    return events


def chance_rate(goals: np.ndarray, episodes: np.ndarray) -> float:
    """Share of all WINDOW-step windows in these episodes (weighted as given) containing a goal."""
    cumulative = np.concatenate((np.zeros((len(goals), 1), int), goals.cumsum(axis=1)), axis=1)
    per_episode = (cumulative[:, WINDOW:] - cumulative[:, :-WINDOW] > 0).mean(axis=1)
    return float(per_episode[episodes].mean())


def ratio_interval(stays: pd.DataFrame, near: pd.DataFrame) -> tuple[float, float]:
    """95% bootstrap interval for the stay/near-miss ratio, resampling whole episodes."""
    episodes = np.union1d(stays["episode"].unique(), near["episode"].unique())
    s = stays.groupby("episode")["goal_follows"].agg(["sum", "count"]).reindex(episodes, fill_value=0).to_numpy()
    n = near.groupby("episode")["goal_follows"].agg(["sum", "count"]).reindex(episodes, fill_value=0).to_numpy()
    ratios = []
    for _ in range(BOOTSTRAP_ROUNDS):
        pick = rng.integers(0, len(episodes), len(episodes))
        ss, nn = s[pick].sum(axis=0), n[pick].sum(axis=0)
        ratios.append((ss[0] / ss[1]) / (nn[0] / nn[1]))
    return tuple(np.percentile(ratios, [2.5, 97.5]))


def report(label: str, stays: pd.DataFrame, near: pd.DataFrame, goals: np.ndarray) -> None:
    rs, rn = stays["goal_follows"].mean(), near["goal_follows"].mean()
    lo, hi = ratio_interval(stays, near)
    print(f"\n{label}")
    print(f"  after hazard stays: {rs:6.1%}  (n={len(stays)}, chance in same episodes {chance_rate(goals, stays['episode'].to_numpy()):.1%})")
    print(f"  after near misses:  {rn:6.1%}  (n={len(near)}, chance in same episodes {chance_rate(goals, near['episode'].to_numpy()):.1%})")
    print(f"  ratio stays / near misses: {rs / rn:.2f}  (95% interval {lo:.2f} to {hi:.2f})")


def main() -> None:
    steps = pd.read_parquet(PROCESSED / "steps.parquet", columns=["episode", "step", "cost", "goal_closeness"])
    events = pd.read_parquet(PROCESSED / "events.parquet")
    goals = goal_steps_by_episode(events)

    tested = goal_follows(events[events["event_type"].isin(["hazard_stay", "near_miss"])], goals)
    stays = tested[tested["event_type"] == "hazard_stay"]
    near = tested[tested["event_type"] == "near_miss"]
    print(f"window: goal within {WINDOW} steps after the event ends")

    report("A. All episodes", stays, near, goals)

    both = np.intersect1d(stays["episode"].unique(), near["episode"].unique())
    report(f"B. Same episodes only ({len(both)} episodes with both)",
           stays[stays["episode"].isin(both)], near[near["episode"].isin(both)], goals)

    episode_cost = steps.groupby("episode")["cost"].sum()
    high = episode_cost[episode_cost > episode_cost.median()].index
    for name, keep in [("higher-cost half", True), ("lower-cost half", False)]:
        s = stays[stays["episode"].isin(high) == keep]
        n = near[near["episode"].isin(high) == keep]
        report(f"Split: {name} of episodes", s, n, goals)

    # Extra: was the robot heading toward the goal during the event?
    closeness = steps["goal_closeness"].to_numpy().reshape(-1, EPISODE_LENGTH)
    for name, group in [("hazard stays", stays), ("near misses", near)]:
        ep, start, stop = group["episode"].to_numpy(), group["start_step"].to_numpy(), group["stop_step"].to_numpy() - 1
        change = closeness[ep, stop] - closeness[ep, start]
        print(f"\ngoal closeness change during {name}: median {np.median(change):+.3f}, "
              f"getting closer in {np.mean(change > 0):.0%}")


if __name__ == "__main__":
    main()
