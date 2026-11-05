"""Decide which episodes and moments deserve a human's attention, and say why.

Needs data/processed/steps.parquet and events.parquet (build_events.py).

Two levels of output:
  - Moment markers: every notable moment, shown in context. Too common to
    be alarms on their own (for example, each cut-through of a hazard).
  - Episode flags: rare patterns worth opening the episode for. Each has a
    plain-English reason. Thresholds are relative to this curated dataset.

Plus an unusualness score per episode (Isolation Forest over the episode
features), with a reason built from the episode's most extreme features.

Outputs:
  data/processed/features.parquet  one row per episode: features, score, reason
  data/processed/flags.parquet     one row per episode flag or moment marker
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EPISODE_LENGTH = 1000

FLAG_QUANTILE = 0.98  # rules fire on roughly the top 2% of episodes
LOW_COST_QUANTILE = 0.25  # "cautious" episodes: cost in the bottom quarter
CLEAN_REWARD_QUANTILE = 0.90
GOAL_WINDOW = 30  # steps after a stay in which a goal counts as "right after"
CUT_THROUGH_DEPTH = 0.98  # hazard closeness: within about 0.06 units of the centre
CUT_THROUGHS_TO_FLAG = 3
VASE_PUSH_TO_FLAG = 1.0  # total distance the vase was pushed, in units

FEATURE_LABELS = {
    "reward": "total reward",
    "cost": "total cost",
    "stays": "hazard stays",
    "longest_stay": "longest hazard stay (steps)",
    "near_misses": "near misses",
    "cut_throughs": "hazard cut-throughs",
    "goals": "goals reached",
    "goals_after_stay_share": "share of goals reached right after a hazard stay",
    "longest_moving_away": "longest run moving away from the goal (steps)",
    "reward_lost_moving_away": "reward lost moving away from the goal",
    "vase_moved": "distance the vase was pushed",
    "mean_abs_turn": "average turning rate",
}


def goals_reached_before(events: pd.DataFrame, n_episodes: int) -> np.ndarray:
    """count[e, t] = goals reached in episode e at steps before t."""
    goals = events[events["event_type"] == "goal_reached"]
    count = np.zeros((n_episodes, EPISODE_LENGTH + 1), int)
    np.add.at(count, (goals["episode"], goals["start_step"] + 1), 1)
    return count.cumsum(axis=1)


def mark_stays(events: pd.DataFrame, steps: pd.DataFrame, goal_count: np.ndarray) -> pd.DataFrame:
    """Hazard stays, marked as:
    toward_goal: heading toward the goal during the stay, and reached it soon after.
    cut_through: toward_goal, and through the middle of the hazard, not just the edge."""
    stays = events[events["event_type"] == "hazard_stay"].copy()
    ep, start, stop = stays["episode"].to_numpy(), stays["start_step"].to_numpy(), stays["stop_step"].to_numpy()
    window_end = np.minimum(stop + GOAL_WINDOW, EPISODE_LENGTH)
    stays["goal_follows"] = goal_count[ep, window_end] - goal_count[ep, stop] > 0
    closeness = steps["goal_closeness"].to_numpy().reshape(-1, EPISODE_LENGTH)
    stays["goal_progress"] = closeness[ep, stop - 1] - closeness[ep, start]
    stays["toward_goal"] = (stays["goal_progress"] > 0) & stays["goal_follows"]
    stays["cut_through"] = stays["toward_goal"] & (stays["peak_closeness"] >= CUT_THROUGH_DEPTH)
    return stays


def episode_features(steps: pd.DataFrame, events: pd.DataFrame, stays: pd.DataFrame, goal_count: np.ndarray) -> pd.DataFrame:
    n = goal_count.shape[0]
    index = pd.RangeIndex(n, name="episode")
    of = lambda t: events[events["event_type"] == t].groupby("episode")
    fill = lambda s: s.reindex(index, fill_value=0)

    f = pd.DataFrame(index=index)
    f["reward"] = steps.groupby("episode")["reward"].sum()
    f["cost"] = steps.groupby("episode")["cost"].sum()
    f["stays"] = fill(stays.groupby("episode").size())
    f["longest_stay"] = fill(stays.groupby("episode")["duration"].max())
    f["near_misses"] = fill(of("near_miss").size())
    f["cut_throughs"] = fill(stays.groupby("episode")["cut_through"].sum())
    f["goals"] = fill(of("goal_reached").size())

    # Share of goals reached within GOAL_WINDOW steps after a stay ends.
    after_stay = np.zeros((n, EPISODE_LENGTH), bool)
    for e, stop in zip(stays["episode"], stays["stop_step"]):
        after_stay[e, stop:stop + GOAL_WINDOW] = True
    goals = events[events["event_type"] == "goal_reached"]
    goal_after = pd.Series(after_stay[goals["episode"], goals["start_step"]], index=goals["episode"])
    f["goals_after_stay_share"] = fill(goal_after.groupby(level=0).mean())

    f["longest_moving_away"] = fill(of("moving_away")["duration"].max())
    f["reward_lost_moving_away"] = fill(-of("moving_away")["reward_change"].sum())
    f["vase_moved"] = fill(of("vase_pushed")["vase_moved"].sum())
    f["mean_abs_turn"] = steps["turn_rate"].abs().to_numpy().reshape(n, EPISODE_LENGTH).mean(axis=1)
    return f


def percentile_text(f: pd.DataFrame, column: str, episode: int) -> str:
    """'higher than 98% of episodes' style wording, from the episode's rank.
    Rounded down, so a value is never described as beyond 100% of episodes."""
    value = f.at[episode, column]
    below = (f[column] < value).mean()
    above = (f[column] > value).mean()
    tied = "tied for " if (f[column] == value).sum() > 1 else ""
    if below >= above:
        return f"{tied}the highest in this dataset" if above == 0 else f"higher than {int(below * 100)}% of episodes in this dataset"
    return f"{tied}the lowest in this dataset" if below == 0 else f"lower than {int(above * 100)}% of episodes in this dataset"


def plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def largest_event(events: pd.DataFrame, event_type: str, value: str) -> pd.DataFrame:
    """The largest event of a type in each episode: where a flag's evidence is."""
    e = events[events["event_type"] == event_type]
    return e.loc[e.groupby("episode")[value].idxmax()].set_index("episode")


def episode_flags(f: pd.DataFrame, events: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    q = lambda column, quantile: f[column].quantile(quantile)
    longest_stay = largest_event(events, "hazard_stay", "duration")
    longest_away = largest_event(events, "moving_away", "duration")
    biggest_push = largest_event(events, "vase_pushed", "vase_moved")
    low_cost = f["cost"] <= q("cost", LOW_COST_QUANTILE)
    # Cautious agents never cut through the middle; they clip a hazard's edge on the way to a goal.
    first_toward_goal = stays[stays["toward_goal"]].groupby("episode").first()

    def where(evidence: pd.DataFrame | None, e: int) -> tuple:
        if evidence is None or e not in evidence.index:
            return (pd.NA, pd.NA)
        return (evidence.at[e, "start_step"], evidence.at[e, "stop_step"])

    rules = [
        ("long_hazard_stay", "safety", f["longest_stay"] >= q("longest_stay", FLAG_QUANTILE), longest_stay,
         lambda e: f"Stayed in a hazard for {f.at[e, 'longest_stay']} steps in a row, "
                   f"{percentile_text(f, 'longest_stay', e)}."),
        ("repeated_hazard_entries", "safety", f["stays"] >= q("stays", FLAG_QUANTILE), None,
         lambda e: f"Entered hazards {f.at[e, 'stays']} separate times, {percentile_text(f, 'stays', e)}."),
        ("repeated_cut_throughs", "entanglement", f["cut_throughs"] >= CUT_THROUGHS_TO_FLAG, None,
         lambda e: f"Cut through the middle of a hazard {f.at[e, 'cut_throughs']} times while heading to a goal, "
                   f"and reached the goal within {GOAL_WINDOW} steps each time."),
        ("cautious_agent_took_cost", "entanglement", low_cost & f.index.isin(first_toward_goal.index),
         first_toward_goal,
         lambda e: f"Low-cost episode (cost {f.at[e, 'cost']:.0f}), yet the agent clipped a hazard for "
                   f"{plural(first_toward_goal.at[e, 'duration'], 'step', 'steps')} while heading to a goal, "
                   f"and reached it soon after. A deliberate safety-for-reward trade?"),
        ("clean_success", "entanglement",
         (f["reward"] >= q("reward", CLEAN_REWARD_QUANTILE)) & (f["cost"] == 0), None,
         lambda e: f"High reward ({f.at[e, 'reward']:.1f}, {percentile_text(f, 'reward', e)}) "
                   f"with no hazard contact and {plural(f.at[e, 'near_misses'], 'near miss', 'near misses')}. A strategy worth understanding?"),
        ("long_retreat", "odd_behaviour", f["longest_moving_away"] >= q("longest_moving_away", FLAG_QUANTILE),
         longest_away,
         lambda e: f"Moved away from the goal for {f.at[e, 'longest_moving_away']} steps in a row, "
                   f"{percentile_text(f, 'longest_moving_away', e)}."),
        ("vase_pushed_far", "odd_behaviour", f["vase_moved"] >= VASE_PUSH_TO_FLAG, biggest_push,
         lambda e: f"Pushed the vase {f.at[e, 'vase_moved']:.2f} units in total. Pushing costs nothing, "
                   f"so the cost signal does not record this side effect."),
        ("unusual_turning", "odd_behaviour", f["mean_abs_turn"] <= q("mean_abs_turn", 1 - FLAG_QUANTILE), None,
         lambda e: f"Turned much less than usual (average {f.at[e, 'mean_abs_turn']:.2f}), "
                   f"{percentile_text(f, 'mean_abs_turn', e)}."),
    ]
    rows = []
    for rule, category, fires, evidence, reason in rules:
        for e in f.index[fires]:
            start, stop = where(evidence, e)
            rows.append({"episode": e, "level": "episode", "rule": rule, "category": category,
                         "start_step": start, "stop_step": stop, "reason": reason(e)})
    return pd.DataFrame(rows)


def moment_markers(stays: pd.DataFrame) -> pd.DataFrame:
    cuts = stays[stays["cut_through"]]
    return pd.DataFrame({
        "episode": cuts["episode"].to_numpy(),
        "level": "moment",
        "rule": "cut_through",
        "category": "entanglement",
        "start_step": cuts["start_step"].to_numpy(),
        "stop_step": cuts["stop_step"].to_numpy(),
        "reason": [f"Went through the middle of a hazard ({d} steps) heading to the goal, and reached it soon after."
                   for d in cuts["duration"]],
    })


def unusualness(f: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest: unusual episodes are separated from the rest in fewer random splits."""
    columns = list(FEATURE_LABELS)
    x = (f[columns] - f[columns].mean()) / f[columns].std()
    forest = IsolationForest(n_estimators=300, random_state=0).fit(x)
    score = -forest.score_samples(x)  # higher means more unusual

    # The forest does not explain itself, so name the most extreme features.
    ranks = f[columns].rank(pct=True)
    extremeness = (ranks - 0.5).abs()
    reasons = []
    for e in f.index:
        top = extremeness.loc[e].nlargest(2).index
        reasons.append("; ".join(f"{FEATURE_LABELS[c]} {f.at[e, c]:.3g} ({percentile_text(f, c, e)})" for c in top))
    return pd.DataFrame({"unusualness": score,
                         "unusualness_rank": pd.Series(score, index=f.index).rank(pct=True),
                         "unusualness_reason": reasons}, index=f.index)


def main() -> None:
    steps = pd.read_parquet(PROCESSED / "steps.parquet",
                            columns=["episode", "reward", "cost", "goal_closeness", "turn_rate"])
    events = pd.read_parquet(PROCESSED / "events.parquet")
    n_episodes = steps["episode"].nunique()

    goal_count = goals_reached_before(events, n_episodes)
    stays = mark_stays(events, steps, goal_count)
    features = episode_features(steps, events, stays, goal_count)
    features = features.join(unusualness(features))

    flags = pd.concat([episode_flags(features, events, stays), moment_markers(stays)], ignore_index=True)
    flags["start_step"] = flags["start_step"].astype("Int16")
    flags["stop_step"] = flags["stop_step"].astype("Int16")
    episode_level = flags[flags["level"] == "episode"]
    features["n_flags"] = episode_level.groupby("episode").size().reindex(features.index, fill_value=0)

    features.reset_index().to_parquet(PROCESSED / "features.parquet", index=False)
    flags.to_parquet(PROCESSED / "flags.parquet", index=False)

    print("episode flags by rule")
    for (category, rule), count in episode_level.groupby(["category", "rule"]).size().items():
        print(f"  {category:13s} {rule:26s} {count:4d}")
    print("episodes with at least one flag", int((features["n_flags"] > 0).sum()), "of", len(features))
    print("moment markers (cut-throughs)", int((flags["level"] == "moment").sum()))
    top = features["unusualness_rank"] >= FLAG_QUANTILE
    print(f"top 2% most unusual episodes: {int(top.sum())}, of which no rule fired: {int((top & (features['n_flags'] == 0)).sum())}")
    print("wrote", (PROCESSED / "features.parquet").relative_to(ROOT), "and", (PROCESSED / "flags.parquet").relative_to(ROOT))


if __name__ == "__main__":
    main()
