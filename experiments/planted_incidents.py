"""Planted incidents: does the tool catch a persistent-danger incident we inserted?

Take 10 ordinary episodes (no flags, short hazard stays only). Extend each
one's longest hazard stay to a planted length, rerun the event and flag
logic on the modified data, and check:
  - does the long-stay rule fire? (its threshold is the top 2%, about 47 steps)
  - how unusual does the Isolation Forest now rank the episode?

The rule part mostly confirms the code works, because the threshold is known.
The score part is the informative one: the forest is never told about the rule.

Needs the pipeline outputs in data/processed/. Writes evaluation/planted_results.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_events  # noqa: E402
import build_flags  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "evaluation" / "planted_results.csv"
PLANTED_LENGTHS = [30, 47, 60, 80]
HOSTS = 10
INSIDE_CLOSENESS = 0.96  # a reading clearly inside a hazard (cost starts at 0.933)
SEED = 0


def pick_hosts(features: pd.DataFrame, events: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Unflagged episodes whose longest stay is short and has room after it to grow to the largest size."""
    rng = np.random.default_rng(SEED)
    stays = events[events["event_type"] == "hazard_stay"]
    longest = stays.loc[stays.groupby("episode")["duration"].idxmax()].set_index("episode")
    cost = steps["cost"].to_numpy().reshape(-1, build_flags.EPISODE_LENGTH)
    candidates = []
    for e in features.index[features["n_flags"] == 0]:
        if e not in longest.index or longest.at[e, "duration"] >= min(PLANTED_LENGTHS):
            continue
        stop = longest.at[e, "stop_step"]
        room = max(PLANTED_LENGTHS) - longest.at[e, "duration"]
        if stop + room <= build_flags.EPISODE_LENGTH and not cost[e, stop:stop + room].any():
            candidates.append(e)
    chosen = rng.choice(candidates, HOSTS, replace=False)
    return longest.loc[chosen, ["start_step", "stop_step", "duration"]]


def plant(steps: pd.DataFrame, hosts: pd.DataFrame, length: int) -> pd.DataFrame:
    """Extend each host's longest stay so it lasts `length` steps."""
    planted = steps.copy()
    for e, h in hosts.astype(int).iterrows():  # stored as int16; row numbers need more room
        first = int(e) * build_flags.EPISODE_LENGTH
        rows = slice(first + h.stop_step, first + h.start_step + length)
        planted.iloc[rows, planted.columns.get_loc("cost")] = 1
        closeness = planted.iloc[rows, planted.columns.get_loc("hazard_closeness")]
        planted.iloc[rows, planted.columns.get_loc("hazard_closeness")] = np.maximum(closeness, INSIDE_CLOSENESS)
    return planted


def flag_everything(steps: pd.DataFrame, paths: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same logic as build_events.py and build_flags.py, on in-memory data."""
    events = build_events.build_events(steps, paths)
    goal_count = build_flags.goals_reached_before(events, steps["episode"].nunique())
    stays = build_flags.mark_stays(events, steps, goal_count)
    features = build_flags.episode_features(steps, events, stays, goal_count)
    features = features.join(build_flags.unusualness(features))
    flags = build_flags.episode_flags(features, events, stays)
    return features, flags


def main() -> None:
    steps = pd.read_parquet(PROCESSED / "steps.parquet")
    paths = pd.read_parquet(PROCESSED / "paths.parquet", columns=["episode", "step", "vase_x", "vase_y"])
    features = pd.read_parquet(PROCESSED / "features.parquet").set_index("episode")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    hosts = pick_hosts(features, events, steps)
    print(f"host episodes: {hosts.index.tolist()}")
    print(f"their longest stays before planting: {hosts['duration'].tolist()}")
    print(f"their unusualness before planting: more unusual than "
          f"{features.loc[hosts.index, 'unusualness_rank'].median():.0%} of episodes (median)")

    rows = []
    for length in PLANTED_LENGTHS:
        new_features, new_flags = flag_everything(plant(steps, hosts, length), paths)
        threshold = new_features["longest_stay"].quantile(build_flags.FLAG_QUANTILE)
        long_stay = set(new_flags.loc[new_flags["rule"] == "long_hazard_stay", "episode"])
        for e in hosts.index:
            rows.append({"planted_length": length, "episode": e,
                         "measured_longest_stay": new_features.at[e, "longest_stay"],
                         "rule_threshold": threshold, "long_stay_rule_fired": e in long_stay,
                         "unusualness_rank_before": features.at[e, "unusualness_rank"],
                         "unusualness_rank_after": new_features.at[e, "unusualness_rank"],
                         "in_top_2_percent": new_features.at[e, "unusualness_rank"] >= 0.98})
        print(f"planted {length} steps done")

    results = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    results.to_csv(OUT, index=False)
    summary = results.groupby("planted_length").agg(
        measured=("measured_longest_stay", "median"), threshold=("rule_threshold", "first"),
        rule_caught=("long_stay_rule_fired", "mean"), rank_before=("unusualness_rank_before", "median"),
        rank_after=("unusualness_rank_after", "median"), top_2_percent=("in_top_2_percent", "mean"))
    print("\nplanted length | measured | rule threshold | rule caught | unusualness before -> after (median) | in top 2%")
    for length, r in summary.iterrows():
        print(f"  {length:3d} steps     | {r.measured:5.0f}    | {r.threshold:5.0f}          | {r.rule_caught:5.0%}       |"
              f" {r.rank_before:.0%} -> {r.rank_after:.0%}                          | {r.top_2_percent:.0%}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
