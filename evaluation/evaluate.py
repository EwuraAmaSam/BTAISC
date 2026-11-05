"""Blind-labelling evaluation of the flags (assumption A8). The plan is in evaluation/PLAN.md.

  python evaluation/evaluate.py sample    draw the labelling sample (once, before labelling)
  python evaluation/evaluate.py analyse   compare your labels with the flags (also saved to results.txt)

The sample file lists episodes only. Which group each episode belongs to is
worked out again from flags.parquet at analysis time, so nothing on disk
tells the labeller which episodes were flagged.
"""

import io
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EVALUATION = ROOT / "evaluation"
SAMPLE = EVALUATION / "label_sample.csv"
LABELS = EVALUATION / "labels.csv"
RELABELS = EVALUATION / "relabels.csv"
RESULTS = EVALUATION / "results.txt"

# The rules aimed at the three in-scope researcher goals. The other three
# rules (long retreat, vase pushed far, unusual turning) are not evaluated.
IN_SCOPE_RULES = {
    "repeated_cut_throughs": "Trade-off",
    "cautious_agent_took_cost": "Trade-off",
    "long_hazard_stay": "Persistent danger",
    "repeated_hazard_entries": "Persistent danger",
    "clean_success": "Instructive success",
}
PER_RULE = 3
UNFLAGGED = 15
RELABEL = 5
SEED = 0

# Success criteria, fixed before labelling (moderate option).
MIN_PRECISION = 0.5
MIN_LIFT = 2.0


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(PROCESSED / "features.parquet").set_index("episode")
    flags = pd.read_parquet(PROCESSED / "flags.parquet")
    return features, flags[flags["level"] == "episode"]


def draw_sample() -> None:
    if SAMPLE.exists() and LABELS.exists():
        raise SystemExit(f"{LABELS.name} already exists. Redrawing now would break the evaluation.")
    features, flags = load()
    rng = np.random.default_rng(SEED)
    chosen: list[int] = []
    # Three per rule, drawn at random (not the most unusual), so no rule dominates.
    for rule in IN_SCOPE_RULES:
        candidates = [e for e in flags.loc[flags["rule"] == rule, "episode"].unique() if e not in chosen]
        chosen += rng.choice(candidates, PER_RULE, replace=False).tolist()
    unflagged = features.index[features["n_flags"] == 0].to_numpy()
    chosen += rng.choice(unflagged, UNFLAGGED, replace=False).tolist()
    order = rng.permutation(chosen)
    relabel = set(rng.choice(order, RELABEL, replace=False).tolist())
    EVALUATION.mkdir(exist_ok=True)
    pd.DataFrame({"episode": order, "relabel": [e in relabel for e in order]}).to_csv(SAMPLE, index=False)
    print(f"wrote {SAMPLE.relative_to(ROOT)}: {len(order)} episodes, {RELABEL} marked for the re-label check")


def groups(sample: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Each sampled episode's group: flagged by an in-scope rule, or unflagged."""
    in_scope = flags[flags["rule"].isin(IN_SCOPE_RULES)]
    rules = in_scope.groupby("episode")["rule"].agg(lambda r: ", ".join(sorted(r)))
    out = sample.set_index("episode")
    out["rules"] = rules.reindex(out.index)
    out["group"] = np.where(out["rules"].notna(), "flagged", "unflagged")
    return out


def analysis_text() -> str:
    """Run the analysis, save it to results.txt, and return the text."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_analysis()
    text = f"Analysis run {datetime.now().isoformat(timespec='seconds')}\n" + buffer.getvalue()
    RESULTS.write_text(text, encoding="utf-8")
    return text


def print_analysis() -> None:
    features, flags = load()
    sample = pd.read_csv(SAMPLE)
    labels = pd.read_csv(LABELS).set_index("episode")
    table = groups(sample, flags).join(labels, how="left")
    missing = table["worth_investigating"].isna().sum()
    if missing:
        print(f"note: {missing} sampled episodes are not labelled yet; results use the labelled ones only")
    table = table.dropna(subset=["worth_investigating"])

    print("\nlabels by group (counts)")
    print(pd.crosstab(table["group"], table["worth_investigating"]).to_string())

    decided = table[table["worth_investigating"] != "Unsure"]
    yes_rate = decided.groupby("group")["worth_investigating"].apply(lambda s: (s == "Yes").mean())
    precision, baseline = yes_rate.get("flagged", np.nan), yes_rate.get("unflagged", np.nan)
    lift = precision / baseline if baseline > 0 else np.inf
    print(f"\nprecision (flagged judged worth investigating): {precision:.0%}")
    print(f"baseline  (unflagged judged worth investigating): {baseline:.0%}")
    print(f"lift: {lift:.2f}" + ("  (baseline is 0, so lift is unbounded)" if baseline == 0 else ""))
    print(f"unsure answers: {(table['worth_investigating'] == 'Unsure').sum()} (left out of the rates above)")
    success = precision >= MIN_PRECISION and lift >= MIN_LIFT
    print(f"\nresult against the plan (precision >= {MIN_PRECISION:.0%} and lift >= {MIN_LIFT}): "
          f"{'SUCCESS' if success else 'NOT MET'}")

    print("\nper rule (3 episodes each, so read these as examples, not statistics)")
    for rule, goal in IN_SCOPE_RULES.items():
        rows = table[table["rules"].fillna("").str.contains(rule)]
        print(f"  {rule:26s} ({goal}): " + ", ".join(f"{e}={w}" for e, w in rows["worth_investigating"].items()))

    print("\nunflagged episodes you judged worth investigating (possible misses)")
    misses = table[(table["group"] == "unflagged") & (table["worth_investigating"] == "Yes")]
    for e, r in misses.iterrows():
        print(f"  episode {e}: {r['kinds']} | {r['note']}")

    if RELABELS.exists():
        again = pd.read_csv(RELABELS).set_index("episode").join(labels, rsuffix="_first")
        agree = (again["worth_investigating"] == again["worth_investigating_first"]).sum()
        print(f"\nself-consistency: {agree} of {len(again)} re-labels match the first answer")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "sample":
        draw_sample()
    elif command == "analyse":
        print(analysis_text())
        print(f"saved to {RESULTS.relative_to(ROOT)}")
    else:
        raise SystemExit(__doc__)
