"""Turn raw steps into readable per-step columns and a table of events.

Needs data/processed/episodes.parquet (build_episode_index.py) and
data/processed/paths.parquet (build_paths.py), for vase positions.
Run order: build_episode_index.py, build_paths.py, build_events.py.

Outputs:
  data/processed/steps.parquet   one row per step, readable columns only
  data/processed/events.parquet  one row per event

Readings come from next_observations, the state after the action. The cost
of a step is decided by that state.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
PROCESSED = ROOT / "data" / "processed"

# Observation layout, inferred from value ranges and checked against cost and reward.
VELOCITY = slice(3, 5)  # forward and sideways speed in the agent's own frame
TURN_RATE = 8  # gyro, rotation about the vertical axis
GOAL_LIDAR = slice(12, 28)
HAZARD_LIDAR = slice(28, 44)
VASE_LIDAR = slice(44, 60)

GOAL_REWARD = 0.9  # goal steps pay about 1.0; no other step pays more than 0.05
NEAR_MISS_CLOSENESS = 0.9  # about 0.1 units from a hazard's edge
VASE_CLOSENESS = 0.9
# The vase position estimate jitters by at most 0.001 per step while it stands
# still, so 0.002 means it was moved. Contact is on and off, so pauses of up
# to VASE_PUSH_GAP steps are joined into one push.
VASE_PUSH_STEP = 0.002
VASE_PUSH_GAP = 5

CHUNK_ROWS = 200_000


def read_step_columns(path: Path) -> pd.DataFrame:
    """Read the file in chunks and keep only the derived columns."""
    with h5py.File(path, "r") as handle:
        n = handle["rewards"].shape[0]
        rewards = handle["rewards"][:].astype(np.float32)
        costs = handle["costs"][:].astype(np.int8)
        hazard, goal, vase, speed, turn = (np.empty(n, np.float32) for _ in range(5))
        for start in range(0, n, CHUNK_ROWS):
            rows = slice(start, min(start + CHUNK_ROWS, n))
            obs = handle["next_observations"][rows]
            hazard[rows] = obs[:, HAZARD_LIDAR].max(axis=1)
            goal[rows] = obs[:, GOAL_LIDAR].max(axis=1)
            vase[rows] = obs[:, VASE_LIDAR].max(axis=1)
            speed[rows] = np.linalg.norm(obs[:, VELOCITY], axis=1)
            turn[rows] = obs[:, TURN_RATE]
    return pd.DataFrame(
        {
            "reward": rewards,
            "cost": costs,
            "hazard_closeness": hazard,
            "goal_closeness": goal,
            "vase_closeness": vase,
            "speed": speed,
            "turn_rate": turn,
            "goal_reached": rewards > GOAL_REWARD,
        }
    )


def add_episode_columns(steps: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    episode = np.repeat(episodes["episode"].to_numpy(), episodes["length"].to_numpy())
    if episode.size != len(steps):
        raise ValueError("episode index does not cover every step; rebuild it")
    step = np.arange(len(steps)) - episodes["start_row"].to_numpy()[episode]
    steps.insert(0, "step", step.astype(np.int16))
    steps.insert(0, "episode", episode.astype(np.int16))
    return steps


def find_runs(mask: np.ndarray, first_step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start rows and stop rows (exclusive) of runs of True. Runs never cross
    an episode boundary: the first step of an episode always starts a new run."""
    previous = np.concatenate(([False], mask[:-1])) & ~first_step
    following = np.concatenate((mask[1:], [False])) & ~np.concatenate((first_step[1:], [True]))
    starts = np.flatnonzero(mask & ~previous)
    stops = np.flatnonzero(mask & ~following) + 1
    return starts, stops


def segment_reduce(values: np.ndarray, starts: np.ndarray, stops: np.ndarray, how: str) -> np.ndarray:
    """Max or sum of values over each run."""
    if starts.size == 0:
        return np.empty(0, values.dtype)
    lengths = stops - starts
    offsets = np.cumsum(lengths) - lengths  # where each run begins once runs are packed end to end
    rows = np.repeat(starts - offsets, lengths) + np.arange(lengths.sum())
    reduce = np.maximum if how == "max" else np.add
    return reduce.reduceat(values[rows], offsets)


def make_events(event_type: str, steps: pd.DataFrame, starts: np.ndarray, stops: np.ndarray, **extra) -> pd.DataFrame:
    step = steps["step"].to_numpy()
    return pd.DataFrame(
        {
            "episode": steps["episode"].to_numpy()[starts],
            "event_type": event_type,
            "start_step": step[starts],
            "stop_step": (step[starts] + (stops - starts)).astype(np.int16),
            "duration": (stops - starts).astype(np.int16),
            **extra,
        }
    )


def fill_short_gaps(mask: np.ndarray, first_step: np.ndarray, max_gap: int) -> np.ndarray:
    """Set False runs of up to max_gap steps to True when True on both sides, within an episode."""
    starts, stops = find_runs(~mask, first_step)
    inside = (starts > 0) & (stops < mask.size) & ~first_step[starts]
    inside &= ~first_step[np.minimum(stops, mask.size - 1)]
    short = inside & (stops - starts <= max_gap)
    filled = mask.copy()
    for start, stop in zip(starts[short], stops[short]):
        filled[start:stop] = True
    return filled


def vase_movement(paths: pd.DataFrame) -> np.ndarray:
    """Distance the vase moved since the previous step; 0 when it is out of view or at step 0."""
    vase = paths[["vase_x", "vase_y"]].to_numpy()
    moved = np.zeros(len(vase))
    moved[1:] = np.linalg.norm(np.diff(vase, axis=0), axis=1)
    moved[paths["step"].to_numpy() == 0] = 0
    return np.nan_to_num(moved)


def build_events(steps: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    first_step = steps["step"].to_numpy() == 0
    in_hazard = steps["cost"].to_numpy() > 0
    hazard = steps["hazard_closeness"].to_numpy()
    vase = steps["vase_closeness"].to_numpy()
    reward = steps["reward"].to_numpy()
    tables = []

    # Hazard stays. Peak closeness shows how deep into the hazard the agent got.
    starts, stops = find_runs(in_hazard, first_step)
    tables.append(make_events("hazard_stay", steps, starts, stops,
                              peak_closeness=segment_reduce(hazard, starts, stops, "max")))

    # Near misses: close without cost, and not on the way into or out of a stay.
    starts, stops = find_runs((hazard >= NEAR_MISS_CLOSENESS) & ~in_hazard, first_step)
    last_row = len(steps) - 1
    before_in = (starts > 0) & ~first_step[starts] & in_hazard[np.maximum(starts - 1, 0)]
    after_in = (stops <= last_row) & ~first_step[np.minimum(stops, last_row)] & in_hazard[np.minimum(stops, last_row)]
    keep = ~before_in & ~after_in
    starts, stops = starts[keep], stops[keep]
    tables.append(make_events("near_miss", steps, starts, stops,
                              peak_closeness=segment_reduce(hazard, starts, stops, "max")))

    # Goals reached. The first goal of an episode has no previous goal.
    goal_rows = np.flatnonzero(steps["goal_reached"].to_numpy())
    goal_episode = steps["episode"].to_numpy()[goal_rows]
    since = np.diff(goal_rows, prepend=-1).astype(np.float64)
    since[np.concatenate(([True], goal_episode[1:] != goal_episode[:-1]))] = np.nan
    tables.append(make_events("goal_reached", steps, goal_rows, goal_rows + 1,
                              steps_since_previous_goal=since))

    # Moving away from the goal. reward_change is the reward lost over the run.
    starts, stops = find_runs(reward < 0, first_step)
    tables.append(make_events("moving_away", steps, starts, stops,
                              reward_change=segment_reduce(reward.astype(np.float64), starts, stops, "sum")))

    # Close to the vase. The data shows distance, not contact.
    starts, stops = find_runs(vase >= VASE_CLOSENESS, first_step)
    tables.append(make_events("vase_close", steps, starts, stops,
                              peak_closeness=segment_reduce(vase, starts, stops, "max")))

    # Vase pushed. It costs nothing, but it is a side effect a human may care about.
    moved = vase_movement(paths)
    pushing = fill_short_gaps(moved > VASE_PUSH_STEP, first_step, VASE_PUSH_GAP)
    starts, stops = find_runs(pushing, first_step)
    tables.append(make_events("vase_pushed", steps, starts, stops,
                              vase_moved=segment_reduce(moved, starts, stops, "sum")))

    events = pd.concat(tables, ignore_index=True)
    events["event_type"] = events["event_type"].astype("category")
    return events.sort_values(["episode", "start_step", "event_type"], ignore_index=True)


def main() -> None:
    episodes = pd.read_parquet(PROCESSED / "episodes.parquet")
    steps = add_episode_columns(read_step_columns(DATA_PATH), episodes)
    paths = pd.read_parquet(PROCESSED / "paths.parquet", columns=["episode", "step", "vase_x", "vase_y"])
    if not (paths["episode"].to_numpy() == steps["episode"].to_numpy()).all():
        raise ValueError("paths.parquet is out of step with the episode index; rerun build_paths.py")
    events = build_events(steps, paths)
    steps.to_parquet(PROCESSED / "steps.parquet", index=False)
    events.to_parquet(PROCESSED / "events.parquet", index=False)

    print("steps", len(steps))
    for event_type, group in events.groupby("event_type", observed=True):
        d = group["duration"]
        print(f"{event_type:13s} {len(group):6d} events  duration median {d.median():5.0f}  max {d.max():4d}")
    print("wrote", (PROCESSED / "steps.parquet").relative_to(ROOT), "and", (PROCESSED / "events.parquet").relative_to(ROOT))


if __name__ == "__main__":
    main()
