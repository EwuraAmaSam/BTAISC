"""Rebuild where the robot went, and the map around it, from its own sensors.

Needs data/processed/episodes.parquet from build_episode_index.py.

Outputs:
  data/processed/paths.parquet  one row per step: position, heading, vase position
  data/processed/map.parquet    one row per hazard or goal, per episode

Method (dead reckoning): the compass gives the heading, the velocity sensor
gives speed in the robot's own frame. Turning that speed into world
directions and adding it up step by step gives the path. Lidar readings,
taken from each point on the path, then place the hazards, goals and vase.

Positions are in the environment's distance units, with the origin where the
episode starts. Every episode has its own layout. Episodes flagged as
continuing the previous one only have a stale first observation (a recording
artefact), so each episode is rebuilt on its own from next_observations.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
PROCESSED = ROOT / "data" / "processed"

# Checked against the data: heading change / turn rate gives 0.02 s per step,
# and lidar closeness = 1 - distance / 3 matches the hazard radius of 0.2.
DT = 0.02
LIDAR_RANGE = 3.0
BIN_WIDTH = 2 * np.pi / 16

VELOCITY = slice(3, 5)
COMPASS = (9, 10)
GOAL_LIDAR = slice(12, 28)
HAZARD_LIDAR = slice(28, 44)
VASE_LIDAR = slice(44, 60)

GOAL_REWARD = 0.9
GOAL_MIN_CLOSENESS = 0.05  # the goal is visible anywhere within 3 units
HAZARD_MIN_CLOSENESS = 0.6  # use only nearby hazards, within 1.2 units
VASE_MIN_CLOSENESS = 0.3
HAZARD_CLUSTER_RADIUS = 0.05
HAZARD_MIN_ESTIMATES = 20


def heading_and_path(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Heading in radians and x, y position after each step, starting at 0, 0."""
    # A compass on a turning body reads the fixed field rotating the other way.
    heading = np.arctan2(-obs[:, COMPASS[1]], obs[:, COMPASS[0]])
    forward, sideways = obs[:, VELOCITY].T
    vx = forward * np.cos(heading) - sideways * np.sin(heading)
    vy = forward * np.sin(heading) + sideways * np.cos(heading)
    return heading, np.cumsum(np.column_stack((vx, vy)) * DT, axis=0)


def lidar_points(lidar: np.ndarray, position: np.ndarray, heading: np.ndarray,
                 min_closeness: float, strongest_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World positions of objects seen by one lidar. Returns step index, x, y, closeness.

    Each object lights up one bin fully and spreads a fraction into its two
    neighbours. The split between the neighbours says where in the bin the
    object sits, which is far more precise than using the bin centre.
    """
    before = np.roll(lidar, 1, axis=1)
    after = np.roll(lidar, -1, axis=1)
    if strongest_only:
        step = np.arange(len(lidar))
        bin_ = lidar.argmax(axis=1)
    else:
        step, bin_ = np.nonzero((lidar >= before) & (lidar >= after) & (lidar > 0))
    closeness = lidar[step, bin_]
    keep = closeness >= min_closeness
    step, bin_, closeness = step[keep], bin_[keep], closeness[keep]

    share_after, share_before = after[step, bin_], before[step, bin_]
    total = share_after + share_before
    within_bin = np.divide(share_after, total, out=np.full(total.shape, 0.5), where=total > 0)
    angle = heading[step] + (bin_ + within_bin) * BIN_WIDTH
    distance = LIDAR_RANGE * (1 - closeness)
    xy = position[step] + distance[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    return step, xy, closeness


def spread(points: np.ndarray, centre: np.ndarray) -> float:
    """Median distance of estimates from their centre: how much they agree."""
    return float(np.median(np.linalg.norm(points - centre, axis=1)))


def locate_hazards(obs, position, heading) -> list[dict]:
    """Hazards never move within an episode, so their estimates form tight clusters."""
    _, xy, _ = lidar_points(obs[:, HAZARD_LIDAR], position, heading, HAZARD_MIN_CLOSENESS, strongest_only=False)
    if len(xy) < HAZARD_MIN_ESTIMATES:
        return []
    labels = DBSCAN(eps=HAZARD_CLUSTER_RADIUS, min_samples=HAZARD_MIN_ESTIMATES).fit_predict(xy)
    hazards = []
    for label in sorted(set(labels) - {-1}):
        points = xy[labels == label]
        centre = np.median(points, axis=0)
        hazards.append({"object": "hazard", "x": centre[0], "y": centre[1],
                        "estimates": len(points), "spread": spread(points, centre)})
    return hazards


def locate_goals(obs, position, heading, goal_reached, episode, step) -> list[dict]:
    """A goal stays put until it is reached, then a new one appears.
    The step that reaches a goal already observes the next one."""
    goal_number = np.cumsum(goal_reached)
    seen, xy, _ = lidar_points(obs[:, GOAL_LIDAR], position, heading, GOAL_MIN_CLOSENESS, strongest_only=True)
    reached_rows = np.flatnonzero(goal_reached)
    goals = []
    for number in np.unique(goal_number[seen]):
        points = xy[goal_number[seen] == number]
        centre = np.median(points, axis=0)
        reached = reached_rows[number] if number < len(reached_rows) else None
        goals.append({"object": "goal", "x": centre[0], "y": centre[1],
                      "estimates": len(points), "spread": spread(points, centre),
                      "goal_number": int(number),
                      "reached_episode": episode[reached] if reached is not None else np.nan,
                      "reached_step": step[reached] if reached is not None else np.nan})
    return goals


def vase_track(obs, position, heading) -> np.ndarray:
    """Vase position at each step it is in view, else NaN. The vase can be pushed."""
    track = np.full((len(obs), 2), np.nan)
    seen, xy, _ = lidar_points(obs[:, VASE_LIDAR], position, heading, VASE_MIN_CLOSENESS, strongest_only=True)
    track[seen] = xy
    return track


def main() -> None:
    episodes = pd.read_parquet(PROCESSED / "episodes.parquet")
    episode_of_row = np.repeat(episodes["episode"].to_numpy(), episodes["length"].to_numpy())
    step_of_row = np.arange(len(episode_of_row)) - episodes["start_row"].to_numpy()[episode_of_row]

    n = len(episode_of_row)
    x, y, heading_all = (np.empty(n, np.float32) for _ in range(3))
    vase = np.empty((n, 2), np.float32)
    map_rows = []

    with h5py.File(DATA_PATH, "r") as handle:
        for episode, start, stop in episodes[["episode", "start_row", "stop_row"]].itertuples(index=False):
            obs = handle["next_observations"][start:stop]
            goal_reached = handle["rewards"][start:stop] > GOAL_REWARD
            heading, position = heading_and_path(obs)
            x[start:stop], y[start:stop] = position.T
            heading_all[start:stop] = heading
            vase[start:stop] = vase_track(obs, position, heading)
            for row in locate_hazards(obs, position, heading) + locate_goals(
                    obs, position, heading, goal_reached, episode_of_row[start:stop], step_of_row[start:stop]):
                map_rows.append({"episode": episode, **row})

    paths = pd.DataFrame({"episode": episode_of_row.astype(np.int16), "step": step_of_row.astype(np.int16),
                          "x": x, "y": y, "heading": heading_all, "vase_x": vase[:, 0], "vase_y": vase[:, 1]})
    world = pd.DataFrame(map_rows)
    world["object_id"] = world.groupby(["episode", "object"]).cumcount()
    paths.to_parquet(PROCESSED / "paths.parquet", index=False)
    world.to_parquet(PROCESSED / "map.parquet", index=False)

    hazards = world[world["object"] == "hazard"]
    goals = world[world["object"] == "goal"]
    per_episode = hazards.groupby("episode").size().reindex(episodes["episode"], fill_value=0)
    print("episodes", len(episodes))
    print("hazards found per episode -> count", per_episode.value_counts().sort_index().to_dict())
    print(f"hazard estimate spread median {hazards['spread'].median():.4f}  p90 {hazards['spread'].quantile(0.9):.4f}")
    print(f"goals located {len(goals)}  reached {goals['reached_step'].notna().sum()}")
    print(f"goal estimate spread median {goals['spread'].median():.4f}  p90 {goals['spread'].quantile(0.9):.4f}")
    print("wrote", (PROCESSED / "paths.parquet").relative_to(ROOT), "and", (PROCESSED / "map.parquet").relative_to(ROOT))


if __name__ == "__main__":
    main()
