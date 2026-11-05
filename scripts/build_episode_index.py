"""Build the episode index for the DSRL SafetyPointGoal1 dataset.

The HDF5 file is one flat sequence of steps. This script records where each
episode starts and stops.

It also marks episodes whose first observation is a stale copy of the previous
episode's final state. That is a recording artefact: the environment was reset
(every episode has its own layout), but the reset observation was not stored.
Many other episodes have a stale first observation too, copied from an
episode that is not in the file, so it cannot be matched. Later steps
therefore read next_observations, the state after each action, and never
observations[0].

Output: data/processed/episodes.parquet, one row per episode.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
OUT_PATH = ROOT / "data" / "processed" / "episodes.parquet"

# Observations are rounded floats, so the copy is matched with a tolerance.
MATCH_ATOL = 1e-5


def episode_bounds(terminals: np.ndarray, timeouts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return start rows and stop rows. Stop is exclusive, as in Python slices."""
    done = np.flatnonzero(terminals | timeouts)
    if done.size == 0 or done[-1] != len(timeouts) - 1:
        raise ValueError("the last step does not end an episode")
    stops = done + 1
    starts = np.concatenate(([0], stops[:-1]))
    return starts, stops


def find_stale_first_observations(handle: h5py.File, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """True where an episode's first observation equals the previous episode's
    last next_observation."""
    # Read only the boundary rows instead of the whole 2M x 60 arrays.
    last_next_obs = handle["next_observations"][stops[:-1] - 1]
    first_obs = handle["observations"][starts[1:]]
    copied = np.all(np.isclose(last_next_obs, first_obs, atol=MATCH_ATOL), axis=1)
    return np.concatenate(([False], copied))


def build_index(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        terminals = handle["terminals"][:].astype(bool)
        timeouts = handle["timeouts"][:].astype(bool)
        starts, stops = episode_bounds(terminals, timeouts)
        stale = find_stale_first_observations(handle, starts, stops)

    return pd.DataFrame(
        {
            "episode": np.arange(len(starts)),
            "start_row": starts,
            "stop_row": stops,
            "length": stops - starts,
            "ended_by_terminal": terminals[stops - 1],
            "stale_first_observation": stale,
        }
    )


def main() -> None:
    index = build_index(DATA_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(OUT_PATH, index=False)

    print("episodes", len(index))
    print("length min/max", index["length"].min(), index["length"].max())
    print("ended by terminal", int(index["ended_by_terminal"].sum()))
    print("first observation copied from the previous episode", int(index["stale_first_observation"].sum()))
    print("wrote", OUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
