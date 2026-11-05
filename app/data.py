"""Load the pipeline outputs for the app. Small tables load once; per-step data
is sliced one episode (1,000 rows) at a time."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DATA_PATH = ROOT / "data" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
EVALUATION = ROOT / "evaluation"
EPISODE_LENGTH = 1000
LIDAR = {"goal": slice(12, 28), "hazard": slice(28, 44), "vase": slice(44, 60)}


@st.cache_data
def features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "features.parquet").set_index("episode")


@st.cache_data
def flags() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "flags.parquet")


@st.cache_data
def events() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "events.parquet")


@st.cache_data
def world_map() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "map.parquet")


@st.cache_resource
def _all_steps() -> pd.DataFrame:
    steps = pd.read_parquet(PROCESSED / "steps.parquet")
    paths = pd.read_parquet(PROCESSED / "paths.parquet", columns=["x", "y", "heading", "vase_x", "vase_y"])
    return pd.concat([steps, paths], axis=1)


def episode_steps(episode: int) -> pd.DataFrame:
    """Per-step columns and path for one episode. Rows are stored episode by episode."""
    rows = slice(episode * EPISODE_LENGTH, (episode + 1) * EPISODE_LENGTH)
    return _all_steps().iloc[rows].reset_index(drop=True)


@st.cache_data(max_entries=20)
def episode_lidar(episode: int) -> dict[str, np.ndarray]:
    """Raw 16-direction lidar for one episode, read from the HDF5 file."""
    with h5py.File(DATA_PATH, "r") as handle:
        obs = handle["next_observations"][episode * EPISODE_LENGTH:(episode + 1) * EPISODE_LENGTH]
    return {name: obs[:, columns] for name, columns in LIDAR.items()}
