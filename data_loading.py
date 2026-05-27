from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import kagglehub
import pandas as pd


DEFAULT_DATASET_REPO = "dartweichen/student-life"
DATASET_ENV_VAR = "STUDENT_LIFE_DATASET_PATH"


def _resolve_dataset_dir(dataset_path: str) -> str:
    """
    Kagglehub sometimes returns a path like:
      .../versions/1
    while this dataset’s data lives under:
      .../versions/1/dataset
    """
    candidate = os.path.join(dataset_path, "dataset")
    return candidate if os.path.isdir(candidate) else dataset_path


def get_dataset_dir(dataset_path: Optional[str] = None) -> str:
    """
    Return the dataset directory containing folders like `EMA/` and `sensing/`.
    """
    if dataset_path:
        return _resolve_dataset_dir(dataset_path)

    env_path = os.getenv(DATASET_ENV_VAR)
    if env_path:
        return _resolve_dataset_dir(env_path)

    downloaded_path = kagglehub.dataset_download(DEFAULT_DATASET_REPO)
    return _resolve_dataset_dir(downloaded_path)


def format_user_id(uid: int) -> str:
    return f"u{uid:02d}"


def parse_user_id_from_filename(filename: str) -> Optional[int]:
    # Matches suffixes like: "_u00.csv" -> 0
    m = re.search(r"_u(\d{2})\.csv$", filename)
    if not m:
        return None
    return int(m.group(1))


def list_users(dataset_dir: Optional[str] = None) -> List[int]:
    """
    Discover user ids from one of the sensing modalities.
    """
    dataset_dir = dataset_dir or get_dataset_dir()
    sensing_dir = os.path.join(dataset_dir, "sensing")

    # Prefer activity because it’s always present in this dataset.
    preferred = os.path.join(sensing_dir, "activity")
    candidates = [preferred]
    # Fallback: gps
    candidates.append(os.path.join(sensing_dir, "gps"))

    for cand_dir in candidates:
        if not os.path.isdir(cand_dir):
            continue
        users: List[int] = []
        for fn in os.listdir(cand_dir):
            if not fn.endswith(".csv"):
                continue
            uid = parse_user_id_from_filename(fn)
            if uid is None:
                continue
            users.append(uid)
        users = sorted(set(users))
        if users:
            return users

    raise FileNotFoundError(
        f"Could not discover users under sensing directory: {sensing_dir}"
    )


def ema_definition_path(dataset_dir: Optional[str] = None) -> str:
    dataset_dir = dataset_dir or get_dataset_dir()
    return os.path.join(dataset_dir, "EMA", "EMA_definition.json")


def load_ema_definition(dataset_dir: Optional[str] = None) -> List[dict]:
    """
    EMA_definition.json describes surveys and question option encodings.
    """
    p = ema_definition_path(dataset_dir)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_ema_user_file(response_survey_dir: str, survey_name: str, uid: int) -> str:
    expected = os.path.join(
        response_survey_dir, f"{survey_name}_u{uid:02d}.json"
    )
    if os.path.exists(expected):
        return expected

    # Fallback for any naming mismatch.
    u_tag = f"_u{uid:02d}.json"
    for fn in os.listdir(response_survey_dir):
        if fn.endswith(u_tag) and fn.startswith(survey_name):
            return os.path.join(response_survey_dir, fn)

    raise FileNotFoundError(
        f"Could not find EMA file for survey={survey_name}, uid={uid} in {response_survey_dir}"
    )


def load_ema_responses(
    survey_name: str,
    users: Optional[Sequence[int]] = None,
    dataset_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load EMA responses for a single survey block (e.g., "Stress", "Sleep", "Mood").
    Returns a DataFrame with at least:
      - resp_time (original)
      - timestamp (datetime)
      - date (python date)
      - user (int uid)
      - survey (survey_name)
    """
    dataset_dir = dataset_dir or get_dataset_dir()
    response_survey_dir = os.path.join(
        dataset_dir, "EMA", "response", survey_name
    )
    if not os.path.isdir(response_survey_dir):
        raise FileNotFoundError(f"Missing EMA survey directory: {response_survey_dir}")

    if users is None:
        users = list_users(dataset_dir)

    frames: List[pd.DataFrame] = []
    for uid in users:
        user_file = _find_ema_user_file(response_survey_dir, survey_name, uid)
        with open(user_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df["user"] = uid
        df["survey"] = survey_name
        if "resp_time" in df.columns:
            df["resp_time"] = pd.to_numeric(df["resp_time"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["resp_time"], unit="s", errors="coerce")
            df["date"] = df["timestamp"].dt.date
        # Clean the dataset’s literal "null" strings.
        df = df.replace({"null": pd.NA, "": pd.NA})
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # Normalize well-known inference column names.
    rename: Dict[str, str] = {}
    if "activity inference" in df.columns:
        rename["activity inference"] = "activity_inference"
    if "audio inference" in df.columns:
        rename["audio inference"] = "audio_inference"
    if rename:
        df = df.rename(columns=rename)
    # Normalize common timestamp columns.
    if "time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    if "start_timestamp" in df.columns and "start" not in df.columns:
        df = df.rename(columns={"start_timestamp": "start"})
    return df


def load_sensing_stream(
    modality: str,
    user_id: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load a sensing modality time series for one or more users.

    Pass `user_id` (00-59) to load only a single user CSV.
    """
    dataset_dir = dataset_dir or get_dataset_dir()
    sensing_dir = os.path.join(dataset_dir, "sensing", modality)
    if not os.path.isdir(sensing_dir):
        raise FileNotFoundError(f"Missing sensing modality directory: {sensing_dir}")

    csv_path = os.path.join(sensing_dir, f"{modality}_u{user_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing sensing file: {csv_path}")
    df = pd.read_csv(csv_path, usecols=usecols, index_col=0)
    df = _standardize_column_names(df)
    df = df.replace({"null": pd.NA, "": pd.NA})
    df["user"] = user_id
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().all():
            recovered_timestamp = pd.to_numeric(
                pd.Series(df.index, index=df.index),
                errors="coerce",
            )
            if not recovered_timestamp.isna().all():
                df["timestamp"] = recovered_timestamp
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].fillna("unknown")
    if "datetime" in df.columns:
        df["datetime"] = df["datetime"].fillna("unknown")
    if "date" in df.columns:
        df["date"] = df["date"].fillna("unknown")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
    df = df.reset_index(drop=True)
    return df


