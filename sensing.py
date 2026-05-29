from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd

import utils


SENSING_MODALITIES: Tuple[str, ...] = (
    "activity",
    "audio",
    "bt",
    "conversation",
    "dark",
    "gps",
    "phonecharge",
    "phonelock",
    "wifi",
    "wifi_location",
)


SENSING_PATH_CONFIG: Dict[str, Tuple[str, str]] = {
    "activity": ("activity", "activity"),
    "audio": ("audio", "audio"),
    "bt": ("bluetooth", "bt"),
    "bluetooth": ("bluetooth", "bt"),
    "conversation": ("conversation", "conversation"),
    "dark": ("dark", "dark"),
    "gps": ("gps", "gps"),
    "phonecharge": ("phonecharge", "phonecharge"),
    "phonelock": ("phonelock", "phonelock"),
    "wifi": ("wifi", "wifi"),
    "wifi_location": ("wifi_location", "wifi_location"),
}


def _resolve_sensing_path_config(modality: str) -> Tuple[str, str]:
    if modality not in SENSING_PATH_CONFIG:
        raise ValueError(f"Unsupported sensing modality: {modality}")
    return SENSING_PATH_CONFIG[modality]


def _resolve_sensing_csv_path(
    modality: str,
    user_id: Optional[str] = None,
    dataset_dir: Optional[str] = None,
) -> str:
    dataset_dir = dataset_dir or utils.get_dataset_dir()
    folder_name, file_prefix = _resolve_sensing_path_config(modality)
    sensing_dir = os.path.join(dataset_dir, "sensing", folder_name)
    return utils.resolve_csv_path(
        directory=sensing_dir,
        file_prefix=file_prefix,
        user_id=user_id,
        missing_dir_label="sensing modality directory",
    )


def _load_sensing_columns(
    modality: str,
    user_id: Optional[str] = None,
    dataset_dir: Optional[str] = None,
) -> List[str]:
    """
    Return column names for a sensing modality.

    If `user_id` is not provided, columns are read from the first available CSV.
    """
    csv_path = _resolve_sensing_csv_path(
        modality=modality,
        user_id=user_id,
        dataset_dir=dataset_dir,
    )
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def load_activity_columns(user_id: Optional[str] = None, dataset_dir: Optional[str] = None) -> List[str]:
    return _load_sensing_columns("activity", user_id=user_id, dataset_dir=dataset_dir)


def load_audio_columns(user_id: Optional[str] = None, dataset_dir: Optional[str] = None) -> List[str]:
    return _load_sensing_columns("audio", user_id=user_id, dataset_dir=dataset_dir)


def load_bt_columns(user_id: Optional[str] = None, dataset_dir: Optional[str] = None) -> List[str]:
    return _load_sensing_columns("bt", user_id=user_id, dataset_dir=dataset_dir)


def load_conversation_columns(
    user_id: Optional[str] = None, dataset_dir: Optional[str] = None
) -> List[str]:
    return _load_sensing_columns("conversation", user_id=user_id, dataset_dir=dataset_dir)


def load_dark_columns(user_id: Optional[str] = None, dataset_dir: Optional[str] = None) -> List[str]:
    return _load_sensing_columns("dark", user_id=user_id, dataset_dir=dataset_dir)


def load_gps_columns(user_id: Optional[str] = None, dataset_dir: Optional[str] = None) -> List[str]:
    return _load_sensing_columns("gps", user_id=user_id, dataset_dir=dataset_dir)


def load_phonecharge_columns(
    user_id: Optional[str] = None, dataset_dir: Optional[str] = None
) -> List[str]:
    return _load_sensing_columns("phonecharge", user_id=user_id, dataset_dir=dataset_dir)


def load_phonelock_columns(user_id: Optional[str] = None, dataset_dir: Optional[str] = None) -> List[str]:
    return _load_sensing_columns("phonelock", user_id=user_id, dataset_dir=dataset_dir)


def load_wifi_columns(user_id: Optional[str] = None, dataset_dir: Optional[str] = None) -> List[str]:
    return _load_sensing_columns("wifi", user_id=user_id, dataset_dir=dataset_dir)


def load_wifi_location_columns(
    user_id: Optional[str] = None, dataset_dir: Optional[str] = None
) -> List[str]:
    return _load_sensing_columns("wifi_location", user_id=user_id, dataset_dir=dataset_dir)


def load_all_sensing_columns(
    user_id: Optional[str] = None,
    dataset_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    return {
        modality: _load_sensing_columns(modality, user_id=user_id, dataset_dir=dataset_dir)
        for modality in SENSING_MODALITIES
    }



# def load_sensing_stream(
#     modality: str,
#     user_id: Optional[str] = None,
#     dataset_dir: Optional[str] = None,
#     usecols: Optional[Sequence[str]] = None,
# ) -> pd.DataFrame:
#     """
#     Load a sensing modality time series for one or more users.

#     Pass `user_id` (00-59) to load only a single user CSV.
#     """
#     csv_path = _resolve_sensing_csv_path(modality=modality, user_id=user_id, dataset_dir=dataset_dir)
#     df = pd.read_csv(csv_path, usecols=usecols)
#     df = df.replace({"null": pd.NA, "": pd.NA})
#     df["user"] = user_id
#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
#         df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
#         df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
#         df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
#     for col in df.columns:
#         if pd.api.types.is_object_dtype(df[col]):
#             df[col] = df[col].fillna("unknown")
#     if "datetime" in df.columns:
#         df["datetime"] = df["datetime"].fillna("unknown")
#     if "date" in df.columns:
#         df["date"] = df["date"].fillna("unknown")
#     for col in df.columns:
#         if pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = df[col].fillna(0)
#     df = df.reset_index(drop=True)
#     return df


