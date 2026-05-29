from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

import utils


SMS_FOLDERS: Tuple[str, ...] = ("sms",)

SMS_PATH_CONFIG: Dict[str, Tuple[str, str]] = {
    "sms": ("sms", "sms"),
}


def _resolve_sms_path_config(folder_name: str) -> Tuple[str, str]:
    if folder_name not in SMS_PATH_CONFIG:
        raise ValueError(f"Unsupported sms folder: {folder_name}")
    return SMS_PATH_CONFIG[folder_name]


def _normalize_user_id(user_id: Union[int, str]) -> str:
    if isinstance(user_id, int):
        if user_id < 0:
            raise ValueError("user_id cannot be negative")
        return f"u{user_id:02d}"

    normalized = user_id.strip().lower()
    if normalized.startswith("u") and normalized[1:].isdigit():
        return f"u{int(normalized[1:]):02d}"
    if normalized.isdigit():
        return f"u{int(normalized):02d}"
    raise ValueError(f"Unsupported user_id format: {user_id}")


def _resolve_sms_csv_path(
    folder_name: str = "sms",
    user_id: Optional[Union[int, str]] = None,
    dataset_dir: Optional[str] = None,
) -> str:
    dataset_dir = dataset_dir or utils.get_dataset_dir()
    resolved_folder_name, file_prefix = _resolve_sms_path_config(folder_name)
    sms_dir = os.path.join(dataset_dir, resolved_folder_name)
    resolved_user_id = _normalize_user_id(user_id) if user_id is not None else None
    return utils.resolve_csv_path(
        directory=sms_dir,
        file_prefix=file_prefix,
        user_id=resolved_user_id,
        missing_dir_label="sms directory",
    )


def _load_sms_columns(
    folder_name: str = "sms",
    user_id: Optional[Union[int, str]] = None,
    dataset_dir: Optional[str] = None,
) -> List[str]:
    csv_path = _resolve_sms_csv_path(folder_name=folder_name, user_id=user_id, dataset_dir=dataset_dir)
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def load_sms_columns(
    user_id: Optional[Union[int, str]] = None,
    dataset_dir: Optional[str] = None,
) -> List[str]:
    return _load_sms_columns("sms", user_id=user_id, dataset_dir=dataset_dir)


def load_all_sms_columns(
    user_id: Optional[Union[int, str]] = None,
    dataset_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    return {
        folder_name: _load_sms_columns(folder_name, user_id=user_id, dataset_dir=dataset_dir)
        for folder_name in SMS_FOLDERS
    }


def format_columns_for_copy(columns_by_folder: Dict[str, List[str]]) -> str:
    """
    Build a copy/paste-friendly markdown block of folder columns.
    """
    lines: List[str] = ["# SMS Columns", ""]

    for folder in sorted(columns_by_folder):
        lines.append(f"## {folder}")
        columns = columns_by_folder[folder]
        if not columns:
            lines.append("(no columns found)")
        else:
            lines.append(", ".join(column.strip() for column in columns))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    columns_by_folder = load_all_sms_columns()
    output = format_columns_for_copy(columns_by_folder)
    print(output)


if __name__ == "__main__":
    main()
