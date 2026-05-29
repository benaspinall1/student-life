import os
import kagglehub
from typing import Optional, List, Dict, Tuple, Union, Sequence
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import utils
from datetime import datetime
DEFAULT_DATASET_REPO = "dartweichen/student-life"
DATASET_ENV_VAR = "STUDENT_LIFE_DATASET_PATH"


directory_to_file_prefix = {
    "app_usage": "running_app",
    "call_log": "call_log",
    "sms": "sms",
    "sensing": ["activity", "audio", "bt", "conversation", "dark", "gps", "phonecharge", "phonelock", "wifi", "wifi_location"],
}

DEFAULT_TIME_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "timestamp",
    "time",
    "time_step",
    "timestep",
    "time_stamp",
    "datetime",
    "date_time",
    "event_time",
    "unix_time",
    "unix_timestamp",
)


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column_name.lower())


def _resolve_time_column_name(
    columns: Sequence[str],
    candidates: Sequence[str] = DEFAULT_TIME_COLUMN_CANDIDATES,
) -> str:
    """
    Resolve the best timestamp-like column from available columns.
    Prefers explicit candidate matches, then falls back to fuzzy token scoring.
    """
    normalized_to_original = {_normalize_column_name(col): col for col in columns}

    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_to_original:
            return normalized_to_original[normalized_candidate]

    token_priority = ("timestamp", "time", "datetime", "date", "unix", "epoch", "step")
    best_match = ""
    best_score = 0
    for column in columns:
        normalized_column = _normalize_column_name(column)
        score = sum(1 for token in token_priority if token in normalized_column)
        if normalized_column.endswith("id"):
            score -= 1
        if score > best_score:
            best_score = score
            best_match = column

    if best_score > 0:
        return best_match

    raise KeyError(
        "No timestamp-like column found. "
        f"Available columns: {list(columns)}"
    )

def _resolve_dataset_dir(dataset_path: str) -> str:
    """
    Kagglehub sometimes returns a path like:
      .../versions/1
    while this dataset’s data lives under:
      .../versions/1/dataset
    """
    candidate = os.path.join(dataset_path, "dataset")
    return candidate if os.path.isdir(candidate) else dataset_path

def format_user_id(index: int) -> str:
    """
    Maps indices 0-59 to a string with leading zeros for single digits.
    For example:
        0  -> "00"
        5  -> "05"
        12 -> "12"
        59 -> "59"
    """
    if not (0 <= index <= 59):
        raise ValueError("Index out of range (should be 0-59).")
    return f"u{index:02d}"


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


def resolve_csv_path(
    directory: str,
    file_prefix: str,
    user_id: Optional[str] = None,
    missing_dir_label: str = "directory",
) -> str:
    """
    Resolve a CSV path by prefix and optional user id.

    If `user_id` is provided, resolves `<file_prefix>_<user_id>.csv`.
    Otherwise, returns the first matching CSV in lexical order.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Missing {missing_dir_label}: {directory}")

    if user_id is not None:
        csv_path = os.path.join(directory, f"{file_prefix}_{utils.format_user_id(user_id)}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing file: {csv_path}")
        return csv_path

    csv_files = sorted(
        fn for fn in os.listdir(directory) if fn.startswith(f"{file_prefix}_") and fn.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No files found in: {directory}")
    return os.path.join(directory, csv_files[0])


def get_all_filenames(directory):
    """
    Returns a list of all file names in the given directory.
    Does not recurse into subdirectories.
    """
    try:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        return []



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


def load_directory_time_stamps(
    directory: str,
    file_prefix: str,
    user_id: Optional[Union[int, str]] = None,
) -> List:
    """
    Load all row values from the timestamp-like column for one app usage CSV.
    """
    csv_path = resolve_csv_path(directory=directory, file_prefix=file_prefix, user_id=user_id)
    df = pd.read_csv(csv_path)
    column_name = _resolve_time_column_name(df.columns)
    timestamp_values = coerce_to_unix_seconds(df[column_name].tolist())

    # Some files (e.g., sensing/gps) contain an extra trailing comma per row,
    # which can make pandas parse the true timestamp column as the row index.
    if timestamp_values:
        return timestamp_values

    index_timestamps = coerce_to_unix_seconds(df.index.tolist())
    if index_timestamps:
        return index_timestamps

    return []


def coerce_to_unix_seconds(
    timestamps: Sequence[Union[int, float, str, pd.Timestamp, datetime]],
) -> List[float]:
    """
    Coerce heterogeneous timestamp values to unix seconds.
    Invalid/unparseable values are dropped.
    """
    if not timestamps:
        return []

    series = pd.Series(list(timestamps))
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.dropna().astype("float64").tolist()


def _parse_timestamp(ts: Union[int, float, str, pd.Timestamp, datetime]) -> datetime:
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        # Try parsing: 2023-11-11 12:34:56 or 2023-11-11T12:34:56
        fmt = "%Y-%m-%d %H:%M:%S" if " " in ts else "%Y-%m-%dT%H:%M:%S"
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            # Try generic fromisoformat fallback (Python 3.7+)
            return datetime.fromisoformat(ts)
    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


def compute_timestamp_deltas(timestamps: List[Union[int, float, str, pd.Timestamp]]) -> List[float]:
    """
    Computes the difference (in seconds) between consecutive timestamps using datetime arithmetic.

    Args:
        timestamps: List of timestamps (as ints, floats, or ISO-formatted strings).

    Returns:
        List of deltas in seconds (length is len(timestamps) - 1).
    """
    if len(timestamps) < 2:
        return []

    dt_list = [_parse_timestamp(ts) for ts in timestamps]
    return [
        (dt_list[i+1] - dt_list[i]).total_seconds()
        for i in range(len(dt_list) - 1)
    ]


def get_delta_timestamp_pair(
    timestamps: List[Union[int, float, str, pd.Timestamp, datetime]],
    delta_index: int,
) -> Tuple[datetime, datetime]:
    """
    Return the two consecutive timestamps that produced `deltas[delta_index]`.

    `delta_index` is zero-based and follows the order returned by
    `compute_timestamp_deltas(timestamps)`.
    """
    if len(timestamps) < 2:
        raise ValueError("At least 2 timestamps are required to compute deltas.")

    deltas_len = len(timestamps) - 1
    if not (0 <= delta_index < deltas_len):
        raise IndexError(
            f"delta_index {delta_index} out of range for {deltas_len} deltas."
        )

    parsed = [_parse_timestamp(ts) for ts in timestamps]
    return parsed[delta_index], parsed[delta_index + 1]


def format_seconds_readable(seconds: Union[int, float]) -> str:
    """
    Convert a duration in seconds to a compact human-readable string.
    Example: 3723 -> "1h 2m 3s"
    """
    total_seconds = abs(float(seconds))
    days, remainder = divmod(int(total_seconds), 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    frac = total_seconds - int(total_seconds)
    sec_display = secs + frac

    parts: List[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")

    if sec_display.is_integer():
        sec_value = int(sec_display)
    else:
        sec_value = round(sec_display, 2)
    parts.append(f"{sec_value}s")

    readable = " ".join(parts)
    return f"-{readable}" if seconds < 0 else readable


def compute_average_samples_per_sample_time(
    timestamps: List[Union[int, float, str]],
) -> float:
    """
    Compute the mean sample count per unique timestamp.

    Example:
        timestamps = [1, 1, 2, 3, 3, 3]
        unique counts = [2, 1, 3]
        mean = 2.0
    """
    if not timestamps:
        return 0.0

    counts = pd.Series(timestamps).value_counts()
    # print(counts)
    return float(counts.mean())


def plot_unique_timestamp_counts(timestamps: List[float], output_path: str) -> pd.Series:
    """
    Save a chart of unique timestamp counts over time.
    Returns a series indexed by datetime with count values.
    """
    numeric_timestamps = pd.to_numeric(pd.Series(timestamps), errors="coerce").dropna()
    if numeric_timestamps.empty:
        return pd.Series(dtype="int64")

    timestamp_counts = numeric_timestamps.value_counts().sort_index()
    datetime_index = pd.to_datetime(timestamp_counts.index, unit="s", errors="coerce")
    valid_mask = datetime_index.notna()
    counts_by_time = pd.Series(
        timestamp_counts.values[valid_mask],
        index=datetime_index[valid_mask],
    )
    if counts_by_time.empty:
        return pd.Series(dtype="int64")

    # If there are too many unique times, scatter points stay readable.
    use_scatter = len(counts_by_time) > 1000
    plt.figure(figsize=(12, 5))
    if use_scatter:
        plt.scatter(counts_by_time.index, counts_by_time.values, s=6, alpha=0.8)
    else:
        plt.plot(counts_by_time.index, counts_by_time.values, linewidth=1.2, marker="o", markersize=2)
    source_name = os.path.basename(os.path.dirname(output_path)).replace("_", " ").title()
    plt.title(f"{source_name} Samples Per Unique Timestamp")
    plt.xlabel("Timestamp")
    plt.ylabel("Number of samples")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return counts_by_time


def plot_timestamp_deltas(deltas: List[float], output_path: str) -> pd.Series:
    """
    Save a chart of timestamp deltas (minutes) across consecutive unique timestamps.
    Returns the plotted delta series in minutes indexed by delta order.
    """
    if not deltas:
        return pd.Series(dtype="float64")

    # Inputs are second-based deltas; plot and return minutes for readability.
    delta_series = pd.Series(deltas, index=range(1, len(deltas) + 1), dtype="float64") / 60.0
    use_scatter = len(delta_series) > 1000

    plt.figure(figsize=(12, 5))
    if use_scatter:
        plt.scatter(delta_series.index, delta_series.values, s=6, alpha=0.8)
    else:
        plt.plot(delta_series.index, delta_series.values, linewidth=1.2, marker="o", markersize=2)
    plt.title("Delta Between Consecutive Unique Timestamps")
    plt.xlabel("Delta index")
    plt.ylabel("Delta (minutes)")
    plt.gca().yaxis.set_major_locator(MultipleLocator(20))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return delta_series


def plot_collection_interval_histogram(
    timestamps: List[Union[int, float, str, pd.Timestamp, datetime]],
    output_path: str,
    bins: int = 1000,
    zoom_std: float = 2.0,
) -> pd.Series:
    """
    Plot a histogram of collection intervals in minutes.

    Samples that share the same timestamp are treated as the same collection
    event by deduplicating timestamps before interval computation.

    The x-axis is zoomed around the mean using mean +/- (zoom_std * std),
    clipped to non-negative values.
    """
    if not timestamps:
        return pd.Series(dtype="float64")

    unique_seconds = sorted(set(coerce_to_unix_seconds(timestamps)))
    if len(unique_seconds) < 2:
        return pd.Series(dtype="float64")

    deltas_seconds = compute_timestamp_deltas(unique_seconds)
    if not deltas_seconds:
        return pd.Series(dtype="float64")

    intervals_minutes = pd.Series(deltas_seconds, dtype="float64") / 60.0
    mean_minutes = float(intervals_minutes.mean())
    std_minutes = float(intervals_minutes.std(ddof=0))

    plt.figure(figsize=(11, 5))
    plt.hist(intervals_minutes, bins=bins, edgecolor="black", alpha=0.8)
    plt.title("Data Collection Interval Histogram")
    plt.xlabel("Interval between unique collection timestamps (minutes)")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)

    if len(intervals_minutes) > 1:
        if std_minutes > 0:
            x_min = max(0.0, mean_minutes - zoom_std * std_minutes)
            x_max = mean_minutes + zoom_std * std_minutes
            if x_max > x_min:
                plt.xlim(x_min, x_max)
        elif mean_minutes > 0:
            # Constant interval case: still create a useful visible window.
            padding = max(0.1, mean_minutes * 0.1)
            plt.xlim(max(0.0, mean_minutes - padding), mean_minutes + padding)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return intervals_minutes


def format_columns_for_copy(title: str, columns_by_folder: Dict[str, List[str]]) -> str:
    """
    Build a copy/paste-friendly markdown block of folder columns.
    """
    lines: List[str] = [title, ""]

    for folder in sorted(columns_by_folder):
        lines.append(f"## {folder}")
        columns = columns_by_folder[folder]
        if not columns:
            lines.append("(no columns found)")
        else:
            lines.append(", ".join(column.strip() for column in columns))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"



    