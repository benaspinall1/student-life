import sensing as sensing_dl
import call_log as call_log_dl
import app_usage as app_usage_dl
import sms as sms_dl
import os
from typing import List, Optional, Tuple, Literal
import utils
import pandas as pd
import math
import statistics


def print_columns():
    sensing_columns = sensing_dl.load_all_sensing_columns()
    sensing_output = utils.format_columns_for_copy("# Student Life Sensing Columns", sensing_columns)
    print(sensing_output)

    call_log_columns = call_log_dl.load_all_call_log_columns()
    call_log_output = utils.format_columns_for_copy("# Student Life Call Log Columns", call_log_columns)
    print(call_log_output)

    app_usage_columns = app_usage_dl.load_all_app_usage_columns()
    app_usage_output = utils.format_columns_for_copy("# Student Life App Usage Columns", app_usage_columns)
    print(app_usage_output)

    sms_columns = sms_dl.load_all_sms_columns()
    sms_output = utils.format_columns_for_copy("# Student Life SMS Columns", sms_columns)
    print(sms_output)


def timestamp_analysis(
    dataset_dir: str,
    folder_name: str,
    file_prefix: str,
    user_id: int,
) -> Optional[Tuple[float, List[float]]]:
    directory = os.path.join(dataset_dir, folder_name)
    time_steps = utils.load_directory_time_stamps(directory=directory, file_prefix=file_prefix, user_id=user_id)
    if not time_steps:
        print("No timestamps found.")
        return

    output_graph_path = os.path.join(os.getcwd(), f"frequency_analysis/{folder_name}/sample_counts/{utils.format_user_id(user_id)}.png")
    output_interval_hist_path = os.path.join(
        os.getcwd(),
        f"frequency_analysis/{folder_name}/interval_histograms/{utils.format_user_id(user_id)}.png",
    )
    os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_interval_hist_path), exist_ok=True)

    counts_by_time = utils.plot_unique_timestamp_counts(timestamps=time_steps, output_path=output_graph_path)

    if counts_by_time.empty:
        print("No unique timestamp counts available to plot.")
        return

    # Compute numeric deltas for plotting, then format only for display.
    deltas = utils.compute_timestamp_deltas(counts_by_time.index.tolist())
    # utils.plot_timestamp_deltas(deltas=deltas, output_path=output_delta_graph_path)
    utils.plot_collection_interval_histogram(
        timestamps=time_steps,
        output_path=output_interval_hist_path,
    )

    return counts_by_time.mean(), deltas


def inspect_user_delta(
    dataset_dir: str,
    folder_name: str,
    file_prefix: str,
    user_id: int,
    delta_index: int,
) -> None:
    """Print the timestamp pair that produced a given delta index for one user."""
    directory = os.path.join(dataset_dir, folder_name)
    time_steps = utils.load_directory_time_stamps(
        directory=directory,
        file_prefix=file_prefix,
        user_id=user_id,
    )
    if not time_steps:
        print(f"{utils.format_user_id(user_id)} has no timestamps to inspect.", flush=True)
        return

    unique_timestamps = pd.to_datetime(
        sorted(set(utils.coerce_to_unix_seconds(time_steps))),
        unit="s",
    ).tolist()
    start_ts, end_ts = utils.get_delta_timestamp_pair(
        timestamps=unique_timestamps,
        delta_index=delta_index,
    )
    delta_seconds = (end_ts - start_ts).total_seconds()
    print(
        (
            f"{utils.format_user_id(user_id)} delta[{delta_index}] => "
            f"{start_ts.isoformat()} -> {end_ts.isoformat()} "
            f"({utils.format_seconds_readable(delta_seconds)})"
        ),
        flush=True,
    )


def find_extreme_delta_timestamps(
    dataset_dir: str,
    folder_name: str,
    file_prefix: str,
    user_ids: List[int],
    extreme: Literal["max", "min"] = "max",
) -> Optional[Tuple[int, int, pd.Timestamp, pd.Timestamp, float]]:
    """
    Find either the largest or smallest delta across users and return:
    (user_id, delta_index, start_timestamp, end_timestamp, delta_seconds)
    """
    if extreme not in ("max", "min"):
        raise ValueError("extreme must be either 'max' or 'min'.")

    record: Optional[Tuple[int, int, pd.Timestamp, pd.Timestamp, float]] = None
    choose = max if extreme == "max" else min

    for user_id in user_ids:
        directory = os.path.join(dataset_dir, folder_name)
        time_steps = utils.load_directory_time_stamps(
            directory=directory,
            file_prefix=file_prefix,
            user_id=user_id,
        )
        if not time_steps:
            continue

        unique_timestamps = pd.to_datetime(
            sorted(set(utils.coerce_to_unix_seconds(time_steps))),
            unit="s",
        ).tolist()
        deltas = utils.compute_timestamp_deltas(unique_timestamps)
        if not deltas:
            continue

        local_index = choose(range(len(deltas)), key=lambda i: deltas[i])
        local_value = deltas[local_index]
        start_ts, end_ts = utils.get_delta_timestamp_pair(
            timestamps=unique_timestamps,
            delta_index=local_index,
        )

        if record is None:
            record = (user_id, local_index, start_ts, end_ts, local_value)
            continue

        if (extreme == "max" and local_value > record[4]) or (
            extreme == "min" and local_value < record[4]
        ):
            record = (user_id, local_index, start_ts, end_ts, local_value)

    return record


def frequency_analysis(
    dataset_dir: str,
    folder_name: str,
    file_prefix: str,
    user_ids: List[int],
    extreme: Literal["max", "min"] = "max",
) -> None:
    dataset_dir = utils.get_dataset_dir()
    user_ids = utils.list_users()
    total_users = len(user_ids)
    means: List[float] = []
    all_deltas: List[float] = []
    for index, user_id in enumerate(user_ids, start=1):
        print(f"[{index}/{total_users}] Processing {utils.format_user_id(user_id)}...", flush=True)
        result = timestamp_analysis(
            dataset_dir=dataset_dir,
            folder_name=folder_name,
            file_prefix=file_prefix,
            user_id=user_id,
        )
        if result is None:
            continue

        mean, user_deltas = result
        means.append(mean)
        all_deltas.extend(user_deltas)

    print(f"Completed processing {len(means)} users.", flush=True)

    if means:
        print(f"Mean samples per unique timestamp: {sum(means) / len(means):.2f}")
    else:
        print("Mean samples per unique timestamp: no valid data")

    if all_deltas:
        print(f"Sample frequency in minutes:")
        print(f"Mean : {(sum(all_deltas) / len(all_deltas)) / 60:.2f}")
        print(f"Min: {(min(all_deltas) / 60):.2f}")
        print(f"Max: {(max(all_deltas) / 60):.2f}")
        print(f"Median: {(statistics.median(all_deltas) / 60):.2f}")
        print(f"Mode: {(statistics.mode(all_deltas) / 60):.2f}")
        print(f"Std: {(statistics.stdev(all_deltas) / 60):.2f}")
        print(f"Var: {(statistics.variance(all_deltas) / 60):.2f}")
    else:
        print("No valid data")

    max_delta_record = find_extreme_delta_timestamps(
        dataset_dir=dataset_dir,
        folder_name=folder_name,
        file_prefix=file_prefix,
        user_ids=user_ids,
        extreme="max",
    )
    if max_delta_record is not None:
        user_id, delta_index, start_ts, end_ts, delta_seconds = max_delta_record
        print(
            (
                f"Global max delta: {utils.format_user_id(user_id)} delta[{delta_index}] => "
                f"{start_ts.isoformat()} -> {end_ts.isoformat()} "
                f"({utils.format_seconds_readable(delta_seconds)})"
            ),
            flush=True,
        )



def main() -> None:
    dataset_dir = utils.get_dataset_dir()
    folder_name = "sensing/activity"
    file_prefix = "activity"
    user_ids = utils.list_users()
    extreme = "max"
    frequency_analysis(
        dataset_dir=dataset_dir,
        folder_name=folder_name,
        file_prefix=file_prefix,
        user_ids=user_ids,
        extreme=extreme,
    )


if __name__ == "__main__":
    
    main()

