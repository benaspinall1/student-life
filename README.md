# StudentLife Dataset Analysis

Tools for exploring the [StudentLife](https://www.kaggle.com/datasets/dartweichen/student-life)
dataset (Dartmouth College). The code inspects the *collection frequency* of the
dataset's sensing, app usage, call log, and SMS streams, and renders timestamp
plots and interval histograms for each user.

## Requirements

- Python 3.8+
- A [Kaggle account](https://www.kaggle.com/) with API access (used to download the
  dataset automatically), **or** a local copy of the dataset.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is resolved in this order (see `utils.get_dataset_dir`):

1. An explicit path passed in code.
2. The `STUDENT_LIFE_DATASET_PATH` environment variable.
3. Automatic download via [`kagglehub`](https://github.com/Kaggle/kagglehub) from
   the `dartweichen/student-life` Kaggle repo.

### Option A — Automatic download (kagglehub)

Authenticate with Kaggle once, then the dataset downloads automatically on first
run. Create an API token at <https://www.kaggle.com/settings> ("Create New Token"),
which downloads `kaggle.json`, then:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Option B — Use a local dataset

Point the tool at an existing copy of the dataset:

```bash
export STUDENT_LIFE_DATASET_PATH=/path/to/student-life/dataset
```

The directory should contain the dataset folders (`sensing/`, `app_usage/`,
`call_log/`, `sms/`, etc.).

## Running

The main entry point runs a frequency analysis over the `app_usage` stream for
every user and prints summary statistics:

```bash
python student-life.py
```

This generates plots under `frequency_analysis/`:

- `frequency_analysis/<stream>/uNN.png` — samples per unique timestamp over time.
- `frequency_analysis/<stream>/interval_hist/uNN.png` — collection interval histogram.

To analyze a different stream, edit `folder_name` / `file_prefix` in `main()` in
`student-life.py`. Supported streams and their file prefixes are defined in
`utils.directory_to_file_prefix`:

| Stream      | Folder      | File prefix    |
|-------------|-------------|----------------|
| App usage   | `app_usage` | `running_app`  |
| Call log    | `call_log`  | `call_log`     |
| SMS         | `sms`       | `sms`          |
| Sensing     | `sensing`   | `activity`, `audio`, `bt`, `conversation`, `dark`, `gps`, `phonecharge`, `phonelock`, `wifi`, `wifi_location` |

