"""Launch Beaker jobs to get Sentinel-2 L1C data."""

import argparse
import multiprocessing
import random
import uuid

import tqdm
from beaker import (
    Beaker,
    DataMount,
    DataSource,
    ExperimentSpec,
    Priority,
)
from rslearn.dataset import Dataset, Window
from upath import UPath

# Sentinel-2 L1C is only used in the res_10 group.
GROUP = "res_10"

# Relevant layers that would be ingested.
LAYER_NAMES = [
    "sentinel2_freq",
    "sentinel2_mo01",
    "sentinel2_mo02",
    "sentinel2_mo03",
    "sentinel2_mo04",
    "sentinel2_mo05",
    "sentinel2_mo06",
    "sentinel2_mo07",
    "sentinel2_mo08",
    "sentinel2_mo09",
    "sentinel2_mo10",
    "sentinel2_mo11",
    "sentinel2_mo12",
]

BEAKER_BUDGET = "ai2/d5"
BEAKER_WORKSPACE = "ai2/earth-systems"


def is_window_pending(window: Window) -> bool:
    """Check if the window needs ingestion for Sentinel-2 L1C data.

    Args:
        window: the window to check.

    Returns:
        whether the window hasn't been ingested yet.
    """
    layer_datas = window.load_layer_datas()
    for layer_name in LAYER_NAMES:
        if layer_name not in layer_datas:
            # Not prepared, so doesn't need ingestion.
            continue
        if not window.is_layer_completed(layer_name):
            return True
    return False


def launch_job(
    image: str,
    clusters: list[str],
    ds_path: str,
    window_names: list[str],
) -> None:
    """Launch a Beaker job that ingests the specified windows.

    Args:
        image: the name of the Beaker image to use.
        clusters: list of Beaker clusters to target.
        ds_path: the dataset path.
        window_names: names of the windows to ingest in this job.
        weka_mounts: list of weka mounts for Beaker job.
    """
    beaker = Beaker.from_env(default_workspace=BEAKER_WORKSPACE)
    with beaker.session():
        # Add random string since experiment names must be unique.
        task_uuid = str(uuid.uuid4())[0:16]
        experiment_name = f"helios-sentinel2-l1c-{task_uuid}"

        command = [
            "python",
            "helios/dataset_creation/scripts/sentinel2_l1c/entrypoint.py",
            "--ds_path",
            ds_path,
            "--windows",
            ",".join(window_names),
        ]
        weka_mount = DataMount(
            source=DataSource(weka="dfive-default"),
            mount_path="/weka/dfive-default",
        )
        experiment_spec = ExperimentSpec.new(
            budget=BEAKER_BUDGET,
            task_name=experiment_name,
            beaker_image=image,
            priority=Priority.high,
            cluster=clusters,
            command=command,
            datasets=[weka_mount],
            resources={"gpuCount": 0},
            preemptible=True,
        )
        beaker.experiment.create(experiment_name, experiment_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Beaker jobs to get Sentinel-2 L1C data",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to the rslearn dataset for dataset creation assuming /weka/dfive-default/ is mounted",
        required=True,
    )
    parser.add_argument(
        "--image_name",
        type=str,
        help="Name of the Beaker image to use for the job",
        required=True,
    )
    parser.add_argument(
        "--clusters",
        type=str,
        help="Comma-separated list of clusters to target",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers",
        default=32,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size, i.e., number of windows to ingest per Beaker job",
        default=10,
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        help="Maximum number of jobs to start",
        default=None,
    )
    args = parser.parse_args()
    clusters = args.clusters.split(",")

    # Check which windows are not done.
    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(groups=[GROUP], workers=args.workers)
    p = multiprocessing.Pool(args.workers)
    is_pending_list = list(
        tqdm.tqdm(
            p.imap(is_window_pending, windows),
            desc="Checking pending windows",
            total=len(windows),
        )
    )

    pending_window_names: list[str] = []
    for window, is_pending in zip(windows, is_pending_list):
        if not is_pending:
            continue
        pending_window_names.append(window.name)
    print(f"got {len(pending_window_names)} pending windows")

    if len(pending_window_names) > args.max_jobs * args.batch_size:
        pending_window_names = random.sample(
            pending_window_names, args.max_jobs * args.batch_size
        )

    # Launch jobs for the pending windows.
    batches = []
    for i in range(0, len(pending_window_names), args.batch_size):
        batch = pending_window_names[i : i + args.batch_size]
        batches.append(batch)

    for batch in tqdm.tqdm(batches, desc="Launching jobs"):
        launch_job(args.image_name, args.clusters.split(","), args.ds_path, batch)
