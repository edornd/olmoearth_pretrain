"""Trying to prototype fitting everything into olmo core."""

import logging

import numpy as np
import torch
from olmo_core.utils import setup_logging
from upath import UPath

from helios.data.collator import per_modality_collate_fn
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset
from helios.dataset.index import DatasetIndexParser
from helios.train.trainer import HeliosTrainer

logger = logging.getLogger(__name__)


## Config does not yet support our new dataset type so we will construct manually for now

if __name__ == "__main__":
    setup_logging()
    # set log level to debug
    logger.setLevel(logging.DEBUG)

    index_path_old = "gs://ai2-helios/data/20250113-sample-dataset-helios/index.csv"
    index_path = "gs://ai2-helios/data/20250115-sample-dataset-helios/index.csv"
    index_parser = DatasetIndexParser(index_path)
    samples = index_parser.samples
    workdir = UPath("/Users/henryh/Desktop/eai-repos/helios-repos/helios/workdir")
    dataloader = HeliosDataLoader.wrap_numpy_dataset(
        dataset=HeliosDataset(
            *samples,
            ignore_data_sources=["openstreetmap"],
            filter_samples_with_missing_inputs=True,
            dtype=np.dtype("float32"),
        ),
        global_batch_size=4,
        dp_world_size=1,
        collator=per_modality_collate_fn,
        work_dir=workdir,
        num_threads=0,
    )

    # # potentially missing dataset prepare
    # for epoch in range(1, 3):
    #     dataloader.reshuffle(epoch=epoch)
    #     batch_iterator = dataloader._iter_batches()
    #     # Need to call reshuffle
    #     batches_found = 0
    #     batch_start = time.time()
    #     for batch in batch_iterator:
    #         batch_end = time.time()
    #         if batches_found > 0:
    #             logger.info(f"batch time {batch_end - batch_start}")
    #         batches_found += 1
    #         time.sleep(10)
    #         batch_start = time.time()
    #         logger.info("batch found")
    #     dataloader.reset()

    # Need an optimizer
    # Need a checkpointer
    # Need a module
    # first lets grab the anysat model already in there repo
    import torch

    model = torch.hub.load(
        "gastruc/anysat",
        "anysat",
        pretrained=False,
        force_reload=True,
        flash_attn=False,
    )
    from olmo_core.optim import AdamWConfig
    from olmo_core.train.checkpoint import CheckpointerConfig
    from olmo_core.train.common import Duration

    max_duration = Duration.steps(4)

    checkpointer_config = CheckpointerConfig(work_dir=workdir)
    checkpointer = checkpointer_config.build()
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optim_config = AdamWConfig()
    optim = optim_config.build(model)
    trainer = HeliosTrainer(
        work_dir=workdir,
        model=model,
        optim=optim,
        data_loader=dataloader,
        device=DEVICE,
        save_folder=workdir / "save_folder",
        callbacks={},
        rank_microbatch_size=4,
        max_duration=max_duration,
        checkpointer=checkpointer,
    )
    trainer.fit()
