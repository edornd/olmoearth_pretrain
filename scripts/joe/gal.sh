#!/bin/bash

#python scripts/joe/galileo.py launch galileo_repro_ ai2/jupiter-cirrascale-2 --launch.priority=urgent --common.launch.num_gpus=8 --train_module.rank_microbatch_size=32
python scripts/joe/galileo.py launch galileo_repro_noema ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=8 --train_module.ema_decay=\[1,1\]
