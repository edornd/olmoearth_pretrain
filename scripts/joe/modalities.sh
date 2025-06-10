#!/bin/bash

# Titan on 128
python scripts/v0.2_sweep/latent_mim_128.py launch v0.2_latent_mim_128_space_time_sen2 ai2/titan-cirrascale --launch.priority=urgent --common.training_modalities=\[sentinel2_l2a,\] --common.launch.num_gpus=1
