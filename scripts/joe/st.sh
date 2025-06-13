#!/bin/bash

python scripts/joe/latent_mim_st.py launch latent_mim_large_st_contrastive ai2/titan-cirrascale --launch.priority=urgent --common.launch.num_gpus=4
