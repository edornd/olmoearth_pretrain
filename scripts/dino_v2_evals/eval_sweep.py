"""Run an evaluation sweep for DINOv2."""

import subprocess


# Evaluation That sweeps over the following:
# Learning Rate
# Normalization
# helios pretrained, dataset norms and imagenet normalization

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

Normalization_MODES = ["imagenet", "dataset", "helios"]

lr_args = " ".join(
    [
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr}",
        # "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr}",
        # "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr}",
        # "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr}",
        # "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr}",
        # "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr}",
        # "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr}",
        # "--trainer.callbacks.downstream_evaluator.tasks.breizhcrops.probe_lr={lr}",
    ]
)

imagenet_args = " ".join(
    [
        "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_method=no_norm",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_method=no_norm",
        "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_method=no_norm",
        "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_method=no_norm",
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_method=no_norm",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_method=no_norm",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_method=no_norm",
        "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=False",
    ]
)

dataset_args = " ".join(
    [
        "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=False",
    ]
)

helios_args = " ".join(
    [
        "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=True",
    ]
)

for norm_mode in Normalization_MODES:
    for lr in LP_LRs:
        print(f"Running with {norm_mode} normalization and {lr} learning rate")
        run_name = f"dino_v2_eval_norm{norm_mode}_{lr}"
        lr_args = lr_args.format(lr=lr)
        if norm_mode == "imagenet":
            lr_args += imagenet_args
        elif norm_mode == "dataset":
            lr_args += dataset_args
        elif norm_mode == "helios":
            lr_args += helios_args
        cmd = f"python3 scripts/dino_v2_evals/dino_v2_eval.py ai2/saturn-cirrascale --launch.priority=high {lr_args} --launch.task_name=eval"
        subprocess.run(cmd, shell=True)  # nosec
