"""Launching Beaker experiments."""

from olmo_core.launch.beaker import BeakerLaunchConfig

HELIOS_DEFAULT_SETUP_STEPS = (
    'git clone "$REPO_URL" .',
    'git checkout "$GIT_REF"',
    "git submodule update --init --recursive",
    "conda shell.bash activate base",
    "pip install -e '.[all]'",
    "pip freeze",
)


class HeliosBeakerLaunchConfig(BeakerLaunchConfig):
    """Beaker launch config for Helios."""
    pass

    # def build_experiment_spec(
    #     self, torchrun: bool = True, entrypoint: Optional[str] = None
    # ) -> ExperimentSpec:
    #     """
    #     Get the Beaker experiment spec corresponding to this config instance.
    #     """
    #     # Get repository account, name, and current ref.
    #     github_account, github_repo, git_ref, is_public = ensure_repo(self.allow_dirty)

    #     if not is_public and self.setup_steps == DEFAULT_SETUP_STEPS:
    #         raise OLMoConfigurationError(
    #             "It looks like your repository is private and private repositories will require "
    #             "custom 'setup_steps' in order to clone the repo."
    #         )

    #     entrypoint_script = [
    #         "#!/usr/bin/env bash",
    #         "set -exuo pipefail",
    #         "[[ -d /var/lib/tcpxo/lib64 ]] && export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH",
    #         # Setup the kernel cache directory used by pytorch
    #         "mkdir -p /root/.cache/torch/kernels && export PYTORCH_KERNEL_CACHE_PATH=/root/.cache/torch/kernels",
    #         "mkdir -p /olmo-core-runtime",
    #         "cd /olmo-core-runtime",
    #         *self.setup_steps,
    #     ]

    #     if torchrun:
    #         if any(["augusta" in cluster for cluster in self.clusters]):
    #             entrypoint_script.append(
    #                 "export BEAKER_REPLICA_RANK=$("
    #                 "python -m olmo_core.launch.reorder_ranks_in_gcp "
    #                 "${BEAKER_REPLICA_RANK} "
    #                 "${BEAKER_REPLICA_COUNT} "
    #                 "${BEAKER_LEADER_REPLICA_HOSTNAME}"
    #                 ")"
    #             )
    #         entrypoint_script.append(" ".join(self._get_torchrun_cmd()) + ' "$@"')
    #     else:
    #         entrypoint = entrypoint or "python"
    #         entrypoint_script.append(f'{entrypoint} "$@"')

    #     entrypoint_dataset = self._create_script_dataset("entrypoint.sh", entrypoint_script)

    #     task_spec = (
    #         TaskSpec.new(
    #             self.task_name,
    #             beaker_image=self.beaker.image.get(self.beaker_image).id,
    #             priority=self.priority,
    #             preemptible=self.preemptible,
    #             arguments=self.cmd,
    #             command=["bash", "/olmo-core/entrypoint.sh"],
    #             replicas=self.num_nodes if self.num_nodes > 1 else None,
    #             leader_selection=self.num_nodes > 1,
    #             host_networking=self.host_networking
    #             if self.host_networking is not None
    #             else (
    #                 self.num_nodes > 1 or any(["augusta" in cluster for cluster in self.clusters])
    #             ),
    #             propagate_failure=False if self.num_nodes > 1 else None,
    #             propagate_preemption=True if self.num_nodes > 1 else None,
    #             synchronized_start_timeout="90m" if self.num_nodes > 1 else None,
    #             resources=TaskResources(gpu_count=self.num_gpus, shared_memory="10GiB"),
    #         )
    #         .with_dataset("/olmo-core", beaker=entrypoint_dataset.id)
    #         .with_constraint(cluster=self.clusters)
    #         .with_env_var("REPO_URL", f"https://github.com/{github_account}/{github_repo}")
    #         .with_env_var("GIT_REF", git_ref)
    #     )

    #     for name, val in self._get_env_vars():
    #         task_spec = task_spec.with_env_var(name=name, value=val)

    #     for env_secret in self.env_secrets or []:
    #         task_spec = task_spec.with_env_var(name=env_secret.name, secret=env_secret.secret)

    #     if self.nfs:
    #         task_spec = task_spec.with_dataset(
    #             "/net/nfs.cirrascale", host_path="/net/nfs.cirrascale"
    #         )
    #         task_spec = task_spec.with_dataset("/net/nfs", host_path="/net/nfs.cirrascale")

    #     if self.weka_buckets:
    #         for bucket in self.weka_buckets:
    #             task_spec = task_spec.with_dataset(bucket.mount, weka=bucket.bucket)

    #     return ExperimentSpec(
    #         description=self.description,
    #         budget=self.budget,
    #         tasks=[task_spec],
    #         retry=None if not self.retries else RetrySpec(allowed_task_retries=self.retries),
    #     )
