from __future__ import annotations

import argparse
import os
from pathlib import Path
from pprint import pprint
import socket

import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from .mixed_grpo_trainer import MixedTemperatureGRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERL_CONFIG_DIR = PROJECT_ROOT / "verl" / "trainer" / "config"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "mixed_temp_grpo_1p7b"
DEFAULT_EXPERIMENT_CONFIG = EXPERIMENT_ROOT / "configs" / "base.yaml"
DEFAULT_REWARD_PATH = EXPERIMENT_ROOT / "reward" / "gsm8k_reward.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Mixed-temperature GRPO mechanism experiment entrypoint")
    parser.add_argument(
        "--experiment-config",
        default=str(DEFAULT_EXPERIMENT_CONFIG),
        help="Path to the experiment-local YAML config fragment.",
    )
    return parser.parse_known_args()


def load_config(experiment_config_path: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(VERL_CONFIG_DIR), version_base=None):
        base_cfg = compose(config_name="ppo_trainer")

    experiment_base_cfg = OmegaConf.load(DEFAULT_EXPERIMENT_CONFIG)
    experiment_override_cfg = (
        OmegaConf.create({})
        if str(Path(experiment_config_path).resolve()) == str(DEFAULT_EXPERIMENT_CONFIG.resolve())
        else OmegaConf.load(experiment_config_path)
    )
    cli_cfg = OmegaConf.from_cli(overrides)
    OmegaConf.set_struct(base_cfg, False)
    config = OmegaConf.merge(base_cfg, experiment_base_cfg, experiment_override_cfg, cli_cfg)

    merged_experiment_cfg = OmegaConf.create({})
    for candidate in (
        OmegaConf.select(experiment_base_cfg, "experiment"),
        OmegaConf.select(experiment_override_cfg, "experiment"),
        OmegaConf.select(cli_cfg, "experiment"),
        OmegaConf.select(config, "experiment"),
        OmegaConf.select(cli_cfg, "mixed_temp_experiment"),
    ):
        if candidate is not None:
            merged_experiment_cfg = OmegaConf.merge(merged_experiment_cfg, candidate)
    config.mixed_temp_experiment = merged_experiment_cfg

    if not config.custom_reward_function.get("path"):
        config.custom_reward_function.path = str(DEFAULT_REWARD_PATH)

    reward_path = Path(config.custom_reward_function.path)
    if not reward_path.is_absolute():
        config.custom_reward_function.path = str((PROJECT_ROOT / reward_path).resolve())

    OmegaConf.resolve(config)
    return config


def main():
    args, overrides = parse_args()
    config = load_config(args.experiment_config, overrides)
    run_mixed_grpo(config)


def run_mixed_grpo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    cuda_available = is_cuda_available() if callable(is_cuda_available) else bool(is_cuda_available)
    if (
        cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = MixedTaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = MixedTaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class MixedTaskRunner:
    def run(self, config):
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        print(f"MixedTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            raise NotImplementedError("The 1.7B mechanistic experiment currently targets the FSDP path only.")
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils.dataset.rl_dataset import collate_fn

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = MixedTemperatureGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
