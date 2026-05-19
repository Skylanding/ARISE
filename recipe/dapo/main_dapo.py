# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Keep `main` separated from `ray_trainer` because `ray_trainer`
is reused by other entrypoints.
"""

import os

import hydra
import ray

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.device import is_cuda_available

from .dapo_ray_trainer import RayDAPOTrainer


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # Initialize a local Ray cluster when not connected to one.
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # Keep the main task off the Ray head node.
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True evaluates symbolic references.
        OmegaConf.resolve(config)

        model_path = os.path.expanduser(str(config.actor_rollout_ref.model.path))
        # For local absolute paths, fail early with an actionable message
        # instead of deferring to huggingface_hub repo-id validation.
        if os.path.isabs(model_path) and not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model path does not exist: {model_path}. "
                "Please set actor_rollout_ref.model.path to a valid local model directory."
            )
        local_path = copy_to_local(model_path)

        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # For multimodal LLMs; may be None.

        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        try:
            n_gpus_per_node = max(1, int(config.trainer.get("n_gpus_per_node", 1) or 1))
        except (TypeError, ValueError):
            n_gpus_per_node = 1
        try:
            nnodes = max(1, int(config.trainer.get("nnodes", 1) or 1))
        except (TypeError, ValueError):
            nnodes = 1
        resource_pool_spec = {
            global_pool_id: [n_gpus_per_node] * nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Multi-source reward flow:
        # - Rule-based reward managers score directly.
        # - Model-based reward managers call reward models.
        # - Code-style prompts may run in a sandbox when test cases exist.
        # - Final rewards are merged in one place.
        # - Reward type routing depends on data tags.
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "naive":
            from verl.workers.reward_manager import NaiveRewardManager

            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == "prime":
            from verl.workers.reward_manager import PrimeRewardManager

            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == "dapo":
            from verl.workers.reward_manager import DAPORewardManager

            reward_manager_cls = DAPORewardManager
        else:
            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_fn_key = config.data.get("reward_fn_key", "data_source")
        skill_reward_cfg = config.get("skill_library", {}).get("reward", {})
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=getattr(config.reward_model, "overlong_buffer", None),
            skill_reward_cfg=skill_reward_cfg,
        )

        # Validation consistently uses function-based reward evaluation.
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=getattr(config.reward_model, "overlong_buffer", None),
            skill_reward_cfg=skill_reward_cfg,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Optional agent_system integration:
        # TrajectoryCollector + math-style environment (for example dapo-math-17k.parquet).
        traj_collector = None
        envs = None
        val_envs = None
        if config.trainer.get("use_agent_system", False):
            from omegaconf import OmegaConf
            # Default config for math environment: single-turn interaction, no val_batch_size required.
            if not hasattr(config, "env") or config.get("env") is None:
                config.env = OmegaConf.create({
                    "env_name": "math",
                    "max_steps": 1,
                    "rollout": {"n": 1},
                    "seed": 0,
                    "resources_per_worker": {},
                })
            elif config.env.get("env_name", "").lower() != "math" and config.data.get("val_batch_size") is None:
                raise ValueError(
                    "trainer.use_agent_system=True only supports math (DAPO math library). "
                    "Set env.env_name=math or leave env unset; data uses TRAIN_FILE (e.g. dapo-math-17k.parquet)."
                )
            if config.env.get("env_name", "").lower() == "math":
                OmegaConf.set_struct(config.env, False)
                config.env.setdefault("max_steps", 1)
                config.env.setdefault("rollout", OmegaConf.create({"n": 1}))
                if not isinstance(config.env.get("rollout", {}).get("n"), int):
                    config.env.rollout = OmegaConf.create({"n": 1})
                if not hasattr(config.env, "resources_per_worker"):
                    config.env.resources_per_worker = {}
                OmegaConf.set_struct(config.env, True)
            from agent_system.environments import make_envs
            from agent_system.multi_turn_rollout import TrajectoryCollector
            envs, val_envs = make_envs(config)
            # Keep math-environment reward behavior aligned with trainer-side DAPO reward logic
            # (custom_reward_function / overlong_buffer / skill_reward_cfg).
            if getattr(envs, "is_math", False) and hasattr(envs, "set_reward_manager"):
                envs.set_reward_manager(reward_fn)
            if getattr(val_envs, "is_math", False) and hasattr(val_envs, "set_reward_manager"):
                val_envs.set_reward_manager(val_reward_fn)
            traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        device_name = "cuda" if is_cuda_available else "npu"

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            device_name=device_name,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
