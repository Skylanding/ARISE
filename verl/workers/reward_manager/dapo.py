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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class DAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        skill_reward_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.skill_reward_cfg = skill_reward_cfg or {}

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            non_tensor_batch = data_item.non_tensor_batch

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            reward_model_info = non_tensor_batch.get("reward_model", {})
            if not isinstance(reward_model_info, dict):
                reward_model_info = {}
            ground_truth = reward_model_info.get(
                "ground_truth",
                non_tensor_batch.get("ground_truth", non_tensor_batch.get("answer", "")),
            )
            if ground_truth is None:
                ground_truth = ""
            elif not isinstance(ground_truth, str):
                ground_truth = str(ground_truth)

            data_source = non_tensor_batch.get(self.reward_fn_key, non_tensor_batch.get("data_source", "math_dapo"))
            if data_source is None:
                data_source = "math_dapo"

            extra_info = non_tensor_batch.get("extra_info", None)

            selected_skill_id = non_tensor_batch.get("selected_skill_id")
            compute_result = self.compute_reward_from_text(
                solution_str=response_str,
                ground_truth=ground_truth,
                data_source=data_source,
                extra_info=extra_info,
                selected_skill_id=selected_skill_id,
            )
            reward = compute_result["reward"]
            result = compute_result["result"]
            skill_used = compute_result["skill_used"]
            skill_reward = compute_result["skill_reward"]

            if self.overlong_buffer_cfg is not None and getattr(self.overlong_buffer_cfg, "enable", False):
                overlong_buffer_len = int(self.overlong_buffer_cfg.len)
                # Guard invalid configurations (e.g., len >= max_resp_len), which
                # otherwise apply penalty to almost every sample and collapse reward.
                if self.max_resp_len is None or overlong_buffer_len <= 0 or overlong_buffer_len >= int(self.max_resp_len):
                    overlong_reward = 0.0
                else:
                    expected_len = int(self.max_resp_len) - overlong_buffer_len
                    exceed_len = int(valid_response_length) - expected_len
                    overlong_penalty_factor = float(self.overlong_buffer_cfg.penalty_factor)
                    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0.0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # Keep DAPO-compatible behavior for empty responses: no synthetic
            # response token is activated; this sample contributes zero reward.
            if int(valid_response_length) > 0:
                reward_tensor[i, valid_response_length - 1] = reward
            reward_extra_info["skill_used"].append(bool(skill_used))
            reward_extra_info["skill_reward"].append(float(skill_reward))
            reward_extra_info["selected_skill_id"].append(selected_skill_id)
            if isinstance(result, dict):
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                reward_extra_info["score"].append(float(result))

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", result)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def compute_reward_from_text(
        self,
        solution_str: str,
        ground_truth: str,
        data_source: str = "math_dapo",
        extra_info=None,
        selected_skill_id=None,
    ) -> dict:
        """Compute DAPO reward from decoded text, including optional skill reward."""
        result = self.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if isinstance(result, dict):
            score = result.get("score", 0.0)
        else:
            score = result
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if not torch.isfinite(torch.tensor(score)):
            score = 0.0

        reward = score
        skill_used = selected_skill_id is not None
        skill_reward = 0.0
        if self.skill_reward_cfg.get("enable", False):
            math_correct = reward > 0
            if skill_used and math_correct:
                reward_with_skill = float(
                    self.skill_reward_cfg.get(
                        "reward_when_skill_used_and_correct",
                        self.skill_reward_cfg.get("skill_bonus_when_used_and_correct", 2.0),
                    )
                )
                skill_reward = reward_with_skill - reward
                reward = reward_with_skill

        return {
            "reward": float(reward),
            "score": float(score),
            "result": result,
            "skill_used": bool(skill_used),
            "skill_reward": float(skill_reward),
        }
