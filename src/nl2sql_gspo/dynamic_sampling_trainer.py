"""DAPO-style dynamic-sampling GRPO trainer.

Implements the DAPO §3.2 *oversample-and-replace* pipeline on top of
TRL's ``GRPOTrainer`` and a vLLM-server rollout backend.

High-level loop inside ``_generate_and_score_completions`` (training only):

1. Run the standard generation+scoring on the dataloader's prompts.
2. For each per-rank prompt-group (``num_generations`` rollouts), check
   if ``std(result_reward) >= dynamic_sampling_min_std``. If so it is
   "heterogeneous" — i.e. has a non-trivial learning signal — and is
   appended to a per-rank ``kept`` buffer. Otherwise discarded.
3. Synchronize across ranks: how many groups does the rank that needs
   the most still need? That number ``need_max`` becomes the per-rank
   prompt count for the next round so TRL's internal
   ``accelerator.gather`` over ``rewards_per_func`` stays uniform.
4. Each rank draws its slice from a shared shuffled backup queue over
    ``train_dataset``. The queue is consumed from the tail in global
    all-rank blocks, then reshuffled only when exhausted. Drawn prompts
    are replicated ``num_generations`` times (mimicking
    ``RepeatRandomSampler``).
5. Repeat until every rank has filled its kept buffer or
   ``--dapo_max_rounds`` is exhausted.
6. If the budget is exhausted before some ranks fill, pad the buffer
   with zero-masked groups so the output shape matches what TRL's
   training loop expects, and the loss contributes 0 from those slots.
7. Concatenate all kept chunks → final output dict. Re-compute
   ``num_items_in_batch`` from the final completion_mask (DAPO loss
   normalizer = total non-pad completion tokens / num_processes).

Key invariants
--------------
* ``len(round_inputs)`` is identical on all ranks (required by TRL's
  ``_calculate_rewards`` gather).
* ``advantages`` returned by super() are normalized over each round's
  global batch — across rounds we accept slightly inconsistent
  advantage scales (for ``scale_rewards="batch"``); the per-group
  centering is unchanged because it uses intra-group means.
* ``num_items_in_batch`` is recomputed at the end so DAPO loss
  normalization matches the final kept set.
"""

from __future__ import annotations

import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional

from accelerate.utils import DistributedType
from accelerate.utils.operations import gather_object
import torch
from trl.data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from trl.models.utils import disable_gradient_checkpointing
from trl import GRPOTrainer
from trl.trainer.utils import nanmax, nanmin, nanstd, pad, use_adapter
from transformers.trainer import clear_device_cache, is_sagemaker_mp_enabled
from transformers.training_args import OptimizerNames


def _pad_to_width(tensor: torch.Tensor, target_width: int, pad_value, side: str) -> torch.Tensor:
    if tensor.dim() != 2 or tensor.size(1) >= target_width:
        return tensor
    extra = target_width - tensor.size(1)
    pad = tensor.new_full((tensor.size(0), extra), pad_value)
    if side == "left":
        return torch.cat([pad, tensor], dim=1)
    return torch.cat([tensor, pad], dim=1)


def _is_truncated_completion(ids: List[int], eos_and_pad: List[int], max_completion_length: int) -> bool:
    if len(ids) < max_completion_length:
        return False
    return bool(ids) and ids[-1] not in eos_and_pad


# Pad value/side for known 2D output keys. ``prompt_ids`` and
# ``completion_ids`` use the trainer tokenizer pad token, so they're handled inline.
_PAD_SPEC: Dict[str, tuple] = {
    "prompt_mask": (0, "left"),
    "completion_mask": (0, "right"),
    "old_per_token_logps": (0.0, "right"),
    "ref_per_token_logps": (0.0, "right"),
    "sampling_per_token_logps": (0.0, "right"),
    "importance_sampling_ratio": (0.0, "right"),
}


class DynamicSamplingGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with DAPO oversample-and-replace dynamic sampling."""

    @staticmethod
    def _all_skip_policy_loss(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(value.all().item())
        if isinstance(value, list):
            return all(value)
        return bool(value)

    def __init__(
        self,
        *args,
        enable_dynamic_sampling: bool = True,
        dynamic_sampling_min_std: float = 1e-6,
        dapo_max_rounds: int = 6,
        dapo_oversample_factor: int = 1,
        dynamic_sampling_reward_name: Optional[str] = None,
        save_latest_full_checkpoint: bool = False,
        latest_full_checkpoint_dir_name: str = "latest-full-checkpoint",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_dynamic_sampling = enable_dynamic_sampling
        self.dynamic_sampling_min_std = float(dynamic_sampling_min_std)
        self.dapo_max_rounds = max(1, int(dapo_max_rounds))
        self.dapo_oversample_factor = max(1, int(dapo_oversample_factor))
        self.dynamic_sampling_reward_name = (
            dynamic_sampling_reward_name if dynamic_sampling_reward_name else None
        )
        self.save_latest_full_checkpoint = bool(save_latest_full_checkpoint)
        self.latest_full_checkpoint_dir_name = (
            latest_full_checkpoint_dir_name.strip() or "latest-full-checkpoint"
        )
        self._dyn_pool_indices: List[int] = []
        self._dyn_pool_cursor: int = 0
        self._dyn_pool_pass: int = 0
        self._last_rewards_per_func: Optional[torch.Tensor] = None

        # Resolve reward-name to column index on the registered reward funcs.
        self._dyn_reward_idx: Optional[int] = None
        if self.dynamic_sampling_reward_name is not None:
            try:
                self._dyn_reward_idx = list(self.reward_func_names).index(
                    self.dynamic_sampling_reward_name
                )
            except ValueError:
                if getattr(self.accelerator, "is_main_process", True):
                    print(
                        f"[dapo] WARNING: reward name "
                        f"'{self.dynamic_sampling_reward_name}' not found in "
                        f"{list(self.reward_func_names)}; falling back to total advantages."
                    )
                self.dynamic_sampling_reward_name = None

        if getattr(self.accelerator, "is_main_process", True):
            criterion = (
                f"single reward '{self.dynamic_sampling_reward_name}'"
                if self.dynamic_sampling_reward_name
                else "total advantages"
            )
            mode = (
                f"single-shot-oversample(K={self.dapo_oversample_factor})"
                if self.dapo_oversample_factor > 1
                else "iterative-oversample-and-replace"
            )
            print(
                f"[dapo] enabled={self.enable_dynamic_sampling} | mode={mode} | "
                f"max_rounds={self.dapo_max_rounds} | min_std={self.dynamic_sampling_min_std} | "
                f"criterion={criterion} | num_generations={self.num_generations}"
            )

    def _wait_for_everyone(self) -> None:
        wait_for_everyone = getattr(self.accelerator, "wait_for_everyone", None)
        if callable(wait_for_everyone):
            wait_for_everyone()

    def _latest_full_checkpoint_dir(self, trial=None) -> str:
        run_dir = self._get_output_dir(trial=trial)
        return os.path.join(run_dir, self.latest_full_checkpoint_dir_name)

    def _save_latest_restart_checkpoint(self, trial=None) -> None:
        output_dir = self._latest_full_checkpoint_dir(trial=trial)
        if self.is_world_process_zero() and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        self._wait_for_everyone()

        self.save_model(output_dir, _internal_call=True)
        self._save_optimizer_and_scheduler(output_dir)
        self._save_scaler(output_dir)
        self._save_rng_state(output_dir)

        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        self._wait_for_everyone()

    def _save_checkpoint(self, model, trial) -> None:
        super()._save_checkpoint(model, trial)
        if self.save_latest_full_checkpoint:
            self._save_latest_restart_checkpoint(trial=trial)

    # ------------------------------------------------------------------ #
    # Backup pool of replacement prompts (shared shuffled tail queue)
    # ------------------------------------------------------------------ #
    def _refresh_dyn_pool(self) -> None:
        if self.train_dataset is None:
            self._dyn_pool_indices = []
            return
        n = len(self.train_dataset)
        # Shared shuffle across ranks. Each rank later takes a different
        # slice from the same global tail block, so backup prompts consumed
        # by rank 1 on step N are skipped by rank 0 on step N+1.
        # ``_dyn_pool_pass`` changes the order on wrap.
        base_seed = int(getattr(getattr(self, "args", None), "seed", 0) or 0)
        seed = (
            base_seed
            + self._dyn_pool_pass * 1000
            + 17
        )
        rng = random.Random(seed)
        idxs = list(range(n))
        rng.shuffle(idxs)
        self._dyn_pool_indices = idxs
        self._dyn_pool_cursor = n
        self._dyn_pool_pass += 1

    def _draw_replacement_inputs(self, k: int) -> List[Dict[str, Any]]:
        if k <= 0 or self.train_dataset is None:
            return []
        world_size = max(1, int(getattr(self.accelerator, "num_processes", 1) or 1))
        rank = int(getattr(self.accelerator, "process_index", 0) or 0)
        global_count = k * world_size

        global_block: List[int] = []
        while len(global_block) < global_count:
            if not self._dyn_pool_indices or self._dyn_pool_cursor <= 0:
                self._refresh_dyn_pool()
            if not self._dyn_pool_indices:
                break

            need = global_count - len(global_block)
            take = min(need, self._dyn_pool_cursor)
            start = self._dyn_pool_cursor - take
            tail_chunk = self._dyn_pool_indices[start:self._dyn_pool_cursor]
            global_block.extend(reversed(tail_chunk))
            self._dyn_pool_cursor = start

        start = rank * k
        picked = global_block[start : start + k]
        return [dict(self.train_dataset[i]) for i in picked]

    def _build_round_inputs(self, unique_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replicate each unique prompt ``num_generations`` times,
        mimicking ``RepeatRandomSampler(repeat_count=num_generations)``."""
        out: List[Dict[str, Any]] = []
        for ui in unique_inputs:
            for _ in range(self.num_generations):
                out.append(dict(ui))
        return out

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    def _calculate_rewards(self, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        # rewards_per_func is gathered (global) — keep a reference so the
        # heterogeneity check can read a single reward column if requested.
        self._last_rewards_per_func = rewards_per_func
        return rewards_per_func

    # ------------------------------------------------------------------ #
    # Heterogeneity check
    # ------------------------------------------------------------------ #
    def _het_mask_for_round(self, round_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Bool mask over per-rank groups (length = local_groups). True = heterogeneous."""
        advantages = round_out["advantages"]
        local_n = advantages.shape[0]
        num_generations = self.num_generations
        if local_n == 0 or local_n % num_generations != 0:
            return torch.zeros(0, dtype=torch.bool, device=advantages.device)

        # Prefer a single reward column (e.g. result_reward). A 0/1-valued
        # group with all rollouts identical has std == 0.
        if (
            self._dyn_reward_idx is not None
            and self._last_rewards_per_func is not None
        ):
            rpf = self._last_rewards_per_func  # global, gathered
            start = self.accelerator.process_index * local_n
            local_rewards = rpf[start : start + local_n, self._dyn_reward_idx]
            grouped = local_rewards.view(-1, num_generations).to(advantages.device)
            return grouped.float().std(dim=1, unbiased=False) >= self.dynamic_sampling_min_std

        # Fallback: aggregated advantages — non-zero somewhere in the group.
        adv_grouped = advantages.view(-1, num_generations)
        return adv_grouped.abs().max(dim=1).values >= self.dynamic_sampling_min_std

    def _group_reward_bucket_counts(self, round_out: Dict[str, torch.Tensor]):
        """Per-rank counts of all-correct and all-wrong groups for the dyn reward."""
        advantages = round_out["advantages"]
        local_n = advantages.shape[0]
        num_generations = self.num_generations
        if local_n == 0 or local_n % num_generations != 0:
            return None
        if self._dyn_reward_idx is None or self._last_rewards_per_func is None:
            return None

        rpf = self._last_rewards_per_func
        start = self.accelerator.process_index * local_n
        local_rewards = rpf[start : start + local_n, self._dyn_reward_idx]
        grouped = local_rewards.view(-1, num_generations).float().to(advantages.device)
        group_min = grouped.min(dim=1).values
        group_max = grouped.max(dim=1).values
        return (
            int((group_min >= 1.0 - 1e-6).long().sum().item()),
            int((group_max <= 1e-6).long().sum().item()),
        )

    # ------------------------------------------------------------------ #
    # Group extraction / concat
    # ------------------------------------------------------------------ #
    def _extract_groups(
        self,
        round_out: Dict[str, Any],
        group_idx: List[int],
        zero_mask: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not group_idx:
            return None
        num_generations = self.num_generations
        local_n = round_out["prompt_ids"].shape[0]
        rows: List[int] = []
        for g in group_idx:
            rows.extend(range(g * num_generations, (g + 1) * num_generations))
        rows_t = torch.tensor(rows, dtype=torch.long)

        chunk: Dict[str, Any] = {"_n_groups": len(group_idx)}
        for k, v in round_out.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == local_n:
                chunk[k] = v[rows_t.to(v.device)]
            else:
                chunk[k] = v  # scalars / non-row tensors carry through

        if zero_mask and "completion_mask" in chunk and isinstance(chunk["completion_mask"], torch.Tensor):
            chunk["completion_mask"] = torch.zeros_like(chunk["completion_mask"])
        return chunk

    def _concat_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        chunks = [c for c in chunks if c is not None]
        assert chunks, "DAPO: no chunks to concat"

        tensor_keys = set()
        for c in chunks:
            for k, v in c.items():
                # Only concat row-batched tensors. 0-dim scalars (e.g.
                # num_items_in_batch) are carried through, not concatenated.
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    tensor_keys.add(k)

        final: Dict[str, Any] = {}
        for k in tensor_keys:
            vals = [
                c.get(k)
                for c in chunks
                if isinstance(c.get(k), torch.Tensor) and c.get(k).dim() >= 1
            ]
            if not vals:
                continue
            v0 = vals[0]
            if v0.dim() == 1:
                final[k] = torch.cat(vals, dim=0)
                continue
            if v0.dim() == 2:
                if k == "prompt_ids":
                    pad_val, side = self._tokenizer.pad_token_id, "left"
                elif k == "completion_ids":
                    pad_val, side = self._tokenizer.pad_token_id, "right"
                else:
                    pad_val, side = _PAD_SPEC.get(k, (0, "right"))
                max_w = max(t.size(1) for t in vals)
                vals = [_pad_to_width(t, max_w, pad_val, side) for t in vals]
                final[k] = torch.cat(vals, dim=0)
                continue
            final[k] = torch.cat(vals, dim=0)

        # Carry through scalar / non-tensor keys from the first chunk
        for k, v in chunks[0].items():
            if k.startswith("_") or k in final:
                continue
            final[k] = v

        return final

    # ------------------------------------------------------------------ #
    # Single-shot oversample path (preferred when K>1)
    # ------------------------------------------------------------------ #
    def _generate_and_score_candidates_no_policy_logps(self, inputs):
        """Generate/reward candidate groups without policy logprob forwards.

        TRL's standard method computes vLLM old logprobs before rewards.
        For DAPO oversampling that means paying the model-forward cost for
        discarded candidate groups. This variant mirrors the generation,
        reward, advantage, and logging portions of TRL's implementation, then
        leaves policy/reference logprobs to be computed after filtering.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]

        if self.environments:
            for prompt, environment, reset_kwargs in zip(prompts, self.environments, inputs, strict=True):
                observation = environment.reset(**reset_kwargs)
                if observation is not None:
                    prompt[-1]["content"] += observation

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs."
                )
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        generated = self._generate(prompts)
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
            *_,
        ) = generated

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self._tokenizer.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self._tokenizer.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")
        else:
            tool_mask = None

        if self.mask_truncated_completions:
            eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
            is_truncated = torch.tensor(
                [
                    _is_truncated_completion(ids, eos_and_pad, self.max_completion_length)
                    for ids in completion_ids_list
                ],
                device=device,
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )
        if "mm_token_type_ids" in forward_kwargs:
            mm_token_type_ids = forward_kwargs["mm_token_type_ids"]
            forward_kwargs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            else:
                raise ValueError("Invalid value for scale_rewards")
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        else:
            raise ValueError("Invalid multi_objective_aggregation")

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards_for_log = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards_for_log.mean().item())
        self._metrics[mode]["reward_std"].append(rewards_for_log.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())
        if images is not None:
            self._logs["images"].extend(gather_object(images))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if "mm_token_type_ids" in forward_kwargs:
            output["mm_token_type_ids"] = forward_kwargs["mm_token_type_ids"]
        if images is not None:
            output["num_images"] = [len(img_list) for img_list in images]
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        return output

    def _add_policy_logps_for_kept(self, output: Dict[str, Any]) -> Dict[str, Any]:
        if self._all_skip_policy_loss(output.get("_skip_policy_loss", False)):
            if self.use_vllm and self.vllm_importance_sampling_correction:
                output["importance_sampling_ratio"] = torch.ones_like(
                    output["completion_mask"], dtype=torch.float32, device=output["completion_mask"].device
                )
            return output

        device = self.accelerator.device
        prompt_ids = output["prompt_ids"]
        prompt_mask = output["prompt_mask"]
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size
        num_images = output.get("num_images")
        forward_kwargs = {
            k: output[k]
            for k in [
                "pixel_values",
                "image_grid_thw",
                "pixel_attention_mask",
                "image_sizes",
                "token_type_ids",
                "mm_token_type_ids",
            ]
            if k in output
        }

        old_per_token_logps = None
        ref_per_token_logps = None
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            generate_every = self.args.steps_per_generation * self.num_iterations
            need_old_logps = self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            )
            if need_old_logps:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )

            if self.use_vllm and self.vllm_importance_sampling_correction:
                sampling_per_token_logps = output["sampling_per_token_logps"]
                mask = completion_mask if "tool_mask" not in output else completion_mask * output["tool_mask"]
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                else:
                    logps_diff = per_token_logps_diff
                vllm_importance_sampling_ratio = torch.exp(logps_diff)
                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap,
                        value=0.0,
                    )
                else:
                    raise ValueError(f"Unknown vLLM importance sampling mode: {self.vllm_importance_sampling_mode}")
                output["importance_sampling_ratio"] = vllm_importance_sampling_ratio

                delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
                delta_mask = mask.bool()
                delta = delta[delta_mask]
                mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
                max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
                self._metrics["train"]["sampling/sampling_logp_difference/mean"].append(
                    self.accelerator.gather(mean_delta).mean().item()
                )
                self._metrics["train"]["sampling/sampling_logp_difference/max"].append(
                    self.accelerator.gather(max_delta).max().item()
                )
                flat_is_ratio = (
                    vllm_importance_sampling_ratio.flatten()
                    if sequence_level_is
                    else vllm_importance_sampling_ratio[delta_mask]
                )
                min_is = torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                mean_is = torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                max_is = torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                self._metrics["train"]["sampling/importance_sampling_ratio/min"].append(
                    nanmin(self.accelerator.gather(min_is)).item()
                )
                self._metrics["train"]["sampling/importance_sampling_ratio/mean"].append(
                    self.accelerator.gather(mean_is).nanmean().item()
                )
                self._metrics["train"]["sampling/importance_sampling_ratio/max"].append(
                    nanmax(self.accelerator.gather(max_is)).item()
                )

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
                else:
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                            num_images=num_images,
                            **forward_kwargs,
                        )

        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = self._training_step_with_dapo_skip(model, inputs, num_items_in_batch)
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        return output

    def _training_step_with_dapo_skip(self, model, inputs, num_items_in_batch):
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if self.model is not model:
                self.model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if self._all_skip_policy_loss(inputs.get("_skip_policy_loss", False)):
                return torch.zeros((), device=self.accelerator.device)

            if is_sagemaker_mp_enabled():
                from transformers.trainer import smp_forward_backward

                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                loss = loss / self.current_gradient_accumulation_steps

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)
            return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _oversample_and_filter(self, inputs, target_local_groups: int):
        """Single-shot oversample by ``dapo_oversample_factor``.

        Each rank generates ``target_local_groups * K`` groups in one
        round. Extra prompts are drawn from the shared shuffled backup
        queue, consumed from the tail in all-rank blocks. Per rank we keep
        the first ``target_local_groups`` heterogeneous groups; if fewer
        are found we fill the remainder with random
        non-het groups whose ``completion_mask`` is zeroed (no gradient
        contribution). Same shape on every rank by construction.
        """
        device = self.accelerator.device
        K = self.dapo_oversample_factor
        extra_count = target_local_groups * (K - 1)

        big_inputs = list(inputs)
        if extra_count > 0:
            extras_unique = self._draw_replacement_inputs(extra_count)
            # Backup pool too small: fall back to whatever we got.
            big_inputs = big_inputs + self._build_round_inputs(extras_unique)

        round_out = self._generate_and_score_candidates_no_policy_logps(big_inputs)

        het = self._het_mask_for_round(round_out)  # [local_groups]
        local_groups = int(het.numel())
        het_idx = torch.nonzero(het).flatten().tolist()
        nonhet_idx = torch.nonzero(~het).flatten().tolist()

        take_het = het_idx[:target_local_groups]
        chunks: List[Dict[str, Any]] = []
        if take_het:
            ck = self._extract_groups(round_out, take_het)
            if ck is not None:
                chunks.append(ck)

        deficit = target_local_groups - len(take_het)
        n_padded_local = 0
        if deficit > 0 and nonhet_idx:
            step = int(getattr(self.state, "global_step", 0))
            rng = random.Random(step * 1000 + int(self.accelerator.process_index) + 53)
            take_pad = rng.sample(nonhet_idx, min(deficit, len(nonhet_idx)))
            ck = self._extract_groups(round_out, take_pad, zero_mask=True)
            if ck is not None:
                chunks.append(ck)
                n_padded_local = len(take_pad)
            deficit -= len(take_pad)

        # Pathological: not enough groups generated at all (dataset tiny).
        # Fall back to first available group with zero-mask.
        if deficit > 0 and local_groups > 0:
            fallback_idx = list(range(min(deficit, local_groups)))
            ck = self._extract_groups(round_out, fallback_idx, zero_mask=True)
            if ck is not None:
                chunks.append(ck)
                n_padded_local += len(fallback_idx)

        out = self._concat_chunks(chunks)

        # Recompute num_items_in_batch (DAPO loss normalizer)
        try:
            local_count = (out["completion_mask"] > 0).long().sum().to(device)
            agg = self.accelerator.gather(local_count.unsqueeze(0)).sum()
            skip_policy_loss = bool(agg.item() == 0)
            out["num_items_in_batch"] = agg.clamp(min=1)
        except Exception:
            local_count = (out["completion_mask"] > 0).long().sum().to(device)
            skip_policy_loss = bool(local_count.item() == 0)
            out["num_items_in_batch"] = local_count.clamp(min=1)
        out["_skip_policy_loss"] = torch.full(
            (out["completion_mask"].size(0),),
            skip_policy_loss,
            dtype=torch.bool,
            device=out["completion_mask"].device,
        )

        # Logging
        try:
            local_candidate_het = int(het.long().sum().item())
            reward_bucket_counts = self._group_reward_bucket_counts(round_out)
            local_all_correct = reward_bucket_counts[0] if reward_bucket_counts is not None else -1
            local_all_wrong = reward_bucket_counts[1] if reward_bucket_counts is not None else -1
            counters = torch.tensor(
                [local_groups, local_candidate_het, len(take_het), n_padded_local, local_all_correct, local_all_wrong],
                device=device,
                dtype=torch.long,
            )
            gathered = self.accelerator.gather(counters.unsqueeze(0))
            g_attempted = int(gathered[:, 0].sum().item())
            g_candidate_het = int(gathered[:, 1].sum().item())
            g_kept = int(gathered[:, 2].sum().item())
            g_padded = int(gathered[:, 3].sum().item())
            g_all_correct = None if (gathered[:, 4] < 0).any() else int(gathered[:, 4].sum().item())
            g_all_wrong = None if (gathered[:, 5] < 0).any() else int(gathered[:, 5].sum().item())
        except Exception:
            g_attempted = local_groups
            g_candidate_het = int(het.long().sum().item())
            g_kept = len(take_het)
            g_padded = n_padded_local
            reward_bucket_counts = self._group_reward_bucket_counts(round_out)
            g_all_correct = reward_bucket_counts[0] if reward_bucket_counts is not None else None
            g_all_wrong = reward_bucket_counts[1] if reward_bucket_counts is not None else None

        candidate_het_rate = (g_candidate_het / g_attempted) if g_attempted > 0 else 0.0
        fill_rate = (g_kept / (g_kept + g_padded)) if (g_kept + g_padded) > 0 else 0.0
        self._metrics["train"]["dapo/rounds_used"].append(1.0)
        self._metrics["train"]["dapo/groups_attempted"].append(float(g_attempted))
        self._metrics["train"]["dapo/groups_heterogeneous"].append(float(g_candidate_het))
        self._metrics["train"]["dapo/groups_kept"].append(float(g_kept))
        self._metrics["train"]["dapo/groups_padded"].append(float(g_padded))
        self._metrics["train"]["dapo/heterogeneity_rate"].append(candidate_het_rate)
        self._metrics["train"]["dapo/selection_fill_rate"].append(fill_rate)

        if getattr(self.accelerator, "is_main_process", True):
            step = int(getattr(self.state, "global_step", 0))
            reward_bits = []
            for reward_name in ["result_reward", "execution_reward", "format_reward"]:
                key = f"rewards/{reward_name}/mean"
                values = self._metrics.get("train", {}).get(key, [])
                if values:
                    reward_bits.append(f"{reward_name}={values[-1]:.4g}")
            reward_suffix = " " + " ".join(reward_bits) if reward_bits else ""
            bucket_suffix = ""
            if g_all_correct is not None and g_all_wrong is not None:
                bucket_suffix = f" all_correct={g_all_correct} all_wrong={g_all_wrong}"
            print(
                f"[dapo] step={step} mode=oversample K={K} "
                f"attempted={g_attempted} candidate_het={g_candidate_het} "
                f"candidate_het_rate={candidate_het_rate:.2%} "
                f"selected={g_kept}/{g_kept + g_padded} padded={g_padded} "
                f"fill_rate={fill_rate:.2%}{bucket_suffix}{reward_suffix}"
            )
        return self._add_policy_logps_for_kept(out)

    # ------------------------------------------------------------------ #
    # Main override — DAPO oversample-and-replace
    # ------------------------------------------------------------------ #
    def _generate_and_score_completions(self, inputs):
        # Eval / disabled → vanilla path.
        if not self.enable_dynamic_sampling or not self.model.training:
            return super()._generate_and_score_completions(inputs)

        num_generations = self.num_generations
        target_local_groups = len(inputs) // num_generations
        if target_local_groups == 0 or len(inputs) % num_generations != 0:
            return super()._generate_and_score_completions(inputs)

        # Single-shot oversample path (preferred).
        if self.dapo_oversample_factor > 1:
            return self._oversample_and_filter(inputs, target_local_groups)

        device = self.accelerator.device
        kept_chunks: List[Dict[str, Any]] = []
        last_round_out: Optional[Dict[str, Any]] = None
        round_inputs = list(inputs)

        rounds_used = 0
        total_groups_attempted = 0
        total_groups_heterogeneous = 0
        total_groups_kept = 0

        for r in range(self.dapo_max_rounds):
            round_out = self._generate_and_score_candidates_no_policy_logps(round_inputs)
            last_round_out = round_out
            rounds_used += 1

            het = self._het_mask_for_round(round_out)
            local_groups_this_round = int(het.numel())
            total_groups_attempted += local_groups_this_round
            total_groups_heterogeneous += int(het.long().sum().item()) if het is not None else 0

            kept_so_far = sum(c["_n_groups"] for c in kept_chunks)
            slots_remaining = target_local_groups - kept_so_far

            het_idx = torch.nonzero(het).flatten().tolist()[:slots_remaining]
            if het_idx:
                chunk = self._extract_groups(round_out, het_idx)
                if chunk is not None:
                    kept_chunks.append(chunk)
                    total_groups_kept += len(het_idx)

            kept_so_far = sum(c["_n_groups"] for c in kept_chunks)
            need_local = target_local_groups - kept_so_far

            need_t = torch.tensor([need_local], device=device, dtype=torch.long)
            need_max = int(self.accelerator.gather(need_t).max().item())

            if need_max == 0:
                break
            if r == self.dapo_max_rounds - 1:
                break  # last allowed round; can't sample more

            replacement_unique = self._draw_replacement_inputs(need_max)
            if len(replacement_unique) < need_max:
                break  # backup pool exhausted
            round_inputs = self._build_round_inputs(replacement_unique)

        # Pad with zero-masked groups if still short.
        kept_so_far = sum(c["_n_groups"] for c in kept_chunks)
        padded_groups = 0
        if kept_so_far < target_local_groups and last_round_out is not None:
            deficit = target_local_groups - kept_so_far
            available = last_round_out["prompt_ids"].shape[0] // num_generations
            take = min(deficit, available)
            if take > 0:
                pad_chunk = self._extract_groups(
                    last_round_out, list(range(take)), zero_mask=True
                )
                if pad_chunk is not None:
                    kept_chunks.append(pad_chunk)
                    padded_groups = take

        out = self._concat_chunks(kept_chunks)

        # Recompute num_items_in_batch (DAPO loss normalizer)
        try:
            local_count = (out["completion_mask"] > 0).long().sum().to(device)
            agg = self.accelerator.gather(local_count.unsqueeze(0)).sum()
            skip_policy_loss = bool(agg.item() == 0)
            out["num_items_in_batch"] = agg.clamp(min=1)
        except Exception:
            local_count = (out["completion_mask"] > 0).long().sum().to(device)
            skip_policy_loss = bool(local_count.item() == 0)
            out["num_items_in_batch"] = local_count.clamp(min=1)
        out["_skip_policy_loss"] = torch.full(
            (out["completion_mask"].size(0),),
            skip_policy_loss,
            dtype=torch.bool,
            device=out["completion_mask"].device,
        )

        # Logging
        try:
            reward_bucket_counts = self._group_reward_bucket_counts(last_round_out) if last_round_out is not None else None
            local_all_correct = reward_bucket_counts[0] if reward_bucket_counts is not None else -1
            local_all_wrong = reward_bucket_counts[1] if reward_bucket_counts is not None else -1
            counters = torch.tensor(
                [
                    rounds_used,
                    total_groups_attempted,
                    total_groups_heterogeneous,
                    total_groups_kept,
                    padded_groups,
                    local_all_correct,
                    local_all_wrong,
                ],
                device=device,
                dtype=torch.long,
            )
            gathered = self.accelerator.gather(counters.unsqueeze(0))
            g_rounds_max = int(gathered[:, 0].max().item())
            g_attempted = int(gathered[:, 1].sum().item())
            g_candidate_het = int(gathered[:, 2].sum().item())
            g_kept = int(gathered[:, 3].sum().item())
            g_padded = int(gathered[:, 4].sum().item())
            g_all_correct = None if (gathered[:, 5] < 0).any() else int(gathered[:, 5].sum().item())
            g_all_wrong = None if (gathered[:, 6] < 0).any() else int(gathered[:, 6].sum().item())
        except Exception:
            g_rounds_max = rounds_used
            g_attempted = total_groups_attempted
            g_candidate_het = total_groups_heterogeneous
            g_kept = total_groups_kept
            g_padded = padded_groups
            reward_bucket_counts = self._group_reward_bucket_counts(last_round_out) if last_round_out is not None else None
            g_all_correct = reward_bucket_counts[0] if reward_bucket_counts is not None else None
            g_all_wrong = reward_bucket_counts[1] if reward_bucket_counts is not None else None

        candidate_het_rate = (g_candidate_het / g_attempted) if g_attempted > 0 else 0.0
        fill_rate = (g_kept / (g_kept + g_padded)) if (g_kept + g_padded) > 0 else 0.0
        self._metrics["train"]["dapo/rounds_used"].append(float(g_rounds_max))
        self._metrics["train"]["dapo/groups_attempted"].append(float(g_attempted))
        self._metrics["train"]["dapo/groups_heterogeneous"].append(float(g_candidate_het))
        self._metrics["train"]["dapo/groups_kept"].append(float(g_kept))
        self._metrics["train"]["dapo/groups_padded"].append(float(g_padded))
        self._metrics["train"]["dapo/heterogeneity_rate"].append(candidate_het_rate)
        self._metrics["train"]["dapo/selection_fill_rate"].append(fill_rate)

        if getattr(self.accelerator, "is_main_process", True):
            step = int(getattr(self.state, "global_step", 0))
            bucket_suffix = ""
            if g_all_correct is not None and g_all_wrong is not None:
                bucket_suffix = f" all_correct={g_all_correct} all_wrong={g_all_wrong}"
            print(
                f"[dapo] step={step} rounds={g_rounds_max}/{self.dapo_max_rounds} "
                f"attempted={g_attempted} candidate_het={g_candidate_het} "
                f"candidate_het_rate={candidate_het_rate:.2%} "
                f"selected={g_kept}/{g_kept + g_padded} padded={g_padded} "
                f"fill_rate={fill_rate:.2%}{bucket_suffix}"
            )

        return self._add_policy_logps_for_kept(out)
