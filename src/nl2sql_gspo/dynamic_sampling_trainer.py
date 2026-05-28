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
   ``num_items_in_batch`` from the final effective completion mask
   (completion padding/truncation mask AND tool-observation mask; DAPO loss
   normalizer = total policy-action completion tokens / num_processes).

Key invariants
--------------
* ``len(round_inputs)`` is identical on all ranks (required by TRL's
  ``_calculate_rewards`` gather).
* ``advantages`` returned by super() are normalized over each round's
  global batch — across rounds we accept slightly inconsistent
  advantage scales (for ``scale_rewards="batch"``); the per-group
  centering is unchanged because it uses intra-group means.
* ``num_items_in_batch`` is recomputed at the end from the effective policy
  mask so DAPO loss normalization matches the final kept set.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from accelerate.utils import DistributedType, broadcast_object_list
from accelerate.utils.operations import gather_object
import torch
from trl.chat_template_utils import parse_response
from trl.data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from trl.models.utils import disable_gradient_checkpointing
from trl import GRPOTrainer
from trl.trainer.utils import nanmax, nanmin, nanstd, pad, use_adapter
from transformers.trainer import clear_device_cache, is_sagemaker_mp_enabled
from transformers.training_args import OptimizerNames

from nl2sql_gspo.inference_tool_executor import extract_tool_calls
from nl2sql_gspo.sql_utils import extract_completion_text, extract_final_answer_sql, extract_sql


_TOOL_CALL_NAME_RE = re.compile(
    r"(?:<\|tool_call\>\s*)?call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{",
    re.IGNORECASE,
)
_TOOL_RESPONSE_RE = re.compile(r"<\|tool_response\>|response:[A-Za-z_][A-Za-z0-9_]*\{", re.IGNORECASE)


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


def _flatten_gathered(items):
    """Flatten gather_object output while preserving scalar dict samples."""

    flattened = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _quantile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


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
        reward_only_eval: bool = False,
        beta_schedule: Optional[Sequence[Tuple[int, float]]] = None,
        static_beta_fallback: Optional[float] = None,
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
        self.reward_only_eval = bool(reward_only_eval)
        self.beta_schedule = sorted(
            ((int(step), float(beta)) for step, beta in (beta_schedule or [])),
            key=lambda item: item[0],
        )
        self._static_beta = float(
            getattr(self, "beta", 0.0) if static_beta_fallback is None else static_beta_fallback
        )
        self._dyn_pool_indices: List[int] = []
        self._dyn_pool_cursor: int = 0
        self._dyn_pool_pass: int = 0
        self._last_rewards_per_func: Optional[torch.Tensor] = None
        self.debug_rollouts = os.environ.get("DAPO_DEBUG_ROLLOUTS", "0") != "0"
        self.debug_rollout_samples = max(0, int(os.environ.get("DAPO_DEBUG_ROLLOUT_SAMPLES", "3")))
        self.debug_rollout_sample_chars = max(0, int(os.environ.get("DAPO_DEBUG_ROLLOUT_SAMPLE_CHARS", "500")))
        self.debug_rollout_every = max(1, int(os.environ.get("DAPO_DEBUG_ROLLOUT_EVERY", "1")))
        self.debug_tool_loop = os.environ.get("TOOL_LOOP_DEBUG", "0") != "0"
        self.debug_attention_mask = os.environ.get("ATTENTION_MASK_DEBUG", "0") != "0"
        self.debug_attention_mask_every = max(1, int(os.environ.get("ATTENTION_MASK_DEBUG_EVERY", "1")))
        self.debug_attention_mask_samples = max(0, int(os.environ.get("ATTENTION_MASK_DEBUG_SAMPLES", "2")))

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
                f"criterion={criterion} | num_generations={self.num_generations} | "
                f"max_tool_calling_iterations={self.max_tool_calling_iterations}"
            )
            if self.debug_rollouts:
                print(
                    "[rollout-debug] enabled "
                    f"every={self.debug_rollout_every} "
                    f"samples={self.debug_rollout_samples} "
                    f"sample_chars={self.debug_rollout_sample_chars}"
                )
            if self.debug_attention_mask:
                print(
                    "[attention-mask-debug] enabled "
                    f"every={self.debug_attention_mask_every} "
                    f"samples={self.debug_attention_mask_samples}"
                )
            if self.beta_schedule:
                formatted = ",".join(f"{step}:{beta:g}" for step, beta in self.beta_schedule)
                print(f"[beta-schedule] enabled {formatted} | static_beta_fallback={self._static_beta:g}")
            else:
                print(f"[beta-schedule] disabled | static_beta={self._static_beta:g}")

    def _current_beta(self) -> float:
        if not self.beta_schedule:
            return self._static_beta

        step = int(getattr(self.state, "global_step", 0))
        current = self._static_beta
        for start_step, beta in self.beta_schedule:
            if step < start_step:
                break
            current = beta
        return float(current)

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
        original_save_only_model = self.args.save_only_model
        if self.save_latest_full_checkpoint:
            self.args.save_only_model = True
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self.args.save_only_model = original_save_only_model
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

    @staticmethod
    def _effective_completion_mask(output: Dict[str, Any]) -> torch.Tensor:
        """Mask of policy-action tokens: completion padding/truncation AND tool observations."""

        completion_mask = output["completion_mask"]
        tool_mask = output.get("tool_mask")
        if isinstance(tool_mask, torch.Tensor):
            return completion_mask * tool_mask
        return completion_mask

    def _global_sum_int(self, value: int) -> int:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return int(value)
        tensor = torch.tensor(int(value), device=self.accelerator.device, dtype=torch.long)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return int(tensor.item())

    def _generate_tool_continuations(self, prompt_ids, images, multimodal_fields):
        """Generate one continuation per active tool-rollout history.

        TRL's server-mode vLLM helper assumes prompts are arranged as
        ``num_generations`` duplicates of each original prompt and therefore
        sends every Nth prompt with ``n=N``. After a tool response is appended,
        each rollout history is unique, so continuation must be ``n=1`` per
        active sequence. This helper also supports zero active local sequences
        so all ranks still participate in the same collectives.
        """
        if not self.use_vllm or getattr(self.vllm_generation, "mode", None) != "server":
            return self._generate_single_turn(prompt_ids, images, multimodal_fields)

        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        all_prompts = gather_object(prompt_ids)
        local_images = images if images is not None else [None] * len(prompt_ids)
        all_images = gather_object(local_images)
        if all(img is None for img in all_images):
            all_images = None
        counts = [int(x) for x in gather_object([len(prompt_ids)])]

        if self.accelerator.is_main_process:
            if all_prompts:
                output = self.vllm_generation.vllm_client.generate(
                    prompts=all_prompts,
                    images=all_images,
                    n=1,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    logprobs=getattr(self.vllm_generation, "logprobs", 0),
                    structured_outputs_regex=getattr(self.vllm_generation, "structured_outputs_regex", None),
                    generation_kwargs=getattr(self.vllm_generation, "generation_kwargs", None),
                )
                payload = (
                    output["prompt_ids"],
                    output["completion_ids"],
                    output["logprobs"],
                    output.get("logprob_token_ids"),
                )
            else:
                payload = ([], [], [], None)
        else:
            payload = None

        obj_list = [payload]
        broadcast_object_list(obj_list, from_process=0)
        _, all_completion_ids, all_logprobs, _ = obj_list[0]

        start = sum(counts[: self.accelerator.process_index])
        end = start + len(prompt_ids)
        completion_ids = all_completion_ids[start:end]
        logprobs = all_logprobs[start:end] if all_logprobs is not None else None
        if logprobs is not None:
            logprobs = [[lp[0] for lp in seq] for seq in logprobs]
        return completion_ids, logprobs

    def _completion_debug_flags(self, completion: Any) -> Dict[str, Any]:
        text = extract_completion_text(completion)
        call_names = [m.group("name") for m in _TOOL_CALL_NAME_RE.finditer(text)]
        final_sql = extract_final_answer_sql(text)
        extracted_sql = extract_sql(text)
        roles = []
        if isinstance(completion, list):
            roles = [str(m.get("role", "")) for m in completion if isinstance(m, dict)]
        return {
            "text": text,
            "call_names": call_names,
            "has_tool_call": bool(call_names),
            "has_tool_response": bool(_TOOL_RESPONSE_RE.search(text)) or "tool" in roles,
            "has_final_answer_tag": "<final_answer" in text.lower(),
            "has_final_sql": bool(final_sql),
            "has_extracted_sql": bool(extracted_sql),
            "sql": final_sql or extracted_sql,
        }

    def _attach_gemma_tool_calls(self, completions) -> int:
        """Populate TRL's expected ``tool_calls`` field from Gemma tool text.

        TRL only runs tools when the decoded assistant message contains a
        structured ``tool_calls`` list. Gemma-4 emits compact native calls like
        ``call:sqlite_query{db_id:<|"|>...,sql:<|"|>...}``; without this bridge
        those calls remain plain text and no tool response is ever appended.
        """

        attached = 0
        if not completions:
            return attached

        for completion in completions:
            if not isinstance(completion, list) or not completion:
                continue
            message = completion[-1]
            if not isinstance(message, dict) or message.get("tool_calls"):
                continue
            content = str(message.get("content", ""))
            parsed_calls = extract_tool_calls(content)
            if not parsed_calls:
                continue
            message["tool_calls"] = [
                {
                    "id": call.get("id", f"call_{idx}"),
                    "type": "function",
                    "function": {
                        "name": call.get("function", {}).get("name", ""),
                        "arguments": call.get("function", {}).get("arguments", {}),
                    },
                }
                for idx, call in enumerate(parsed_calls)
            ]
            attached += len(message["tool_calls"])

        return attached

    def _gemma_tool_parse_stats(self, completions) -> Dict[str, Any]:
        raw_seq = 0
        parsed_seq = 0
        parsed_calls = 0
        names = Counter()
        raw_names = Counter()
        for completion in completions or []:
            if not isinstance(completion, list) or not completion:
                continue
            message = completion[-1]
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", ""))
            raw_matches = re.findall(r"call:([A-Za-z_][A-Za-z0-9_]*)\{", content)
            if raw_matches:
                raw_seq += 1
                raw_names.update(raw_matches)
            parsed = extract_tool_calls(content)
            if parsed:
                parsed_seq += 1
                parsed_calls += len(parsed)
                names.update(call.get("function", {}).get("name", "unknown") for call in parsed)
        return {
            "raw_seq": raw_seq,
            "parsed_seq": parsed_seq,
            "parsed_calls": parsed_calls,
            "raw_names": raw_names,
            "names": names,
        }

    def _tool_dict_index(self, completion_index: int) -> int:
        """Map a generated completion back to an available tool dictionary.

        TRL builds tool dictionaries for its configured generation batch size,
        but DAPO oversampling can ask vLLM for a much larger temporary rollout
        batch. Plain callable tools are identical across rows, so wrapping the
        index keeps tool execution valid for oversampled completions.
        """

        tool_dict_count = len(self._sync_tool_dicts)
        if tool_dict_count == 0:
            return 0
        return completion_index % tool_dict_count

    @staticmethod
    def _format_tool_result(result):
        """Return chat-template-safe tool content plus any image payloads."""

        images = []
        if isinstance(result, list) and all(isinstance(part, dict) and "type" in part for part in result):
            for part in result:
                if part.get("type") == "image":
                    images.append(part.get("image"))
            return result, images
        if isinstance(result, str):
            return result, images
        try:
            return json.dumps(result, ensure_ascii=False, default=str), images
        except TypeError:
            return str(result), images

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs, images, multimodal_fields):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt.
        mode = "train" if self.model.training else "eval"
        parse_stats = self._gemma_tool_parse_stats(completions)
        attached_initial = self._attach_gemma_tool_calls(completions)
        if getattr(self.accelerator, "is_main_process", True):
            step = int(getattr(self.state, "global_step", 0))
            raw_names = ",".join(f"{k}:{v}" for k, v in parse_stats["raw_names"].most_common(8)) or "none"
            names = ",".join(f"{k}:{v}" for k, v in parse_stats["names"].most_common(8)) or "none"
            print(
                f"[tool-parse] mode={mode} step={step} rollouts={len(completions)} "
                f"raw_tool_seq={parse_stats['raw_seq']} parsed_tool_seq={parse_stats['parsed_seq']} "
                f"attached_gemma_tool_calls={attached_initial} raw_names={raw_names} parsed_names={names}",
                flush=True,
            )

        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_images = [[] for _ in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        global_active_tool_sequences = self._global_sum_int(len(idxs_with_tool))

        while global_active_tool_sequences > 0 and iteration_num < self.max_tool_calling_iterations:
            if self.debug_tool_loop and getattr(self.accelerator, "is_main_process", True):
                names = Counter(
                    call.get("function", {}).get("name", "unknown")
                    for call_list in tool_calls
                    for call in (call_list or [])
                )
                name_summary = ",".join(f"{k}:{v}" for k, v in names.most_common(8)) or "none"
                print(
                    f"[tool-loop] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                    f"iter={iteration_num} seqs={len(idxs_with_tool)} calls={sum(len(x) for x in tool_calls)} "
                    f"global_seqs={global_active_tool_sequences} names={name_summary}",
                    flush=True,
                )
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]
            completions_len_before = [len(completions[i]) for i in idxs_with_tool]
            tool_images_len_before = [len(tool_images[i]) for i in idxs_with_tool]
            prompts_len_before = [len(prompts[i]) for i in idxs_with_tool]

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                tool_dict_index = self._tool_dict_index(idx_with_tool)
                sync_tool_dict = self._sync_tool_dicts[tool_dict_index]
                async_tool_dict = self._async_tool_dicts[tool_dict_index]
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []

                for tool_call in tool_call_list:
                    tool_call_count += 1
                    tool_call_id = tool_call.get("id")
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in sync_tool_dict:
                                tool_call_results.append(
                                    (tool_call_id, name, sync_tool_dict[name](**function["arguments"]))
                                )
                            elif name in async_tool_dict:
                                async_coros.append(
                                    (tool_call_id, name, async_tool_dict[name](**function["arguments"]))
                                )
                            else:
                                raise ValueError(f"Tool {name} not found.")
                        except Exception as exc:
                            tool_failure_count += 1
                            tool_call_results.append((tool_call_id, name, {"error": str(exc)}))
                    else:
                        tool_failure_count += 1
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append(
                            (tool_call_id, name, {"error": f"Unsupported tool call type: {tool_call['type']}"})
                        )

                if async_coros:

                    async def _run_async_tools(coros_with_names):
                        coros = [coro for _, _, coro in coros_with_names]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [
                            (tool_call_id, name, result)
                            for (tool_call_id, name, _), result in zip(coros_with_names, results, strict=False)
                        ]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for tool_call_id, name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((tool_call_id, name, {"error": str(result)}))
                        else:
                            tool_call_results.append((tool_call_id, name, result))

                for tool_call_id, name, result in tool_call_results:
                    content, images_from_tool = self._format_tool_result(result)
                    tool_message = {"role": "tool", "name": name, "content": content}
                    if tool_call_id is not None:
                        tool_message["tool_call_id"] = tool_call_id
                    for image in images_from_tool:
                        if image is not None:
                            tool_images[idx_with_tool].append(image)
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            if self.debug_tool_loop and getattr(self.accelerator, "is_main_process", True):
                print(
                    f"[tool-loop] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                    f"iter={iteration_num} executed_calls={tool_call_count} failures={tool_failure_count}",
                    flush=True,
                )

            prompt_completion_tool_ids = []
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_messages = []
                for message in reversed(completions[idx_with_tool]):
                    if message["role"] == "tool":
                        tool_messages.insert(0, message)
                    else:
                        break
                suffix_ids = self._get_tool_suffix_ids(tool_messages)
                prompt_completion_tool_ids.append(
                    prompt_ids[idx_with_tool] + completion_ids[idx_with_tool] + suffix_ids
                )

            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.vllm_generation.llm.llm_engine.model_config.max_model_len
            else:
                config = self.model.config.text_config if self._is_vlm else self.model.config
                env_max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN") or 0)
                batch_max_prompt_len = max((len(ids) for ids in prompt_ids), default=0)
                max_model_len = (
                    getattr(config, "max_position_embeddings", None)
                    or getattr(self.vllm_generation, "max_model_length", None)
                    or env_max_model_len
                    or (batch_max_prompt_len + self.max_completion_length)
                )
            overlong = [
                len(pct) - len(prompt_ids[i]) > self.max_completion_length or len(pct) >= max_model_len
                for i, pct in zip(idxs_with_tool, prompt_completion_tool_ids, strict=True)
            ]
            if self.debug_tool_loop and getattr(self.accelerator, "is_main_process", True):
                kept = len(overlong) - sum(1 for value in overlong if value)
                max_len = max((len(pct) for pct in prompt_completion_tool_ids), default=0)
                print(
                    f"[tool-loop] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                    f"iter={iteration_num} post_tool_prompt_seqs={len(prompt_completion_tool_ids)} "
                    f"kept={kept} overlong={len(overlong) - kept} max_prompt_completion_tool_len={max_len}",
                    flush=True,
                )
            for idx in range(len(idxs_with_tool)):
                if overlong[idx]:
                    idx_with_tool = idxs_with_tool[idx]
                    del completions[idx_with_tool][completions_len_before[idx] :]
                    del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                    del prompts[idx_with_tool][prompts_len_before[idx] :]

            idxs_with_tool = [idx for idx, over in zip(idxs_with_tool, overlong, strict=True) if not over]
            prompt_completion_tool_ids = [
                pct for pct, over in zip(prompt_completion_tool_ids, overlong, strict=True) if not over
            ]
            global_active_tool_sequences = self._global_sum_int(len(idxs_with_tool))
            if global_active_tool_sequences == 0:
                break

            if idxs_with_tool:
                merged_images = images
                if any(imgs for imgs in tool_images):
                    if merged_images is None:
                        merged_images = [imgs if imgs else None for imgs in tool_images]
                    else:
                        merged_images = [
                            (existing or []) + new for existing, new in zip(merged_images, tool_images, strict=True)
                        ]
                loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None
                if multimodal_fields:
                    loop_multimodal_fields = {}
                    for key, value in multimodal_fields.items():
                        selected = [value[i] for i in idxs_with_tool]
                        if selected and isinstance(selected[0], list):
                            selected = [
                                item + [0] * (len(pct) - len(item))
                                for item, pct in zip(selected, prompt_completion_tool_ids, strict=True)
                            ]
                        loop_multimodal_fields[key] = selected
                else:
                    loop_multimodal_fields = {}
            else:
                loop_images = None
                loop_multimodal_fields = {}

            if self.debug_tool_loop and getattr(self.accelerator, "is_main_process", True):
                print(
                    f"[tool-loop] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                    f"iter={iteration_num} post_tool_generate_start seqs={len(prompt_completion_tool_ids)} "
                    f"global_seqs={global_active_tool_sequences}",
                    flush=True,
                )
            post_tool_ids, post_tool_logprobs = self._generate_tool_continuations(
                prompt_completion_tool_ids, loop_images, loop_multimodal_fields
            )
            if self.debug_tool_loop and getattr(self.accelerator, "is_main_process", True):
                print(
                    f"[tool-loop] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                    f"iter={iteration_num} post_tool_generate_done "
                    f"mean_new_tokens={sum(len(ids) for ids in post_tool_ids) / max(len(post_tool_ids), 1):.1f}",
                    flush=True,
                )

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(prompt_ids[idx_with_tool])
                excess_length = completion_tool_length + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
                    new_len = len(post_tool_ids[idx]) - excess_length
                    post_tool_ids[idx] = post_tool_ids[idx][:new_len]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:new_len]

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            post_tool_completions = [parse_response(self._tokenizer, ids) if ids else {} for ids in post_tool_ids]
            self._attach_gemma_tool_calls([[completion] for completion in post_tool_completions if completion])

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:
                    completions[idx_with_tool].append(post_tool_completions[idx])

            tool_calls = [completion.get("tool_calls") for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            if self.debug_tool_loop and getattr(self.accelerator, "is_main_process", True):
                print(
                    f"[tool-loop] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                    f"iter={iteration_num} next_tool_seqs={len(idxs_with_tool)}",
                    flush=True,
                )
            global_active_tool_sequences = self._global_sum_int(len(idxs_with_tool))
            iteration_num += 1

        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count, tool_images

    def _log_rollout_debug(
        self,
        *,
        mode: str,
        inputs,
        completions,
        completion_ids_list,
        tool_mask_list,
        rewards_per_func: torch.Tensor,
        num_generations: int,
    ) -> None:
        if not self.debug_rollouts or mode != "train":
            return
        step = int(getattr(self.state, "global_step", 0))
        if step % self.debug_rollout_every != 0:
            return

        eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
        lengths = [len(ids) for ids in completion_ids_list]
        truncated = [
            _is_truncated_completion(ids, eos_and_pad, self.max_completion_length)
            for ids in completion_ids_list
        ]
        flags = [self._completion_debug_flags(completion) for completion in completions]

        tool_mask_sequences = 0
        tool_mask_zero_tokens = 0
        tool_mask_total_tokens = 0
        if tool_mask_list is not None:
            for mask in tool_mask_list:
                zeros = sum(1 for x in mask if int(x) == 0)
                tool_mask_zero_tokens += zeros
                tool_mask_total_tokens += len(mask)
                if zeros > 0:
                    tool_mask_sequences += 1

        call_counts: Dict[str, int] = {}
        for flag in flags:
            for name in flag["call_names"]:
                call_counts[name] = call_counts.get(name, 0) + 1

        local_counts = torch.tensor(
            [
                len(completions),
                sum(1 for flag in flags if flag["has_tool_call"]),
                sum(1 for flag in flags if flag["has_tool_response"]),
                sum(1 for flag in flags if flag["has_final_answer_tag"]),
                sum(1 for flag in flags if flag["has_final_sql"]),
                sum(1 for flag in flags if flag["has_extracted_sql"]),
                sum(1 for x in truncated if x),
                tool_mask_sequences,
                tool_mask_zero_tokens,
                tool_mask_total_tokens,
            ],
            device=self.accelerator.device,
            dtype=torch.long,
        )
        gathered_counts = self.accelerator.gather(local_counts.unsqueeze(0))
        totals = gathered_counts.sum(dim=0).tolist()

        gathered_lengths = _flatten_gathered(gather_object(lengths))
        gathered_call_counts = _flatten_gathered(gather_object([call_counts]))
        merged_call_counts: Dict[str, int] = {}
        for item in gathered_call_counts:
            if not isinstance(item, dict):
                continue
            for name, count in item.items():
                merged_call_counts[name] = merged_call_counts.get(name, 0) + int(count)

        gathered_samples = []
        if self.debug_rollout_samples > 0:
            local_samples = []
            for i, (flag, length, is_truncated) in enumerate(zip(flags, lengths, truncated)):
                if len(local_samples) >= self.debug_rollout_samples:
                    break
                if flag["has_final_sql"] and flag["has_tool_response"]:
                    continue
                db_id = ""
                try:
                    db_id = str(inputs[i].get("db_id", ""))
                except Exception:
                    pass
                local_samples.append(
                    {
                        "rank": int(self.accelerator.process_index),
                        "idx": i,
                        "db_id": db_id,
                        "len": length,
                        "truncated": bool(is_truncated),
                        "tool_call": flag["has_tool_call"],
                        "tool_response": flag["has_tool_response"],
                        "final_sql": flag["has_final_sql"],
                        "extractable_sql": flag["has_extracted_sql"],
                        "calls": flag["call_names"],
                        "sql": flag["sql"][:240],
                        "text": flag["text"][: self.debug_rollout_sample_chars].replace("\n", "\\n"),
                    }
                )
            gathered_samples = _flatten_gathered(gather_object([local_samples]))

        if not getattr(self.accelerator, "is_main_process", True):
            return

        n = max(int(totals[0]), 1)
        len_mean = sum(gathered_lengths) / max(len(gathered_lengths), 1)
        mask_frac = float(totals[8]) / float(totals[9]) if totals[9] else 0.0
        call_suffix = (
            " tool_calls_by_name="
            + ",".join(f"{name}:{count}" for name, count in sorted(merged_call_counts.items()))
            if merged_call_counts
            else " tool_calls_by_name=none"
        )
        print(
            f"[rollout-debug] step={step} candidates={totals[0]} groups={totals[0] // max(num_generations, 1)} "
            f"num_generations={num_generations} "
            f"tool_call={totals[1]}({totals[1] / n:.1%}) "
            f"tool_response={totals[2]}({totals[2] / n:.1%}) "
            f"final_tag={totals[3]}({totals[3] / n:.1%}) "
            f"final_sql={totals[4]}({totals[4] / n:.1%}) "
            f"extractable_sql={totals[5]}({totals[5] / n:.1%}) "
            f"truncated={totals[6]}({totals[6] / n:.1%}) "
            f"tool_masked_seq={totals[7]}({totals[7] / n:.1%}) "
            f"tool_masked_token_frac={mask_frac:.1%}{call_suffix}"
        )
        print(
            f"[rollout-debug] step={step} completion_tokens "
            f"mean={len_mean:.1f} p50={_quantile(gathered_lengths, 0.50):.0f} "
            f"p90={_quantile(gathered_lengths, 0.90):.0f} "
            f"p99={_quantile(gathered_lengths, 0.99):.0f} "
            f"max={max(gathered_lengths) if gathered_lengths else 0}"
        )

        if rewards_per_func is not None and rewards_per_func.numel() > 0:
            reward_bits = []
            for i, name in enumerate(self.reward_func_names):
                vals = rewards_per_func[:, i].detach().float()
                finite = vals[~torch.isnan(vals)]
                if finite.numel() == 0:
                    continue
                reward_bits.append(
                    f"{name}:mean={finite.mean().item():.3g},"
                    f"nz={(finite != 0).float().mean().item():.1%},"
                    f"min={finite.min().item():.3g},max={finite.max().item():.3g}"
                )
            if reward_bits:
                print(f"[rollout-debug] step={step} rewards " + " | ".join(reward_bits))

            if self._dyn_reward_idx is not None:
                vals = rewards_per_func[:, self._dyn_reward_idx].detach().float()
                try:
                    grouped = vals.view(-1, num_generations)
                    successes = grouped.sum(dim=1)
                    buckets = [(successes == i).sum().item() for i in range(num_generations + 1)]
                    compact = ",".join(f"{i}:{int(c)}" for i, c in enumerate(buckets) if c)
                    print(
                        f"[rollout-debug] step={step} group_successes reward={self.dynamic_sampling_reward_name} "
                        f"successes_per_{num_generations}={compact or 'none'}"
                    )
                except Exception:
                    pass

        printed = 0
        for sample in gathered_samples:
            if printed >= self.debug_rollout_samples:
                break
            if not isinstance(sample, dict):
                continue
            print(
                "[rollout-sample] "
                f"step={step} rank={sample['rank']} idx={sample['idx']} db_id={sample['db_id']} "
                f"len={sample['len']} trunc={sample['truncated']} "
                f"tool_call={sample['tool_call']} tool_response={sample['tool_response']} "
                f"final_sql={sample['final_sql']} extractable_sql={sample['extractable_sql']} "
                f"calls={sample['calls']} sql={sample['sql']!r} text={sample['text']!r}"
            )
            printed += 1

    def _log_attention_mask_debug(self, output: Dict[str, Any], attention_mask: torch.Tensor) -> None:
        """Validate whether tool observations are visible to logprob forwards."""

        if not self.debug_attention_mask:
            return
        step = int(getattr(self.state, "global_step", 0))
        if step % self.debug_attention_mask_every != 0:
            return

        completion_mask = output.get("completion_mask")
        tool_mask = output.get("tool_mask")
        completion_ids = output.get("completion_ids")
        prompt_ids = output.get("prompt_ids")
        if not (
            isinstance(completion_mask, torch.Tensor)
            and isinstance(tool_mask, torch.Tensor)
            and isinstance(completion_ids, torch.Tensor)
            and isinstance(prompt_ids, torch.Tensor)
        ):
            return

        prompt_width = prompt_ids.size(1)
        completion_attention = attention_mask[:, prompt_width:]
        tool_positions = tool_mask == 0
        effective_loss_mask = completion_mask * tool_mask
        loss_masked_positions = effective_loss_mask == 0
        attention_zero_positions = completion_attention == 0
        tool_token_count = int(tool_positions.long().sum().item())
        tool_tokens_hidden = int((tool_positions & attention_zero_positions).long().sum().item())
        tool_tokens_visible = int((tool_positions & ~attention_zero_positions).long().sum().item())
        loss_masked_tokens = int(loss_masked_positions.long().sum().item())
        attention_zero_tokens = int(attention_zero_positions.long().sum().item())
        rows_with_tool = int(tool_positions.any(dim=1).long().sum().item())

        local_counts = torch.tensor(
            [
                completion_ids.size(0),
                rows_with_tool,
                tool_token_count,
                tool_tokens_hidden,
                tool_tokens_visible,
                loss_masked_tokens,
                attention_zero_tokens,
            ],
            device=completion_ids.device,
            dtype=torch.long,
        )
        try:
            totals = self.accelerator.gather(local_counts.unsqueeze(0)).sum(dim=0).tolist()
        except Exception:
            totals = local_counts.tolist()

        if getattr(self.accelerator, "is_main_process", True):
            hidden_frac = float(totals[3]) / float(totals[2]) if totals[2] else 0.0
            print(
                f"[attention-mask-debug] step={step} rows={totals[0]} "
                f"rows_with_tool_response={totals[1]} tool_response_tokens={totals[2]} "
                f"tool_response_hidden_by_attention={totals[3]}({hidden_frac:.1%}) "
                f"tool_response_visible_to_attention={totals[4]} "
                f"loss_masked_completion_tokens={totals[5]} "
                f"attention_zero_completion_tokens={totals[6]}",
                flush=True,
            )

        if self.debug_attention_mask_samples <= 0:
            return

        samples = []
        for row in range(completion_ids.size(0)):
            if len(samples) >= self.debug_attention_mask_samples:
                break
            zero_idxs = torch.nonzero(tool_positions[row], as_tuple=False).flatten()
            if zero_idxs.numel() == 0:
                continue
            start = int(zero_idxs[0].item())
            end = start
            zero_set = set(int(x.item()) for x in zero_idxs[:256])
            while end + 1 in zero_set:
                end += 1
            lo = max(0, start - 12)
            hi = min(completion_ids.size(1), end + 13)
            ids = completion_ids[row, lo:hi].detach().cpu().tolist()
            try:
                text = self.processing_class.decode(ids, skip_special_tokens=False)
            except Exception:
                text = str(ids)
            samples.append(
                {
                    "rank": int(self.accelerator.process_index),
                    "row": row,
                    "span": [start, end],
                    "span_len": end - start + 1,
                    "attention_sum_on_span": int(completion_attention[row, start : end + 1].long().sum().item()),
                    "loss_sum_on_span": int(effective_loss_mask[row, start : end + 1].long().sum().item()),
                    "window": text[:500].replace("\n", "\\n"),
                }
            )

        try:
            gathered_samples = _flatten_gathered(gather_object([samples]))
        except Exception:
            gathered_samples = samples

        if not getattr(self.accelerator, "is_main_process", True):
            return
        printed = 0
        for sample in gathered_samples:
            if printed >= self.debug_attention_mask_samples:
                break
            if not isinstance(sample, dict):
                continue
            print(
                "[attention-mask-sample] "
                f"step={step} rank={sample['rank']} row={sample['row']} "
                f"tool_span={sample['span']} span_len={sample['span_len']} "
                f"attention_sum_on_span={sample['attention_sum_on_span']} "
                f"loss_sum_on_span={sample['loss_sum_on_span']} "
                f"window={sample['window']!r}",
                flush=True,
            )
            printed += 1

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

        if getattr(self.accelerator, "is_main_process", True):
            print(
                f"[rollout-stage] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                f"generate_start prompts={len(prompts)}",
                flush=True,
            )
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
        if getattr(self.accelerator, "is_main_process", True):
            lengths = [len(ids) for ids in completion_ids_list]
            mean_len = sum(lengths) / max(len(lengths), 1)
            print(
                f"[rollout-stage] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                f"generate_done completions={len(completion_ids_list)} mean_completion_tokens={mean_len:.1f} "
                f"max_completion_tokens={max(lengths, default=0)}",
                flush=True,
            )

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

        # Keep completion_mask as the attention/padding mask. Tool responses
        # must remain visible to later assistant tokens, but they are excluded
        # from the policy loss via completion_mask * tool_mask in TRL's loss.

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

        if getattr(self.accelerator, "is_main_process", True):
            print(
                f"[rollout-stage] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                f"reward_start completions={len(completions)}",
                flush=True,
            )
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if getattr(self.accelerator, "is_main_process", True):
            print(
                f"[rollout-stage] mode={mode} step={int(getattr(self.state, 'global_step', 0))} reward_done",
                flush=True,
            )
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        self._log_rollout_debug(
            mode=mode,
            inputs=inputs,
            completions=completions,
            completion_ids_list=completion_ids_list,
            tool_mask_list=tool_mask_list,
            rewards_per_func=rewards_per_func,
            num_generations=num_generations,
        )

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
        self._log_attention_mask_debug(output, attention_mask)
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

            current_beta = self._current_beta()
            if current_beta != 0.0:
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
                    peft_config = getattr(model, "peft_config", None)
                    if peft_config is None:
                        raise RuntimeError(
                            "Current beta is nonzero, but no ref_model was initialized and the model "
                            "does not expose PEFT adapters. If using --beta_schedule with a static "
                            "--beta of 0, initialize GRPOConfig with a positive beta so TRL builds the "
                            "reference model."
                        )
                    with use_adapter(model, adapter_name="ref" if "ref" in peft_config else None):
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

    @staticmethod
    def _reward_only_batch_to_examples(batch: Any) -> List[Dict[str, Any]]:
        if isinstance(batch, list):
            return list(batch)
        if isinstance(batch, tuple):
            return list(batch)
        if not isinstance(batch, dict):
            return [batch]

        n = None
        for value in batch.values():
            if isinstance(value, (list, tuple)):
                n = len(value)
                break
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                n = int(value.size(0))
                break
        if n is None:
            return [batch]

        examples: List[Dict[str, Any]] = []
        for i in range(n):
            example: Dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, (list, tuple)):
                    example[key] = value[i]
                elif isinstance(value, torch.Tensor) and value.dim() > 0:
                    example[key] = value[i]
                else:
                    example[key] = value
            examples.append(example)
        return examples

    def _collect_reward_only_eval_inputs(self, dataloader) -> List[Dict[str, Any]]:
        inputs: List[Dict[str, Any]] = []
        for batch in dataloader:
            inputs.extend(self._reward_only_batch_to_examples(batch))
        return inputs

    def _mean_metric_values(self, mode: str, prefix: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key, values in self._metrics.get(mode, {}).items():
            numeric_values = [
                float(v)
                for v in values
                if isinstance(v, (int, float)) and v == v
            ]
            if numeric_values:
                metrics[f"{prefix}_{key}"] = sum(numeric_values) / len(numeric_values)
        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if not self.reward_only_eval:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(dataset, dict):
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        dataloader = self.get_eval_dataloader(eval_dataset)
        inputs = self._collect_reward_only_eval_inputs(dataloader)
        start_time = time.time()
        was_training = self.model.training
        self.model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        num_generations = int(getattr(self, "num_generations_eval", self.num_generations))
        if getattr(self.accelerator, "is_main_process", True):
            print(
                f"[reward-only-eval] step={int(getattr(self.state, 'global_step', 0))} "
                f"prompt_groups={len(inputs) / max(num_generations, 1):.0f} "
                f"generations={len(inputs)}",
                flush=True,
            )

        with torch.no_grad():
            if inputs:
                self._generate_and_score_candidates_no_policy_logps(inputs)

        runtime = time.time() - start_time
        reward_metrics = self._mean_metric_values("eval", metric_key_prefix)
        samples_per_second = len(inputs) / runtime if runtime > 0 else 0.0
        base_metrics = {
            f"{metric_key_prefix}_runtime": runtime,
            f"{metric_key_prefix}_samples_per_second": samples_per_second,
            f"{metric_key_prefix}_num_generations": float(len(inputs)),
            f"{metric_key_prefix}_num_prompt_groups": float(len(inputs) / max(num_generations, 1)),
            f"{metric_key_prefix}_reward_only": 1.0,
        }
        metrics = {**base_metrics, **reward_metrics}
        self.log(base_metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        if was_training:
            self.model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
        return metrics

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
        original_beta = self.beta
        current_beta = self._current_beta()
        self.beta = current_beta
        try:
            self._metrics["train"]["beta"].append(current_beta)
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            self.beta = original_beta

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
            effective_mask = self._effective_completion_mask(out)
            local_count = (effective_mask > 0).long().sum().to(device)
            agg = self.accelerator.gather(local_count.unsqueeze(0)).sum()
            skip_policy_loss = bool(agg.item() == 0)
            out["num_items_in_batch"] = agg.clamp(min=1)
        except Exception:
            effective_mask = self._effective_completion_mask(out)
            local_count = (effective_mask > 0).long().sum().to(device)
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
            mode = "train" if self.model.training else "eval"
            if getattr(self.accelerator, "is_main_process", True):
                print(
                    f"[rollout-stage] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                    f"generate_score_start prompts={len(inputs)}",
                    flush=True,
                )
            out = super()._generate_and_score_completions(inputs)
            if getattr(self.accelerator, "is_main_process", True):
                print(
                    f"[rollout-stage] mode={mode} step={int(getattr(self.state, 'global_step', 0))} "
                    "generate_score_done",
                    flush=True,
                )
            return out

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
            effective_mask = self._effective_completion_mask(out)
            local_count = (effective_mask > 0).long().sum().to(device)
            agg = self.accelerator.gather(local_count.unsqueeze(0)).sum()
            skip_policy_loss = bool(agg.item() == 0)
            out["num_items_in_batch"] = agg.clamp(min=1)
        except Exception:
            effective_mask = self._effective_completion_mask(out)
            local_count = (effective_mask > 0).long().sum().to(device)
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
