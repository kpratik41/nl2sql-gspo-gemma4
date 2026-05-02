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
4. Each rank independently draws ``need_max`` fresh prompts from a
   per-rank shuffled backup pool over ``train_dataset`` and replicates
   each prompt ``num_generations`` times (mimicking ``RepeatRandomSampler``).
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

import random
from typing import Any, Dict, List, Optional

import torch
from trl import GRPOTrainer


def _pad_to_width(tensor: torch.Tensor, target_width: int, pad_value, side: str) -> torch.Tensor:
    if tensor.dim() != 2 or tensor.size(1) >= target_width:
        return tensor
    extra = target_width - tensor.size(1)
    pad = tensor.new_full((tensor.size(0), extra), pad_value)
    if side == "left":
        return torch.cat([pad, tensor], dim=1)
    return torch.cat([tensor, pad], dim=1)


# Pad value/side for known 2D output keys. ``prompt_ids`` and
# ``completion_ids`` use ``self.pad_token_id`` so they're handled inline.
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

    def __init__(
        self,
        *args,
        enable_dynamic_sampling: bool = True,
        dynamic_sampling_min_std: float = 1e-6,
        dapo_max_rounds: int = 6,
        dynamic_sampling_reward_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_dynamic_sampling = enable_dynamic_sampling
        self.dynamic_sampling_min_std = float(dynamic_sampling_min_std)
        self.dapo_max_rounds = max(1, int(dapo_max_rounds))
        self.dynamic_sampling_reward_name = (
            dynamic_sampling_reward_name if dynamic_sampling_reward_name else None
        )
        self._dyn_pool_indices: List[int] = []
        self._dyn_pool_cursor: int = 0
        self._dyn_pool_seed_step: int = -1
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
            print(
                f"[dapo] enabled={self.enable_dynamic_sampling} | mode=oversample-and-replace | "
                f"max_rounds={self.dapo_max_rounds} | min_std={self.dynamic_sampling_min_std} | "
                f"criterion={criterion} | num_generations={self.num_generations}"
            )

    # ------------------------------------------------------------------ #
    # Backup pool of replacement prompts (per-rank shuffle)
    # ------------------------------------------------------------------ #
    def _refresh_dyn_pool(self) -> None:
        if self.train_dataset is None:
            self._dyn_pool_indices = []
            return
        n = len(self.train_dataset)
        step = int(getattr(self.state, "global_step", 0))
        # Per-rank shuffle: each rank draws different replacements within
        # the same step so vLLM doesn't redundantly re-generate the same
        # prompts on multiple ranks.
        seed = step * 1000 + int(self.accelerator.process_index) + 17
        rng = random.Random(seed)
        idxs = list(range(n))
        rng.shuffle(idxs)
        self._dyn_pool_indices = idxs
        self._dyn_pool_cursor = 0
        self._dyn_pool_seed_step = step

    def _draw_replacement_inputs(self, k: int) -> List[Dict[str, Any]]:
        if k <= 0 or self.train_dataset is None:
            return []
        step = int(getattr(self.state, "global_step", 0))
        if (
            not self._dyn_pool_indices
            or self._dyn_pool_seed_step != step
            or self._dyn_pool_cursor + k > len(self._dyn_pool_indices)
        ):
            self._refresh_dyn_pool()
        picked = self._dyn_pool_indices[self._dyn_pool_cursor : self._dyn_pool_cursor + k]
        self._dyn_pool_cursor += k
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
                if isinstance(v, torch.Tensor):
                    tensor_keys.add(k)

        final: Dict[str, Any] = {}
        for k in tensor_keys:
            vals = [c.get(k) for c in chunks if isinstance(c.get(k), torch.Tensor)]
            if not vals:
                continue
            v0 = vals[0]
            if v0.dim() == 1:
                final[k] = torch.cat(vals, dim=0)
                continue
            if v0.dim() == 2:
                if k == "prompt_ids":
                    pad_val, side = self.pad_token_id, "left"
                elif k == "completion_ids":
                    pad_val, side = self.pad_token_id, "right"
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

        device = self.accelerator.device
        kept_chunks: List[Dict[str, Any]] = []
        last_round_out: Optional[Dict[str, Any]] = None
        round_inputs = list(inputs)

        rounds_used = 0
        total_groups_attempted = 0
        total_groups_kept = 0

        for r in range(self.dapo_max_rounds):
            round_out = super()._generate_and_score_completions(round_inputs)
            last_round_out = round_out
            rounds_used += 1

            het = self._het_mask_for_round(round_out)
            local_groups_this_round = int(het.numel())
            total_groups_attempted += local_groups_this_round

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
            out["num_items_in_batch"] = agg
        except Exception:
            pass  # keep whatever super provided

        # Logging
        try:
            counters = torch.tensor(
                [rounds_used, total_groups_attempted, total_groups_kept, padded_groups],
                device=device,
                dtype=torch.long,
            )
            gathered = self.accelerator.gather(counters.unsqueeze(0))
            g_rounds_max = int(gathered[:, 0].max().item())
            g_attempted = int(gathered[:, 1].sum().item())
            g_kept = int(gathered[:, 2].sum().item())
            g_padded = int(gathered[:, 3].sum().item())
        except Exception:
            g_rounds_max = rounds_used
            g_attempted = total_groups_attempted
            g_kept = total_groups_kept
            g_padded = padded_groups

        het_rate = (g_kept / g_attempted) if g_attempted > 0 else 0.0
        self._metrics["train"]["dapo/rounds_used"].append(float(g_rounds_max))
        self._metrics["train"]["dapo/groups_attempted"].append(float(g_attempted))
        self._metrics["train"]["dapo/groups_kept"].append(float(g_kept))
        self._metrics["train"]["dapo/groups_padded"].append(float(g_padded))
        self._metrics["train"]["dapo/heterogeneity_rate"].append(het_rate)

        if getattr(self.accelerator, "is_main_process", True):
            step = int(getattr(self.state, "global_step", 0))
            print(
                f"[dapo] step={step} rounds={g_rounds_max}/{self.dapo_max_rounds} "
                f"attempted={g_attempted} kept={g_kept} padded={g_padded} "
                f"het_rate={het_rate:.2%}"
            )

        return out
