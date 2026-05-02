"""Dynamic-sampling GRPO trainer (DAPO §3.2).

Implements *true* DAPO-style dynamic sampling: after rollouts and reward
computation we identify groups whose intra-group reward standard deviation
is below ``dynamic_sampling_min_std`` (i.e. accuracy 0 or 1 in the binary
case — no learning signal) and *replace* those prompts by drawing fresh
prompts from a shuffled backup pool. We re-run rollouts for the
replacement prompts and splice them into the original tensors. We retry
up to ``dynamic_sampling_max_attempts`` times. Any groups that are still
flat after the final attempt are masked out (their ``completion_mask`` is
zeroed) so they contribute no gradient — this matches the DAPO objective
constraint ``0 < |{o_i : R_i is correct}| < G``.

Cross-process correctness
-------------------------
TRL's ``_generate_and_score_completions`` performs an internal
``accelerator.gather`` to compute global advantages. Each call therefore
requires every process to pass the same number of prompts. We honour this
by computing the per-attempt replacement count ``K`` as the *max* number
of bad groups across all processes; processes with fewer bad groups
generate extra rollouts that simply aren't spliced in. This wastes some
compute relative to a fully synchronous oversampling pass but keeps
TRL's batching invariants intact.

Notes
-----
* If ``dynamic_sampling_max_attempts == 0`` we fall back to the masking
  variant (faster, no extra rollouts; identical PG effect on flat groups).
* Combine with ``loss_type="dapo"`` so the loss is normalized by the
  valid token count, keeping per-step gradient magnitude stable as the
  effective batch size varies.
"""

from __future__ import annotations

import random
from typing import Dict, List

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


class DynamicSamplingGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with DAPO-style oversample-and-replace dynamic sampling."""

    def __init__(
        self,
        *args,
        enable_dynamic_sampling: bool = True,
        dynamic_sampling_min_std: float = 1e-6,
        dynamic_sampling_max_attempts: int = 0,
        dynamic_sampling_reward_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_dynamic_sampling = enable_dynamic_sampling
        self.dynamic_sampling_min_std = dynamic_sampling_min_std
        self.dynamic_sampling_max_attempts = int(dynamic_sampling_max_attempts)
        self.dynamic_sampling_reward_name = (
            dynamic_sampling_reward_name if dynamic_sampling_reward_name else None
        )
        self._dyn_pool_indices: List[int] = []
        self._dyn_pool_cursor: int = 0
        self._last_rewards_per_func: torch.Tensor | None = None

        # Resolve reward-name to column index on the registered reward funcs.
        self._dyn_reward_idx: int | None = None
        if self.dynamic_sampling_reward_name is not None:
            try:
                self._dyn_reward_idx = list(self.reward_func_names).index(
                    self.dynamic_sampling_reward_name
                )
            except ValueError:
                if getattr(self.accelerator, "is_main_process", True):
                    print(
                        f"[dyn-sampling] WARNING: reward name "
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
                f"oversample-and-replace (max_attempts={self.dynamic_sampling_max_attempts})"
                if self.dynamic_sampling_max_attempts > 0
                else "masking-only"
            )
            print(
                f"[dyn-sampling] enabled={self.enable_dynamic_sampling} | mode={mode} | "
                f"min_std={self.dynamic_sampling_min_std} | criterion={criterion} | "
                f"num_generations={self.num_generations}"
            )

    # ------------------------------------------------------------------ #
    # Backup pool of replacement prompts
    # ------------------------------------------------------------------ #
    def _refresh_dyn_pool(self) -> None:
        if self.train_dataset is None:
            self._dyn_pool_indices = []
            return
        n = len(self.train_dataset)
        rng = random.Random(int(getattr(self.state, "global_step", 0)) + 17)
        idxs = list(range(n))
        rng.shuffle(idxs)
        self._dyn_pool_indices = idxs
        self._dyn_pool_cursor = 0

    def _draw_replacement_inputs(self, k: int) -> List[Dict]:
        if k <= 0 or self.train_dataset is None:
            return []
        if not self._dyn_pool_indices or self._dyn_pool_cursor + k > len(self._dyn_pool_indices):
            self._refresh_dyn_pool()
        picked = self._dyn_pool_indices[self._dyn_pool_cursor : self._dyn_pool_cursor + k]
        self._dyn_pool_cursor += k
        return [dict(self.train_dataset[i]) for i in picked]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _calculate_rewards(self, *args, **kwargs):
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        # rewards_per_func is gathered (global) — keep a reference so the
        # heterogeneity check can read a single reward column if requested.
        self._last_rewards_per_func = rewards_per_func
        return rewards_per_func

    def _zero_std_group_mask(self, advantages: torch.Tensor) -> torch.Tensor:
        num_generations = self.num_generations
        if advantages.shape[0] == 0 or advantages.shape[0] % num_generations != 0:
            return torch.zeros(0, dtype=torch.bool, device=advantages.device)

        # Optional: judge heterogeneity using a single reward column (e.g.
        # result_reward) instead of the aggregated/normalized advantages.
        if (
            self._dyn_reward_idx is not None
            and self._last_rewards_per_func is not None
        ):
            rpf = self._last_rewards_per_func  # global, gathered
            local_n = advantages.shape[0]
            start = self.accelerator.process_index * local_n
            local_rewards = rpf[start : start + local_n, self._dyn_reward_idx]
            grouped = local_rewards.view(-1, num_generations).to(advantages.device)
            # Group is flat when intra-group std is ~0 (all rollouts got the
            # same reward value, e.g. all 0 or all 1 for binary rewards).
            return grouped.float().std(dim=1, unbiased=False) < self.dynamic_sampling_min_std

        adv_grouped = advantages.view(-1, num_generations)
        return adv_grouped.abs().max(dim=1).values < self.dynamic_sampling_min_std

    def _splice_groups(
        self,
        out: Dict[str, torch.Tensor],
        new_out: Dict[str, torch.Tensor],
        bad_local_groups: torch.Tensor,
        n_replace: int,
    ) -> Dict[str, torch.Tensor]:
        if n_replace <= 0:
            return out

        num_generations = self.num_generations
        bad_idx = torch.nonzero(bad_local_groups, as_tuple=False).flatten()[:n_replace]
        if bad_idx.numel() == 0:
            return out

        bad_rows = (
            bad_idx.unsqueeze(1) * num_generations
            + torch.arange(num_generations, device=bad_idx.device).unsqueeze(0)
        ).flatten()
        new_rows = torch.arange(n_replace * num_generations, device=bad_idx.device)

        spliced: Dict[str, torch.Tensor] = {}
        for key, value in out.items():
            if not isinstance(value, torch.Tensor) or key not in new_out:
                spliced[key] = value
                continue
            new_value = new_out[key]
            if not isinstance(new_value, torch.Tensor):
                spliced[key] = value
                continue

            if value.dim() == 1:
                merged = value.clone()
                merged[bad_rows.to(merged.device)] = new_value[new_rows.to(new_value.device)]
                spliced[key] = merged
                continue

            if value.dim() == 2:
                target_w = max(value.size(1), new_value.size(1))
                pad_val = 0
                side = "right"
                if key in ("prompt_ids", "prompt_mask"):
                    side = "left"
                    pad_val = self.pad_token_id if key == "prompt_ids" else 0
                elif key == "completion_ids":
                    pad_val = self.pad_token_id
                value_p = _pad_to_width(value, target_w, pad_val, side).clone()
                new_p = _pad_to_width(new_value, target_w, pad_val, side)
                value_p[bad_rows.to(value_p.device)] = new_p[new_rows.to(new_p.device)]
                spliced[key] = value_p
                continue

            spliced[key] = value
        return spliced

    def _mask_zero_std(self, out: Dict[str, torch.Tensor], log_key: str) -> None:
        advantages = out.get("advantages")
        completion_mask = out.get("completion_mask")
        if advantages is None or completion_mask is None:
            return
        flat_mask = self._zero_std_group_mask(advantages)
        if flat_mask.numel() == 0:
            return
        zero_std_fraction = float(flat_mask.float().mean().item())
        mode = "train" if self.model.training else "eval"
        self._metrics[mode][log_key].append(zero_std_fraction)

        # Per-rank counts; gather to log a global view once on rank 0.
        local_bad = int(flat_mask.sum().item())
        local_total = int(flat_mask.numel())
        try:
            counts = self.accelerator.gather(
                torch.tensor(
                    [local_bad, local_total],
                    device=self.accelerator.device,
                    dtype=torch.long,
                )
            )
            global_bad = int(counts[0::2].sum().item())
            global_total = int(counts[1::2].sum().item())
        except Exception:
            global_bad, global_total = local_bad, local_total

        if getattr(self.accelerator, "is_main_process", True) and mode == "train":
            step = int(getattr(self.state, "global_step", 0))
            print(
                f"[dyn-sampling] step={step} masked_groups={global_bad}/{global_total} "
                f"({zero_std_fraction:.2%} local)"
            )

        if flat_mask.any():
            keep = (~flat_mask).repeat_interleave(self.num_generations).to(
                completion_mask.device, dtype=completion_mask.dtype
            )
            out["completion_mask"] = completion_mask * keep.unsqueeze(1)

    # ------------------------------------------------------------------ #
    # Main override
    # ------------------------------------------------------------------ #
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)

        if not self.enable_dynamic_sampling or not self.model.training:
            return out

        if out.get("advantages") is None:
            return out

        max_attempts = self.dynamic_sampling_max_attempts
        mode = "train" if self.model.training else "eval"

        attempts_used = 0
        if max_attempts > 0:
            for _ in range(max_attempts):
                bad_local = self._zero_std_group_mask(out["advantages"])
                bad_count_local = int(bad_local.sum().item()) if bad_local.numel() else 0

                bad_count_t = torch.tensor(
                    [bad_count_local], device=self.accelerator.device, dtype=torch.long
                )
                gathered = self.accelerator.gather(bad_count_t)
                k_uniform = int(gathered.max().item())
                if k_uniform == 0:
                    break

                replacements = self._draw_replacement_inputs(k_uniform)
                if len(replacements) < k_uniform:
                    break

                new_out = super()._generate_and_score_completions(replacements)
                out = self._splice_groups(out, new_out, bad_local, bad_count_local)
                attempts_used += 1

        self._mask_zero_std(out, log_key="dynamic_sampling/zero_std_group_fraction")
        self._metrics[mode]["dynamic_sampling/resample_attempts"].append(float(attempts_used))
        return out
