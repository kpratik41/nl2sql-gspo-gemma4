"""Watermarked and unwatermarked generation from a Hugging Face causal LM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from synthmark.config import to_hf_config
from synthmark.keys import WatermarkKey

DEFAULT_MODEL = "google/gemma-4-E4B-it"


@dataclass
class GenerationOutput:
    """Text plus the bookkeeping the experiments need."""

    texts: list[str]
    token_ids: list[list[int]]
    """Completion token ids exactly as sampled, before any decode/re-encode round trip."""
    num_new_tokens: list[int]
    wall_time_s: float
    watermarked: bool
    key_fingerprint: str | None = None
    sampling: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return sum(self.num_new_tokens) / self.wall_time_s if self.wall_time_s else 0.0


class WatermarkedLM:
    """A model wrapper whose only added surface is ``key=...`` on ``generate``.

    The design goal is that turning the watermark on is a one-argument change to
    an existing call site, so that an A/B comparison is genuinely
    apples-to-apples: same weights, same prompts, same sampling parameters, same
    seed, with the watermark as the only difference.

    SynthID needs a sampling distribution to work with.  Under greedy decoding
    there is no residual randomness to steer, so no watermark can be embedded;
    :meth:`generate` refuses ``do_sample=False`` with the watermark on rather
    than silently producing unmarked text.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device_map: str | dict = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str | None = None,
    ):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"dtype": dtype, "device_map": device_map}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self.model.device

    # ---------------------------------------------------------------- prompts

    def chat_prompt(self, user_message: str, system: str | None = None) -> str:
        """Render one user turn through the model's chat template."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        return self.tokenizer.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=False
        )[0]

    def chat_prompts(self, user_messages: Sequence[str], system: str | None = None) -> list[str]:
        return [self.chat_prompt(m, system) for m in user_messages]

    # ------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str] | str,
        *,
        key: WatermarkKey | None = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 64,
        top_p: float = 0.95,
        do_sample: bool = True,
        batch_size: int = 8,
        seed: int | None = None,
    ) -> GenerationOutput:
        """Generate completions, optionally watermarked with ``key``.

        Args:
            key: When given, the SynthID watermark is applied.  When ``None``,
                generation is byte-for-byte the ordinary HF path.
            seed: Seeds the sampler before *each* batch, so that a watermarked
                and an unwatermarked run over the same prompts start from the
                same random state.  They will still diverge -- the watermark
                consumes randomness differently -- but the comparison is not
                confounded by one run getting a luckier seed.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if key is not None and not do_sample:
            raise ValueError(
                "SynthID watermarking requires sampling: it steers the choice among "
                "plausible next tokens, and greedy decoding leaves no choice to steer. "
                "Pass do_sample=True, or generate without a key."
            )

        wm_config = to_hf_config(key) if key is not None else None
        texts: list[str] = []
        token_ids: list[list[int]] = []
        counts: list[int] = []

        # Time only the forward passes; tokenisation and decoding are excluded so
        # that the watermark overhead measurement is not diluted.
        elapsed = 0.0
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start : start + batch_size])
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
            prompt_len = enc["input_ids"].shape[1]
            if seed is not None:
                torch.manual_seed(seed)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                watermarking_config=wm_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed += time.perf_counter() - t0

            for row in out:
                comp = row[prompt_len:]
                # Strip trailing pad; keep everything else exactly as sampled.
                keep = comp.tolist()
                while keep and keep[-1] == self.tokenizer.pad_token_id:
                    keep.pop()
                token_ids.append(keep)
                counts.append(len(keep))
                texts.append(self.tokenizer.decode(keep, skip_special_tokens=True))

        return GenerationOutput(
            texts=texts,
            token_ids=token_ids,
            num_new_tokens=counts,
            wall_time_s=elapsed,
            watermarked=key is not None,
            key_fingerprint=key.fingerprint if key is not None else None,
            sampling={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
            },
        )

    # ------------------------------------------------------------- likelihood

    @torch.no_grad()
    def perplexity(self, texts: Sequence[str], *, batch_size: int = 8, max_len: int = 1024) -> list[float]:
        """Token-level perplexity of ``texts`` under this model.

        Two details that are easy to get wrong and silently corrupt the numbers:

        * A BOS token is prepended.  Gemma's tokenizer does not add one here, and
          scoring without it leaves the first tokens conditioned on nothing,
          which inflates perplexity by orders of magnitude.
        * Padding is applied on the **right**.  The tokenizer is configured for
          left padding because that is what batched *generation* requires; left
          padding during scoring misaligns positions and produces garbage.

        Note on interpretation: perplexity measured under the *same* model that
        produced the text is biased **against** watermarked text by
        construction.  SynthID selects among near-equally-likely tokens using the
        g-function rather than raw probability, so watermarked text sits slightly
        off the model's own argmax path and scores slightly higher perplexity
        whether or not a reader would notice any difference.  For a fair fluency
        comparison, score with an independent model -- see
        ``experiments/03_quality.py``.
        """
        bos = self.tokenizer.bos_token_id
        pad = self.tokenizer.pad_token_id or 0
        out: list[float] = []

        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            seqs = []
            for t in chunk:
                body = self.tokenizer(t, add_special_tokens=False)["input_ids"][: max_len - 1]
                seqs.append(([bos] if bos is not None else []) + body)

            width = max(len(s_) for s_ in seqs)
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            attn = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, s_ in enumerate(seqs):
                ids[i, : len(s_)] = torch.tensor(s_, dtype=torch.long)
                attn[i, : len(s_)] = 1
            ids, attn = ids.to(self.device), attn.to(self.device)

            logits = self.model(input_ids=ids, attention_mask=attn).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            m = attn[:, 1:].float()
            nll = -(tok_lp * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            out.extend(torch.exp(nll).cpu().tolist())
        return out
