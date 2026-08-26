"""Does watermarking degrade output quality or task accuracy?

Scope of the claim being tested
------------------------------
SynthID only ever acts on a *sampling* distribution: it re-ranks tokens the
model was already prepared to emit, and never introduces a token the model would
not have sampled.  Two consequences shape this experiment:

* Benchmarks scored by comparing option log-probabilities -- MMLU, HellaSwag,
  ARC and most multiple-choice suites -- are **unaffected by construction**,
  because no sampling happens.  Reporting "no MMLU delta" would be measuring
  nothing.  It is not evidence, and this study does not claim it as such.
* The claim that can be tested is about *sampled, free-form* generation.  So we
  test exactly that: a reasoning benchmark answered by sampled chain-of-thought
  (GSM8K, exact match on the final number), plus reference-free fluency and
  diversity measures on the open-ended corpus.

Everything is paired: the same prompts and the same seeds under both conditions,
so the comparison is within-prompt and the intervals are correspondingly tight.
A confidence interval that contains zero does not prove there is no effect --
it bounds how large an effect the data can still hide, and that bound is what
gets reported.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import DATA, DEMO_MASTER_SECRET, MODEL_ID, PRIMARY_KEY_ID, RESULTS, banner, save_json

from synthmark import WatermarkedLM, derive_key, paired_bootstrap_diff, two_proportion_diff_ci

ANSWER_RE = re.compile(r"(-?[\d,]*\.?\d+)")

GSM8K_INSTRUCTION = (
    "Solve the problem step by step. End your response with the final numeric "
    "answer on its own line in the form '#### <answer>'.\n\nProblem: {q}"
)


def extract_answer(text: str) -> str | None:
    """Pull the final number out of a chain-of-thought response."""
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = ANSWER_RE.search(tail.replace(",", ""))
        if m:
            return m.group(1)
    nums = ANSWER_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def numbers_equal(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return False


def distinct_n(text: str, n: int) -> float:
    """Fraction of distinct n-grams: a simple, reference-free diversity proxy."""
    toks = text.split()
    if len(toks) < n:
        return float("nan")
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that appear more than once -- degeneracy detector."""
    toks = text.split()
    if len(toks) < n:
        return float("nan")
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    counts = defaultdict(int)
    for g in grams:
        counts[g] += 1
    return sum(c for c in counts.values() if c > 1) / len(grams)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus.json"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--gsm8k-n", type=int, default=250)
    ap.add_argument("--gsm8k-max-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-gsm8k", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.8-27B",
                    help="Independent model used to score fluency. Must not be the model under test.")
    ap.add_argument("--judge-device", default="cuda:1")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--out", default=None,
                    help="Results JSON path; defaults to results/03_quality.json. Pass a\n                          distinct path when evaluating a second model.")
    args = ap.parse_args()

    banner("Quality and accuracy under watermarking")
    corpus = json.loads(open(args.corpus).read())
    records = corpus["records"]
    key = derive_key(DEMO_MASTER_SECRET, PRIMARY_KEY_ID)

    results: dict = {"meta": {"model": args.model, "key": key.public_summary()}}

    # ------------------------------------------------- 1. diversity / degeneracy
    banner("1. Diversity and degeneracy on the generated corpus (paired by prompt)")
    paired = defaultdict(dict)
    for r in records:
        paired[(r["suite"], r["prompt_index"], r["replicate"])][r["condition"]] = r

    div_results = {}
    for suite in sorted({r["suite"] for r in records}):
        rows = [v for k, v in paired.items() if k[0] == suite and len(v) == 2]
        if not rows:
            continue
        stats = {}
        for label, fn in (
            ("distinct_2", lambda t: distinct_n(t, 2)),
            ("distinct_3", lambda t: distinct_n(t, 3)),
            ("repetition_4gram", lambda t: repetition_rate(t, 4)),
            ("length_tokens", None),
        ):
            if label == "length_tokens":
                a = np.array([r["watermarked"]["num_new_tokens"] for r in rows], dtype=float)
                b = np.array([r["unwatermarked"]["num_new_tokens"] for r in rows], dtype=float)
            else:
                a = np.array([fn(r["watermarked"]["text"]) for r in rows])
                b = np.array([fn(r["unwatermarked"]["text"]) for r in rows])
            stats[label] = paired_bootstrap_diff(a, b)
        div_results[suite] = stats

        d2, rep = stats["distinct_2"], stats["repetition_4gram"]
        print(f"{suite:12s} n={d2['n_pairs']:4d}  distinct-2 wm {d2['mean_a']:.4f} vs {d2['mean_b']:.4f} "
              f"(diff {d2['mean_diff']:+.4f} [{d2['ci_low']:+.4f},{d2['ci_high']:+.4f}])")
        print(f"{'':12s}        rep-4gram  wm {rep['mean_a']:.4f} vs {rep['mean_b']:.4f} "
              f"(diff {rep['mean_diff']:+.4f} [{rep['ci_low']:+.4f},{rep['ci_high']:+.4f}])")
    results["diversity"] = div_results

    # -------------------------------------------------------- 2. fluency (PPL)
    lm = WatermarkedLM(args.model, device_map=args.device)

    if not args.skip_judge:
        banner(f"2. Fluency: perplexity under an independent judge ({args.judge_model})")
        print("The judge must be a different model. Scoring watermarked text with the model")
        print("that produced it is biased against it by construction: SynthID selects among")
        print("near-ties using the g-function rather than raw probability, so watermarked")
        print("text sits slightly off that model's own argmax path regardless of quality.")
        judge = WatermarkedLM(args.judge_model, device_map=args.judge_device)
        ppl_results = {}
        for suite in ("creative", "open_ended", "financial"):
            rows = [v for k, v in paired.items() if k[0] == suite and len(v) == 2][:200]
            if not rows:
                continue
            wm_ppl = np.array(judge.perplexity([r["watermarked"]["text"] for r in rows], batch_size=8))
            un_ppl = np.array(judge.perplexity([r["unwatermarked"]["text"] for r in rows], batch_size=8))
            d = paired_bootstrap_diff(wm_ppl, un_ppl)
            ppl_results[suite] = d
            print(f"{suite:12s} n={d['n_pairs']:4d}  PPL wm {d['mean_a']:.3f} vs un {d['mean_b']:.3f}  "
                  f"diff {d['mean_diff']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  "
                  f"{'no significant difference' if d['contains_zero'] else 'SIGNIFICANT'}")
        results["perplexity_independent_judge"] = {"judge_model": args.judge_model, "by_suite": ppl_results}
        del judge
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------ 3. task accuracy
    if not args.skip_gsm8k:
        banner(f"3. Task accuracy: GSM8K ({args.gsm8k_n} problems, sampled CoT, temperature 1.0)")
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.gsm8k_n))
        questions = [row["question"] for row in ds]
        golds = [row["answer"].split("####")[-1].strip().replace(",", "") for row in ds]
        prompts = lm.chat_prompts([GSM8K_INSTRUCTION.format(q=q) for q in questions])

        acc = {}
        preds = {}
        for cond, k in (("watermarked", key), ("unwatermarked", None)):
            out = lm.generate(
                prompts, key=k, max_new_tokens=args.gsm8k_max_tokens,
                temperature=1.0, top_k=64, top_p=0.95,
                batch_size=args.batch_size, seed=7,
            )
            got = [extract_answer(t) for t in out.texts]
            correct = [numbers_equal(g, gold) for g, gold in zip(got, golds)]
            acc[cond] = correct
            preds[cond] = out.texts
            print(f"  {cond:14s} {sum(correct)}/{len(correct)} = {np.mean(correct):.3f}  "
                  f"({out.tokens_per_second:.0f} tok/s)")

        k_wm, k_un = sum(acc["watermarked"]), sum(acc["unwatermarked"])
        n = len(golds)
        ci = two_proportion_diff_ci(k_wm, n, k_un, n)
        # Paired view: how many items flipped in each direction.
        flips_wm_only = sum(1 for a, b in zip(acc["watermarked"], acc["unwatermarked"]) if a and not b)
        flips_un_only = sum(1 for a, b in zip(acc["watermarked"], acc["unwatermarked"]) if b and not a)
        results["gsm8k"] = {
            **ci,
            "n_problems": n,
            "correct_watermarked": k_wm,
            "correct_unwatermarked": k_un,
            "flips_watermarked_only": flips_wm_only,
            "flips_unwatermarked_only": flips_un_only,
        }
        print(f"\n  accuracy delta (watermarked - unwatermarked): {ci['diff']:+.4f} "
              f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
        print(f"  items correct only when watermarked:   {flips_wm_only}")
        print(f"  items correct only when unwatermarked: {flips_un_only}")
        print("\n  Both arms sample at temperature 1.0, so item-level flips in both")
        print("  directions are expected from sampling noise alone.")
        save_json(
            {"questions": questions[:20], "gold": golds[:20],
             "watermarked": preds["watermarked"][:20], "unwatermarked": preds["unwatermarked"][:20]},
            Path(str(args.out).replace(".json", "_gsm8k_samples.json"))
            if args.out else RESULTS / "03_gsm8k_samples.json",
        )

    save_json(results, Path(args.out) if args.out else RESULTS / "03_quality.json")


if __name__ == "__main__":
    main()
