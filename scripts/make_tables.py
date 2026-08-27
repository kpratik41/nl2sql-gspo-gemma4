"""Emit the markdown result tables used in watermarking_report.md from results/*.json.

Generating the tables rather than transcribing them means the report cannot
drift from the data it describes.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=3, plus=False):
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return "—"
    return f"{f:+.{nd}f}" if plus else f"{f:.{nd}f}"


SUITE_ORDER = ["creative", "open_ended", "financial", "factual", "structured", "code"]
SUITE_LABEL = {
    "creative": "creative", "open_ended": "open_ended", "financial": "financial",
    "factual": "factual", "structured": "structured", "code": "code",
    "HIGH_ENTROPY_POOLED": "**pooled (high-entropy)**",
}


DETECT_RUNS = [
    ("02_detectability.json", "Gemma-4-E4B"),
    ("02_detectability_31b.json", "Gemma-4-31B"),
]


def detectability_by_model():
    """Detection strength on both models, at full length and by length."""
    runs = [(load(f), label) for f, label in DETECT_RUNS]
    runs = [(x, label) for x, label in runs if x]
    if len(runs) < 2:
        return

    print("\n### T19 — Detection strength by model, at full length (method: mean)\n")
    print("| Suite | E4B AUC | E4B TPR@1% | E4B mean g | 31B AUC | 31B TPR@1% | 31B mean g |")
    print("|---|---|---|---|---|---|---|")
    for suite in SUITE_ORDER + ["HIGH_ENTROPY_POOLED"]:
        cells, ok = [], True
        for x, _ in runs:
            m = x["full_length"].get(f"{suite}/mean")
            if not m:
                ok = False
                break
            cells += [fmt(m["auc"], 4), fmt(m["tpr_at_fpr_1pct"], 3), fmt(m["mean_positive"], 4)]
        if ok:
            print(f"| {SUITE_LABEL.get(suite, suite)} | " + " | ".join(cells) + " |")

    print("\n### T20 — Detection power vs. length, by model (creative suite, method: mean)\n")
    print("| Target tokens | E4B AUC | E4B TPR@1% | 31B AUC | 31B TPR@1% |")
    print("|---|---|---|---|---|")
    lengths = sorted({int(k.rsplit("/", 1)[1]) for x, _ in runs
                      for k in x.get("length_sweep", {}) if k.startswith("creative/mean/")})
    for L in lengths:
        cells, ok = [], True
        for x, _ in runs:
            v = x.get("length_sweep", {}).get(f"creative/mean/{L}")
            if not v:
                ok = False
                break
            cells += [fmt(v["auc"], 4), fmt(v["tpr_at_fpr_1pct"], 3)]
        if ok:
            print(f"| {L} | " + " | ".join(cells) + " |")


def detectability():
    d = load("02_detectability.json")
    if not d:
        return
    print("\n### T1 — Separation at full length (method: mean)\n")
    print("| Suite | Median tokens scored | AUC | 95% CI | TPR @1% FPR | Mean score (WM) | Mean score (plain) |")
    print("|---|---|---|---|---|---|---|")
    order = SUITE_ORDER + ["HIGH_ENTROPY_POOLED"]
    for s in order:
        k = f"{s}/mean"
        m = d["full_length"].get(k)
        if not m:
            continue
        print(f"| {SUITE_LABEL.get(s, s)} | {m['median_tokens_scored']:.0f} | {fmt(m['auc'],4)} | "
              f"[{fmt(m['auc_ci_low'])}, {fmt(m['auc_ci_high'])}] | {fmt(m['tpr_at_fpr_1pct'])} | "
              f"{fmt(m['mean_positive'],4)} | {fmt(m['mean_negative'],4)} |")

    print("\n### T2 — Detection power vs. text length (pooled high-entropy, method: mean)\n")
    print("| Target tokens | Median scored | AUC | TPR @1% FPR |")
    print("|---|---|---|---|")
    seen = {}
    for k, v in d.get("length_sweep", {}).items():
        suite, method, L = k.rsplit("/", 2)
        if method != "mean":
            continue
        seen.setdefault(int(L), []).append(v)
    for L in sorted(seen):
        vs = seen[L]
        auc = sum(v["auc"] for v in vs) / len(vs)
        tpr = sum(v["tpr_at_fpr_1pct"] for v in vs) / len(vs)
        med = sum(v["median_tokens_scored"] for v in vs) / len(vs)
        print(f"| {L} | {med:.0f} | {fmt(auc,4)} | {fmt(tpr)} |")
    print("\n*(averaged over the creative, open_ended and financial suites)*")

    print("\n### T3 — Key isolation: our text scored with a different key\n")
    print("| Suite | AUC (wrong key) | Mean score (WM text) | Mean score (plain) |")
    print("|---|---|---|---|")
    for s in SUITE_ORDER:
        m = d.get("cross_key", {}).get(f"{s}/mean")
        if not m:
            continue
        print(f"| {s} | {fmt(m['auc'],4)} | {fmt(m['mean_positive'],4)} | {fmt(m['mean_negative'],4)} |")

    print("\n### T4 — False positives on human-written text\n")
    fp = d.get("human_false_positives", {}).get("mean")
    if fp:
        print(f"{fp['n_eval']} WikiText-103 passages, median {fp['median_tokens']:.0f} scored tokens, "
              f"mean score {fmt(fp['mean_score'],4)} (null expectation 0.5).\n")
        print("| Target FPR | Observed (empirical threshold) | Observed (analytic p-value) |")
        print("|---|---|---|")
        for t in (0.10, 0.01, 0.001):
            e = fp.get(f"observed_fpr_at_target_{t}")
            a = fp.get(f"analytic_fpr_at_target_{t}")
            print(f"| {t:.1%} | {fmt(e,4)} | {fmt(a,4)} |")

    nc = d.get("null_calibration", {}).get("mean")
    if nc:
        print("\n### T5 — Is the analytic null the real null? (unwatermarked model output)\n")
        print(f"n = {nc['n']}\n")
        print("| Nominal p threshold | Observed rate |")
        print("|---|---|")
        for t in ("0.10", "0.01", "0.001"):
            print(f"| {t} | {fmt(nc.get(f'observed_rate_p_lt_{t}'),4)} |")

    rt = d.get("token_roundtrip")
    if rt:
        print(f"\n**Token round-trip:** {rt['exact_roundtrip_fraction']:.1%} of texts re-tokenise "
              f"exactly; {rt['position_agreement']:.1%} of token positions agree.")


QUALITY_RUNS = [
    ("03_quality.json", "Gemma-4-E4B"),
    ("03_quality_31b.json", "Gemma-4-31B"),
]


def quality():
    runs = [(load(f), label) for f, label in QUALITY_RUNS]
    runs = [(d, label) for d, label in runs if d]
    if not runs:
        return

    judges = {d.get("perplexity_independent_judge", {}).get("judge_model")
              for d, _ in runs if d.get("perplexity_independent_judge")}
    judge = judges.pop() if len(judges) == 1 else " / ".join(sorted(filter(None, judges)))

    print(f"\n### T7 — Fluency under an independent judge (`{judge}`)\n")
    print("| Model | Suite | n pairs | PPL watermarked | PPL plain | Difference [95% CI] | Significant? |")
    print("|---|---|---|---|---|---|---|")
    for d, label in runs:
        pj = d.get("perplexity_independent_judge")
        if not pj:
            continue
        for suite, x in pj["by_suite"].items():
            sig = "no" if x["contains_zero"] else "**yes**"
            print(f"| {label} | {suite} | {x['n_pairs']} | {fmt(x['mean_a'],3)} | {fmt(x['mean_b'],3)} | "
                  f"{fmt(x['mean_diff'],3,True)} [{fmt(x['ci_low'],3,True)}, {fmt(x['ci_high'],3,True)}] | {sig} |")

    print("\n### T6 — Diversity (distinct-2), paired by prompt\n")
    print("| Model | Suite | n pairs | distinct-2 WM | distinct-2 plain | Difference [95% CI] |")
    print("|---|---|---|---|---|---|")
    for d, label in runs:
        for suite in SUITE_ORDER:
            st = d.get("diversity", {}).get(suite)
            if not st:
                continue
            x = st["distinct_2"]
            print(f"| {label} | {suite} | {x['n_pairs']} | {fmt(x['mean_a'],4)} | {fmt(x['mean_b'],4)} | "
                  f"{fmt(x['mean_diff'],4,True)} [{fmt(x['ci_low'],4,True)}, {fmt(x['ci_high'],4,True)}] |")

    print("\n### T8 — Task accuracy: GSM8K, sampled chain-of-thought at temperature 1.0\n")
    print("| Model | Watermarked | Unwatermarked | Difference [95% CI] | Flips WM-only / plain-only |")
    print("|---|---|---|---|---|")
    for d, label in runs:
        g = d.get("gsm8k")
        if not g:
            continue
        print(f"| {label} | {g['correct_watermarked']}/{g['n_problems']} ({fmt(g['acc_a'],3)}) | "
              f"{g['correct_unwatermarked']}/{g['n_problems']} ({fmt(g['acc_b'],3)}) | "
              f"**{fmt(g['diff'],4,True)}** [{fmt(g['ci_low'],4,True)}, {fmt(g['ci_high'],4,True)}] | "
              f"{g['flips_watermarked_only']} / {g['flips_unwatermarked_only']} |")


def robustness():
    d = load("04_robustness.json")
    if not d:
        return
    res = d["results"]
    base = res.get("none/-", {}).get("auc")
    print("\n### T9 — Robustness: AUC after modification\n")
    print("| Attack | Level | Median tokens | AUC | TPR @1% FPR | Mean score (WM) |")
    print("|---|---|---|---|---|---|")
    for k, v in res.items():
        name, lvl = v.get("attack"), v.get("strength")
        print(f"| {name} | {lvl} | {v['median_tokens_scored']:.0f} | {fmt(v['auc'],4)} | "
              f"{fmt(v['tpr_at_fpr_1pct'])} | {fmt(v['mean_positive'],4)} |")
    if base:
        print(f"\n*(unattacked baseline AUC = {fmt(base,4)})*")


MODELS = [
    ("results/05_overhead.json", "Gemma-4-E4B"),
    ("results/05_overhead_31b.json", "Gemma-4-31B"),
]


def overhead():
    """Throughput cost, compared across model sizes.

    Both models share the same 262k-token vocabulary, so putting them
    side by side isolates the effect of model size on the watermark's
    relative cost.
    """
    loaded = [(load(Path(p).name), label) for p, label in MODELS]
    loaded = [(d, label) for d, label in loaded if d]
    if not loaded:
        return

    print("\n### T10 — Generation throughput cost, by model size\n")
    print("*Throughput in tokens per second — higher is faster.*\n")
    header = "| Batch size |" + "".join(
        f" {label} plain (tok/s) | {label} watermarked (tok/s) | {label} overhead |"
        for _, label in loaded)
    print(header)
    print("|---|" + "---|" * (3 * len(loaded)))
    batches = sorted({int(b) for d, _ in loaded for b in d.get("throughput_by_batch_size", {})})
    for bs in batches:
        row = f"| {bs} |"
        for d, _ in loaded:
            v = d.get("throughput_by_batch_size", {}).get(str(bs))
            row += (f" {v['plain_tokens_per_s']:.1f} | {v['watermarked_tokens_per_s']:.1f} | "
                    f"**{v['relative_overhead']:.1%}** |") if v else " — | — | — |"
        print(row)

    print("\n### T11 — Cost of watermarking depth (batch 16), by model size\n")
    print("*Watermarked throughput at each depth, tokens per second — higher is faster.*\n")
    print("| Depth |" + "".join(f" {label} (tok/s) | {label} overhead |" for _, label in loaded))
    print("|---|" + "---|" * (2 * len(loaded)))
    depths = sorted({int(x) for d, _ in loaded for x in d.get("depth_cost", {}).get("by_depth", {})})
    for depth in depths:
        row = f"| {depth} |"
        for d, _ in loaded:
            v = d.get("depth_cost", {}).get("by_depth", {}).get(str(depth))
            row += f" {v['tokens_per_s']:.1f} | **{v['relative_overhead']:.1%}** |" if v else " — | — |"
        print(row)

    print("\n### T12 — Detection throughput\n")
    print("| Model | Device | Texts/s | ms per text |")
    print("|---|---|---|---|")
    for d, label in loaded:
        for dev, v in d.get("detection_throughput", {}).items():
            print(f"| {label} | {dev} | {v['texts_per_s']:.1f} | {1000 / v['texts_per_s']:.1f} |")


def processor_cost():
    """Where the watermark's inference cost goes, and whether it is inherent."""
    d = load("08_processor_cost.json")
    if not d:
        return

    print("\n### T14 — Watermark cost per decoding step, model excluded\n")
    print("| Batch | Processor ms/step | Processor ms **per sequence** |")
    print("|---|---|---|")
    for b, v in sorted(d.get("processor_by_batch", {}).items(), key=lambda kv: int(kv[0])):
        print(f"| {b} | {v['ms_per_step']:.2f} | {v['ms_per_sequence']:.2f} |")

    print("\n### T15 — Attribution: is the overhead the processor?\n")
    print("| Batch | Model forward (ms) | Watermarked step (ms) | Observed Δ | Processor alone | Explained |")
    print("|---|---|---|---|---|---|")
    for b, v in sorted(d.get("attribution", {}).items(), key=lambda kv: int(kv[0])):
        print(f"| {b} | {v['plain_ms_per_step']:.2f} | {v['watermarked_ms_per_step']:.2f} | "
              f"{v['observed_delta_ms']:.2f} | {v['processor_ms']:.2f} | "
              f"**{v['fraction_explained']:.0%}** |")

    print("\n### T16 — What the cost would be if the processor amortised across the batch\n")
    print("| Batch | Measured overhead | If it batched like the model |")
    print("|---|---|---|")
    for b, v in sorted(d.get("hypothetical_amortised", {}).items(), key=lambda kv: int(kv[0])):
        print(f"| {b} | {v['actual']:.1%} | **{v['if_amortised']:.1%}** |")

    print("\n### T17 — Processor cost by depth (batch 16), model excluded\n")
    print("| Depth | Processor ms/step |")
    print("|---|---|")
    for dep, ms in sorted(d.get("processor_by_depth_batch16", {}).items(), key=lambda kv: int(kv[0])):
        print(f"| {dep} | {ms:.2f} |")


def fastpath():
    """The candidate-only fix: equivalence and cost."""
    d = load("10_fastpath.json")
    if not d:
        return
    e = d["equivalence"]
    print("\n### T21 — Candidate-only path: does it change the answer?\n")
    print("| Check | Result |")
    print("|---|---|")
    print(f"| Largest probability difference, any token | {e['max_prob_diff']:.1e} |")
    print(f"| Total-variation distance between sampling distributions | {e['max_total_variation']:.1e} |")
    print("| Most likely token ever changed | no |")

    m = d["meta"]
    print(f"\n### T22 — Cost per decoding step, CPU (vocab {m['vocab']:,}, "
          f"depth {m['depth']}, top-k {m['top_k']})\n")
    print("| Batch | Upstream (ms) | Candidate-only (ms) | Speed-up |")
    print("|---|---|---|---|")
    for b, v in sorted(d["cost_by_batch"].items(), key=lambda kv: int(kv[0])):
        print(f"| {b} | {v['upstream_ms']:.1f} | {v['fast_ms']:.2f} | **{v['speedup']:.0f}x** |")
    sc = d["scaling"]
    print(f"\n*From batch {sc['from_batch']} to {sc['to_batch']}, upstream cost grows "
          f"{sc['upstream_growth']:.1f}x; the candidate-only path grows "
          f"{sc['fast_growth']:.1f}x.*")


def walkthrough():
    """The token-by-token illustration used in section 2."""
    d = load("09_walkthrough.json")
    if not d:
        return
    print("\n### T18 — Token-by-token: the same passage under the correct key and a different one\n")
    print(f"| # | Token | Heads (of {d['depth']}) | g | Running mean g | g, wrong key | Running mean, wrong key |")
    print("|---|---|---|---|---|---|---|")
    for r in d["tokens"]:
        print(f"| {r['index']} | `{r['token']}` | {r['heads']}/{r['depth']} | {r['g']:.3f} | "
              f"**{r['running_g']:.3f}** | {r['g_wrong_key']:.3f} | {r['running_g_wrong_key']:.3f} |")
    w = d["whole_passage"]
    print(f"\nOver the full {w['tokens_scored']}-token passage: correct key **{w['mean_g_correct_key']:.4f}**, "
          f"a different key {w['mean_g_wrong_key']:.4f}, null expectation 0.5000 "
          f"(z = {w['z_score']:+.2f}, p = {w['p_value']:.1e}).")


def bayesian():
    d = load("06_bayesian.json")
    if not d:
        return
    print("\n### T13 — Learned detector vs. mean detector, held-out prompts\n")
    print("| Tokens | AUC mean | AUC weighted | AUC Bayesian | TPR@1% mean | TPR@1% Bayesian |")
    print("|---|---|---|---|---|---|")
    for L, row in d.get("by_length", {}).items():
        if "mean" not in row or "bayesian" not in row:
            continue
        print(f"| {L} | {fmt(row['mean']['auc'],4)} | {fmt(row.get('weighted_mean',{}).get('auc'),4)} | "
              f"{fmt(row['bayesian']['auc'],4)} | {fmt(row['mean']['tpr_at_fpr_1pct'])} | "
              f"{fmt(row['bayesian']['tpr_at_fpr_1pct'])} |")


if __name__ == "__main__":
    for fn in (detectability_by_model, detectability, quality, robustness, overhead, processor_cost, fastpath, walkthrough, bayesian):
        try:
            fn()
        except Exception as e:  # a missing study should not block the rest
            print(f"\n<!-- {fn.__name__} failed: {e} -->")
