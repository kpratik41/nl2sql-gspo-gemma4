"""Emit the markdown result tables used in report.md from results/*.json.

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


def quality():
    d = load("03_quality.json")
    if not d:
        return
    print("\n### T6 — Diversity and degeneracy (paired)\n")
    print("| Suite | n pairs | distinct-2 WM | distinct-2 plain | Difference [95% CI] |")
    print("|---|---|---|---|---|")
    for s in SUITE_ORDER:
        st = d.get("diversity", {}).get(s)
        if not st:
            continue
        x = st["distinct_2"]
        print(f"| {s} | {x['n_pairs']} | {fmt(x['mean_a'],4)} | {fmt(x['mean_b'],4)} | "
              f"{fmt(x['mean_diff'],4,True)} [{fmt(x['ci_low'],4,True)}, {fmt(x['ci_high'],4,True)}] |")

    pj = d.get("perplexity_independent_judge")
    if pj:
        print(f"\n### T7 — Fluency under an independent judge (`{pj['judge_model']}`)\n")
        print("| Suite | n pairs | PPL watermarked | PPL plain | Difference [95% CI] | Significant? |")
        print("|---|---|---|---|---|---|")
        for s, x in pj["by_suite"].items():
            sig = "no" if x["contains_zero"] else "**yes**"
            print(f"| {s} | {x['n_pairs']} | {fmt(x['mean_a'],2)} | {fmt(x['mean_b'],2)} | "
                  f"{fmt(x['mean_diff'],2,True)} [{fmt(x['ci_low'],2,True)}, {fmt(x['ci_high'],2,True)}] | {sig} |")

    g = d.get("gsm8k")
    if g:
        print("\n### T8 — Task accuracy: GSM8K, sampled chain-of-thought at temperature 1.0\n")
        print("| Condition | Correct | Accuracy |")
        print("|---|---|---|")
        print(f"| watermarked | {g['correct_watermarked']}/{g['n_problems']} | {fmt(g['acc_a'],4)} |")
        print(f"| unwatermarked | {g['correct_unwatermarked']}/{g['n_problems']} | {fmt(g['acc_b'],4)} |")
        print(f"\nDifference **{fmt(g['diff'],4,True)}**, 95% CI "
              f"[{fmt(g['ci_low'],4,True)}, {fmt(g['ci_high'],4,True)}].  \n"
              f"Item-level flips: {g['flips_watermarked_only']} correct only when watermarked, "
              f"{g['flips_unwatermarked_only']} correct only when not.")


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


def overhead():
    d = load("05_overhead.json")
    if not d:
        return
    print("\n### T10 — Generation throughput, watermark on vs off\n")
    print("| Batch size | Plain (tok/s) | Watermarked (tok/s) | Overhead |")
    print("|---|---|---|---|")
    for bs, v in d.get("throughput_by_batch_size", {}).items():
        print(f"| {bs} | {v['plain_tokens_per_s']:.1f} | {v['watermarked_tokens_per_s']:.1f} | "
              f"{v['relative_overhead']:.1%} |")

    dc = d.get("depth_cost")
    if dc:
        print("\n### T11 — Cost of watermarking depth (batch 16)\n")
        print("| Depth | Throughput (tok/s) | Overhead vs. plain |")
        print("|---|---|---|")
        for depth, v in dc["by_depth"].items():
            print(f"| {depth} | {v['tokens_per_s']:.1f} | {v['relative_overhead']:.1%} |")

    dt = d.get("detection_throughput")
    if dt:
        print("\n### T12 — Detection throughput\n")
        print("| Device | Texts/s | ms per text |")
        print("|---|---|---|")
        for dev, v in dt.items():
            print(f"| {dev} | {v['texts_per_s']:.1f} | {1000 / v['texts_per_s']:.1f} |")


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
    for fn in (detectability, quality, robustness, overhead, bayesian):
        try:
            fn()
        except Exception as e:  # a missing study should not block the rest
            print(f"\n<!-- {fn.__name__} failed: {e} -->")
