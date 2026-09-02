"""Can we detect our own watermark, and how often do we flag text that is not ours?

Three questions, in the order a reviewer will ask them:

1. **Does it work?**  Separation between watermarked and unwatermarked output
   of the same model, per prompt suite, as AUC and as TPR at a fixed
   false-positive rate.
2. **How much text do we need?**  The same numbers as a function of the number
   of scored tokens.  This is the single most important operational figure:
   below some length the detector should not be asked for a verdict at all.
3. **Whom do we wrongly accuse?**  The false-positive rate measured on
   human-written text, and on text watermarked with a *different* key.  The
   second is the multi-tenant property: one desk's key must not detect another
   desk's output.

The analytic p-value is also checked against reality: if the closed-form null is
right, then exactly 1% of true negatives should fall below p = 0.01.  A detector
whose stated false-positive rate is not the observed one cannot be governed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from common import (
    DATA,
    DEMO_MASTER_SECRET,
    FIGURES,
    OTHER_KEY_ID,
    PRIMARY_KEY_ID,
    RESULTS,
    banner,
    save_json,
)
from transformers import AutoTokenizer

from synthmark_eval import Detector, derive_key, evaluate_detection
from synthmark.data import HIGH_ENTROPY_SUITES

LENGTHS = [25, 50, 100, 200, 300, 400]
METHODS = ["mean", "weighted_mean"]


def truncate_ids(token_ids: list[int], n: int, tokenizer) -> str:
    """Cut a completion to exactly ``n`` sampled tokens and decode it.

    Truncating on token ids rather than on words gives exact control of the
    x-axis of the length curve, which is the quantity detection power actually
    depends on.
    """
    return tokenizer.decode(token_ids[:n], skip_special_tokens=True)


def score_all(detector, texts, method, batch_size=32):
    scores, ntok = [], []
    for i in range(0, len(texts), batch_size):
        s, n = detector.score(texts[i : i + batch_size], method=method)
        scores.append(s)
        ntok.append(n)
    if not scores:
        return np.array([]), np.array([])
    return np.concatenate(scores), np.concatenate(ntok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus.json"))
    ap.add_argument("--human", default=str(DATA / "human_texts.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None,
                    help="Results JSON path; defaults to results/02_detectability.json.")
    args = ap.parse_args()

    banner("Detectability")
    corpus = json.loads(open(args.corpus).read())
    meta, records = corpus["meta"], corpus["records"]
    tokenizer = AutoTokenizer.from_pretrained(meta["model"])
    human_texts = json.loads(open(args.human).read())

    key = derive_key(DEMO_MASTER_SECRET, PRIMARY_KEY_ID)
    other_key = derive_key(DEMO_MASTER_SECRET, OTHER_KEY_ID)
    det = Detector(key, tokenizer, device=args.device)
    det_other = Detector(other_key, tokenizer, device=args.device)
    print(f"primary key fp={key.fingerprint}   other key fp={other_key.fingerprint}")

    by_suite = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_suite[r["suite"]][r["condition"]].append(r)

    results: dict = {
        "meta": {
            "model": meta["model"],
            "key": key.public_summary(),
            "other_key": other_key.public_summary(),
            "n_human_texts": len(human_texts),
        },
        "full_length": {},
        "length_sweep": {},
        "cross_key": {},
        "human_false_positives": {},
        "null_calibration": {},
    }

    # ------------------------------------------------ 1. full-length separation
    banner("1. Separation at full length, by prompt suite")
    print(f"{'suite':12s} {'method':14s} {'n_tok':>6s} {'AUC':>7s} {'95% CI':>16s} "
          f"{'TPR@1%':>7s} {'TPR@.1%':>8s} {'mean+':>7s} {'mean-':>7s}")
    for suite in sorted(by_suite):
        wm = [r["text"] for r in by_suite[suite]["watermarked"]]
        un = [r["text"] for r in by_suite[suite]["unwatermarked"]]
        for method in METHODS:
            s_wm, n_wm = score_all(det, wm, method)
            s_un, _ = score_all(det, un, method)
            m = evaluate_detection(s_wm, s_un, tokens_scored=n_wm)
            results["full_length"][f"{suite}/{method}"] = m.to_dict()
            print(f"{suite:12s} {method:14s} {np.median(n_wm):6.0f} {m.auc:7.4f} "
                  f"[{m.auc_ci_low:.3f},{m.auc_ci_high:.3f}] {m.tpr_at_fpr_1pct:7.3f} "
                  f"{m.tpr_at_fpr_0p1pct:8.3f} {m.mean_positive:7.4f} {m.mean_negative:7.4f}")

    # Pooled across the high-entropy suites. Per-suite negative sets are too
    # small to resolve a 1% false-positive rate; pooling gives enough negatives
    # for the operating point that actually gets deployed.
    pooled_wm = [r["text"] for s_ in HIGH_ENTROPY_SUITES for r in by_suite.get(s_, {}).get("watermarked", [])]
    pooled_un = [r["text"] for s_ in HIGH_ENTROPY_SUITES for r in by_suite.get(s_, {}).get("unwatermarked", [])]
    for method in METHODS:
        s_wm, n_wm = score_all(det, pooled_wm, method)
        s_un, _ = score_all(det, pooled_un, method)
        m = evaluate_detection(s_wm, s_un, tokens_scored=n_wm)
        results["full_length"][f"HIGH_ENTROPY_POOLED/{method}"] = m.to_dict()
        print(f"{'POOLED':12s} {method:14s} {np.median(n_wm):6.0f} {m.auc:7.4f} "
              f"[{m.auc_ci_low:.3f},{m.auc_ci_high:.3f}] {m.tpr_at_fpr_1pct:7.3f} "
              f"{m.tpr_at_fpr_0p1pct:8.3f} {m.mean_positive:7.4f} {m.mean_negative:7.4f}")

    # ---------------------------------------------------- 2. length dependence
    banner("2. Detection power vs. number of scored tokens (high-entropy suites)")
    print(f"{'suite':12s} {'method':14s} {'target':>7s} {'n_tok':>6s} {'AUC':>7s} {'TPR@1%':>7s} {'TPR@.1%':>8s}")
    for suite in HIGH_ENTROPY_SUITES:
        if suite not in by_suite:
            continue
        wm_ids = [r["token_ids"] for r in by_suite[suite]["watermarked"]]
        un_ids = [r["token_ids"] for r in by_suite[suite]["unwatermarked"]]
        for method in METHODS:
            for L in LENGTHS:
                wm_t = [truncate_ids(t, L, tokenizer) for t in wm_ids]
                un_t = [truncate_ids(t, L, tokenizer) for t in un_ids]
                s_wm, n_wm = score_all(det, wm_t, method)
                s_un, _ = score_all(det, un_t, method)
                ok = ~np.isnan(s_wm)
                if ok.sum() < 5:
                    continue
                m = evaluate_detection(s_wm, s_un, tokens_scored=n_wm)
                results["length_sweep"][f"{suite}/{method}/{L}"] = {
                    **m.to_dict(), "target_tokens": L,
                }
                print(f"{suite:12s} {method:14s} {L:7d} {np.median(n_wm):6.0f} {m.auc:7.4f} "
                      f"{m.tpr_at_fpr_1pct:7.3f} {m.tpr_at_fpr_0p1pct:8.3f}")

    # ------------------------------------------------------- 3. wrong-key test
    banner("3. Key isolation: our watermarked text, scored with a different key")
    print(f"{'suite':12s} {'method':14s} {'AUC(wrong key)':>15s} {'mean+':>8s} {'mean-':>8s}")
    for suite in sorted(by_suite):
        wm = [r["text"] for r in by_suite[suite]["watermarked"]]
        un = [r["text"] for r in by_suite[suite]["unwatermarked"]]
        for method in METHODS:
            s_wm, _ = score_all(det_other, wm, method)
            s_un, _ = score_all(det_other, un, method)
            m = evaluate_detection(s_wm, s_un)
            results["cross_key"][f"{suite}/{method}"] = m.to_dict()
            print(f"{suite:12s} {method:14s} {m.auc:15.4f} {m.mean_positive:8.4f} {m.mean_negative:8.4f}")
    print("\nAUC ~= 0.5 means the wrong key sees nothing: keys are independent.")

    # -------------------------------------------- 4. false positives on humans
    banner("4. False positives on human-written text")
    # Split the human corpus: half sets the threshold, half measures the rate.
    half = len(human_texts) // 2
    cal_texts, eval_texts = human_texts[:half], human_texts[half:]
    for method in METHODS:
        cal = det.calibrate(cal_texts, method=method)
        s_h, n_h = score_all(det, eval_texts, method)
        ok = n_h > 0
        s_h, n_h = s_h[ok], n_h[ok]

        rates = {}
        for fpr in (0.10, 0.01, 0.001):
            thr = np.array([cal.threshold(int(n), fpr) for n in n_h])
            rates[f"observed_fpr_at_target_{fpr}"] = float(np.mean(s_h >= thr))
        # Analytic p-values on the same texts.
        analytic = []
        for t in eval_texts:
            r = det.detect(t, method=method)
            if r.p_value is not None:
                analytic.append(r.p_value)
        analytic = np.array(analytic)
        for fpr in (0.10, 0.01, 0.001):
            rates[f"analytic_fpr_at_target_{fpr}"] = float(np.mean(analytic < fpr))
        rates["n_eval"] = int(len(s_h))
        rates["median_tokens"] = float(np.median(n_h))
        rates["mean_score"] = float(np.mean(s_h))
        results["human_false_positives"][method] = rates
        cal.save(RESULTS / f"calibration_human_{method}"
                 f"{'_31b' if '31b' in str(args.corpus) else ''}.json")

        print(f"\n[{method}] {len(s_h)} human texts, median {np.median(n_h):.0f} scored tokens, "
              f"mean score {np.mean(s_h):.4f}")
        print("  empirical thresholds (calibrated on a disjoint half of the human corpus):")
        for fpr in (0.10, 0.01, 0.001):
            print(f"    target {fpr:6.3f} -> observed {rates[f'observed_fpr_at_target_{fpr}']:.4f}")
        print("  analytic p-value thresholds (no calibration):")
        for fpr in (0.10, 0.01, 0.001):
            print(f"    target {fpr:6.3f} -> observed {rates[f'analytic_fpr_at_target_{fpr}']:.4f}")

    # ---------------------------- 5. is the analytic null actually the null?
    banner("5. Analytic null vs. observed null (unwatermarked model output)")
    all_un = [r["text"] for r in records if r["condition"] == "unwatermarked"]
    for method in METHODS:
        ps = []
        for i in range(0, len(all_un), 32):
            for t in all_un[i : i + 32]:
                r = det.detect(t, method=method)
                if r.p_value is not None:
                    ps.append(r.p_value)
        ps = np.array(ps)
        entry = {
            "n": int(len(ps)),
            "observed_rate_p_lt_0.10": float(np.mean(ps < 0.10)),
            "observed_rate_p_lt_0.01": float(np.mean(ps < 0.01)),
            "observed_rate_p_lt_0.001": float(np.mean(ps < 0.001)),
        }
        results["null_calibration"][method] = entry
        print(f"[{method}] n={entry['n']}  "
              f"p<0.10: {entry['observed_rate_p_lt_0.10']:.4f}  "
              f"p<0.01: {entry['observed_rate_p_lt_0.01']:.4f}  "
              f"p<0.001: {entry['observed_rate_p_lt_0.001']:.4f}")
    print("\nA well-calibrated analytic null makes these match their targets.")

    # ------------------------------------ 6. cost of the decode/re-encode round trip
    banner("6. Token round-trip fidelity")
    # The watermark is embedded in *token* choices, but a real detector is handed
    # *text* and must re-tokenise it. If detokenise-then-retokenise is not the
    # identity, some watermarked positions are destroyed before scoring even
    # begins. This measures that loss directly, with no attack involved.
    sample = [r for r in records if r["condition"] == "watermarked"][:400]
    exact, total_orig, total_kept = 0, 0, 0
    for r in sample:
        orig = r["token_ids"]
        retok = tokenizer(r["text"], add_special_tokens=False)["input_ids"]
        total_orig += len(orig)
        total_kept += sum(1 for a, b in zip(orig, retok) if a == b)
        if orig == retok:
            exact += 1
    rt = {
        "n_texts": len(sample),
        "exact_roundtrip_fraction": exact / len(sample),
        "position_agreement": total_kept / max(total_orig, 1),
    }
    results["token_roundtrip"] = rt
    print(f"  {len(sample)} texts: {rt['exact_roundtrip_fraction']:.1%} re-tokenise exactly, "
          f"{rt['position_agreement']:.1%} of token positions agree")

    # Score the same texts from stored ids vs from re-tokenised text.
    ids_scores = []
    for i in range(0, len(sample), 32):
        chunk = sample[i : i + 32]
        texts = [tokenizer.decode(r["token_ids"], skip_special_tokens=True) for r in chunk]
        s_, _ = score_all(det, texts, "mean")
        ids_scores.append(s_)
    print(f"  mean score via decoded text: {np.nanmean(np.concatenate(ids_scores)):.4f}")

    save_json(results, Path(args.out) if args.out else RESULTS / "02_detectability.json")


if __name__ == "__main__":
    main()
