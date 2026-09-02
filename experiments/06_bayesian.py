"""Does the learned detector help where the mean detector is weakest -- short text?

The mean detector weights every position and every depth equally.  Tournament
sampling does not distribute signal equally across depths, so a detector that
learns the per-depth pattern should extract more from the same g-values.  The
place that matters is short text, where the mean detector runs out of signal
first.

This study needs no new generation: it trains on the g-values of the corpus that
already exists, which makes it by far the cheapest way to buy detection power.
Training and evaluation use **disjoint prompts**, not just disjoint samples, so
the reported AUC cannot be inflated by the detector memorising prompt-specific
token patterns.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from common import (
    DATA,
    DEMO_MASTER_SECRET,
    PRIMARY_KEY_ID,
    RESULTS,
    banner,
    save_json,
)
from transformers import AutoTokenizer

from synthmark_eval import Detector, derive_key, evaluate_detection
from synthmark.bayesian import build_dataset, save_detector, train_bayesian_detector

LENGTHS = [25, 50, 100, 200, 400]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-length", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    banner("Learned (Bayesian) detector vs. the mean detector")
    corpus = json.loads(open(args.corpus).read())
    meta, records = corpus["meta"], corpus["records"]
    tokenizer = AutoTokenizer.from_pretrained(meta["model"])
    key = derive_key(DEMO_MASTER_SECRET, PRIMARY_KEY_ID)
    det = Detector(key, tokenizer, device=args.device)

    # High-entropy suites only: there is no point training a detector on text
    # that carries no watermark to begin with.
    suites = ("creative", "open_ended", "financial")
    rows = [r for r in records if r["suite"] in suites]

    # Split by prompt index, so train and test share no prompts.
    all_idx = sorted({r["prompt_index"] for r in rows})
    n_train = int(len(all_idx) * 0.6)
    train_idx, test_idx = set(all_idx[:n_train]), set(all_idx[n_train:])
    print(f"{len(train_idx)} training prompts / {len(test_idx)} held-out prompts per suite")

    def texts_for(cond, idx_set, length=None):
        out = []
        for r in rows:
            if r["condition"] != cond or r["prompt_index"] not in idx_set:
                continue
            ids = r["token_ids"][:length] if length else r["token_ids"]
            out.append(tokenizer.decode(ids, skip_special_tokens=True))
        return out

    tr_wm = texts_for("watermarked", train_idx, args.train_length)
    tr_un = texts_for("unwatermarked", train_idx, args.train_length)
    print(f"training on {len(tr_wm)} watermarked / {len(tr_un)} unwatermarked texts "
          f"truncated to {args.train_length} tokens")

    g, m, y = build_dataset(det, tr_wm, tr_un)
    print(f"g-values tensor {tuple(g.shape)}")
    model, history = train_bayesian_detector(
        g, m, y, watermarking_depth=key.depth, epochs=args.epochs, device=args.device
    )
    print(f"best held-out (within-train) AUC during fitting: "
          f"{max(h.val_auc for h in history):.4f}")

    banner("Held-out prompts: AUC by text length")
    print(f"{'tokens':>7s} {'mean':>8s} {'weighted':>9s} {'bayesian':>9s}   "
          f"{'TPR@1% mean':>12s} {'TPR@1% bayes':>13s}")
    table = {}
    for L in LENGTHS:
        te_wm = texts_for("watermarked", test_idx, L)
        te_un = texts_for("unwatermarked", test_idx, L)
        row = {}
        for method, kwargs in (
            ("mean", {}),
            ("weighted_mean", {}),
            ("bayesian", {"bayesian_model": model}),
        ):
            s_wm, n_wm = det.score(te_wm, method=method, **kwargs)
            s_un, _ = det.score(te_un, method=method, **kwargs)
            if np.isnan(s_wm).all():
                continue
            row[method] = evaluate_detection(s_wm, s_un, tokens_scored=n_wm).to_dict()
        table[L] = row
        if {"mean", "bayesian"} <= row.keys():
            print(f"{L:7d} {row['mean']['auc']:8.4f} {row['weighted_mean']['auc']:9.4f} "
                  f"{row['bayesian']['auc']:9.4f}   "
                  f"{row['mean']['tpr_at_fpr_1pct']:12.3f} "
                  f"{row['bayesian']['tpr_at_fpr_1pct']:13.3f}")

    # Write results first: persisting the detector goes through
    # save_pretrained, which pulls in accelerate -> deepspeed and fails on hosts
    # without CUDA_HOME. That is a packaging quirk of the environment, not a
    # result, and it must not cost us the measurements.
    save_json({"meta": {"model": meta["model"], "suites": list(suites),
                        "train_length": args.train_length,
                        "n_train_prompts": len(train_idx),
                        "n_test_prompts": len(test_idx)},
               "by_length": table}, RESULTS / "06_bayesian.json")

    out_dir = RESULTS / "bayesian_detector"
    try:
        save_detector(model, out_dir, model_name=meta["model"],
                      watermarking_config={"ngram_len": key.ngram_len, "keys": list(key.keys),
                                           "sampling_table_size": key.sampling_table_size,
                                           "sampling_table_seed": key.sampling_table_seed,
                                           "context_history_size": key.context_history_size})
        print(f"\nThe saved detector at {out_dir} embeds the secret key.")
        print("Treat that directory exactly as you would the key file itself.")
    except Exception as exc:
        print(f"\ncould not persist the detector ({type(exc).__name__}: {exc});")
        print("results above are unaffected. Set CUDA_HOME to enable save_pretrained.")


if __name__ == "__main__":
    main()
