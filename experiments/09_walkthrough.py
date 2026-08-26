"""A token-by-token illustration of the watermark, on real generated text.

The mechanism is easy to state and hard to picture, because the whole point is
that no individual token gives anything away. This script makes that concrete:
it takes a passage the model actually generated, and prints, for each token, the
coin flips the g-function produced under the correct key and under a different
one.

The pedagogical payload is in the two running averages. Token by token the
values are noisy and indistinguishable; only after hundreds of tokens does the
correct key's average separate from 0.5 while the wrong key's stays put. That is
exactly why the watermark is invisible to a reader and why short text cannot be
judged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import DATA, DEMO_MASTER_SECRET, OTHER_KEY_ID, PRIMARY_KEY_ID, RESULTS, banner, save_json
from transformers import AutoTokenizer

from synthmark import Detector, derive_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus.json"))
    ap.add_argument("--suite", default="creative")
    ap.add_argument("--tokens", type=int, default=14, help="How many tokens to display.")
    ap.add_argument("--out", default=str(RESULTS / "09_walkthrough.json"))
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    rec = next(r for r in corpus["records"]
               if r["condition"] == "watermarked" and r["suite"] == args.suite)
    tokenizer = AutoTokenizer.from_pretrained(corpus["meta"]["model"])

    key = derive_key(DEMO_MASTER_SECRET, PRIMARY_KEY_ID)
    other = derive_key(DEMO_MASTER_SECRET, OTHER_KEY_ID)
    right = Detector(key, tokenizer, device="cpu")
    wrong = Detector(other, tokenizer, device="cpu")

    text = rec["text"]
    g_right, _ = right.g_values([text])
    g_wrong, _ = wrong.g_values([text])
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    depth, ngram = key.depth, key.ngram_len

    banner("Token-by-token walkthrough")
    print(f"prompt : {rec['prompt']}")
    print(f"text   : {text[:100]}...\n")
    print(f"Each token gets {depth} independent coin flips, one per key in the bundle, seeded")
    print(f"by the secret key and the {ngram - 1} tokens immediately before it.\n")

    rows, run_r, run_w = [], 0.0, 0.0
    print(f"{'#':>3} {'token':<16} {'heads':>8} {'g':>6} {'running':>8}   {'wrong g':>8} {'running':>8}")
    for i in range(min(args.tokens, g_right.shape[1])):
        token = tokenizer.decode([ids[i + ngram - 1]])
        gr, gw = float(g_right[0, i].mean()), float(g_wrong[0, i].mean())
        run_r += gr
        run_w += gw
        rows.append({"index": i + 1, "token": token,
                     "heads": int(g_right[0, i].sum()), "depth": depth,
                     "g": gr, "running_g": run_r / (i + 1),
                     "g_wrong_key": gw, "running_g_wrong_key": run_w / (i + 1)})
        print(f"{i+1:3d} {token!r:<16} {int(g_right[0,i].sum()):>3}/{depth:<4} {gr:6.3f} "
              f"{run_r/(i+1):8.3f}   {gw:8.3f} {run_w/(i+1):8.3f}")

    s_r, n = right.score([text])
    s_w, _ = wrong.score([text])
    result = right.detect(text)
    print(f"\nOver the whole {int(n[0])}-token passage:")
    print(f"  correct key      mean g = {s_r[0]:.4f}")
    print(f"  a different key  mean g = {s_w[0]:.4f}")
    print(f"  null expectation          0.5000")
    print(f"  -> z = {result.z_score:+.2f}, p = {result.p_value:.2e}")
    print("\nNo single token is a tell. The signal exists only in the average.")

    save_json({"prompt": rec["prompt"], "text_preview": text[:200],
               "model": corpus["meta"]["model"], "depth": depth, "ngram_len": ngram,
               "tokens": rows,
               "whole_passage": {"tokens_scored": int(n[0]),
                                 "mean_g_correct_key": float(s_r[0]),
                                 "mean_g_wrong_key": float(s_w[0]),
                                 "z_score": result.z_score, "p_value": result.p_value}},
              Path(args.out))


if __name__ == "__main__":
    main()
