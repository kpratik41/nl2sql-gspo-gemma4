"""How much watermark survives editing, paraphrasing and dilution?

This is the study that decides what the detector may honestly be used for.  A
watermark that only survives verbatim copy-paste is a provenance signal for
honest users; one that survives paraphrase would be an anti-abuse control.  They
support very different policies, so the difference has to be measured rather
than assumed.

Method
------
Every attack is applied to **both** arms -- watermarked and unwatermarked text --
and the AUC is recomputed on the attacked pair.  Attacking only the positives
would confound "the watermark was destroyed" with "the attack changed the score
distribution", and would overstate the damage.

Attacks are grouped by what they cost the attacker:

*Free*        lowercasing, removing markdown, truncation -- no loss of meaning.
*Cheap*       random word deletion / swapping / substitution -- meaning degrades
              visibly, so a real adversary would not go far up this curve.
*Expensive*   LLM paraphrase and round-trip translation -- meaning preserved,
              text fully re-sampled.  This is the real adversary.
*Structural*  dilution with human text -- not an attack at all, but the common
              case of a part-machine, part-human document.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from common import DATA, DEMO_MASTER_SECRET, MODEL_ID, PRIMARY_KEY_ID, RESULTS, banner, save_json

from synthmark_eval import Detector, WatermarkedLM, derive_key, evaluate_detection
from synthmark_eval import attacks as A


def score_all(det, texts, method="mean", batch_size=32):
    s, n = [], []
    for i in range(0, len(texts), batch_size):
        a, b = det.score(texts[i : i + batch_size], method=method)
        s.append(a)
        n.append(b)
    return np.concatenate(s), np.concatenate(n)


def evaluate(det, wm_texts, un_texts, name, strength, table, method="mean"):
    s_wm, n_wm = score_all(det, wm_texts, method)
    s_un, _ = score_all(det, un_texts, method)
    ok = ~np.isnan(s_wm)
    if ok.sum() < 5:
        return None
    m = evaluate_detection(s_wm, s_un, tokens_scored=n_wm)
    table[f"{name}/{strength}"] = {**m.to_dict(), "attack": name, "strength": strength}
    print(f"  {name:22s} {strength:>7} {np.median(n_wm):6.0f} {m.auc:7.4f} "
          f"{m.tpr_at_fpr_1pct:8.3f} {m.mean_positive:8.4f}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus.json"))
    ap.add_argument("--human", default=str(DATA / "human_texts.json"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--n-texts", type=int, default=192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-llm-attacks", action="store_true")
    ap.add_argument("--n-attack-texts", type=int, default=64)
    ap.add_argument("--attack-max-tokens", type=int, default=400)
    args = ap.parse_args()

    banner("Robustness of detection to text modification")
    corpus = json.loads(open(args.corpus).read())
    records = corpus["records"]
    human_texts = json.loads(open(args.human).read())

    # Use the high-entropy suites: these carry the strongest watermark, so the
    # damage measured here is a best case for the defender.  Low-entropy text is
    # already weakly marked before any attack (see 02_detectability).
    suites = ("creative", "open_ended", "financial")
    wm = [r for r in records if r["condition"] == "watermarked" and r["suite"] in suites][: args.n_texts]
    un = [r for r in records if r["condition"] == "unwatermarked" and r["suite"] in suites][: args.n_texts]
    wm_texts = [r["text"] for r in wm]
    un_texts = [r["text"] for r in un]
    print(f"{len(wm_texts)} watermarked / {len(un_texts)} unwatermarked texts from {suites}")

    from transformers import AutoTokenizer

    key = derive_key(DEMO_MASTER_SECRET, PRIMARY_KEY_ID)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    det = Detector(key, tokenizer, device=args.device)

    table: dict = {}
    print(f"\n  {'attack':22s} {'level':>7s} {'n_tok':>6s} {'AUC':>7s} {'TPR@1%':>8s} {'mean+':>8s}")
    evaluate(det, wm_texts, un_texts, "none", "-", table)

    # ------------------------------------------------------------ free attacks
    banner("Free attacks (no loss of meaning)")
    print(f"  {'attack':22s} {'level':>7s} {'n_tok':>6s} {'AUC':>7s} {'TPR@1%':>8s} {'mean+':>8s}")
    for fn in (A.lowercase, A.strip_formatting):
        a = fn(wm_texts)
        b = fn(un_texts)
        evaluate(det, a.texts, b.texts, a.name, "-", table)
    for frac in (0.75, 0.5, 0.25, 0.125):
        a = A.truncate(wm_texts, frac)
        b = A.truncate(un_texts, frac)
        evaluate(det, a.texts, b.texts, "truncate", frac, table)

    # ----------------------------------------------------------- cheap attacks
    banner("Cheap attacks (meaning degrades)")
    print(f"  {'attack':22s} {'level':>7s} {'n_tok':>6s} {'AUC':>7s} {'TPR@1%':>8s} {'mean+':>8s}")
    for name, fn in (("delete_words", A.delete_words), ("swap_words", A.swap_words),
                     ("substitute_words", A.substitute_words)):
        for p in (0.05, 0.1, 0.2, 0.4):
            a = fn(wm_texts, p, seed=1)
            b = fn(un_texts, p, seed=1)
            evaluate(det, a.texts, b.texts, name, p, table)

    # ------------------------------------------------------- dilution (not an attack)
    banner("Dilution with human-written text")
    print(f"  {'attack':22s} {'level':>7s} {'n_tok':>6s} {'AUC':>7s} {'TPR@1%':>8s} {'mean+':>8s}")
    for frac in (0.75, 0.5, 0.25, 0.1):
        a = A.mix_with_human(wm_texts, human_texts, frac, seed=2)
        b = A.mix_with_human(un_texts, human_texts, frac, seed=2)
        evaluate(det, a.texts, b.texts, "mix_with_human", frac, table)

    # ------------------------------------------------------- expensive attacks
    if not args.skip_llm_attacks:
        banner("Semantic-preserving attacks (LLM paraphrase, round-trip translation)")
        lm = WatermarkedLM(args.model, device_map=args.device)

        def gen(prompts):
            rendered = lm.chat_prompts(list(prompts))
            # The attacker generates *without* a watermark: laundering the text
            # through a watermarked model would simply re-mark it.
            return lm.generate(
                rendered, key=None, max_new_tokens=args.attack_max_tokens, temperature=1.0,
                top_k=64, top_p=0.95, batch_size=32, seed=5,
            ).texts

        n_attack = min(args.n_attack_texts, len(wm_texts))
        print(f"  (using {n_attack} texts per arm; LLM attacks are the slow ones)")
        print(f"  {'attack':22s} {'level':>7s} {'n_tok':>6s} {'AUC':>7s} {'TPR@1%':>8s} {'mean+':>8s}")

        # Paraphrase rounds are chained rather than recomputed, so round 2 costs
        # one extra pass instead of two.
        cur_wm, cur_un = wm_texts[:n_attack], un_texts[:n_attack]
        para_samples = None
        for rounds in (1, 2):
            cur_wm = A.paraphrase(cur_wm, gen, rounds=1).texts
            cur_un = A.paraphrase(cur_un, gen, rounds=1).texts
            evaluate(det, cur_wm, cur_un, "paraphrase", rounds, table)
            if rounds == 1:
                para_samples = list(cur_wm[:3])

        a = A.round_trip_translate(wm_texts[:n_attack], gen, language="French")
        b = A.round_trip_translate(un_texts[:n_attack], gen, language="French")
        evaluate(det, a.texts, b.texts, "round_trip_translate", "fr", table)

        save_json(
            {"original": wm_texts[:3], "paraphrased_once": para_samples,
             "round_trip_translated": a.texts[:3]},
            RESULTS / "04_attack_samples.json",
        )

    save_json({"meta": {"model": args.model, "suites": list(suites),
                        "n_texts": len(wm_texts)}, "results": table},
              RESULTS / "04_robustness.json")


if __name__ == "__main__":
    main()
