"""Re-score the fluency comparison with a different, independent judge model.

Fluency has no ground truth, so it is measured as perplexity. The model under
test cannot score its own output: SynthID picks among near-tied tokens by
g-value rather than raw probability, so watermarked text sits slightly off that
model's own argmax path whether or not a reader would notice. Self-perplexity
therefore shows a penalty that is not a quality penalty.

A *different* model has a different argmax path and does not inherit that bias.
The strongest control is a model from an unrelated family -- different
architecture, different tokenizer, different training data -- because it shares
none of the generator's idiosyncrasies.

This script recomputes only the fluency table and patches it into an existing
quality-results JSON, leaving diversity and task-accuracy untouched (neither
depends on the judge).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import DATA, RESULTS, banner, save_json

from synthmark import WatermarkedLM, paired_bootstrap_diff

HIGH_ENTROPY = ("creative", "open_ended", "financial")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus.json"))
    ap.add_argument("--judge-model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--judge-device", default="cuda:1")
    ap.add_argument("--patch", default=str(RESULTS / "03_quality.json"),
                    help="Quality-results JSON to update in place.")
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    records = corpus["records"]
    generator_model = corpus["meta"]["model"]

    banner(f"Fluency re-scored | generator {generator_model} | judge {args.judge_model}")
    if args.judge_model.split("/")[0].lower() == generator_model.split("/")[0].lower():
        print("NOTE: judge and generator come from the same model family; an unrelated")
        print("      architecture would be a stronger control.")
    else:
        print("Judge is from an unrelated model family: different architecture, tokenizer")
        print("and training data, so it shares none of the generator's idiosyncrasies.")

    paired = defaultdict(dict)
    for r in records:
        paired[(r["suite"], r["prompt_index"], r["replicate"])][r["condition"]] = r

    judge = WatermarkedLM(args.judge_model, device_map=args.judge_device)
    print(f"judge loaded: {type(judge.model).__name__}\n")

    print(f"{'suite':12s} {'n':>5s} {'PPL wm':>9s} {'PPL plain':>10s} {'diff':>8s} {'95% CI':>20s}")
    by_suite = {}
    for suite in HIGH_ENTROPY:
        rows = [v for k, v in paired.items() if k[0] == suite and len(v) == 2][: args.max_pairs]
        if not rows:
            continue
        wm = np.array(judge.perplexity([r["watermarked"]["text"] for r in rows],
                                       batch_size=args.batch_size))
        un = np.array(judge.perplexity([r["unwatermarked"]["text"] for r in rows],
                                       batch_size=args.batch_size))
        d = paired_bootstrap_diff(wm, un)
        by_suite[suite] = d
        print(f"{suite:12s} {d['n_pairs']:5d} {d['mean_a']:9.3f} {d['mean_b']:10.3f} "
              f"{d['mean_diff']:+8.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
              f"{'' if d['contains_zero'] else '  SIGNIFICANT'}")

    patch_path = Path(args.patch)
    data = json.loads(patch_path.read_text()) if patch_path.exists() else {"meta": {}}
    data["perplexity_independent_judge"] = {
        "judge_model": args.judge_model,
        "judge_family_independent": args.judge_model.split("/")[0].lower()
        != generator_model.split("/")[0].lower(),
        "generator_model": generator_model,
        "by_suite": by_suite,
    }
    save_json(data, patch_path)


if __name__ == "__main__":
    main()
