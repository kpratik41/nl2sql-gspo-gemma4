"""Command line interface: ``synthmark <command>``.

The commands mirror the three things a team needs to do -- mint a key, generate
watermarked text, and check text -- without writing any Python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _add_key_source(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("key source")
    g.add_argument("--key-file", help="Path to a key JSON written by 'synthmark keygen'.")
    g.add_argument("--key-id", help="Derive the key from the master secret using this label.")
    g.add_argument(
        "--master-secret-env",
        default="SYNTHMARK_MASTER_SECRET",
        help="Environment variable holding the master secret (default: SYNTHMARK_MASTER_SECRET).",
    )


def _need(module: str, distribution: str):
    """Import an optional sibling distribution, or exit explaining which to install."""
    from importlib import import_module

    try:
        return import_module(module)
    except ModuleNotFoundError:
        sys.exit(
            f"error: this command needs the {distribution} package.\n"
            f"  pip install {distribution}"
        )


def _master_secret(env_var: str) -> bytes:
    from .keys import load_master_secret

    return load_master_secret(env_var)


def _resolve_key(args):
    from .keys import WatermarkKey, derive_key

    if args.key_file:
        return WatermarkKey.load(args.key_file)
    if args.key_id:
        secret = os.environ.get(args.master_secret_env)
        if not secret:
            sys.exit(
                f"error: {args.master_secret_env} is not set.\n"
                f'  export {args.master_secret_env}="$(openssl rand -hex 32)"'
            )
        return derive_key(secret, args.key_id)
    sys.exit("error: supply either --key-file or --key-id")


# ------------------------------------------------------------------- commands


def cmd_keygen(args) -> None:
    from .keys import derive_key, generate_key

    if args.from_master:
        secret = os.environ.get(args.master_secret_env)
        if not secret:
            sys.exit(f"error: {args.master_secret_env} is not set")
        key = derive_key(secret, args.key_id, depth=args.depth, ngram_len=args.ngram_len, notes=args.notes)
        source = f"derived from ${args.master_secret_env}"
    else:
        key = generate_key(args.key_id, depth=args.depth, ngram_len=args.ngram_len, notes=args.notes)
        source = "freshly generated"

    if args.out:
        path = key.save(args.out, overwrite=args.overwrite)
        print(f"wrote {path} (mode 0600, {source})")
    else:
        print(json.dumps(key.to_dict(), indent=2))
    print(json.dumps(key.public_summary(), indent=2), file=sys.stderr)
    if not args.from_master:
        print(
            "\nThis key file is the only copy of the secret. Anyone holding it can both\n"
            "detect and forge this watermark. Consider --from-master instead, which lets\n"
            "you re-derive the key from a single escrowed secret.",
            file=sys.stderr,
        )


def cmd_generate(args) -> None:
    WatermarkedLM = _need("synthmark_eval.generate", "synthmark-eval").WatermarkedLM

    key = None if args.no_watermark else _resolve_key(args)
    lm = WatermarkedLM(args.model, device_map=args.device)

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        prompts = [l for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    else:
        prompts = [l for l in sys.stdin.read().splitlines() if l.strip()]

    rendered = lm.chat_prompts(prompts) if not args.raw_prompt else prompts
    out = lm.generate(
        rendered, key=key, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        batch_size=args.batch_size, seed=args.seed,
    )
    if args.json:
        print(json.dumps({
            "watermarked": out.watermarked,
            "key_fingerprint": out.key_fingerprint,
            "tokens_per_second": out.tokens_per_second,
            "outputs": [{"prompt": p, "text": t, "num_new_tokens": n}
                        for p, t, n in zip(prompts, out.texts, out.num_new_tokens)],
        }, indent=2))
    else:
        for t in out.texts:
            print(t)
            print("-" * 70)
        print(f"[{'watermarked' if out.watermarked else 'not watermarked'}; "
              f"{out.tokens_per_second:.1f} tok/s]", file=sys.stderr)


def cmd_detect(args) -> None:
    from transformers import AutoTokenizer

    _d = _need("synthmark_detect", "synthmark-detect")
    Calibration, Detector = _d.Calibration, _d.Detector

    key = _resolve_key(args)
    if args.text:
        texts = [args.text]
    elif args.file:
        texts = [Path(args.file).read_text()]
    else:
        texts = [sys.stdin.read()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    det = Detector(key, tokenizer, device=args.device)
    cal = Calibration.load(args.calibration) if args.calibration else None

    out = []
    for t in texts:
        r = det.detect(t, method=args.method, calibration=cal, target_fpr=args.target_fpr)
        out.append(r.to_dict())
        if not args.json:
            verdict = (
                "WATERMARK DETECTED" if r.is_watermarked
                else "no watermark detected" if r.is_watermarked is not None
                else ("WATERMARK DETECTED" if (r.p_value or 1) < args.target_fpr else "no watermark detected")
            )
            print(f"{verdict}")
            print(f"  score              {r.score:.5f}")
            print(f"  tokens scored      {r.num_tokens_scored}")
            if r.z_score is not None:
                print(f"  z-score            {r.z_score:+.3f}")
                print(f"  p-value            {r.p_value:.3e}")
            if r.empirical_p_value is not None:
                print(f"  empirical p-value  {r.empirical_p_value:.4f}")
            if r.threshold is not None:
                print(f"  threshold @{args.target_fpr:.1%} FPR {r.threshold:.5f}")
            print(f"  key                {r.key_id} (fp {r.key_fingerprint})")
            if r.num_tokens_scored < 40:
                print("  WARNING: very short text; detection is unreliable below ~40 scored tokens.")
    if args.json:
        print(json.dumps(out, indent=2))


def cmd_calibrate(args) -> None:
    from transformers import AutoTokenizer

    Detector = _need("synthmark_detect", "synthmark-detect").Detector

    key = _resolve_key(args)
    texts = json.loads(Path(args.texts).read_text())
    if isinstance(texts, dict):
        texts = texts.get("texts", [])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    det = Detector(key, tokenizer, device=args.device)
    cal = det.calibrate(texts, method=args.method)
    path = cal.save(args.out)
    total = sum(len(v) for v in cal.scores.values())
    print(f"wrote {path}: {total} null scores across {len(cal.buckets)} length buckets")
    for b in cal.buckets:
        n = len(cal.scores.get(b, []))
        if n:
            print(f"  >={b:4d} tokens: n={n:4d}  threshold@1%FPR={cal.threshold(b + 1, 0.01):.5f}")


def cmd_serve(args) -> None:
    import uvicorn

    Calibration = _need("synthmark_detect", "synthmark-detect").Calibration
    from .registry import KeyRegistry
    _s = _need("synthmark_detect.serve", "synthmark-detect[serve]")
    build_app, build_served = _s.build_app, _s.build_served

    master = _master_secret(args.master_secret_env)
    registry = KeyRegistry.load(args.registry)

    calibrations = {}
    if args.calibration_dir:
        for entry in registry:
            path = Path(args.calibration_dir) / f"{entry.key_id.replace('/', '_')}.json"
            if path.exists():
                calibrations[entry.key_id] = Calibration.load(path)

    served = build_served(registry, master, calibrations=calibrations, device=args.device)
    print(
        f"serving {len(served)} keys over {len(registry.tokenizers())} tokenizers:",
        file=sys.stderr,
    )
    for sk in served.values():
        print(
            f"  {sk.entry.key_id:40s} {sk.entry.status:8s} fp={sk.key.fingerprint} "
            f"model={sk.entry.model_id}",
            file=sys.stderr,
        )
    uvicorn.run(build_app(served), host=args.host, port=args.port)


def cmd_registry(args) -> None:
    """Stamp or verify the fingerprints that bind a registry to a master secret."""
    from .registry import KeyRegistry, RegistryError

    master = _master_secret(args.master_secret_env)
    registry = KeyRegistry.load(args.registry)

    if args.stamp:
        stamped = registry.stamp_fingerprints(master)
        stamped.save(args.registry)
        print(f"stamped {len(stamped)} fingerprints into {args.registry}", file=sys.stderr)
        registry = stamped

    failures = 0
    for entry in registry:
        try:
            key = entry.resolve(master, verify=True)
            state = "ok" if entry.fingerprint else "ok (unstamped)"
            print(f"{state:15s} {entry.key_id:40s} fp={key.fingerprint} model={entry.model_id}")
        except RegistryError as exc:
            failures += 1
            print(f"{'MISMATCH':15s} {entry.key_id:40s} {exc}")
    if failures:
        sys.exit(f"\n{failures} key(s) do not match the master secret; refusing to pass.")


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synthmark",
        description="SynthID-Text watermarking, detection and evaluation for open-weight LLMs.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    k = sub.add_parser("keygen", help="Mint a watermark key.")
    k.add_argument("key_id", help="Non-secret label, e.g. 'markets-research/v1'.")
    k.add_argument("--out", help="Where to write the key JSON (mode 0600). Omit to print.")
    k.add_argument("--from-master", action="store_true",
                   help="Derive deterministically from the master secret instead of drawing a new key.")
    k.add_argument("--master-secret-env", default="SYNTHMARK_MASTER_SECRET")
    k.add_argument("--depth", type=int, default=30, help="Watermarking depth (default 30).")
    k.add_argument("--ngram-len", type=int, default=5)
    k.add_argument("--notes", default="")
    k.add_argument("--overwrite", action="store_true")
    k.set_defaults(func=cmd_keygen)

    g = sub.add_parser("generate", help="Generate watermarked text.")
    g.add_argument("--model", default="google/gemma-4-E4B-it")
    g.add_argument("--prompt")
    g.add_argument("--prompt-file", help="One prompt per line. Reads stdin if omitted.")
    g.add_argument("--raw-prompt", action="store_true", help="Do not apply the chat template.")
    g.add_argument("--no-watermark", action="store_true", help="Generate an unwatermarked baseline.")
    g.add_argument("--max-new-tokens", type=int, default=256)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=64)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--seed", type=int)
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--json", action="store_true")
    _add_key_source(g)
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("detect", help="Check text for a watermark.")
    d.add_argument("--model", default="google/gemma-4-E4B-it", help="Tokenizer to use; must match the generating model.")
    d.add_argument("--text")
    d.add_argument("--file", help="Read the candidate text from a file. Reads stdin if omitted.")
    d.add_argument("--method", choices=["mean", "weighted_mean"], default="mean")
    d.add_argument("--calibration", help="Calibration JSON for empirical thresholds.")
    d.add_argument("--target-fpr", type=float, default=0.01)
    d.add_argument("--device", default="cpu")
    d.add_argument("--json", action="store_true")
    _add_key_source(d)
    d.set_defaults(func=cmd_detect)

    c = sub.add_parser("calibrate", help="Build empirical thresholds from non-watermarked text.")
    c.add_argument("--texts", required=True, help="JSON list of non-watermarked texts.")
    c.add_argument("--out", required=True)
    c.add_argument("--model", default="google/gemma-4-E4B-it")
    c.add_argument("--method", choices=["mean", "weighted_mean"], default="mean")
    c.add_argument("--device", default="cpu")
    _add_key_source(c)
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("serve", help="Run the multi-model detection HTTP service.")
    s.add_argument("--registry", required=True, help="Path to the key registry JSON.")
    s.add_argument("--master-secret-env", default="SYNTHMARK_MASTER_SECRET")
    s.add_argument("--calibration-dir", help="Dir of <key_id with / as _>.json calibrations.")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_serve)

    r = sub.add_parser("registry", help="Check a key registry against the master secret.")
    r.add_argument("--registry", required=True)
    r.add_argument("--master-secret-env", default="SYNTHMARK_MASTER_SECRET")
    r.add_argument("--stamp", action="store_true",
                   help="Fill in missing fingerprints and rewrite the file.")
    r.set_defaults(func=cmd_registry)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
