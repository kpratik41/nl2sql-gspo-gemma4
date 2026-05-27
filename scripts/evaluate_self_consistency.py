#!/usr/bin/env python3
"""Evaluate execution-result self-consistency over pass@k candidates.

For each checkpoint, this script loads 16 pass@k candidates per example plus
the checkpoint's temperature-0 inference output. It executes candidate SQL,
clusters candidates by BIRD execution result, selects a consensus prediction,
and reports BIRD execution accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nl2sql_gspo.sql_utils import bird_execute_sql, bird_get_gold_rows, bird_result_match


RowsKey = Tuple[str, Optional[frozenset], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passk-dir-template",
        default="outputs/passk/maskfix_ckpt-{ckpt}_old-dev-schema-tool_full1534_temp1p2_tp1_ctx45k",
    )
    parser.add_argument(
        "--checkpoint-dir-template",
        default=(
            "outputs/training/train-6601-schema-bare-tool/gemma-4-E4B-it/"
            "grpo_deepspeed_p15500_c8000_g16_t1p2_bs4_ga8_lr2e-6_"
            "e4b_bare_lr2e6_maskfix_20260526_044450/checkpoint-{ckpt}"
        ),
    )
    parser.add_argument("--database-dir", default="/home/ec2-user/nl2sql-gspo-gemma4/databases/dev_databases")
    parser.add_argument("--ckpts", default="0,20,40,60,80,100,120")
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--eval-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", default="outputs/analysis/maskfix_self_consistency")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rows_key(executed: bool, rows: Optional[List[Tuple[Any, ...]]], error: str) -> RowsKey:
    if not executed:
        return ("error", None, error or "execution failed")
    return ("ok", rows_to_set(rows), "")


def rows_to_set(rows: Optional[List[Tuple[Any, ...]]]) -> frozenset:
    if not rows:
        return frozenset()
    converted: List[Tuple[Any, ...]] = []
    for row in rows:
        try:
            converted.append(tuple(row))
        except Exception:
            converted.append((repr(row),))
    try:
        return frozenset(converted)
    except TypeError:
        return frozenset(tuple(repr(cell) for cell in row) for row in converted)


def execute_unique_sqls(
    sql_jobs: Iterable[Tuple[str, str]],
    database_dir: str,
    timeout_s: float,
    workers: int,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    unique_jobs = sorted(set(sql_jobs))
    print(f"[self-consistency] executing {len(unique_jobs)} unique predicted SQL/db pairs")

    def run(job: Tuple[str, str]) -> Tuple[Tuple[str, str], Dict[str, Any]]:
        db_id, sql = job
        if not sql.strip():
            executed, rows, error = False, None, "empty sql"
        else:
            executed, rows, error = bird_execute_sql(
                sql=sql,
                db_id=db_id,
                database_dir=database_dir,
                timeout_s=timeout_s,
            )
        key = rows_key(executed, rows, error)
        return job, {
            "executed": executed,
            "rows": rows,
            "error": error,
            "cluster_key": key,
        }

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(16, workers)) as pool:
        futures = [pool.submit(run, job) for job in unique_jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            job, result = future.result()
            results[job] = result
            if completed == 1 or completed == len(futures) or completed % 1000 == 0:
                print(f"[self-consistency] executed {completed}/{len(futures)}")
    return results


def execute_gold_rows(
    gold_jobs: Iterable[Tuple[str, str]],
    database_dir: str,
    timeout_s: float,
    workers: int,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    unique_jobs = sorted(set(gold_jobs))
    print(f"[self-consistency] executing {len(unique_jobs)} unique gold SQL/db pairs")

    def run(job: Tuple[str, str]) -> Tuple[Tuple[str, str], Dict[str, Any]]:
        db_id, sql = job
        executed, row_set, error = bird_get_gold_rows(sql, db_id, database_dir, timeout_s=timeout_s)
        return job, {"executed": executed, "row_set": row_set, "error": error}

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(16, workers)) as pool:
        futures = [pool.submit(run, job) for job in unique_jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            job, result = future.result()
            results[job] = result
            if completed == 1 or completed == len(futures) or completed % 1000 == 0:
                print(f"[self-consistency] gold executed {completed}/{len(futures)}")
    return results


def load_checkpoint_inputs(
    ckpt: int,
    passk_dir: Path,
    checkpoint_dir: Path,
    num_generations: int,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]], Dict[int, str]]:
    passk_path = passk_dir / "passk_candidates.jsonl"
    temp0_path = checkpoint_dir / "eval_results.jsonl"
    if not passk_path.exists():
        raise FileNotFoundError(passk_path)
    if not temp0_path.exists():
        raise FileNotFoundError(temp0_path)

    by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    gold_by_idx: Dict[int, str] = {}
    for row in read_jsonl(passk_path):
        idx = int(row["idx"])
        by_idx[idx].append(row)
        gold_by_idx.setdefault(idx, row.get("gold_sql", ""))

    temp0_by_idx = {int(row["idx"]): row for row in read_jsonl(temp0_path)}

    passk_indices = set(by_idx)
    temp0_indices = set(temp0_by_idx)
    if passk_indices != temp0_indices:
        missing_temp0 = sorted(passk_indices - temp0_indices)[:10]
        missing_passk = sorted(temp0_indices - passk_indices)[:10]
        raise ValueError(
            f"checkpoint-{ckpt}: passk/temp0 idx mismatch; "
            f"missing_temp0={missing_temp0} missing_passk={missing_passk}"
        )

    for idx, candidates in by_idx.items():
        candidates.sort(key=lambda item: int(item.get("sample_id", 0)))
        sample_ids = [int(item.get("sample_id", -1)) for item in candidates]
        expected = list(range(num_generations))
        if sample_ids != expected:
            raise ValueError(f"checkpoint-{ckpt} idx={idx}: expected sample_ids={expected}, got={sample_ids}")
        temp0 = temp0_by_idx[idx]
        if candidates[0].get("db_id") != temp0.get("db_id"):
            raise ValueError(
                f"checkpoint-{ckpt} idx={idx}: db mismatch passk={candidates[0].get('db_id')} "
                f"temp0={temp0.get('db_id')}"
            )
        if candidates[0].get("gold_sql", "") != temp0.get("gold_sql", ""):
            raise ValueError(f"checkpoint-{ckpt} idx={idx}: gold_sql mismatch")

    return by_idx, temp0_by_idx, gold_by_idx


def sorted_winning_groups(groups: Dict[RowsKey, List[Dict[str, Any]]]) -> List[Tuple[RowsKey, List[Dict[str, Any]]]]:
    def sample_sort_value(candidate: Dict[str, Any]) -> int:
        sample_id = candidate.get("sample_id", 10**9)
        if sample_id == "temp0":
            return 10**8
        try:
            return int(sample_id)
        except Exception:
            return 10**9

    return sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            0 if item[0][0] == "ok" else 1,
            min(sample_sort_value(candidate) for candidate in item[1]),
        ),
    )


def is_valid_cluster_key(key: RowsKey) -> bool:
    status, row_set, _error = key
    return status == "ok" and row_set is not None and len(row_set) > 0


def valid_sorted_winning_groups(groups: Dict[RowsKey, List[Dict[str, Any]]]) -> List[Tuple[RowsKey, List[Dict[str, Any]]]]:
    return sorted_winning_groups({key: group for key, group in groups.items() if is_valid_cluster_key(key)})


def evaluate_checkpoint(
    ckpt: int,
    passk_dir: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print(f"[self-consistency] checkpoint-{ckpt}")
    by_idx, temp0_by_idx, gold_by_idx = load_checkpoint_inputs(
        ckpt=ckpt,
        passk_dir=passk_dir,
        checkpoint_dir=checkpoint_dir,
        num_generations=args.num_generations,
    )

    pred_jobs: List[Tuple[str, str]] = []
    gold_jobs: List[Tuple[str, str]] = []
    for idx, candidates in by_idx.items():
        db_id = candidates[0].get("db_id", "")
        gold_jobs.append((db_id, gold_by_idx[idx]))
        for candidate in candidates:
            pred_jobs.append((db_id, candidate.get("pred_sql", "")))
        pred_jobs.append((db_id, temp0_by_idx[idx].get("pred_sql", "")))

    pred_exec = execute_unique_sqls(
        pred_jobs,
        database_dir=args.database_dir,
        timeout_s=args.eval_timeout,
        workers=args.workers,
    )
    gold_exec = execute_gold_rows(
        gold_jobs,
        database_dir=args.database_dir,
        timeout_s=args.eval_timeout,
        workers=args.workers,
    )

    per_example: List[Dict[str, Any]] = []
    option1_correct = 0
    option2_correct = 0
    option1_ties = 0
    option1_tie_temp0_matches = 0
    option2_ties = 0
    option1_cluster_sizes: Counter[int] = Counter()
    option2_cluster_sizes: Counter[int] = Counter()

    for idx in sorted(by_idx):
        candidates = by_idx[idx]
        temp0 = temp0_by_idx[idx]
        db_id = candidates[0].get("db_id", "")
        gold_sql = gold_by_idx[idx]
        gold_result = gold_exec[(db_id, gold_sql)]

        groups16: Dict[RowsKey, List[Dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            exec_result = pred_exec[(db_id, candidate.get("pred_sql", ""))]
            groups16[exec_result["cluster_key"]].append(candidate)

        temp0_exec = pred_exec[(db_id, temp0.get("pred_sql", ""))]
        temp0_key = temp0_exec["cluster_key"]
        winners16 = valid_sorted_winning_groups(groups16)
        max16 = len(winners16[0][1]) if winners16 else 0
        tied16 = [(key, group) for key, group in winners16 if len(group) == max16]

        if not tied16:
            selected1 = {
                "source": "no_valid_cluster",
                "sample_id": None,
                "pred_sql": "",
                "exec": {"executed": False, "rows": None, "error": "no valid non-empty executed cluster"},
                "cluster_key": ("error", None, "no valid non-empty executed cluster"),
            }
        elif len(tied16) > 1:
            option1_ties += 1
            if is_valid_cluster_key(temp0_key):
                selected1 = {
                    "source": "temp0_tie_break",
                    "sample_id": None,
                    "pred_sql": temp0.get("pred_sql", ""),
                    "exec": temp0_exec,
                    "cluster_key": temp0_key,
                }
                if any(key == temp0_key for key, _ in tied16):
                    option1_tie_temp0_matches += 1
            else:
                key, group = tied16[0]
                candidate = group[0]
                selected1 = {
                    "source": "passk16_tie_fallback_temp0_invalid",
                    "sample_id": int(candidate.get("sample_id", -1)),
                    "pred_sql": candidate.get("pred_sql", ""),
                    "exec": pred_exec[(db_id, candidate.get("pred_sql", ""))],
                    "cluster_key": key,
                }
        else:
            key, group = tied16[0]
            candidate = group[0]
            selected1 = {
                "source": "passk16_majority",
                "sample_id": int(candidate.get("sample_id", -1)),
                "pred_sql": candidate.get("pred_sql", ""),
                "exec": pred_exec[(db_id, candidate.get("pred_sql", ""))],
                "cluster_key": key,
            }

        option1_res = int(
            selected1["exec"]["executed"]
            and gold_result["executed"]
            and bird_result_match(selected1["exec"]["rows"], gold_result["row_set"])
        )
        option1_correct += option1_res
        option1_cluster_sizes[max16] += 1

        groups17: Dict[RowsKey, List[Dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            exec_result = pred_exec[(db_id, candidate.get("pred_sql", ""))]
            groups17[exec_result["cluster_key"]].append(candidate)
        groups17[temp0_key].append({**temp0, "sample_id": "temp0"})
        winners17 = valid_sorted_winning_groups(groups17)
        max17 = len(winners17[0][1]) if winners17 else 0
        tied17 = [(key, group) for key, group in winners17 if len(group) == max17]
        if len(tied17) > 1:
            option2_ties += 1
        if tied17:
            key17, group17 = tied17[0]
            chosen17 = group17[0]
            selected2_exec = pred_exec[(db_id, chosen17.get("pred_sql", ""))]
            option2_source = "temp0" if chosen17.get("sample_id") == "temp0" else "passk"
            option2_sample_id = chosen17.get("sample_id")
        else:
            selected2_exec = {"executed": False, "rows": None, "error": "no valid non-empty executed cluster"}
            option2_source = "no_valid_cluster"
            option2_sample_id = None
        option2_res = int(
            selected2_exec["executed"]
            and gold_result["executed"]
            and bird_result_match(selected2_exec["rows"], gold_result["row_set"])
        )
        option2_correct += option2_res
        option2_cluster_sizes[max17] += 1

        per_example.append(
            {
                "ckpt": ckpt,
                "idx": idx,
                "db_id": db_id,
                "gold_executed": gold_result["executed"],
                "gold_error": gold_result["error"],
                "num_groups_16": len(groups16),
                "num_valid_groups_16": len(winners16),
                "largest_group_16": max16,
                "option1_tied": len(tied16) > 1,
                "option1_source": selected1["source"],
                "option1_sample_id": selected1["sample_id"],
                "option1_correct": option1_res,
                "option1_pred_executed": selected1["exec"]["executed"],
                "option1_pred_error": selected1["exec"]["error"],
                "num_groups_17": len(groups17),
                "num_valid_groups_17": len(winners17),
                "largest_group_17": max17,
                "option2_tied": len(tied17) > 1,
                "option2_source": option2_source,
                "option2_sample_id": option2_sample_id,
                "option2_correct": option2_res,
                "option2_pred_executed": selected2_exec["executed"],
                "option2_pred_error": selected2_exec["error"],
                "temp0_correct": int(temp0.get("res", 0)),
                "temp0_pred_executed": bool(temp0.get("pred_executed", False)),
                "passk_candidate_correct_count": sum(int(candidate.get("correct", 0)) for candidate in candidates),
            }
        )

    total = len(per_example)
    summary = {
        "ckpt": ckpt,
        "examples": total,
        "option1_correct": option1_correct,
        "option1_accuracy": 100.0 * option1_correct / max(1, total),
        "option1_ties": option1_ties,
        "option1_tie_temp0_matches_tied_group": option1_tie_temp0_matches,
        "option1_largest_group_size_counts": dict(sorted(option1_cluster_sizes.items())),
        "option2_correct": option2_correct,
        "option2_accuracy": 100.0 * option2_correct / max(1, total),
        "option2_ties_after_adding_temp0": option2_ties,
        "option2_largest_group_size_counts": dict(sorted(option2_cluster_sizes.items())),
        "temp0_correct": sum(int(row.get("res", 0)) for row in temp0_by_idx.values()),
        "temp0_accuracy": 100.0 * sum(int(row.get("res", 0)) for row in temp0_by_idx.values()) / max(1, total),
    }
    return summary, per_example


def write_outputs(output_dir: Path, summaries: List[Dict[str, Any]], per_examples: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "self_consistency_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    write_jsonl(output_dir / "self_consistency_per_example.jsonl", per_examples)

    csv_path = output_dir / "self_consistency_summary.csv"
    fields = [
        "ckpt",
        "examples",
        "option1_correct",
        "option1_accuracy",
        "option1_ties",
        "option1_tie_temp0_matches_tied_group",
        "option2_correct",
        "option2_accuracy",
        "option2_ties_after_adding_temp0",
        "temp0_correct",
        "temp0_accuracy",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})

    md_lines = [
        "# Self-Consistency Summary",
        "",
        "| ckpt | option1 acc | option1 correct | ties | option2 acc | option2 correct | ties after +temp0 | temp0 acc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        md_lines.append(
            f"| {row['ckpt']} | `{row['option1_accuracy']:.2f}%` | "
            f"`{row['option1_correct']} / {row['examples']}` | `{row['option1_ties']}` | "
            f"`{row['option2_accuracy']:.2f}%` | `{row['option2_correct']} / {row['examples']}` | "
            f"`{row['option2_ties_after_adding_temp0']}` | `{row['temp0_accuracy']:.2f}%` |"
        )
    (output_dir / "self_consistency_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ckpts = [int(value) for value in args.ckpts.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is non-empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[self-consistency] starting")
    print(f"[self-consistency] ckpts={ckpts}")
    print(f"[self-consistency] workers={max(16, args.workers)} timeout={args.eval_timeout}")
    started = time.time()

    summaries: List[Dict[str, Any]] = []
    all_per_examples: List[Dict[str, Any]] = []
    for ckpt in ckpts:
        passk_dir = Path(args.passk_dir_template.format(ckpt=ckpt))
        checkpoint_dir = Path(args.checkpoint_dir_template.format(ckpt=ckpt))
        summary, per_examples = evaluate_checkpoint(ckpt, passk_dir, checkpoint_dir, args)
        summaries.append(summary)
        all_per_examples.extend(per_examples)
        write_outputs(output_dir, summaries, all_per_examples)
        print(
            f"[self-consistency] ckpt-{ckpt} option1={summary['option1_accuracy']:.2f}% "
            f"option2={summary['option2_accuracy']:.2f}% temp0={summary['temp0_accuracy']:.2f}%"
        )

    print(f"[self-consistency] complete in {time.time() - started:.1f}s")
    print(f"[self-consistency] wrote {output_dir / 'self_consistency_summary.md'}")


if __name__ == "__main__":
    main()
