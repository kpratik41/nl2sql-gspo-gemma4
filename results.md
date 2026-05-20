# Results

## BIRD Dev Tool Pass@K: Gemma 4 31B, Async vLLM, Temperature 0.8

Result folder:

```text
outputs/bird_dev_tool_passk16_vllm_async_temp08_limit1534
```

This run evaluated pass@k on the full BIRD dev set by sampling 16 independent tool-calling completions per example, executing each predicted SQL against the dev databases, and comparing execution results with the gold SQL.

### Run Configuration

| setting | value |
| --- | --- |
| model | `google/gemma-4-31B-it` |
| input file | `outputs/dev-20251106-schema-bare-tool.jsonl` |
| database dir | `databases/dev_databases` |
| difficulty file | `data/bird_dev_data/raw/dev_20251106.json` |
| examples | `1534` |
| generations per example | `16` |
| total candidates | `24544` |
| temperature | `0.8` |
| top_p | `1.0` |
| max_new_tokens | `8000` |
| max_tool_rounds | `8` |
| vLLM tensor parallel size | `8` |
| vLLM async concurrency | `16` |

### Output Artifacts

| file | description |
| --- | --- |
| `passk_candidates_raw.jsonl` | raw generated candidates before SQL execution scoring |
| `passk_candidates.jsonl` | generated candidates with execution status, gold SQL, and correctness |
| `passk_per_example.jsonl` | per-question aggregate over the 16 candidates |
| `passk_summary.json` | machine-readable summary |
| `passk_summary.md` | human-readable summary |

### Summary Metrics

| metric | value |
| --- | ---: |
| candidate accuracy | `67.62%` |
| correct candidates | `16597 / 24544` |
| estimated pass@1 | `67.62%` |
| prefix pass@1 | `67.14%` |
| estimated pass@16 | `74.19%` |
| prefix pass@16 | `74.19%` |
| predicted SQL executed | `24173` |
| predicted SQL execution failed | `371` |
| total tool calls | `38691` |
| average tool calls per candidate | `1.58` |

### Pass@K Table

| k | estimated pass@k | prefix pass@k |
| ---: | ---: | ---: |
| 1 | 67.62 | 67.14 |
| 2 | 70.29 | 69.88 |
| 3 | 71.37 | 70.80 |
| 4 | 71.98 | 71.58 |
| 5 | 72.39 | 72.43 |
| 6 | 72.69 | 72.75 |
| 7 | 72.93 | 72.82 |
| 8 | 73.14 | 72.95 |
| 9 | 73.31 | 73.14 |
| 10 | 73.47 | 73.53 |
| 11 | 73.61 | 73.66 |
| 12 | 73.73 | 73.79 |
| 13 | 73.86 | 73.86 |
| 14 | 73.97 | 73.99 |
| 15 | 74.08 | 74.12 |
| 16 | 74.19 | 74.19 |

### Stop Reasons

| stop reason | count |
| --- | ---: |
| finished | `24096` |
| max_tool_rounds | `429` |
| max_new_tokens | `19` |

### Timing

| phase | seconds | approximate |
| --- | ---: | ---: |
| generation | `21019.74` | `5.84 h` |
| evaluation | `449.02` | `7.48 min` |
| total | `21470.74` | `5.96 h` |

### Notes

- `candidate_accuracy` treats every sampled candidate independently.
- `estimated pass@k` uses the standard combinatorial pass@k estimate from the 16 sampled candidates for each example.
- `prefix pass@k` checks whether at least one of the first `k` candidates by sample id was correct.
- At `k=16`, estimated and prefix pass@k match because both use all 16 candidates.
- This run used the bare tool-calling dev file, so prompts included tool access but not the heavier full schema stats/few-shot variant.
