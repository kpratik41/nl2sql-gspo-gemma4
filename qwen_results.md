# Qwen Results

## Qwen3.6-35B-A3B - BIRD Dev Temp 0

Run:

- Model: `Qwen/Qwen3.6-35B-A3B`
- Data: `outputs/old-dev-schema-tool-unpatched.jsonl`
- Output: `outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.6-35B-A3B/full1534_tp2_shards4_temp0_openai_tool_qwen_native_required_retryfix`
- Inference: async vLLM server, `TP=2`, `SHARDS=4`, `temperature=0.0`
- Tool calling: Qwen native structured tool calls, `tool_choice_policy=required_first`, `empty_tool_retries=1`

Overall BIRD EX:

- Correct: `1028 / 1534`
- Accuracy: `67.01%`

SQL execution:

- Pred SQL extracted: `1528 / 1534`
- Pred SQL missing: `6`
- Pred SQL executed: `1525 / 1534`
- Extracted SQL execution failures: `3`
- Both pred and gold executed: `1524`

Tool usage:

- Total tool calls: `3958`
- Avg tool calls overall: `2.58`
- Avg tool calls on EX-correct examples: `2.22`
- Avg tool calls on EX-incorrect examples: `3.30`
- Tool counts: `sqlite_query=2975`, `sqlite_peek=509`, `bm25_search_sqlite=474`
- Rejected parsed tool calls: `110`, all parsed as invalid `scratch_pad`
- Forced-final examples: `86`
- Empty-tool retries: `271`

By shard:

| Shard | Rows | Correct | Accuracy | Avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| `shard_0` | 383 | 251 | 65.54% | 2.50 |
| `shard_1` | 384 | 263 | 68.49% | 2.47 |
| `shard_2` | 383 | 261 | 68.15% | 2.71 |
| `shard_3` | 384 | 253 | 65.89% | 2.65 |

By difficulty:

| Difficulty | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| Simple | 672 | 925 | 72.65% |
| Moderate | 273 | 464 | 58.84% |
| Challenging | 83 | 145 | 57.24% |

By database:

| Database | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| `california_schools` | 57 | 89 | 64.04% |
| `card_games` | 116 | 191 | 60.73% |
| `codebase_community` | 133 | 186 | 71.51% |
| `debit_card_specializing` | 42 | 64 | 65.62% |
| `european_football_2` | 88 | 129 | 68.22% |
| `financial` | 69 | 106 | 65.09% |
| `formula_1` | 105 | 174 | 60.34% |
| `student_club` | 130 | 158 | 82.28% |
| `superhero` | 112 | 129 | 86.82% |
| `thrombosis_prediction` | 81 | 163 | 49.69% |
| `toxicology` | 95 | 145 | 65.52% |

## Qwen3.6-27B Dense - BIRD Dev Temp 0

Run:

- Model: `Qwen/Qwen3.6-27B`
- Data: `outputs/old-dev-schema-tool-unpatched.jsonl`
- Output: `outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.6-27B/full1534_tp2_shards4_temp0_openai_tool_qwen_native_required_retryfix`
- Inference: async vLLM server, `TP=2`, `SHARDS=4`, `temperature=0.0`
- Tool calling: Qwen native structured tool calls, `tool_choice_policy=required_first`, `empty_tool_retries=1`

Overall BIRD EX:

- Correct: `1063 / 1534`
- Accuracy: `69.30%`

SQL execution:

- Pred SQL extracted: `1525 / 1534`
- Pred SQL missing: `9`
- Pred SQL executed: `1524 / 1534`
- Extracted SQL execution failures: `1`
- Total missing or execution-failed pred SQL: `10`
- Both pred and gold executed: `1523`

Tool usage:

- Total tool calls: `4438`
- Avg tool calls overall: `2.89`
- Avg tool calls on EX-correct examples: `2.55`
- Avg tool calls on EX-incorrect examples: `3.67`
- Avg tool calls where pred SQL executed: `2.87`
- Avg tool calls where pred SQL was missing or did not execute: `6.30`
- Tool counts: `sqlite_query=2707`, `bm25_search_sqlite=1292`, `sqlite_peek=439`
- Rejected parsed tool calls: `3`
- Forced-final examples: `71`
- Empty-tool retries: `206`

By shard:

| Shard | Rows | Correct | Accuracy | Avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| `shard_0` | 383 | 260 | 67.89% | 2.76 |
| `shard_1` | 384 | 267 | 69.53% | 2.84 |
| `shard_2` | 383 | 275 | 71.80% | 2.86 |
| `shard_3` | 384 | 261 | 67.97% | 3.12 |

By difficulty:

| Difficulty | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| Simple | 683 | 925 | 73.84% |
| Moderate | 296 | 464 | 63.79% |
| Challenging | 84 | 145 | 57.93% |

By database:

| Database | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| `california_schools` | 56 | 89 | 62.92% |
| `card_games` | 122 | 191 | 63.87% |
| `codebase_community` | 129 | 186 | 69.35% |
| `debit_card_specializing` | 41 | 64 | 64.06% |
| `european_football_2` | 91 | 129 | 70.54% |
| `financial` | 72 | 106 | 67.92% |
| `formula_1` | 111 | 174 | 63.79% |
| `student_club` | 131 | 158 | 82.91% |
| `superhero` | 118 | 129 | 91.47% |
| `thrombosis_prediction` | 90 | 163 | 55.21% |
| `toxicology` | 102 | 145 | 70.34% |

## California Schools Comparison

### Qwen3.6-27B Dense - California Schools Temp 0

Run:

- Model: `Qwen/Qwen3.6-27B`
- Data: `outputs/old-dev-schema-tool-unpatched-california_schools.jsonl`
- Output: `outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.6-27B/california_schools_tp2_shards4_temp0_openai_tool_qwen_native_required_retryfix`
- Inference: async vLLM server, `TP=2`, `SHARDS=4`, `temperature=0.0`
- Tool calling: Qwen native structured tool calls, `tool_choice_policy=required_first`, `empty_tool_retries=1`

Overall California-schools EX:

- Correct: `56 / 89`
- Accuracy: `62.92%`

SQL execution:

- Pred SQL extracted: `88 / 89`
- Pred SQL missing: `1`
- Pred SQL executed: `88 / 89`
- Extracted SQL execution failures: `0`

Tool usage:

- Total tool calls: `331`
- Avg tool calls overall: `3.72`
- Avg tool calls on EX-correct examples: `3.14`
- Avg tool calls on EX-incorrect examples: `4.70`
- Tool counts: `sqlite_query=211`, `bm25_search_sqlite=79`, `sqlite_peek=41`
- Rejected parsed tool calls: `0`
- Forced-final examples: `8`
- Empty-tool retries: `2`

By shard:

| Shard | Rows | Correct | Accuracy | Avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| `shard_0` | 22 | 18 | 81.82% | 3.14 |
| `shard_1` | 22 | 9 | 40.91% | 4.59 |
| `shard_2` | 22 | 14 | 63.64% | 3.45 |
| `shard_3` | 23 | 15 | 65.22% | 3.70 |

By difficulty:

| Difficulty | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| Simple | 37 | 54 | 68.52% |
| Moderate | 18 | 30 | 60.00% |
| Challenging | 1 | 5 | 20.00% |

### Qwen3.8-27B Dense - California Schools Temp 0

Run:

- Model: `Qwen/Qwen3.8-27B`
- Data: `outputs/old-dev-schema-tool-unpatched-california_schools.jsonl`
- Output: `outputs/inference/dev/old-dev-schema-tool-unpatched/Qwen3.8-27B/california_schools_tp2_shards4_temp0_openai_tool_qwen3_coder`
- Inference: async vLLM server, `TP=2`, `SHARDS=4`, `temperature=0.0`
- Tool calling: Qwen native structured tool calls, `tool_call_parser=qwen3_coder`, `tool_choice_policy=required_first`, `empty_tool_retries=1`

Overall California-schools EX:

- Correct: `57 / 89`
- Accuracy: `64.04%`

SQL execution:

- Pred SQL extracted: `89 / 89`
- Pred SQL missing: `0`
- Pred SQL executed: `89 / 89`
- Extracted SQL execution failures: `0`

Tool usage:

- Total tool calls: `270`
- Avg tool calls overall: `3.03`
- Avg tool calls on EX-correct examples: `2.39`
- Avg tool calls on EX-incorrect examples: `4.19`
- Tool counts: `sqlite_query=202`, `bm25_search_sqlite=61`, `sqlite_peek=7`
- Rejected parsed tool calls: `0`
- Forced-final examples: `10`

By shard:

| Shard | Rows | Correct | Accuracy | Avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| `shard_0` | 22 | 17 | 77.27% | 2.32 |
| `shard_1` | 22 | 12 | 54.55% | 3.41 |
| `shard_2` | 22 | 14 | 63.64% | 3.14 |
| `shard_3` | 23 | 14 | 60.87% | 3.26 |

By difficulty:

| Difficulty | Correct | Rows | Accuracy |
| --- | ---: | ---: | ---: |
| Simple | 40 | 54 | 74.07% |
| Moderate | 15 | 30 | 50.00% |
| Challenging | 2 | 5 | 40.00% |

### California Schools Model Comparison

| Model | Correct | Rows | Accuracy | Avg tools | Avg tools on correct | Pred SQL executed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.6-35B-A3B` | 57 | 89 | 64.04% | 3.28 | 2.82 | 89 / 89 |
| `Qwen3.6-27B` | 56 | 89 | 62.92% | 3.72 | 3.14 | 88 / 89 |
| `Qwen3.8-27B` | 57 | 89 | 64.04% | 3.03 | 2.39 | 89 / 89 |

Takeaway: on `california_schools`, `Qwen3.8-27B` matches the `Qwen3.6-35B-A3B` MoE accuracy while using fewer tool calls on average. It is one example ahead of `Qwen3.6-27B` and extracted/executed SQL for every sample.


---

## Qwen3.8-27B RL — BIRD Dev Results

All numbers on BIRD **dev** (1534 questions), execution accuracy (EX).
Run from the `qwen-3p8` branch, async in-process vLLM engine, tp=2 with 4 data-parallel
shards on 8x RTX PRO 6000 Blackwell, ctx 43k, 16 concurrent requests per shard.

- RL run: `outputs/qwen-rl1/checkpoint-{0,5,...,35}` (checkpoint-0 = the Qwen3.8-27B base model)
- Eval data: `outputs/qwen-old-dev-schema-tool-unpatched.jsonl` (3 few-shots per prompt, tool dialect)
- Databases: `databases/dev_databases`, gold diff `data/bird_dev_data/raw/bird_dev_unpatched.json`

Every RL checkpoint ships without `preprocessor_config.json` / `video_preprocessor_config.json`
(the trainer saves only text-side files, but Qwen3.8-27B is multimodal). The runner backfills
both from the base snapshot; without them the engine dies on `OSError: Can't load image processor`.

### 1. Temperature-0 checkpoint series

| ckpt | EX | correct | simple | moderate | challenging |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (base) | 71.06% | 1090 | 76.22 | 63.79 | 61.38 |
| 5 | 71.32% | 1094 | 76.65 | 63.58 | 62.07 |
| 10 | 72.88% | 1118 | 77.73 | 67.03 | 60.69 |
| 15 | 73.27% | 1124 | 77.73 | 68.97 | 58.62 |
| 20 | 73.79% | 1132 | 78.59 | 68.32 | 60.69 |
| 25 | 73.66% | 1130 | 77.95 | 69.18 | 60.69 |
| 30 | 73.21% | 1123 | 78.05 | 68.32 | 57.93 |
| **35** | **74.05%** | **1136** | 79.57 | 67.24 | 60.69 |
| 40 | 73.60% | 1129 | 79.14 | 65.73 | 63.45 |
| 45 | 73.14% | 1122 | 79.03 | **64.44** | 63.45 |

RL is worth **+2.99 points** over the base model, but **all of it arrives by step 20**. The total
then looks flat - steps 20-45 span 14 questions, inside the +/-2.2 point binomial interval of a
single 1534-question eval - but the flat total is masking a real compositional shift.

**`moderate` falls strictly monotonically for five consecutive checkpoints:**

    ckpt        25     30     35     40     45
    moderate  69.18  68.32  67.24  65.73  64.44     -4.74 points = -22 questions (n=464)

Five consecutive decreases has a ~0.8% chance under a random-ordering null, and 22 questions is
~2.2 sigma on that stratum. The total stays flat only because the loss is cancelled elsewhere:

    moderate    -22 questions
    simple      +10
    challenging  +4
                ----
    net          -8      (1130 at step 25 -> 1122 at step 45)

So the run is not plateauing, it is **degrading on mid-complexity questions** while making small
gains on easy and hard ones. That matches the sharpening evidence in section 6: unanimous
54.6% -> 60.2%, never-solved 17.8% -> 18.7%, pass@16 -14 questions at step 40.

### 2. pass@16

| ckpt | temp | pass@1 | pass@4 | pass@8 | **pass@16** | candidates correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.2 | 72.32 | 78.99 | 80.85 | **82.20** | 17750/24544 |
| 20 | 0.8 | 73.21 | 78.29 | 79.86 | **81.10** | 17968/24544 |
| 35 | 1.2 | 71.81 | 78.34 | 80.50 | **82.14** | 17625/24544 |
| 40 | 1.2 | 72.84 | 78.34 | 80.02 | **81.29** | 17879/24544 |

Lower temperature raises pass@1 (+0.89) and lowers pass@16 (-1.10) — it moves probability mass
inward from both tails. 15 further RL steps (20 -> 35) left pass@16 **unchanged** (one question),
then step 40 **lost 14 questions of coverage** (1261 -> 1247).

Note ckpt-40 at temperature 1.2 reproduces almost exactly what ckpt-20 looked like at temperature
0.8 (pass@1 72.84 vs 73.21, pass@16 81.29 vs 81.10). **Continued RL is acting like a temperature
reduction baked into the weights.**

### 3. Self-consistency

Majority vote over execution result sets; empty/non-executing results discarded.
Option 1 = pure vote over the 16 samples. Option 2 = vote with the temp-0 prediction as an extra vote.

| ckpt | temp | temp-0 | SC (option 1) | SC (option 2) | SC gain over temp-0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.2 | 73.79% | 74.71% (1146) | **74.77% (1147)** | **+0.98** |
| 20 | 0.8 | 73.79% | 74.71% (1146) | 74.71% (1146) | +0.92 |
| **35** | 1.2 | **74.05%** | 73.79% (1132) | 73.92% (1134) | **-0.13** |
| 40 | 1.2 | 73.60% | 74.25% (1139) | **74.38% (1141)** | **+0.78** |

**Temperature is not an SC lever**: 1.2 and 0.8 both give exactly 74.71%, despite pass@1 differing
by +0.89 and pass@16 by -1.10. Voting is blind to both tails.

**SC gain is not a trend**: +0.98 (step 20), -0.13 (step 35), +0.78 (step 40). Step 35 is an
outlier, not the start of a decline. SC beats greedy at three of four measurements.

#### How SC actually accumulates its score (ckpt-40)

| candidate correct-count | questions | SC got right | rate |
| --- | ---: | ---: | ---: |
| 0 (never solved) | 287 | 0 | 0.0% |
| 1-7 (minority) | 116 | 9 | 7.8% |
| 8 (tie) | 10 | 9 | 90.0% |
| 9-15 (majority) | 197 | 197 | 100.0% |
| 16 (unanimous) | 924 | 924 | 100.0% |

Majority and unanimous convert at exactly 100%, so `(unanimous + majority) / n` is a **floor**, not
a ceiling — 73.08% here. The remaining +1.17 comes from ties and from minority-correct questions
that SC still wins. It wins them because the vote runs over **usable** candidates only
(non-executing and empty-result candidates are discarded first, leaving 1.46 valid groups per
question on average), so a correct cluster of fewer than 8 of 16 can still be the plurality.
Predicting SC from the correct-count distribution alone understates it.

### 4. Best configuration

| rank | configuration | EX | inference cost |
| ---: | --- | ---: | --- |
| 1 | **ckpt-20 + SC (option 2), temp 1.2** | **74.77%** (1147) | 16 rollouts + vote |
| 2 | ckpt-20 + SC (option 1) | 74.71% (1146) | 16 rollouts + vote |
| 3 | ckpt-40 + SC (option 2) | 74.38% (1141) | 16 rollouts + vote |
| 4 | ckpt-35 temp-0 | 74.05% (1136) | 1 greedy pass |
| 5 | ckpt-20 temp-0 | 73.79% (1132) | 1 greedy pass |

0.72 points for 16x the compute. ckpt-35 temp-0 is the best single-pass number; ckpt-20 + SC is the
best overall and beats the next-best SC (ckpt-40) by 6 questions.

### 5. Post-hoc interventions — all measured, none adopted

Everything below was tested on saved generations or as a controlled A/B. None is in use.

| intervention | result | why it failed |
| --- | ---: | --- |
| Self-consistency | +0.98 (ckpt-20), **-0.13 (ckpt-35)** | see above |
| **LLM-as-a-verifier rerank** | **-6.13** | discards the vote prior |
| Column-shape gating | +0.52 ceiling | 84.8% of shape errors unwinnable |
| Prompt rule ("rank/top N -> include the quantity") | **0.00** | model ignored it entirely |

#### LLM-as-a-verifier (ckpt-20, temp 1.2)

Pairwise pivot tournament, same checkpoint as judge, soft verdict from the logprob mass on the
verdict token, each pair judged twice with candidates swapped. 521 contested questions,
949 comparisons, 1898 forward passes, all verdicts parsed cleanly.

Self-consistency **74.64%** -> verifier **68.51%** (**-6.13**, 37 gained / 131 lost).

Not an implementation artefact: the champion won 65% of comparisons (no polarity inversion),
mean score 0.554, 0 unparsed verdicts. The cause is that **sample support is a strong correctness
prior** — P(correct | support=1) = 5.6%, P(correct | support>=12) = 81.4% — and the verifier
discards it, flipping from picks with mean support 10.83/16 to 2.29/16 while being near chance
itself. Hybrid rules (override only on small support gaps, or only on exact ties) were swept
offline; all collapse back to SC (best +2 questions).

An earlier version of this experiment reported an exact 0.00 delta. That run was invalid: both
chat templates open a reasoning span before the answer (gemma `<|channel>thought`, qwen `<think>`),
so a 4-token judge was reading letters out of the thought stream, never a verdict. The fixed judge
lets the model reason and reads the distribution at the token it actually answered with.

*(The verifier harness reconstructs SC from the raw candidates and lands on 1145/1534 = 74.64%,
one question below the official 1146; the gemma reconstruction matched exactly. The 1-question gap
does not affect the conclusion.)*

#### Column-shape analysis (ckpt-20 SC selections)

66 of 1532 SC picks (4.31%) return the wrong number of columns — **17.0% of all SC errors**
(25 extra columns, 41 missing). Of the 57 with support > 8:

- **57.9% have provably correct query logic** — right rows, different column list
- 45.5% of them were selected **unanimously** (16/16); mean support 13.17/16
- only 12.1% have the correct answer in the second cluster; **84.8% are unwinnable**

Ceiling for any shape-detection fix: **+8 questions (+0.52 pts)**.

Root cause is that BIRD gold encodes output conventions the question text does not state — it
appends identifiers ("list the expenses" -> `expense_id, expense_description`), materialises the
sort key on rank questions, and returns every variant of a field. In ~11 cases gold is itself
deficient, answering only half a two-part question.

#### Prompt rule A/B (ckpt-35, temp-0)

One line swapped in the system prompt, everything else byte-identical (verified: same few-shots,
same schema, same `gold_sql`/`question`/`tools`):

    - Return exactly the columns requested in the question, and no extras.
    + - Return the columns the question asks for.
    + - If the question asks to rank, or to list the top N by some quantity, include that quantity as a column.

Result: **74.05% -> 74.05%**, net 0 (7 gained, 7 lost). 181 questions (11.8%) produced different
SQL, but **all 10 targeted rank questions produced byte-identical SQL** — at temp-0 with a changed
prompt, meaning the model ignored the instruction completely on exactly the questions it targeted.

#### Rule mining over the 389 SC errors

Each candidate rule tested by **re-executing the repaired SQL**, not by pattern-matching:

| rule | verified repairs |
| --- | ---: |
| add `DISTINCT` | 2 (0.5%) |
| `CAST` for integer division | ~1 |
| IDs-vs-names hard rule | 5 (1.3%), and 39.3% wrong on its own trigger |
| any column-convention rule | 37 (9.5%) — absolute ceiling |

**65.3% of errors share no rows at all with gold** — a different query, unreachable by output
conventions. Standard BIRD prompt hygiene is saturated on this model: gold has `DISTINCT` where the
prediction does not in 63 cases (16.2%), but adding it repairs **2**. Co-occurrence with failure is
not causation; every number above is a verified repair count.

### 6. Sample diversity

| | ckpt-20 (t=1.2) | ckpt-35 (t=1.2) | ckpt-40 (t=1.2) |
| --- | ---: | ---: | ---: |
| never solved (0/16) | 17.8% | 17.9% (274) | **18.7% (287)** |
| minority correct (voting loses these) | 9.3% | 9.4% (144) | **7.6% (116)** |
| tie (8/16) | — | 0.6% (9) | 0.7% (10) |
| majority correct | — | 17.2% (264) | 12.8% (197) |
| unanimous (16/16) | 54.6% | 55.0% (843) | **60.2% (924)** |
| distinct SQL strings / question | — | 11.60 (median 13) | 11.57 (median 13) |
| distinct **result sets** / contested question | 2.82 | — | — |
| avg tool calls / candidate | — | 2.685 | 2.458 |
| rollouts hitting the 8-round cap | — | 841 | 253 |

Three monotone trends across steps 20 -> 35 -> 40: unanimous **rises** (54.6 -> 55.0 -> 60.2),
minority-correct **falls** (9.3 -> 9.4 -> 7.6), never-solved **rises** (17.8 -> 17.9 -> 18.7).
The policy is progressively concentrating, and it is paying for that concentration in coverage:
13 questions that step 35 could still reach with at least one of 16 samples, step 40 cannot reach
at all.

High syntactic diversity, low semantic diversity: 16 textually different queries collapse to a
handful of distinct answers. Majority-vote ceiling for ckpt-35 is (843+264)/1534 = **72.16%** plus
ties, which is why SC lands near 74%.

### 7. Reproduction

    # temp-0 eval for one checkpoint
    RUN_TAG=rl1_ckpt35 MODEL_PATH=<abs>/outputs/qwen-rl1/checkpoint-35 \
    INPUT_FILE=<abs>/outputs/qwen-old-dev-schema-tool-unpatched.jsonl \
    GPUS=0,1,2,3,4,5,6,7 TP=2 NUM_SHARDS=4 TEMPERATURE=0.0 MAX_MODEL_LEN=43000 \
    OUTPUT_DIR=<ckpt>/temp0_olddev_schema_tool_unpatched_vllm_async_tp2_shards4_ctx43k \
    bash scripts/qwen/run_qwen38_temp0_ckpt_eval.sh

    # pass@16 (defaults are already temp 1.2 / tp=2 / 4 shards / 16 generations / ctx 43k)
    MODEL_PATH=... INPUT_FILE=... BASE_OUT=<ckpt>/passk16_..._temp1p2_tp2_shards4_ctx43k \
    bash scripts/qwen/run_qwen38_27b_passk16_async_engine_tp2_shards4_temp1p2.sh

    # self-consistency over the merged candidates (CPU only)
    PASSK_DIR=<passk>/merged TEMP0_DIR=<ckpt>/temp0_... OUTPUT_DIR=<ckpt>/sc_passk16_... \
    bash scripts/qwen/queue_qwen38_sc_after_passk16.sh

Wall clock on 8 GPUs: temp-0 ~19 min, pass@16 ~2h03m (24,544 rollouts), SC a few minutes.

### 8. Conclusions

1. **RL delivered +2.99 points** (71.06 -> 74.05), and **all of it arrived by step 20**. Steps
   20-45 show no gain in the total, and `moderate` declines monotonically across the last five
   checkpoints (-22 questions). **This run should be stopped, not extended.**
2. **The run has plateaued or regressed on every axis measured**: temp-0 flat across five
   consecutive checkpoints (20-40), pass@16 flat from 20 to 35 then -14 questions at step 40,
   and SC never again reaching its step-20 value.
3. **Post-hoc selection is exhausted.** Four interventions measured between -6.13 and +0.52. The
   residual 25% error is dominated by systematic misreadings shared across the whole sample pool,
   not by picking the wrong candidate from a pool that contains the right one.
4. **17.9% of questions are never solved by any of 16 samples.** That is the real ceiling, and only
   training-side changes can move it.
