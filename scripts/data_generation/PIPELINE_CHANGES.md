# Data-pipeline changes, August 2026

Everything the train/dev data generation pipeline gained on `qwen-3p8`, written
down so it can be ported to `consensus`, `consensus-sft`, and `submssion2`
(the BIRD leaderboard submission branch — note the remote spells it with one
`s` and no hyphen).

As of 2026-08-16 all three of those branches carry **both** bugs below, byte
for byte. The port is a straight application of the same two edits.

---

## Why any of this matters

SQLite is dynamically typed. A column declared `TEXT` can hold `'0.25'`,
`'4200'`, `'nan'` — and 197 columns across the BIRD train and dev databases do.
Under TEXT affinity every text value sorts above every number, so

```sql
WHERE Sentiment_Polarity > 0        -- column is TEXT holding '1.0', '-0.5', 'nan'
```

is not a numeric comparison. It matches nearly every row, returns a plausible
answer, and never raises. Nothing downstream — not execution, not the reward —
signals the mistake.

---

## Change 1 — type labels come from `typeof()`, not from sniffed values

`schema_build.py::classify_column`

**Before.** The declared type was consulted for numeric/date keywords, then
*overridden* by sniffing sampled values: any column whose first 20 values all
parsed as floats was labelled `NUMERIC`.

That fabricated a numeric label for 160 train and 37 dev columns that BIRD's own
`train_tables.json`/`dev_tables.json` and the SQLite DDL both declare `text`.
Verified: for all 4337 columns the two schema sources agree with the DDL
exactly, 0 disagreements — the schema was never wrong, the pipeline overrode it.

**After.** Layered, with the order chosen by what each signal is authoritative
for:

1. **Declared `DATE`/`TIME` keyword.** First, because SQLite has no date storage
   class — dates live as text, and `typeof()` would flatten all 195 date columns
   to `TEXT` and destroy the signal the prompt's `strftime`/`date` rules need.
2. **Date-looking samples.** Catches 24 date columns declared `TEXT`.
3. **`typeof()` storage class.** The truthful answer for everything else, and
   why it outranks sniffing.
4. **Declared numeric keyword**, only as a fallback for the 38 columns that are
   empty or all-NULL so `typeof()` has nothing to report.

Value sniffing (`_is_numeric`) no longer decides anything. `_looks_like_date`
stays, because SQLite has no date storage class for `typeof()` to consult.

**Impact, measured:**

| | old label | new label | columns |
|---|---|---|---|
| train, pre-fix **with** sampling | NUMERIC | TEXT | 160 / 3539 |
| train, pre-fix under `--no-stats` | TEXT | DATE | 18 / 3539 |
| dev, pre-fix **with** sampling | NUMERIC | TEXT | 37 / 798 |
| dev, pre-fix under `--no-stats` | TEXT | DATE | 6 / 798 |

Each comparison collapses to a single transition, which is the sign the change
does one well-defined thing rather than churning labels.

## Change 2 — sampling no longer sits behind `include_stats`

Same file, `build_mschema_from_db`. The `LIMIT 20` read was wrapped in
`if include_stats:`, so `--no-stats` silently disabled *date inference* as well
as stats. That is how the bare train build lost its `DATE` labels on 18 columns.

The read is now unconditional. It is a `LIMIT 20` scan used only to pick the
rendered label, which bare builds need exactly as much as full ones.

## Change 3 — text-affinity defect scanner (new file)

`scripts/data_generation/find_text_affinity_defects.py`

Finds golds the affinity bug demonstrably breaks: it executes each gold against
a `CAST(... AS REAL)`-corrected variant and keeps the rows whose answers differ.

- Covers the three **unsafe** numeric contexts: comparison vs a literal,
  `MIN`/`MAX`, and `ORDER BY`.
- Leaves `AVG`, `SUM` and arithmetic alone — SQLite coerces text for those, so
  they are already correct.
- Normalizes `'1.0'` against `1.0`, so a storage-class formatting change is not
  mistaken for a behaviour change.
- `--output` writes the input minus the defective rows.

**Findings:** 27 / 6601 train rows, 6 / 1534 old-dev rows, 5 / 1534
dev-20251106 rows. Repeat offenders are `formula_1.results.fastestLapSpeed`,
`thrombosis_prediction.Laboratory.DNA`, `card_games.cards.cardKingdomFoilId`.

Golds are **never rewritten** — the rule on this project is to drop provably
wrong training rows, not to correct ground truth.

## Change 4 — text-affinity guidance in the system prompt

`prompts.py`, in the "Dates and numeric scales" section:

```
- A TEXT column may hold numeric-looking values. Comparisons and ORDER BY on it
  use string ordering, not numeric ordering, and return wrong rows without erroring.
- Check such a column with sqlite_peek, and use CAST(col AS REAL) only if its
  values are numeric.
```

Two deliberate choices:

- **Conditional, not a standing order to peek.** Prompt variant B established
  that raising tool-eagerness against a fixed round budget costs more accuracy
  than it buys.
- **Names `sqlite_peek`, not the rendered example values.** The train file is
  built `--no-stats` and carries no example values, so guidance keyed to them
  would be inert for every training prompt while working in dev.

Applied via `_insert_affinity_guidance`, called inside `_to_qwen_template` — so
**on the Qwen side only**. The Gemma templates must stay byte-identical to what
is baked into `outputs/old-dev-schema-tool-*.jsonl`, because
`build_qwen_eval_data.py` asserts on that exact text before converting. Porting
to a Gemma branch means either regenerating those files or applying the same
insertion on the Gemma side there.

---

## Porting checklist

For each of `consensus`, `consensus-sft`, `submssion2`:

1. Cherry-pick `5f016d8` — changes 1 and 2, plus the defect scanner.
2. Decide on change 4. On a Gemma branch it cannot go in via the Qwen
   transform; insert at `_AFFINITY_ANCHOR` in the base template and regenerate
   any file whose system prompt is asserted on elsewhere.
3. Regenerate that branch's train/dev artifacts. Schema build is the slow step
   (~35 min for train, 69 DBs); the tool-dataset step is a few minutes.
4. Re-run the defect scanner on the regenerated train file and drop the rows.

`submssion2` needs extra care: the BIRD team runs that pipeline against the
**held-out test databases**, which were never inspected here. Change 1 is
data-independent and safe. The defect scanner is not applicable — there are no
gold SQLs to validate against, and it must not run as part of submission.

## Reproducing the artifacts on `qwen-3p8`

```bash
python scripts/data_generation/schema_build.py \
  --split train --n-examples -1 --output outputs/qwen-train-6601-schema-bare.jsonl \
  --messages-only --no-fewshots --no-stats --no-nullability --no-comments

python scripts/data_generation/build_tool_dataset.py \
  --input  outputs/qwen-train-6601-schema-bare.jsonl \
  --output outputs/qwen-train-6601-schema-bare-tool.jsonl \
  --prompt-template default_qwen

python scripts/data_generation/find_text_affinity_defects.py \
  --input  outputs/qwen-train-6601-schema-bare-tool.jsonl \
  --report outputs/train-text-affinity-defects.json \
  --output outputs/qwen-train-6574-schema-bare-tool.jsonl

python scripts/qwen/build_qwen_eval_data.py \
  --input  outputs/old-dev-schema-tool-unpatched.jsonl \
  --output outputs/qwen-old-dev-schema-tool-unpatched.jsonl --overwrite
```

Dev stays at 1534 rows throughout; no existing Gemma-era `outputs/*.jsonl` is
ever modified.

## Known-wrong rows not covered by the scanner

`talkingdata.events_relevant.timestamp` is declared `DATETIME` but stores 64-bit
hashes (range ±9.22e18), not times — BIRD anonymised the `*_relevant` tables.
Train row 5710 (6601-file indexing) asks for "the top 2 oldest events" and
orders by that hash:

```
gold answer (ordering by the hash)     : ['online shopping navigation', '"online shopping']
true answer (ordering by real datetime): ['Finance', 'Pay']
```

Four rows also tie at the minimum, so its `LIMIT 2` is not deterministic. The
scanner cannot see this — `typeof()` says integer and the column genuinely *is*
integer; the semantics are what is wrong. Handled as an explicit exclusion.
