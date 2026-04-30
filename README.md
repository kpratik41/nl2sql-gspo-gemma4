# NL2SQL GSPO Gemma4

## Training

Start the vLLM server first:

```bash
bash scripts/launch_vllm.sh
```

Then launch training:

```bash
bash scripts/launch_train.sh
```

To resume training from a saved checkpoint:

```bash
RESUME_FROM_CHECKPOINT=outputs/gemma4_31b_gspo_bird/checkpoint-100 bash scripts/launch_train.sh
```

## Data Generation

Generate BM25-based few-shot files for both train and dev splits:

```bash
c:/Users/kprat/OneDrive/Desktop/Project/nl2sql-gspo-gemma4/.venv/Scripts/python.exe scripts/data_generation/few_shot_bm25.py --top-n 5
```

This writes:

- `data/bird_train_data/raw/train-6601-few-shot.jsonl`
- `data/bird_dev_data/raw/dev_20251106-few-shot.json`

The train few-shot file excludes examples from the same `db_id` as the source record.
