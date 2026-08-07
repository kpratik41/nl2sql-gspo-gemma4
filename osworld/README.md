# OSWorld 2.0 integration files

These two files are our additions to [OSWorld 2.0](https://github.com/xlang-ai/OSWorld-V2).
They live here because the upstream checkout is 1.5GB with its own git history and
is `.gitignore`d — but the paths below mirror where they belong inside it.

| File | Destination in OSWorld-V2 |
|---|---|
| `mm_agents/qwen36_agent.py` | `mm_agents/qwen36_agent.py` |
| `scripts/python/run_multienv_qwen36.py` | `scripts/python/run_multienv_qwen36.py` |

## Install

```bash
git clone https://github.com/xlang-ai/OSWorld-V2.git
cd OSWorld-V2 && uv sync                      # needs Python 3.12+
cp -r ../osworld/mm_agents/*  mm_agents/
cp -r ../osworld/scripts/python/* scripts/python/
```

## What they are

**`qwen36_agent.py`** — subclasses the shipped `Qwen35VLAgent` rather than forking
its 680 lines, so upstream prompt/folding/pyautogui fixes keep flowing through.
Two overrides: inject `chat_template_kwargs={"enable_thinking": ...}` (the base
class forwards `payload["extra_body"]` but never sets it), and strip
`<think>...</think>` before parsing, because the base tool-call extractor scans
the whole response and would otherwise execute a hypothetical call the model
drafted and then rejected.

**`run_multienv_qwen36.py`** — derived from the generic `run_multienv.py`, with
`PromptAgent` swapped for `Qwen36Agent` and mapped to that agent's real
signature. It also re-exposes `--checkpoint_eval_mode` / `--checkpoint_steps`:
`lib_run_single.py` already implements inline checkpoint evaluation, but the
generic runner never surfaced the flags that gate it.

## Run

```bash
.venv/bin/python scripts/python/run_multienv_qwen36.py \
    --provider_name aws --headless \
    --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --base_url http://<gpu-host>:8000/v1 \
    --max_steps 500 \
    --checkpoint_eval_mode inline --checkpoint_steps 50,100,150,250
```

`--max_steps` defaults to **15** upstream; the paper's primary metric is **500**.

See [../OSWORLD.md](../OSWORLD.md) for the full setup, infrastructure notes, and
cost estimates.
