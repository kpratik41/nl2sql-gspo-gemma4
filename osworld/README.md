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

## 7-task smoke subset

`evaluation_examples/test_v2_smoke7.json` — copy it alongside `test_v2.json`, then:

```bash
.venv/bin/python scripts/python/run_multienv_qwen36.py \
    --provider_name aws --headless --max_steps 500 \
    --base_url http://<gpu-host>:8000/v1 \
    --test_all_meta_path evaluation_examples/test_v2_smoke7.json \
    --checkpoint_eval_mode inline --checkpoint_steps 50,100,150,250
```

Selected from the 55 tasks that survive every blocker filter (no `proxy=True`,
no human-in-the-loop, no streaming/dynamic-environment flakiness, no GitLab —
which is the one site you must self-host — and no `intermediate_eval_safe=False`,
so all seven can produce a step-budget curve). Then chosen for diversity:
7 distinct primary apps, all 7 non-blocked challenge phenomena covered, capped
at 2 creative/CAD "ceiling probes" and 1 video-tutorial dependency so the batch
returns usable signal instead of seven zeros.

| Task | Apps | Phenomena | Why it is here |
|---|---|---|---|
| 010 | writer, calc, pdfviewer, thunderbird | conflict-disambig, cross-source, implicit-state, tutorial | classic multi-app office+mail workflow |
| 019 | studio.streamview | multimodal-editing, visual-spatial | the one video-tutorial-following task |
| 040 | excel | cross-source | shortest brief; pure spreadsheet reasoning |
| 046 | chrome, mailhub, vaultbank, teamchat, file_manager, gimp | 5 phenomena | exercises the self-hosted web stack end-to-end |
| 049 | wps | visual-spatial | slide editing against a source PDF |
| 059 | geogebra | multimodal-editing, visual-spatial | precise geometric construction from an image |
| 064 | vscode, libero | implicit-state, multimodal-editing | code-repair ceiling probe |

Requires `WEBSITE_HOST_SUFFIX` to be set (e.g. the team-hosted `web.hku.icu`) —
`desktop_env/controllers/website.py` raises at import if it is missing.
