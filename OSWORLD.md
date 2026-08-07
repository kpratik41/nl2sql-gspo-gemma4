# Running OSWorld 2.0 with Qwen3.6-35B-A3B

Notes and local setup for evaluating this harness's model on
[OSWorld 2.0](https://arxiv.org/abs/2606.29537)
([code](https://github.com/xlang-ai/OSWorld-V2),
[site](https://osworld-v2.xlang.ai/)).

## The benchmark

| | |
|---|---|
| Tasks | **108** (confirmed in `evaluation_examples/test_v2.json` and the release manifest) |
| Environments | 31 self-hosted websites + desktop apps, Ubuntu at **1920×1080** |
| Scoring | partial credit, ~27.25 checkpoints/task (≈2,900 assertions) |
| Per-task scale | median **1.6h** human time; 69.6% over an hour; ~318 tool calls |
| Primary metric | binary completion at **500 steps** |
| SOTA | Claude Opus 4.8 — 20.6% binary / 54.8% partial. GPT-5.5 ~13–14%. Nothing above 10% on tasks over 137 min |

Tasks are also tagged by challenge phenomenon (task counts overlap):
cross-source reasoning 46, visual-spatial precision 45, implicit-state inference 43,
multi-item state tracking 43, conflict disambiguation 39, multimodal editing 30,
tutorial following 22, dynamic environment 10, streaming interaction 6,
human-in-the-loop 6.

## What the 108 tasks actually contain

Downloaded (`osworld-v2-2026.06.24`, 108 task classes) and profiled. Each task is
a Python class subclassing `BaseTask` with `instruction`, `setup()` (downloads
assets, launches apps via `SetupController`), and `evaluate()`.

| | |
|---|---|
| Instruction length | median **443** chars, max **3,932** — these are briefs, not one-liners |
| Distinct apps | **43** |
| Most common | chrome 45, wps 13, vscode 8, mailhub 6, shotcut 6, gimp 5, zotero 5 |
| `intermediate_eval_safe=False` | **33** tasks |
| Needs proxy | 8 tasks |
| References `HOST_SUFFIX` | 6 tasks (the self-hosted site suffix) |

Checkpoints are not inline weight dicts — they are `metric_name` +
`result_spec`/`expected_spec` blocks, with granular partial credit inside rule
lists (e.g. one task expects 9 specific renamed PDFs, each worth credit). Leaf
assertions across the suite total ~3,500, consistent with the paper's ~27.25
checkpoints/task.

**Two findings that change plans:**

**33 tasks set `intermediate_eval_safe=False`.** `lib_run_single.py` falls back
to final-only evaluation for those, so `--checkpoint_steps` yields a step-budget
curve for roughly 3/4 of the suite, not all of it. Still worth enabling; just do
not expect 108 curves.

**There is no browser-only subset.** 17 tasks list `chrome` as their only
`related_apps`, which looks like a cheap entry point for our Playwright harness.
It is not: task_024 is chrome-only yet says "All required documents are on the
desktop" and its evaluators use `vm_file`/`execute` to inspect VM filesystem
state; task_073 is chrome-only yet references TeamChat and Mail. `related_apps`
is unreliable as a capability filter. Every task assumes the desktop VM.

Example of the flavour (task_006, ~680-char instruction): review lab
applications in Thunderbird, cross-reference a LibreOffice Calc spreadsheet,
apply multi-clause eligibility rules, then save 9 CVs as PDFs into a Desktop
folder with a specific renaming convention. Three apps, cross-source reasoning,
and a filesystem-checked result.

## Architecture decision: don't port tasks in, plug the policy out

OSWorld owns the VM images, the environment abstraction, the checkpoint
evaluators, and the scoring. Reimplementing any of it means the numbers stop
being comparable to the leaderboard, which is the only reason to run it.

So [BrowserEnv](harness/env.py) is **not** the eval substrate — OSWorld 2.0 is a
full desktop OS and Playwright cannot drive it. Keep BrowserEnv for fast local
iteration; use OSWorld's `desktop_env/` for measurement.

What does carry over: the vLLM serving setup (their agent takes a `base_url`),
and the coordinate finding — their agent defaults to `coordinate_type="relative"`
with `x_scale = original_width / 999`, independently confirming what
[calibrate.py](scripts/calibrate.py) measured.

## Infrastructure: AWS provider, not Docker

**This GPU box cannot run the Docker provider.** Checked: no `/dev/kvm`, zero
CPUs exposing `vmx`/`svm`, running under Nitro. It is a `p5.48xlarge`, so nested
virtualization is unavailable and OSWorld's Docker provider (which needs KVM) is
out.

That is fine, and arguably better: use `--provider_name aws`. vLLM stays on this
box; environment VMs are separate EC2 instances; the agent talks to both over
HTTP. The release manifest pins `ami-01017272139e01feb` (us-east-1, 1920×1080).
Client VM security groups must allow ports **3000** and **8000** on top of the
standard OSWorld ports.

## The agent adapter

[`OSWorld-V2/mm_agents/qwen36_agent.py`](OSWorld-V2/mm_agents/qwen36_agent.py)
subclasses the shipped `Qwen35VLAgent` rather than forking its 680 lines, so
upstream prompt/folding/pyautogui fixes keep flowing through. Two overrides:

**1. Thinking control.** The base agent forwards `payload["extra_body"]` to the
OpenAI client but never sets it — that is the hook. We inject
`chat_template_kwargs={"enable_thinking": ...}`, which vLLM passes to the Jinja
template.

This is a genuine trade-off, not a formality:

- `True` (default) matches how the reported SOTA was produced (Opus with maximum
  thinking) and should help the constraint-tracking and state-recovery failures
  the paper highlights. Costs 10–30× more decode tokens per step.
- `False` gives ~0.5s steps. In our browser harness this was mandatory, but only
  because `max_tokens=1024` truncated the reasoning. OSWorld's agent uses 32768,
  so that specific failure does not apply here.

Measure both on a few tasks before committing to a full run.

**2. Reasoning stripped before parsing.** The base agent extracts tool calls with
`re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)` over the
*entire* response. With thinking on, a hypothetical call the model drafts and
then rejects mid-reasoning would be executed as if it were the decision. The
adapter strips `<think>...</think>` first.

### Verified locally

Driven against the live vLLM server with no AWS involved, on our calibration
image whose target "9" sits at a known (1120, 680):

```
enable_thinking=False -> pyautogui.click(1119, 676)   (238 chars)
enable_thinking=True  -> pyautogui.click(1119, 682)   (595 chars)
```

Both land on target, through OSWorld's own coordinate pipeline.

## The runner

[`OSWorld-V2/scripts/python/run_multienv_qwen36.py`](OSWorld-V2/scripts/python/run_multienv_qwen36.py),
derived from the generic `run_multienv.py` (582 lines) rather than the
Claude-specific one (1,079 lines of Anthropic plumbing). Changes: swap
`PromptAgent` → `Qwen36Agent`, map to that agent's actual signature
(`history_n`/`image_max`/`fold_size`/`coordinate_type`/`base_url`, not
`max_trajectory_length`/`client_password`), and expose the extra flags.

**Two defaults that would quietly invalidate your numbers:**

- **`--max_steps` defaults to 15.** The paper's primary metric is 500 steps.
  Leaving the default measures something unrelated to the leaderboard.
- **Checkpoint evaluation is off and was unavailable.** `lib_run_single.py`
  already implements inline checkpoint evals, but the generic runner never
  exposed the flags that gate them (only the Claude/GLM/GPT/M3 runners did). Our
  runner now exposes `--checkpoint_eval_mode inline --checkpoint_steps ...`,
  which evaluates at intermediate step counts *within a single run*. Given the
  paper's headline finding — no model above 10% on tasks over 137 minutes —
  a step-budget curve is worth far more than a single endpoint, and it costs
  nothing extra.

```bash
.venv/bin/python scripts/python/run_multienv_qwen36.py \
    --provider_name aws --headless \
    --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --base_url http://<gpu-host>:8000/v1 \
    --max_steps 500 \
    --checkpoint_eval_mode inline --checkpoint_steps 50,100,150,250 \
    --test_all_meta_path evaluation_examples/test_v2.json
```

Add `--no_thinking` to compare the fast/no-reasoning configuration.

## Scoring output

`env.evaluate()` may return a float (legacy) or a dict. When it returns a dict:
`dict["score"]` goes to `result.txt` for legacy aggregation, and the full dict is
serialized to `result.json` alongside it. `partial_scores` is keyed by criterion,
each with `score` ∈ [0,1], `weight` (all weights sum to 1.0), and a
`description`; final score = `sum(score × weight)`. Checkpoint evals accumulate
in `checkpoint_results.json`. Aggregate with `show_result.py`.

## Serving config for OSWorld

Two changes from the browser-harness defaults, both now in
[scripts/serve.sh](scripts/serve.sh):

```bash
MAX_LEN=131072 MAX_IMAGES=24 ./scripts/serve.sh 8
```

- **`MAX_LEN=131072`.** 32k is far too small here. OSWorld screens are 1920×1080
  = `1920*1080/1024` = **2,025 vision tokens each**, and the reference agent keeps
  `image_max=20` — ~40k tokens of images before any text. There is KV headroom:
  one card held 1.47M tokens.
- **`--enable-prefix-caching`** (now on by default). Agent history is
  append-only, so this is a large win. Caveat: history folding rewrites older
  messages every `fold_size=10` steps and invalidates the prefix, so you get hits
  *between* folds, not across a whole episode.

Run 8 replicas so environments can execute in parallel — the bottleneck is
concurrent episodes, not single-request latency.

## SSH key for the AWS provider

`KEY_NAME` in the AWS provider config refers to an **EC2-registered key pair
name**, not a file path. Generated locally (ed25519, no passphrase — the provider
SSHes programmatically, so an encrypted key would block automation; the private
key is therefore unencrypted on disk at mode 600):

```
~/.ssh/osworld_aws       private (600) -- never share, never commit
~/.ssh/osworld_aws.pub   public  (644)
SHA256:NTM8C25zEUfGpaU8kcHUNTEItJW4pPnG/tAQ6VTkhhk
```

Import the public half into EC2 and reference it by name (AWS CLI 2.36.7 is
already installed):

```bash
aws ec2 import-key-pair --key-name osworld_key \
    --public-key-material fileb://~/.ssh/osworld_aws.pub --region us-east-1
```

Then set `KEY_NAME=osworld_key`. If a region rejects ed25519, fall back to
`ssh-keygen -t rsa -b 4096 -f ~/.ssh/osworld_aws_rsa -N ""`.

The security group needs ports **22**, **3000**, and **8000** plus the standard
OSWorld backend/control ports.

## What you need to do next

**1. Accept the dataset gates** (both are `gated: auto`, so approval is
instant — no waiting on a human). While logged into HF, click through:

- https://huggingface.co/datasets/xlangai/osworld_v2_tasks
- https://huggingface.co/datasets/xlangai/osworld_v2_assets_gated

**2. Provide a fresh HF token.** Rotate the one pasted into chat first, then
`export HF_TOKEN=...` — do not paste it into a message.

**3. Download the pinned release:**

```bash
cd OSWorld-V2
uv run scripts/tools/download_osworld_v2_tasks.py  --benchmark-release osworld-v2-2026.06.24
uv run scripts/tools/download_osworld_v2_assets.py --benchmark-release osworld-v2-2026.06.24 \
    --target-dir cache/osworld_v2_assets --clean
export OSWORLD_FILE_BASE_URL="$(pwd)/cache/osworld_v2_assets"
```

Pin every component to the same release — the maintainers explicitly warn
against mixing releases or substituting `main`/`latest`.

## Cost before you commit

108 tasks × up to 500 steps ≈ 54k model calls worst case. At ~2–4s/step that is
30–60 hours serial, or roughly 4–8 hours across 8 parallel environments — plus
8+ EC2 VMs running for that whole window, plus website hosting. Thinking mode
multiplies the decode cost substantially.

Expect low binary completion: Opus 4.8 manages 20.6%, so a 35B open model will
likely be low single digits. **Use partial score as the signal** — that is what
checkpoint scoring exists for.

For a first run you can use the team-hosted sites (`web.hku.icu` suffix) rather
than self-hosting all 31. GitLab must be self-hosted regardless
(`Task-Web/gitlab`).

## Status

- [x] Repo cloned, dependencies installed (`uv sync`, Python 3.12)
- [x] 108 tasks confirmed; infrastructure requirements mapped
- [x] KVM blocker identified → AWS provider
- [x] `Qwen36Agent` written and validated end-to-end against live vLLM
- [x] `run_multienv_qwen36.py` runner derived, compiles, flags verified
- [x] Checkpoint evaluation re-enabled on the generic runner path
- [x] Serving config updated for OSWorld's context and image budget
- [x] ed25519 key pair generated for the AWS provider
- [x] Dataset access; 108 task classes downloaded and profiled
- [ ] `aws configure` + import key pair + security group
- [ ] Inspect individual tasks interactively with `manual_examine.py` (needs a provider)
- [ ] AWS provider setup (`docs/PROVIDER_SETUP.md`)
- [ ] Smoke test on 2–3 tasks
