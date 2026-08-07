# Computer-use agent harness — Qwen3.6-35B-A3B

A small, hackable harness for driving a browser with a self-hosted VLM.
Browser-only today; the `Env` interface is the seam where an Xvfb desktop
backend drops in later without touching the agent loop.

## Layout

```
harness/
  actions.py     action space + tolerant parsing of model output
  env.py         Env interface + BrowserEnv (Playwright/Chromium, set-of-marks)
  coords.py      model coordinate space -> screen pixels
  model.py       OpenAI-compatible vLLM client (round-robins over replicas)
  agent.py       the observe -> think -> act loop
  trajectory.py  per-episode logging (screenshots + JSONL)
scripts/
  serve.sh       launch N vLLM replicas
  calibrate.py   determine the model's coordinate convention
run_task.py      CLI entry point
runs/            one directory per episode
```

## Quick start

```bash
screen -dmS vllm ./scripts/serve.sh      # start once, leave running
curl -s localhost:8000/v1/models         # wait for this to answer (~8 min cold)

.venv/bin/python run_task.py "Find the top story on news.ycombinator.com"
```

Each run writes `runs/<ts>_<slug>/` with `meta.json`, `steps.jsonl`, and the
screenshot the model saw at every step. Over Remote-SSH you can open those PNGs
directly in the VSCode editor.

## Findings from bring-up

These were all measured on this setup, not assumed. Re-check them if you change
model or preprocessing.

**Thinking must be off.** Qwen3.6 reasons by default. Asked to click a target it
could not find, it produced 13k characters of chain-of-thought, fell into a
repetition loop, and never emitted an action. `VLMConfig.enable_thinking=False`
sends `chat_template_kwargs={"enable_thinking": false}`, which the chat template
turns into an empty prefilled `<think></think>`. Steps then take ~0.5s.

**The model grounds in 0..1000 normalised space, not pixels.** This is the
opposite of what the preprocessor config suggests (`patch_size=16, merge_size=2`
and a 16.7M-pixel size cap mean a 1280×800 screenshot is passed through
unresized, so pixels *look* like the natural space). `scripts/calibrate.py`
settles it empirically — median error 15px under `norm_1000` vs 263px under
`pixel`. This is exactly the bug the calibration script exists to catch.

**The model emits `[x, y]` pairs regardless of the prompt.** It will return
`{"point": [x, y]}`, or pack both into `{"x": [132, 138]}`, or emit outright
invalid JSON like `{"x": 500, 490}`. `actions.py` normalises all of these rather
than fighting the prior.

**Pixel grounding is not precise enough for dense text.** Measured against the
real DOM on Hacker News: y was accurate to 1.3px, but x was 38px off — enough to
miss a 24px-wide nav link and land on the adjacent one. Snapping to the nearest
clickable element does not help here, because the wrong element is itself
clickable.

**Set-of-marks fixes it.** `BrowserEnv(marks=True)` (default) outlines
interactive elements with numbered badges and the agent calls `click_id(id)`.
The task that clicked the wrong nav link with pixels now lands on `/newest`
first try. Marks are drawn deliberately faint — heavy badges bury the page text
on link-dense pages, which costs more than imprecise aiming.

## The two dials that matter

**Vision tokens.** Tokens per screenshot = `W × H / 1024`. At 1280×800 that is
exactly **1000 tokens**. `--keep-images N` (default 3) caps how many screenshots
stay in context at full resolution; older ones become text stubs, making
per-step cost `O(keep_images)` instead of `O(steps)`.

**Replicas, not tensor parallelism.** An episode is a serial chain
(screenshot → infer → act) that uses a sliver of one H100. FP8 weights are ~35GB,
so one replica fits per 80GB card with 31.65 GiB left for KV — about 1.47M
tokens, or ~1,400 screenshots. Scale concurrent episodes, not single-request
speed:

```bash
./scripts/serve.sh 8     # 8 replicas, GPUs 0-7, ports 8000-8007
.venv/bin/python run_task.py --base-url http://127.0.0.1:8000/v1 \
                             --base-url http://127.0.0.1:8001/v1 "..."
```

## Serving notes

- **Startup is ~8.5 min cold, and it is not weight loading** (that is 23s for
  34 GiB). `init engine (profile, create kv cache, warmup) took 509.74 s`, almost
  all CUDA graph capture over 51 batch sizes. Add `--enforce-eager` for fast
  harness iteration at some throughput cost. Compiled artifacts cache in
  `~/.cache/vllm`.
- **`.venv/bin` must be on `PATH`.** vLLM JIT-compiles the Gated-DeltaNet kernels
  and shells out to `ninja`; invoking `.venv/bin/python` directly without the
  venv on `PATH` fails engine init with `FileNotFoundError: 'ninja'`.
- `--disable-log-requests` was removed in vLLM 0.26; it is `--no-enable-log-requests`.

## Model notes

`Qwen/Qwen3.6-35B-A3B-FP8` — 35B total / 3B active MoE,
`Qwen3_5MoeForConditionalGeneration`, supported natively by vLLM 0.26. Multimodal
(27-layer vision tower), 262k native context. The text stack is a Gated-DeltaNet /
gated-attention hybrid, so KV grows far more slowly with context than pure
attention — hence 1.47M tokens of KV in 31 GiB.

`Qwen/Qwen-AgentWorld-35B-A3B` is **not** an agent policy — it is a world model
that simulates environments (predicting the next env state from an action).
Potentially useful later as a cheap simulator for harness testing without real
browsers.

## Next steps

- Task suite + scoring, so changes can be measured rather than eyeballed.
- Native tool calling with `xgrammar` constrained decoding, if format errors
  become a problem — makes schema-valid output guaranteed rather than parsed.
- `qwen3_5_mtp` speculative decoding (this model was trained with MTP) for
  decode latency.
- Desktop backend: Xvfb + window manager + `xdotool` behind the same `Env`
  interface (move it behind HTTP at that point, one container per episode).
- A replay viewer over `runs/`.
