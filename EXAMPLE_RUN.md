# Annotated end-to-end run

One complete episode, captured verbatim: every prompt sent to the model, every
response it returned, and what the harness did in between.

Reproduce with `.venv/bin/python scripts/capture_example.py`, which wraps
`VLMClient.chat` to record both sides of each call. (The normal trajectory log
keeps model *output* but not the outgoing prompt, so this script exists to
capture the request side too.)

| | |
|---|---|
| **Task** | `Search Wikipedia for 'Hedy Lamarr' and tell me what invention she is credited as a co-inventor of.` |
| **Start URL** | `https://en.wikipedia.org` |
| **Model** | `Qwen/Qwen3.6-35B-A3B-FP8` served by vLLM 0.26 on one H100 |
| **Sampling** | `temperature=0.2, top_p=0.8, max_tokens=1024, enable_thinking=false` |
| **LLM turns** | **3** (one per step; the loop makes exactly one call per step) |
| **Wall clock** | 4.37s total, of which 2.04s was model time |
| **Outcome** | `finished` — answer verified correct |
| **Artifacts** | `runs/20260807-042105_search-wikipedia-for-hedy-lamarr-and-tel/` |

---

## Pipeline

What happens once per step, in order:

1. **Observe** — `BrowserEnv.observe()` injects the set-of-marks overlay
   (numbered badges on every visible interactive element), takes a 1280×800 PNG
   screenshot, then **removes the overlay** so badges never leak into the live
   page. Returns the PNG plus `{url, title, tabs, marks[]}`.
2. **Log** — the screenshot is written to `runs/<episode>/step_NNN.png`.
3. **Build the user message** — step counter, URL, page title, the numbered
   element listing as *text*, any error from the previous action, and the task.
   The PNG is attached as a base64 `data:` URL.
4. **Prune context** — keep only the newest `keep_images=3` screenshots at full
   resolution; older ones are replaced by the stub
   `[earlier screenshot omitted to save context]`. Never triggered in this run,
   which had exactly 3.
5. **Call the model** — one `POST /v1/chat/completions`, with
   `chat_template_kwargs={"enable_thinking": false}`.
6. **Parse** — `parse_action()` tolerantly extracts one JSON action, stripping
   `<think>` blocks, unwrapping fences and `<tool_call>` tags, and normalising
   the several ways the model expresses a point.
7. **Transform coordinates** — for pixel-based actions only, map `norm_1000` →
   pixels and clamp to the viewport. **Not used in this run**: all three actions
   were index- or text-based.
8. **Execute** — `BrowserEnv.step()` applies the action, waits 400ms for the UI
   to settle, and re-observes. Errors are caught and fed back as text rather
   than crashing the episode.
9. **Append** — the raw assistant text is appended to the message list, and the
   loop repeats until a terminal action (`finish`/`give_up`) or `max_steps`.

Message growth per turn: `system + (user_obs + assistant)*N`, so turn *N* sends
`2N` messages. Here: 2, 4, 6.

---

## The system prompt

Sent unchanged on every turn (2,072 chars). `{width}`/`{height}` are filled from
the actual viewport, and the action list is generated from `ACTION_DOCS` in
[actions.py](harness/actions.py), so it cannot drift from the executable schema.

```
You are a computer-use agent operating a web browser. Each turn you receive a screenshot of the current 1280x800 browser viewport, and you reply with exactly one action.

Available actions:
click_id(id)                 - PREFERRED. Click the element with that numbered badge.
click(point, button="left")  - click at point [x, y]. Only when no badge covers the target.
double_click(point)          - double-click at point [x, y].
type(text, enter=false)      - type text into the focused element. Set enter=true to press Enter after.
key(keys)                    - press a key combo, e.g. "Enter", "Control+a", "Escape", "Tab".
scroll(dy, point=null)       - scroll vertically by dy pixels (positive = down). Optionally centred on point [x, y].
goto(url)                    - navigate directly to a URL.
back()                       - browser back.
wait(ms=1000)                - wait for the page to settle.
finish(answer)               - the task is complete; answer is your result string.
give_up(reason)              - the task cannot be completed; explain why.

Rules:
- Reply with ONE JSON object and nothing else, in this shape:
  {"reasoning": "<one short sentence on what you see and why this action>", "action": "<name>", "args": {...}}
- Interactive elements are outlined with a small coloured number badge. To click
  one, use its number: {"reasoning": "...", "action": "click_id", "args": {"id": 7}}
  This is far more reliable than aiming at a pixel -- always prefer it.
- Only if no badge covers your target, give a location as "point": [x, y], e.g.
  {"reasoning": "empty canvas area", "action": "click", "args": {"point": [512, 331]}}
- Look at the screenshot before every action. Do not assume the page changed the way you expected -- verify, then act.
- Prefer clicking visible UI over guessing URLs, but goto() is fine for a known site.
- If a click did nothing, do not repeat it identically. Scroll, wait, or try a different target.
- When the task is done, call finish(answer). If it is genuinely impossible, call give_up(reason). Do not loop forever.
```

---

## Turn 1 — locate and focus the search box

**Messages sent:** 2 — `system`, `user`. The user message is
`[image, text]`: a 361,365-byte PNG (1280×800 → **1000 vision tokens**) followed
by 1,661 chars of text.

**User text, verbatim:**

```
Step 1/12.
Current URL: https://en.wikipedia.org/wiki/Main_Page
Page title: Wikipedia, the free encyclopedia
Numbered elements on screen:
[0] A  [1] Search Wikipedia  [2] Search  [3] Donate  [4] Create account  [5] Log in  [6] Main Page  [7] Talk  [8] Read  [9] View source  [10] View history  [11] hide  [12] Wikipedia  [13] free  [14] anyone can edit  [15] 251,364  [16] 7,220,395  [17] English  [18] A  [19] Henry Macandrew  [20] British Indian Army  [21] Tirah campaign  [22] staff officer  [23] Boer War  [24] Distinguished Service Order  [25] 5th Bengal Cavalry  [26] 1st Indian Cavalry Division  [27] brigadier-general  [28] Indian Cavalry Corps  [29] Battle of Neuve-Chapelle  [30] 2nd Indian Cavalry Division  [31] Battle of the Somme  [32] Battle of Cambrai  [33] 5th Cavalry Division  [34] Sinai and Palestine campaign  [35] Capture of Damascus  [36] Battle of Aleppo  [37] Full article...  [38] Big Duck  [39] Yeah Yeah Yeahs (EP)  [40] Black Prince's chevauchée of 1356  [41] A  [42] An avalanche  [43] Broad Peak  [44] About 80,000 migrants cross  [45] Morocco–Spain border  [46] Ceuta  [47] A magnitude 6.8 earthquake  [48] Kumamoto Prefecture  [49] Gaelic football  [50] the All-Ireland Senior Football Championship  [51] Mayo  [52] Kerry  [53] the final  [54] Iran war  [55] Lebanon war  [56] Russo-Ukrainian war  [57] timeline  [58] Recent deaths  [59] Franco Baresi  [60] Beto Cabrera  [61] Glen Hansard  [62] Nirmal Purja  [63] Kavinsky  [64] Samara Colina  [65] More current events  [66] Nominate an article
Task: Search Wikipedia for 'Hedy Lamarr' and tell me what invention she is credited as a co-inventor of.
Reply with one JSON action.
```

**Response, verbatim (239 chars, 0.824s):**

```json
{
  "reasoning": "I need to search for 'Hedy Lamarr' on Wikipedia. The search bar is visible at the top of the page with badge [1]. I will click on it to focus and then type the search query.",
  "action": "click_id",
  "args": {"id": 1}
}
```

**Harness did:** parsed → `click_id(id=1)`. No coordinate transform (index-based).
Looked up mark 1 in `_marks`, clicked its box centre, waited 400ms, re-observed.
URL unchanged — as expected, focusing a search box does not navigate.

---

## Turn 2 — type the query

**Messages sent:** 4 — `system`, `user`(turn 1), `assistant`(turn 1 raw text),
`user`(new). Both screenshots still at full resolution (2 of the 3 allowed).

The new user text is byte-identical to turn 1's except `Step 2/12` — same URL,
same 67 marks, because the click only moved focus.

**Response, verbatim (226 chars, 0.447s):**

```json
{
  "reasoning": "The search bar is already focused from the previous action. I need to type 'Hedy Lamarr' into it and press Enter to perform the search.",
  "action": "type",
  "args": {"text": "Hedy Lamarr", "enter": true}
}
```

**Harness did:** parsed → `type(text='Hedy Lamarr', enter=True)`. Typed into the
focused element at 15ms/char, pressed Enter, waited for `domcontentloaded` plus
400ms. Wikipedia resolved the exact title and navigated straight to the article
rather than a results page.

---

## Turn 3 — read the answer and terminate

**Messages sent:** 6 — three screenshots now in context (exactly at the
`keep_images=3` limit; a fourth step would have stubbed out turn 1's image).

**New user text, verbatim:**

```
Step 3/12.
Current URL: https://en.wikipedia.org/wiki/Hedy_Lamarr
Page title: Hedy Lamarr - Wikipedia
Numbered elements on screen:
[0] A  [1] Search Wikipedia  [2] Search  [3] Donate  [4] Create account  [5] Log in  [6] hide  [7] (Top)  [8] Early life  [9] Film career  [10] Toggle Film career subsection  [11] Producer  [12] Inventing career  [13] Later years  [14] Toggle Later years subsection  [15] Marriages and children  [16] Awards, honors, and tributes  [17] Toggle Awards, honors, and tributes subsection  [18] In popular culture  [19] Toggle In popular culture subsection  [20] List of works  [21] Toggle List of works subsection  [22] See also  [23] Explanatory notes  [24] References  [25] Toggle References subsection  [26] Further reading  [27] External links  [28] Article  [29] Talk  [30] Read  [31] Edit  [32] View history  [33] hide  [34] A  [35] Casselberry, Florida  [36] [1]  [37] Stateless  [38] /ˈhɛdi/  [39] [a]  [40] World War II  [41] Czechoslovakia  [42] Ecstasy  [43] Friedrich Mandl  [44] Louis B. Mayer  [45] Algiers  [46] [2]  [47] Boom Town  [48] White Cargo  [49] Samson and Delilah  [50] [3]  [51] Hollywood Walk of Fame  [52] World War II  [53] George Antheil  [54] Allied  [55] torpedoes  [56] spread spectrum  [57] frequency hopping  [58] radio jamming  [59] Axis powers
Task: Search Wikipedia for 'Hedy Lamarr' and tell me what invention she is credited as a co-inventor of.
Reply with one JSON action.
```

**Response, verbatim (619 chars, 0.771s):**

```json
{
  "reasoning": "The search for 'Hedy Lamarr' has been completed, and the Wikipedia article is displayed. The text clearly states that she 'co-invented a radio guidance system during World War II' and later mentions she 'co-invented a radio guidance system for Allied torpedoes that used spread spectrum and frequency hopping technology'. This is the invention she is credited with. The task is complete.",
  "action": "finish",
  "args": {"answer": "Hedy Lamarr is credited as a co-inventor of a radio guidance system (specifically for Allied torpedoes) that used spread spectrum and frequency hopping technology."}
}
```

**Harness did:** `finish` is terminal — logged the step, wrote
`meta.json` with `status="finished"`, and exited the loop without executing
anything in the browser.

---

## Per-step summary

| Step | URL at observation | Action | Images in context | Model latency |
|---|---|---|---|---|
| 0 | `/wiki/Main_Page` | `click_id(id=1)` | 1 | 0.82s |
| 1 | `/wiki/Main_Page` | `type(text='Hedy Lamarr', enter=True)` | 2 | 0.45s |
| 2 | `/wiki/Hedy_Lamarr` | `finish(answer=…)` | 3 | 0.77s |

No parse errors, no action errors, no retries.

## Token accounting

Each 1280×800 screenshot is `1280 × 800 / 1024` = **exactly 1000 vision tokens**
(`patch_size=16`, `merge_size=2`). Vision tokens sent per turn: 1000, 2000, 3000
— 6,000 total. With `keep_images=3` this plateaus at 3000/turn regardless of
episode length; without pruning, a 25-step episode would send 25,000 vision
tokens on its final turn alone.

## Verification

The model's claim of success was checked independently against the DOM rather
than taken at its word — a necessary habit, since in an earlier run on Hacker
News the agent confidently reported it had reached `/newest` while actually
sitting on the front page:

```
$ .venv/bin/python -c "... env.page.locator('.titleline a').first.inner_text()"
```

Here `steps.jsonl` confirms the final observation URL really was
`https://en.wikipedia.org/wiki/Hedy_Lamarr`, and the answer matches the article.

## Why this run is easy

Worth stating plainly so the document is not read as a benchmark: three steps, no
error recovery, no scrolling, no ambiguous targets, and every action
index-or-text based, so the coordinate transform — the part most likely to
break — was never exercised. See [README.md](README.md) for the calibration
results and the pixel-grounding limits that a harder task would hit.
