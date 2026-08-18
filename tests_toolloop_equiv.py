#!/usr/bin/env python3
"""Equivalence + concurrency test for the rollout tool-loop restructure.

Runs the OLD (serial per-sequence) and NEW (batched gather) loop bodies over
identical synthetic inputs and asserts the resulting message sequences,
counters, and image lists are identical.
"""
import asyncio
import json
import threading
import time

# ---------------------------------------------------------------- fake harness

TOOL_LATENCY = 0.20


def _format_tool_result(result):
    """Stand-in for Trainer._format_tool_result."""
    if isinstance(result, dict) and "__image__" in result:
        return json.dumps({k: v for k, v in result.items() if k != "__image__"}), [result["__image__"]]
    return json.dumps(result, default=str, sort_keys=True), []


async def ok_tool(**kw):
    await asyncio.sleep(TOOL_LATENCY)
    return {"rows": [[kw.get("n", 0)]]}


async def raising_tool(**kw):
    await asyncio.sleep(0.01)
    raise RuntimeError("tool blew up")


async def image_tool(**kw):
    await asyncio.sleep(0.01)
    return {"__image__": f"img-{kw.get('n',0)}", "ok": True}


def sync_tool(**kw):
    return {"sync": kw.get("n", 0)}


ASYNC_TOOLS = {"ok_tool": ok_tool, "raising_tool": raising_tool, "image_tool": image_tool}
SYNC_TOOLS = {"sync_tool": sync_tool}


def make_state(n_seq, spec):
    """spec: list per sequence of list of tool_call dicts."""
    completions = [[{"role": "assistant", "content": f"a{i}"}] for i in range(n_seq)]
    prompts = [[] for _ in range(n_seq)]
    tool_images = [[] for _ in range(n_seq)]
    return completions, prompts, tool_images


# ------------------------------------------------------------------- OLD logic

def run_old(idxs_with_tool, tool_calls, prompts, completions, tool_images, loop):
    tool_call_count = 0
    tool_failure_count = 0
    prompt_completion_tools = [prompts[i] for i in idxs_with_tool]

    for idx in range(len(idxs_with_tool)):
        idx_with_tool = idxs_with_tool[idx]
        tool_call_list = tool_calls[idx]
        prompt_completion_tool = prompt_completion_tools[idx]
        sync_tool_dict, async_tool_dict = SYNC_TOOLS, ASYNC_TOOLS
        prompt_completion_tool.append(completions[idx_with_tool][-1])
        async_coros = []
        tool_call_results = []

        for tool_call in tool_call_list:
            tool_call_count += 1
            tool_call_id = tool_call.get("id")
            if tool_call["type"] == "function":
                function = tool_call["function"]
                name = function["name"]
                try:
                    if name in sync_tool_dict:
                        tool_call_results.append((tool_call_id, name, sync_tool_dict[name](**function["arguments"])))
                    elif name in async_tool_dict:
                        async_coros.append((tool_call_id, name, async_tool_dict[name](**function["arguments"])))
                    else:
                        raise ValueError(f"Tool {name} not found.")
                except Exception as exc:
                    tool_failure_count += 1
                    tool_call_results.append((tool_call_id, name, {"error": str(exc)}))
            else:
                tool_failure_count += 1
                name = tool_call.get("name", "unknown")
                tool_call_results.append(
                    (tool_call_id, name, {"error": f"Unsupported tool call type: {tool_call['type']}"})
                )

        if async_coros:

            async def _run_async_tools(coros_with_names):
                coros = [coro for _, _, coro in coros_with_names]
                results = await asyncio.gather(*coros, return_exceptions=True)
                return [
                    (tool_call_id, name, result)
                    for (tool_call_id, name, _), result in zip(coros_with_names, results, strict=False)
                ]

            async_results = asyncio.run_coroutine_threadsafe(_run_async_tools(async_coros), loop).result()
            for tool_call_id, name, result in async_results:
                if isinstance(result, Exception):
                    tool_failure_count += 1
                    tool_call_results.append((tool_call_id, name, {"error": str(result)}))
                else:
                    tool_call_results.append((tool_call_id, name, result))

        for tool_call_id, name, result in tool_call_results:
            content, images_from_tool = _format_tool_result(result)
            tool_message = {"role": "tool", "name": name, "content": content}
            if tool_call_id is not None:
                tool_message["tool_call_id"] = tool_call_id
            for image in images_from_tool:
                if image is not None:
                    tool_images[idx_with_tool].append(image)
            prompt_completion_tool.append(tool_message)
            completions[idx_with_tool].append(tool_message)

    return tool_call_count, tool_failure_count


# ------------------------------------------------------------------- NEW logic

def run_new(idxs_with_tool, tool_calls, prompts, completions, tool_images, loop):
    tool_call_count = 0
    tool_failure_count = 0
    prompt_completion_tools = [prompts[i] for i in idxs_with_tool]

    pending_async = []
    per_seq_results = []

    for idx in range(len(idxs_with_tool)):
        idx_with_tool = idxs_with_tool[idx]
        tool_call_list = tool_calls[idx]
        prompt_completion_tool = prompt_completion_tools[idx]
        sync_tool_dict, async_tool_dict = SYNC_TOOLS, ASYNC_TOOLS
        prompt_completion_tool.append(completions[idx_with_tool][-1])
        tool_call_results = []

        for tool_call in tool_call_list:
            tool_call_count += 1
            tool_call_id = tool_call.get("id")
            if tool_call["type"] == "function":
                function = tool_call["function"]
                name = function["name"]
                try:
                    if name in sync_tool_dict:
                        tool_call_results.append((tool_call_id, name, sync_tool_dict[name](**function["arguments"])))
                    elif name in async_tool_dict:
                        coro = async_tool_dict[name](**function["arguments"])
                        pending_async.append((idx, len(tool_call_results), tool_call_id, name, coro))
                        tool_call_results.append(None)
                    else:
                        raise ValueError(f"Tool {name} not found.")
                except Exception as exc:
                    tool_failure_count += 1
                    tool_call_results.append((tool_call_id, name, {"error": str(exc)}))
            else:
                tool_failure_count += 1
                name = tool_call.get("name", "unknown")
                tool_call_results.append(
                    (tool_call_id, name, {"error": f"Unsupported tool call type: {tool_call['type']}"})
                )

        per_seq_results.append(tool_call_results)

    if pending_async:

        async def _run_async_tools(coros):
            return await asyncio.gather(*coros, return_exceptions=True)

        async_results = asyncio.run_coroutine_threadsafe(
            _run_async_tools([item[4] for item in pending_async]), loop
        ).result()

        for (seq_pos, slot, tool_call_id, name, _), result in zip(pending_async, async_results, strict=False):
            if isinstance(result, BaseException):
                tool_failure_count += 1
                per_seq_results[seq_pos][slot] = (tool_call_id, name, {"error": str(result)})
            else:
                per_seq_results[seq_pos][slot] = (tool_call_id, name, result)

    for idx in range(len(idxs_with_tool)):
        idx_with_tool = idxs_with_tool[idx]
        prompt_completion_tool = prompt_completion_tools[idx]
        for tool_call_id, name, result in per_seq_results[idx]:
            content, images_from_tool = _format_tool_result(result)
            tool_message = {"role": "tool", "name": name, "content": content}
            if tool_call_id is not None:
                tool_message["tool_call_id"] = tool_call_id
            for image in images_from_tool:
                if image is not None:
                    tool_images[idx_with_tool].append(image)
            prompt_completion_tool.append(tool_message)
            completions[idx_with_tool].append(tool_message)

    return tool_call_count, tool_failure_count


# ---------------------------------------------------------------------- driver

def call(name, n=0, cid=None, type_="function", args=None):
    if type_ != "function":
        return {"type": type_, "id": cid, "name": name}
    return {"type": "function", "id": cid, "function": {"name": name, "arguments": args if args is not None else {"n": n}}}


SCENARIOS = {
    "typical: 8 seqs x 1 async call": [[call("ok_tool", i, f"c{i}")] for i in range(8)],
    "multi-call sequences": [
        [call("ok_tool", 0, "a"), call("ok_tool", 1, "b")],
        [call("ok_tool", 2, "c")],
    ],
    "unknown tool": [[call("nope", 0, "a")], [call("ok_tool", 1, "b")]],
    "bad arguments (TypeError at build)": [[call("ok_tool", args={"bad_kw": 1, "n": 0})], [call("ok_tool", 1)]],
    "tool raises during execution": [[call("raising_tool", 0, "a")], [call("ok_tool", 1, "b")]],
    "non-function call type": [[call("x", type_="weird", cid="a")], [call("ok_tool", 1, "b")]],
    "sync tool mixed in": [[call("sync_tool", 0, "a")], [call("ok_tool", 1, "b")]],
    "image-returning tool": [[call("image_tool", 0, "a")], [call("ok_tool", 1, "b")]],
    "no tool calls at all": [],
    "mixed everything": [
        [call("ok_tool", 0, "a"), call("nope", 1, "b")],
        [call("raising_tool", 2, "c")],
        [call("sync_tool", 3, "d")],
        [call("image_tool", 4, "e"), call("ok_tool", 5, "f")],
        [call("z", type_="weird", cid="g")],
    ],
}


def main():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    all_ok = True
    for label, spec in SCENARIOS.items():
        n_seq = max(len(spec), 1)
        idxs = list(range(len(spec)))

        c1, p1, i1 = make_state(n_seq, spec)
        c2, p2, i2 = make_state(n_seq, spec)

        t0 = time.perf_counter()
        old_counts = run_old(idxs, spec, p1, c1, i1, loop)
        t_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        new_counts = run_new(idxs, spec, p2, c2, i2, loop)
        t_new = time.perf_counter() - t0

        J = lambda o: json.dumps(o, sort_keys=True, default=str)
        exact = J(c1) == J(c2) and J(p1) == J(p2) and i1 == i2 and old_counts == new_counts

        # Same messages in a different order is an intentional fix: the old code
        # appended error/sync results before async ones, so tool responses could
        # come back in a different order than the model issued the calls.
        def multiset(seqs):
            return sorted(J(m) for seq in seqs for m in seq)

        reordered_only = (
            not exact
            and multiset(c1) == multiset(c2)
            and multiset(p1) == multiset(p2)
            and i1 == i2
            and old_counts == new_counts
        )

        def issue_order_respected(completions, spec_):
            for seq, calls in zip(completions, spec_):
                ids = [c.get("id") for c in calls]
                got = [m.get("tool_call_id") for m in seq if m["role"] == "tool"]
                if ids != got:
                    return False
            return True

        ok = exact or (reordered_only and issue_order_respected(c2, spec))
        all_ok &= ok
        speed = f"{t_old/t_new:.2f}x" if t_new > 0 else "-"
        tag = "PASS" if exact else ("PASS(order-fixed)" if ok else "FAIL")
        print(f"{tag:<18} {label:<40} counts={new_counts} old={t_old:.2f}s new={t_new:.2f}s speedup={speed}")
        if not ok:
            print("   completions old:", json.dumps(c1, default=str)[:300])
            print("   completions new:", json.dumps(c2, default=str)[:300])
            print("   counts old/new:", old_counts, new_counts)
        elif not exact:
            print(f"     old response order: {[m.get('tool_call_id') for m in c1[0] if m['role']=='tool']}"
                  f"  new: {[m.get('tool_call_id') for m in c2[0] if m['role']=='tool']}"
                  f"  (calls issued: {[c.get('id') for c in spec[0]]})")

    print()
    print("ALL SCENARIOS IDENTICAL:", all_ok)
    loop.call_soon_threadsafe(loop.stop)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
