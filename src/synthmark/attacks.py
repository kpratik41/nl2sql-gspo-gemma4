"""Edits that a watermarked text might undergo before someone tries to detect it.

Everything here is an *evasion or attrition model*, not an exploit: the point is
to measure how much watermark signal survives realistic handling.  Detection
strength is proportional to the number of watermarked token positions that
survive intact, so any edit that changes tokens, or shifts the token boundaries
around surviving tokens, costs signal.

Two families:

*Non-semantic edits* (truncation, deletion, swapping, casing) are cheap to apply
and cheap to defend against in a report, because they visibly damage the text.

*Semantic-preserving edits* (paraphrase, round-trip translation) are the ones
that matter.  They produce text a reader would accept as equivalent while
re-sampling nearly every token, and they are the honest limit of what
watermarking can promise.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class AttackResult:
    name: str
    strength: float
    """Attack-specific intensity parameter, for plotting a dose-response curve."""
    texts: list[str]
    description: str = ""


# --------------------------------------------------------------------- helpers

def _words(text: str) -> list[str]:
    return text.split()


def _rejoin(words: Sequence[str]) -> str:
    return " ".join(words)


# ------------------------------------------------------- non-semantic attacks

def truncate(texts: Sequence[str], keep_fraction: float, *, seed: int = 0) -> AttackResult:
    """Keep only the first ``keep_fraction`` of each text.

    This is less an attack than a measurement of the length/power trade-off in
    disguise: it isolates the effect of having fewer tokens to score, with no
    corruption of the tokens that remain.
    """
    out = []
    for t in texts:
        w = _words(t)
        k = max(1, int(len(w) * keep_fraction))
        out.append(_rejoin(w[:k]))
    return AttackResult("truncate", keep_fraction, out, "keep leading fraction of the text")


def delete_words(texts: Sequence[str], p: float, *, seed: int = 0) -> AttackResult:
    """Delete each word independently with probability ``p``."""
    rng = random.Random(seed)
    out = []
    for t in texts:
        w = [x for x in _words(t) if rng.random() >= p]
        out.append(_rejoin(w) if w else t[:20])
    return AttackResult("delete_words", p, out, "independent random word deletion")


def swap_words(texts: Sequence[str], p: float, *, seed: int = 0) -> AttackResult:
    """Swap each word with its neighbour with probability ``p``.

    Preserves the multiset of words but destroys local n-gram context, which is
    what the g-function keys on.
    """
    rng = random.Random(seed)
    out = []
    for t in texts:
        w = _words(t)
        i = 0
        while i < len(w) - 1:
            if rng.random() < p:
                w[i], w[i + 1] = w[i + 1], w[i]
                i += 2
            else:
                i += 1
        out.append(_rejoin(w))
    return AttackResult("swap_words", p, out, "swap adjacent word pairs")


def substitute_words(
    texts: Sequence[str], p: float, *, vocabulary: Sequence[str] | None = None, seed: int = 0
) -> AttackResult:
    """Replace a fraction of words with other words drawn from the corpus.

    Stands in for a careless human editor: meaning degrades, but the token
    stream is disrupted in the same way a careful editor would disrupt it.
    """
    rng = random.Random(seed)
    if vocabulary is None:
        vocabulary = sorted({w for t in texts for w in _words(t) if w.isalpha()})
    vocabulary = list(vocabulary) or ["the"]
    out = []
    for t in texts:
        w = _words(t)
        for i in range(len(w)):
            if rng.random() < p:
                w[i] = rng.choice(vocabulary)
        out.append(_rejoin(w))
    return AttackResult("substitute_words", p, out, "random word substitution")


def lowercase(texts: Sequence[str]) -> AttackResult:
    """Lowercase everything.

    Semantically almost free for a reader, but it re-tokenises much of the text,
    so it is a clean probe of how brittle the watermark is to token-boundary
    shifts that leave the *words* untouched.
    """
    return AttackResult("lowercase", 1.0, [t.lower() for t in texts], "lowercase the text")


def strip_formatting(texts: Sequence[str]) -> AttackResult:
    """Remove markdown formatting and collapse whitespace.

    Models the very common case of pasting output into a plain-text field.
    """
    out = []
    for t in texts:
        s = re.sub(r"[*_`#>]+", "", t)
        s = re.sub(r"\s+", " ", s).strip()
        out.append(s)
    return AttackResult("strip_formatting", 1.0, out, "remove markdown, collapse whitespace")


def mix_with_human(
    texts: Sequence[str], human_texts: Sequence[str], model_fraction: float, *, seed: int = 0
) -> AttackResult:
    """Dilute watermarked text with human text to a target model fraction.

    This is the realistic "Claude helped me write it" case, and the one where a
    detector's output is most easily over-interpreted: a diluted document is
    genuinely part-machine, and a weak score is the correct answer rather than a
    failure.
    """
    rng = random.Random(seed)
    out = []
    for i, t in enumerate(texts):
        mw = _words(t)
        human = _words(human_texts[i % len(human_texts)])
        n_model = max(1, int(len(mw) * model_fraction))
        n_human = max(0, len(mw) - n_model)
        segment = mw[:n_model]
        filler = human[:n_human] if n_human else []
        # Interleave at paragraph granularity rather than word level so the
        # result reads like a co-written document, not word salad.
        if rng.random() < 0.5:
            out.append(_rejoin(filler + segment))
        else:
            out.append(_rejoin(segment + filler))
    return AttackResult("mix_with_human", model_fraction, out, "dilute with human-written text")


# --------------------------------------------------- model-based (semantic)

PARAPHRASE_INSTRUCTION = (
    "Rewrite the following text in different words. Preserve every fact and the "
    "overall length, but change the phrasing and sentence structure substantially. "
    "Output only the rewritten text, with no preamble.\n\nText:\n{text}"
)

TRANSLATE_TO = (
    "Translate the following English text into {language}. Output only the "
    "translation.\n\n{text}"
)

TRANSLATE_BACK = (
    "Translate the following {language} text into English. Output only the "
    "translation.\n\n{text}"
)


def paraphrase(
    texts: Sequence[str],
    generate_fn: Callable[[Sequence[str]], list[str]],
    *,
    rounds: int = 1,
) -> AttackResult:
    """Paraphrase with a language model, optionally repeatedly.

    ``generate_fn`` takes a list of fully-rendered prompts and returns
    completions.  Keeping it a callable means the attack model can be a
    different model, on a different GPU, from the one under test -- which is the
    realistic threat: an adversary will not use your model to launder your
    watermark.
    """
    current = list(texts)
    for _ in range(max(1, rounds)):
        prompts = [PARAPHRASE_INSTRUCTION.format(text=t) for t in current]
        current = [c.strip() for c in generate_fn(prompts)]
    return AttackResult(
        "paraphrase", float(rounds), current, f"LLM paraphrase x{rounds}"
    )


def round_trip_translate(
    texts: Sequence[str],
    generate_fn: Callable[[Sequence[str]], list[str]],
    *,
    language: str = "French",
) -> AttackResult:
    """Translate to another language and back."""
    fwd = generate_fn([TRANSLATE_TO.format(language=language, text=t) for t in texts])
    back = generate_fn([TRANSLATE_BACK.format(language=language, text=t.strip()) for t in fwd])
    return AttackResult(
        "round_trip_translate", 1.0, [t.strip() for t in back], f"EN -> {language} -> EN"
    )


NON_SEMANTIC_SWEEPS: dict[str, tuple[Callable, Sequence[float]]] = {
    "truncate": (truncate, (1.0, 0.75, 0.5, 0.25, 0.125)),
    "delete_words": (delete_words, (0.0, 0.05, 0.1, 0.2, 0.4)),
    "swap_words": (swap_words, (0.0, 0.05, 0.1, 0.2, 0.4)),
    "substitute_words": (substitute_words, (0.0, 0.05, 0.1, 0.2, 0.4)),
    "mix_with_human": (None, (1.0, 0.75, 0.5, 0.25, 0.1)),
}
