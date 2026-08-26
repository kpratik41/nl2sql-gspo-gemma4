"""Prompt suites and negative (non-watermarked) corpora for the evaluations.

The prompt suites are deliberately split by *entropy*, because entropy is the
single variable that governs watermark strength.  SynthID can only steer a
choice that exists: where the model is nearly certain of the next token there is
no residual randomness to encode a signal in.  Reporting one aggregate detection
number over a mixed prompt set therefore hides the effect that matters most in
practice, so every suite is scored separately.

Suites, in descending order of expected watermark strength:

``creative``   free-form writing; the best case
``open_ended`` explanatory prose; the realistic average case
``factual``    short answers pinned down by the facts; weak
``structured`` JSON with a fixed schema; very weak
``code``       program text; weakest, often undetectable
``financial``  domain-relevant mix, for a bank-specific read
"""

from __future__ import annotations

import re
from typing import Sequence

CREATIVE: list[str] = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe a rainy afternoon in a small coastal town, in vivid detail.",
    "Write a story about the last bookshop in a city of screens.",
    "Describe a fictional mountain village and the people who live there.",
    "Write a letter from a polar explorer to their family back home.",
    "Tell a story about two strangers who share a train compartment overnight.",
    "Describe a night market through the eyes of someone visiting for the first time.",
    "Write a scene in which an old clockmaker teaches an apprentice.",
    "Describe an abandoned amusement park being reclaimed by a forest.",
    "Write a story about a musician who loses their hearing and finds a new way to compose.",
    "Describe the morning routine of a baker in a small town.",
    "Write about a gardener who tends a rooftop garden above a busy city.",
    "Tell the story of a ferry that runs between two islands and the people it carries.",
    "Describe a house that has been in the same family for two hundred years.",
    "Write about a cartographer mapping a coastline that keeps changing.",
    "Describe a winter festival in a place where the sun does not rise for weeks.",
    "Write a story about someone who inherits a shop full of unlabelled keys.",
    "Describe a long walk taken to think something through.",
    "Write about a radio station that broadcasts to almost nobody.",
    "Tell a story set entirely inside a greenhouse.",
    "Describe the sea from the point of view of someone who has never seen it before.",
    "Write about a translator who becomes attached to a writer they have never met.",
    "Describe a village that moves twice a year with the seasons.",
    "Write a story about a photograph found in a second-hand book.",
]

OPEN_ENDED: list[str] = [
    "Explain why interest rates affect bond prices, for a general audience.",
    "Explain the water cycle to a curious ten-year-old.",
    "Discuss the advantages and disadvantages of nuclear power.",
    "Explain what makes a good user interface, with examples.",
    "Write an opinion piece about remote work and its effect on cities.",
    "Describe the process of making sourdough bread and why it works.",
    "Explain how vaccines train the immune system.",
    "Discuss whether cities should be designed around cars or people.",
    "Explain the history of jazz in New Orleans and why it developed there.",
    "Describe how honeybees organise a hive and divide labour.",
    "Explain the trade-offs between renting and buying a home.",
    "Discuss how photography changed the way people remember events.",
    "Explain why languages change over time.",
    "Describe the invention of the printing press and its consequences.",
    "Explain how a modern supply chain moves a product from factory to shelf.",
    "Discuss the role of public libraries in a digital age.",
    "Explain what causes ocean currents and why they matter for climate.",
    "Describe how a courtroom trial is structured and why.",
    "Explain the appeal of long-distance running to people who do not run.",
    "Discuss how urban parks affect the health of a city's residents.",
    "Explain how film editing shapes an audience's emotional response.",
    "Describe the challenges of preserving old buildings in a growing city.",
    "Explain why some scientific results fail to replicate.",
    "Discuss how streaming changed the economics of the music industry.",
]

FACTUAL: list[str] = [
    "What is the capital of Australia? Answer in one sentence.",
    "In what year did the Berlin Wall fall? Answer briefly.",
    "What is the chemical symbol for gold, and what is its atomic number?",
    "Who wrote the novel 'Pride and Prejudice'? Answer in one sentence.",
    "What is the boiling point of water at sea level in Celsius and Fahrenheit?",
    "How many bones are in the adult human body? Answer briefly.",
    "What is the largest planet in the solar system? Answer in one sentence.",
    "In what year did the first human land on the Moon? Answer briefly.",
    "What is the speed of light in a vacuum? Give the standard value.",
    "Which river is the longest in Africa? Answer in one sentence.",
    "What does DNA stand for? Answer briefly.",
    "Who painted the ceiling of the Sistine Chapel? Answer in one sentence.",
    "What is the smallest prime number greater than 100?",
    "What currency is used in Japan? Answer briefly.",
    "How many continents are there, and what are they?",
    "What is the freezing point of water in Kelvin?",
    "Which element has the atomic number 6? Answer briefly.",
    "In what year was the United Nations founded? Answer briefly.",
    "What is the tallest mountain above sea level? Answer in one sentence.",
    "What is the square root of 144?",
    "Who developed the theory of general relativity? Answer briefly.",
    "What is the largest ocean on Earth? Answer in one sentence.",
    "How many minutes are in a full day?",
    "What gas do plants absorb during photosynthesis? Answer briefly.",
]

CODE: list[str] = [
    "Write a Python function that reverses a linked list. Output only code.",
    "Write a Python function to check whether a string is a palindrome. Code only.",
    "Implement binary search in Python. Output only the code.",
    "Write a Python function that merges two sorted lists. Code only.",
    "Write a SQL query that returns the top 10 customers by total order value.",
    "Implement a Python class for a stack with push, pop, and peek. Code only.",
    "Write a Python function that computes the nth Fibonacci number iteratively.",
    "Write a Python decorator that retries a function three times on exception.",
    "Implement quicksort in Python. Output only code.",
    "Write a Python function that flattens a nested list of arbitrary depth.",
    "Write a SQL query joining orders and customers, filtered to the last 30 days.",
    "Write a Python function that parses a CSV file and returns a list of dicts.",
    "Implement a least-recently-used cache in Python. Code only.",
    "Write a Python function that validates an email address with a regex.",
    "Write a bash script that finds and deletes files older than 30 days.",
    "Implement a Python generator that yields prime numbers. Code only.",
    "Write a Python function that computes the edit distance between two strings.",
    "Write a Python context manager that times the enclosed block.",
    "Implement a breadth-first search over a graph in Python. Code only.",
    "Write a Python function that safely deep-merges two dictionaries.",
    "Write a SQL query computing a 7-day moving average of daily revenue.",
    "Write a Python function that batches an iterable into chunks of size n.",
    "Implement a simple binary tree with in-order traversal in Python.",
    "Write a Python function that retries an HTTP GET with exponential backoff.",
]

STRUCTURED: list[str] = [
    'Return a JSON object with keys "name", "currency", "country" for the Bank of Japan. JSON only.',
    'Return JSON with keys "ticker", "sector", "exchange" for Apple Inc. JSON only.',
    'Output a JSON array of the four quarters of a fiscal year, each with "quarter" and "months".',
    'Return a JSON object describing a bond with keys "issuer", "coupon", "maturity", "currency". Use plausible values. JSON only.',
    'Return JSON with keys "country", "capital", "currency_code" for Brazil. JSON only.',
    'Output a JSON array of three risk categories, each with "name" and "description". JSON only.',
    'Return a JSON object with keys "instrument", "asset_class", "settlement_days" for a US Treasury note. JSON only.',
    'Return JSON listing the G7 countries as an array under the key "members". JSON only.',
    'Output a JSON object with keys "metric", "value", "unit" for a portfolio duration of 4.2 years.',
    'Return a JSON object with keys "trade_id", "side", "quantity", "price" for a plausible equity trade. JSON only.',
    'Return JSON with keys "index", "region", "constituent_count" for the FTSE 100. JSON only.',
    'Output a JSON array of three payment methods, each with "method" and "settlement_time". JSON only.',
    'Return a JSON object with keys "rating", "agency", "grade" for a BBB- corporate rating. JSON only.',
    'Return JSON with keys "currency_pair", "base", "quote" for EURUSD. JSON only.',
    'Output a JSON object describing a savings account with keys "product", "rate", "min_balance". JSON only.',
    'Return a JSON array of the four main financial statements, each with "name" and "purpose". JSON only.',
]

FINANCIAL: list[str] = [
    "Summarise, for a retail client, what a floating-rate note is and when it is attractive.",
    "Explain the difference between a stock split and a reverse stock split, for a client newsletter.",
    "Draft a short internal note explaining why a yield curve inversion draws attention.",
    "Explain to a small business owner how a revolving credit facility works.",
    "Write a paragraph explaining counterparty risk to a non-specialist colleague.",
    "Explain the role of a central counterparty in cleared derivatives markets.",
    "Draft a client-facing explanation of what an exchange-traded fund is.",
    "Explain, in plain language, what happens during a corporate bond issuance.",
    "Write a short briefing on how currency hedging works for an international portfolio.",
    "Explain the concept of duration in fixed income to a new analyst.",
    "Describe how a bank's net interest margin responds to rising rates.",
    "Explain what 'know your customer' obligations mean in practice.",
    "Write a short note on why liquidity matters in stressed markets.",
    "Explain the difference between a market order and a limit order to a new investor.",
    "Draft an internal summary of the purpose of stress testing for a bank.",
    "Explain what a credit default swap does, for a general business audience.",
    "Write a paragraph on how inflation expectations feed into long-term bond yields.",
    "Explain to a client why diversification reduces portfolio risk.",
    "Describe the function of a prime brokerage relationship.",
    "Explain settlement risk in foreign exchange transactions.",
    "Write a short explanation of what an initial public offering involves.",
    "Explain the difference between fiscal and monetary policy for a client briefing.",
    "Describe how collateral requirements change in a volatile market.",
    "Explain what operational risk means for a large bank, with examples.",
]

SUITES: dict[str, list[str]] = {
    "creative": CREATIVE,
    "open_ended": OPEN_ENDED,
    "factual": FACTUAL,
    "code": CODE,
    "structured": STRUCTURED,
    "financial": FINANCIAL,
}

# Suites where the model is genuinely free to choose wording.  Used as the
# default for studies that are about the detector rather than about entropy.
HIGH_ENTROPY_SUITES = ("creative", "open_ended", "financial")


def get_suite(name: str, limit: int | None = None) -> list[str]:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; choose from {sorted(SUITES)}")
    prompts = SUITES[name]
    return prompts[:limit] if limit else list(prompts)


def all_prompts(suites: Sequence[str] | None = None, limit_per_suite: int | None = None) -> dict[str, list[str]]:
    names = list(suites) if suites else list(SUITES)
    return {n: get_suite(n, limit_per_suite) for n in names}


def load_human_texts(
    n: int = 500,
    *,
    min_words: int = 40,
    max_words: int = 400,
    cache_dir: str | None = None,
    seed: int = 0,
) -> list[str]:
    """Load human-written passages to measure the detector's false-positive rate.

    Uses WikiText-103 (raw), which is human-written encyclopaedic prose that
    predates the model.  It is deliberately *not* model output: the question this
    corpus answers is "how often would we wrongly flag a person's writing?", and
    that is the number a compliance reviewer will ask for first.
    """
    import random

    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="test", cache_dir=cache_dir
    )
    out: list[str] = []
    buf: list[str] = []
    for row in ds:
        line = _detokenize_wikitext(row["text"].strip())
        if not line:
            continue
        if re.match(r"^=+ .* =+$", line):  # section heading: paragraph boundary
            buf = []
            continue
        buf.append(line)
        joined = " ".join(buf)
        if len(joined.split()) >= min_words:
            words = joined.split()
            out.append(" ".join(words[:max_words]))
            buf = []
        if len(out) >= n * 3:
            break
    random.Random(seed).shuffle(out)
    return out[:n]

def _detokenize_wikitext(line: str) -> str:
    """Undo WikiText's tokenised surface form.

    WikiText ships pre-tokenised: ``@-@`` for hyphens, ``@,@`` inside numbers,
    and a space before every punctuation mark.  Left as-is, those artefacts
    would be tokenised very differently from ordinary prose, and the null
    distribution measured on them would not be the null distribution of the
    human writing we actually care about.
    """
    line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    line = re.sub(r"\s+([,.;:!?%\)\]])", r"\1", line)
    line = re.sub(r"([\(\[])\s+", r"\1", line)
    line = re.sub(r"\s+('s|n't|'re|'ve|'ll|'d|'m)\b", r"\1", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()
