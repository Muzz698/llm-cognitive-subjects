"""Cognitive Reflection Test (CRT): short puzzles with a tempting wrong answer.

Each answer is scored as `correct`, `intuitive` (the tempting wrong answer, a sign
of fast, unreflective thinking) or `other`.

The three original items (Frederick, 2005) are very likely in LLM training data,
so a model may simply remember them. The "new" items keep the same structure with
different numbers and stories. A model that solves the originals but fails the new
versions is probably recalling, not reasoning. Comparing the two sets is part of
the analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    id: str
    kind: str  # "original" or "new"
    question: str
    correct: float
    intuitive: float


ITEMS = [
    # Frederick (2005), Journal of Economic Perspectives 19(4).
    Item("bat_ball", "original",
         "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
         "How much does the ball cost, in cents?", 5, 10),
    Item("widgets", "original",
         "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take "
         "100 machines to make 100 widgets, in minutes?", 5, 100),
    Item("lily_pads", "original",
         "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. "
         "If it takes 48 days for the patch to cover the entire lake, how long would it "
         "take for the patch to cover half of the lake, in days?", 47, 24),
    # New items with the same structure (written for this project).
    Item("notebook_pen", "new",
         "A notebook and a pen cost $2.20 in total. The notebook costs $2.00 more than "
         "the pen. How much does the pen cost, in cents?", 10, 20),
    Item("pizza_drink", "new",
         "A pizza and a drink cost $13 in total. The pizza costs $12 more than the drink. "
         "How much does the drink cost, in cents?", 50, 100),
    Item("printers", "new",
         "If 3 printers take 3 minutes to print 3 pages, how long would it take 60 "
         "printers to print 60 pages, in minutes?", 3, 60),
    Item("cooks", "new",
         "If 8 cooks need 8 hours to bake 8 cakes, how long would 40 cooks need to bake "
         "40 cakes, in hours?", 8, 40),
    Item("mold", "new",
         "Mold on a slice of bread doubles in area every day. If it takes 20 days for the "
         "mold to cover the whole slice, how long does it take to cover half of the slice, "
         "in days?", 19, 10),
    Item("followers", "new",
         "The number of followers of an account doubles every week. If it takes 30 weeks "
         "to reach one million followers, how many weeks did it take to reach half a "
         "million followers?", 29, 15),
]

# Human reference (original 3 items): Frederick (2005), N = 3,428 participants,
# mean 1.24 of 3 correct. Check the paper before quoting exact numbers.
HUMAN_REFERENCE = {"source": "Frederick (2005)", "n": 3428, "mean_correct_of_3": 1.24}

PROMPT = ("Answer the following question. Give only the final number, with no units "
          "and no explanation.\n\nQuestion: {q}\nAnswer:")


def extract_number(text: str) -> float | None:
    """First number in the answer, or None if there is none."""
    match = re.search(r"-?\d+(?:[.,]\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def score(item: Item, answer: str) -> str:
    value = extract_number(answer)
    if value is None:
        return "other"
    candidates = {value}
    if "cents" in item.question:
        candidates.add(round(value * 100, 6))  # answer given in dollars, e.g. "0.05"
    if item.correct in candidates:
        return "correct"
    if item.intuitive in candidates:
        return "intuitive"
    return "other"


def run_crt(client, n_samples: int = 10, seed: int = 0) -> list[dict]:
    """Ask every item `n_samples` times (different seeds) and score each answer."""
    rows = []
    for item in ITEMS:
        for s in range(n_samples):
            answer = client.generate(PROMPT.format(q=item.question), seed=seed + s,
                                     max_tokens=20)
            rows.append({"item": item.id, "kind": item.kind, "sample": s,
                         "answer": answer.strip(), "score": score(item, answer)})
    return rows
