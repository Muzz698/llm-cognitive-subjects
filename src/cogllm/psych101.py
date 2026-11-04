"""Look inside Psych-101 (Binz et al., 2025), a collection of 160 psychology
experiments written as text, with trial-by-trial human choices.

`peek` finds experiments that look like bandit tasks and saves example transcripts,
so the exact format can be checked before parsing. `convert_lefebvre` then turns the
one experiment that matches our task into the common CSV format (see bandit.COLUMNS).

Lefebvre et al. (2017), exp. 1: 4 "casinos", each with two machines paying 0 or 0.5
points; each casino is visited 24 times, interleaved. Pooled over all 44 participants,
the payout rates per casino and machine (first-listed, second-listed) are:
casino 1: 25% / 25%, casino 2: 75% / 25%, casino 3: 25% / 75%, casino 4: 75% / 75%.
Only casinos 2 and 3 have a better machine, so only they are kept, as two games of
24 trials. Rewards 0.5 -> 1.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .bandit import COLUMNS

KEYWORDS = ("slot machine", "casino", "two-armed", "bandit")

LEFEBVRE = "lefebvre2017behavioural/exp1.csv"
# casino -> better machine (0 = first-listed letter, 1 = second); casinos 1 and 4 have none
LEFEBVRE_BEST_ARM = {2: 0, 3: 1}
_LEFEBVRE_TRIAL = re.compile(r"casino (\d)\. You can choose between machines (\w) and (\w)\. "
                             r"You press <<(\w)>> and receive (-?[\d.]+) points")


def _load_psych101():
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise RuntimeError('Install the extra first: pip install -e ".[human]"') from err
    return load_dataset("marcelbinz/Psych-101", split="train", streaming=True)


def peek(out_path: str | Path, keywords=KEYWORDS, max_rows: int = 60_000,
         n_examples: int = 3, chars: int = 2500) -> Path:
    stream = _load_psych101()
    names, examples, columns = Counter(), {}, None
    for i, row in enumerate(stream):
        if i >= max_rows:
            break
        columns = columns or list(row)
        text = str(row.get("text", ""))
        name = str(row.get("experiment", "unknown"))
        head = text[:1500].lower()
        if any(k in head for k in keywords):
            names[name] += 1
            if name not in examples and len(examples) < n_examples * 3:
                examples[name] = text[:chars]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Columns: {columns}", f"Rows scanned: {min(i + 1, max_rows)}",
             "", "Matching experiments (participants found):"]
    lines += [f"  {n}: {c}" for n, c in names.most_common()]
    for name, text in list(examples.items())[: n_examples * 3]:
        lines += ["", "=" * 80, f"EXPERIMENT: {name}", "=" * 80, text]
    out.write_text("\n".join(lines))
    return out


def parse_lefebvre(text: str, participant) -> pd.DataFrame:
    """One participant's transcript -> rows in the common format."""
    rows, visits = [], Counter()
    for m in _LEFEBVRE_TRIAL.finditer(text):
        casino, first, second, pressed, reward = m.groups()
        casino = int(casino)
        if casino not in LEFEBVRE_BEST_ARM:
            continue
        if pressed not in (first, second):
            raise ValueError(f"participant {participant}: pressed {pressed}, "
                             f"not {first} or {second}")
        rows.append((participant, casino, visits[casino], int(pressed == second),
                     int(float(reward) > 0), LEFEBVRE_BEST_ARM[casino]))
        visits[casino] += 1
    return pd.DataFrame(rows, columns=COLUMNS)


def convert_lefebvre(out_path: str | Path) -> pd.DataFrame:
    frames = [parse_lefebvre(row["text"], f"human_{row['participant']}")
              for row in _load_psych101() if row["experiment"] == LEFEBVRE]
    if not frames:
        raise RuntimeError(f"No rows for {LEFEBVRE} found in Psych-101")
    df = pd.concat(frames, ignore_index=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
