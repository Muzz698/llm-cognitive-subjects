"""Two-armed bandit task: choose between two slot machines and learn from rewards.

Each "game" has two machines with hidden win probabilities. A participant plays
several games of a fixed number of trials. The same task is given to LLMs and
(through a common CSV format) compared with human data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Columns every dataset (LLM, simulated or human) is converted to.
COLUMNS = ["participant", "game", "trial", "choice", "reward", "best_arm"]


@dataclass
class TwoArmedBandit:
    """Bernoulli bandit: arm k pays 1 point with probability p[k], else 0."""

    probs: tuple[float, float]
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.probs))

    def pull(self, arm: int) -> int:
        if arm not in (0, 1):
            raise ValueError(f"arm must be 0 or 1, got {arm}")
        return int(self.rng.random() < self.probs[arm])


def make_games(n_games: int, seed: int, low: float = 0.1, high: float = 0.9,
               min_gap: float = 0.2) -> list[tuple[float, float]]:
    """Random reward probabilities for each game, with the two arms at least
    `min_gap` apart so there is something to learn."""
    rng = np.random.default_rng(seed)
    games = []
    while len(games) < n_games:
        p = tuple(float(x) for x in np.round(rng.uniform(low, high, 2), 2))
        if abs(p[0] - p[1]) >= min_gap:
            games.append(p)
    return games


def play(agent, participant: int | str, games: list[tuple[float, float]], n_trials: int,
         seed: int) -> pd.DataFrame:
    """Let one agent play all games. The agent must have
    `reset()` and `choose(history) -> int` where history is a list of
    (choice, reward) pairs from the current game."""
    rows = []
    for g, probs in enumerate(games):
        env = TwoArmedBandit(probs, np.random.default_rng([seed, g]))
        agent.reset()
        history: list[tuple[int, int]] = []
        for t in range(n_trials):
            choice = int(agent.choose(history))
            reward = env.pull(choice)
            history.append((choice, reward))
            rows.append((participant, g, t, choice, reward, env.best_arm))
    return pd.DataFrame(rows, columns=COLUMNS)


def behaviour_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Model-free measures per participant.

    - accuracy: share of trials where the better machine was chosen
    - switch_rate: share of trials where the choice differs from the last one
    - win_stay / lose_shift: repeat after a win, switch after a loss
    """
    out = []
    for pid, d in df.sort_values(["participant", "game", "trial"]).groupby("participant"):
        prev = d.groupby("game")[["choice", "reward"]].shift(1)
        has_prev = prev["choice"].notna()
        same = d["choice"] == prev["choice"]
        won = prev["reward"] == 1
        out.append({
            "participant": pid,
            "accuracy": float((d["choice"] == d["best_arm"]).mean()),
            "switch_rate": float((~same[has_prev]).mean()),
            "win_stay": float(same[has_prev & won].mean()) if (has_prev & won).any() else np.nan,
            "lose_shift": float((~same)[has_prev & ~won].mean())
            if (has_prev & ~won).any() else np.nan,
            "total_reward": int(d["reward"].sum()),
        })
    return pd.DataFrame(out)
