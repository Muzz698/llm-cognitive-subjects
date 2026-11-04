"""Agents that play the bandit: simulated learners (for testing and parameter
recovery) and an LLM agent that answers through a local Ollama server."""
from __future__ import annotations

import re

import numpy as np

from .llm import OllamaClient


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def choose(self, history):
        return int(self.rng.integers(2))


class QLearningAgent:
    """Rescorla-Wagner learner with softmax choice. Ground truth for recovery.
    kappa > 0 adds a bonus for repeating the previous choice (perseveration)."""

    def __init__(self, alpha: float, beta: float, seed: int = 0, q0: float = 0.5,
                 kappa: float = 0.0):
        self.alpha, self.beta, self.q0, self.kappa = alpha, beta, q0, kappa
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.q = np.full(2, self.q0)

    def choose(self, history):
        if history:
            c, r = history[-1]
            self.q[c] += self.alpha * (r - self.q[c])
        logits = self.beta * self.q
        if history:
            logits[history[-1][0]] += self.kappa
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return int(self.rng.random() < p[1])


INSTRUCTIONS = (
    "You are going to visit a casino with two slot machines, Machine {a} and Machine {b}. "
    "Each machine pays out 1 point or 0 points. The machines differ in how often they pay "
    "out, but you do not know which one is better. Your goal is to win as many points as "
    "possible over {n} rounds. You can only learn about a machine by playing it.\n"
)


class LLMAgent:
    """Plays the bandit by reading a text description of everything so far.

    The two machines get neutral letters (F and J, as in Binz & Schulz, 2023). On
    each game the letters are randomly assigned to the arms, so a model that simply
    prefers one letter cannot look like a good learner.
    """

    LETTERS = ("F", "J")

    def __init__(self, client: OllamaClient, n_trials: int, seed: int = 0,
                 max_retries: int = 3):
        self.client = client
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.max_retries = max_retries
        self.invalid = 0  # answers that could not be parsed, even after retries
        self.calls = 0
        self.reset()

    def reset(self):
        # label_of[arm] is the letter shown for that arm in this game
        self.label_of = list(self.LETTERS)
        if self.rng.random() < 0.5:
            self.label_of.reverse()

    def prompt(self, history) -> str:
        a, b = sorted(self.label_of)
        text = INSTRUCTIONS.format(a=a, b=b, n=self.n_trials)
        if history:
            text += "\nSo far:\n"
            for t, (c, r) in enumerate(history, 1):
                points = "point" if r == 1 else "points"
                text += f"Round {t}: you played Machine {self.label_of[c]} and won {r} {points}.\n"
        text += (f"\nRound {len(history) + 1}: which machine do you play? "
                 f"Answer with only the letter {a} or {b}.\nAnswer: Machine")
        return text

    def parse(self, text: str) -> int | None:
        match = re.search(r"\b([FJ])\b", text.strip().upper())
        if not match:
            return None
        return self.label_of.index(match.group(1))

    def choose(self, history):
        prompt = self.prompt(history)
        for attempt in range(self.max_retries):
            self.calls += 1
            seed = self.seed * 1000 + len(history) * 10 + attempt
            answer = self.client.generate(prompt, seed=seed)
            arm = self.parse(answer)
            if arm is not None:
                return arm
        self.invalid += 1
        return int(self.rng.integers(2))
