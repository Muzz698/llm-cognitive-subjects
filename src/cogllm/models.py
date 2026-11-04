"""Cognitive models fitted to bandit choices by maximum likelihood.

- Q-learning (Rescorla-Wagner + softmax): alpha = learning rate (how fast values
  update after a reward), beta = inverse temperature (how strictly the agent picks
  the higher value; low beta = more random exploration).
- Q-learning with perseveration: as above, plus kappa = a bonus for repeating the
  previous choice regardless of reward (kappa > 0 sticky, kappa < 0 alternating).
- Win-stay lose-shift (WSLS) with noise epsilon.
- Random choice (baseline).

Models are compared with BIC (lower is better). `parameter_recovery` checks that
the fitting procedure can find known parameters in simulated data, which must be
true before fitted parameters are interpreted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .bandit import make_games, play

BOUNDS_Q = [(1e-3, 1.0), (1e-3, 30.0)]
BOUNDS_Q_PERSEV = [*BOUNDS_Q, (-10.0, 10.0)]


def _games(df: pd.DataFrame):
    for _, g in df.sort_values(["game", "trial"]).groupby("game"):
        yield g["choice"].to_numpy(int), g["reward"].to_numpy(float)


def nll_q(params, df: pd.DataFrame, q0: float = 0.5) -> float:
    return nll_q_persev([*params, 0.0], df, q0)


def nll_q_persev(params, df: pd.DataFrame, q0: float = 0.5) -> float:
    alpha, beta, kappa = params
    total = 0.0
    for choices, rewards in _games(df):
        q = np.full(2, q0)
        prev = None
        for c, r in zip(choices, rewards, strict=True):
            logits = beta * q
            if prev is not None:
                logits[prev] += kappa
            logits -= logits.max()
            total -= logits[c] - np.log(np.exp(logits).sum())
            q[c] += alpha * (r - q[c])
            prev = c
    return float(total)


def nll_wsls(params, df: pd.DataFrame) -> float:
    (eps,) = params
    total = 0.0
    for choices, rewards in _games(df):
        total -= np.log(0.5)  # first choice of a game
        for t in range(1, len(choices)):
            stay = rewards[t - 1] == 1
            target = choices[t - 1] if stay else 1 - choices[t - 1]
            p = 1 - eps / 2 if choices[t] == target else eps / 2
            total -= np.log(p)
    return float(total)


def _minimize(fun, df, bounds, n_starts, seed):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
        res = minimize(fun, x0, args=(df,), bounds=bounds, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return best


def bic(nll: float, k: int, n: int) -> float:
    return 2 * nll + k * np.log(n)


def fit_participant(df: pd.DataFrame, n_starts: int = 10, seed: int = 0) -> dict:
    n = len(df)
    q = _minimize(nll_q, df, BOUNDS_Q, n_starts, seed)
    qp = _minimize(nll_q_persev, df, BOUNDS_Q_PERSEV, n_starts, seed)
    w = _minimize(nll_wsls, df, [(1e-3, 1.0)], n_starts, seed)
    nll_random = n * np.log(2)
    bics = {"q_learning": bic(q.fun, 2, n), "q_persev": bic(qp.fun, 3, n),
            "wsls": bic(w.fun, 1, n), "random": bic(nll_random, 0, n)}
    return {
        "alpha": float(q.x[0]),
        "beta": float(q.x[1]),
        "persev_alpha": float(qp.x[0]),
        "persev_beta": float(qp.x[1]),
        "kappa": float(qp.x[2]),
        "wsls_epsilon": float(w.x[0]),
        "nll_q_learning": float(q.fun),
        **{f"bic_{k}": float(v) for k, v in bics.items()},
        "best_model": min(bics, key=bics.get),
    }


def fit_all(df: pd.DataFrame, n_starts: int = 10, seed: int = 0) -> pd.DataFrame:
    rows = []
    for pid, d in df.groupby("participant"):
        rows.append({"participant": pid, **fit_participant(d, n_starts, seed)})
    return pd.DataFrame(rows)


def parameter_recovery(n_agents: int = 30, n_games: int = 4, n_trials: int = 20,
                       seed: int = 0, n_starts: int = 5, persev: bool = False) -> pd.DataFrame:
    """Simulate Q-learners with known parameters, fit them, return true vs fitted.
    With persev, the learners also have a stickiness kappa and the perseveration
    model's fitted parameters are returned."""
    from .agents import QLearningAgent

    rng = np.random.default_rng(seed)
    games = make_games(n_games, seed)
    rows = []
    for i in range(n_agents):
        alpha, beta = rng.uniform(0.05, 0.95), rng.uniform(1.0, 15.0)
        kappa = rng.uniform(-2.0, 4.0) if persev else 0.0
        agent = QLearningAgent(alpha, beta, seed=seed + i, kappa=kappa)
        df = play(agent, i, games, n_trials, seed + i)
        fit = fit_participant(df, n_starts, seed)
        if persev:
            rows.append({"true_alpha": alpha, "true_beta": beta, "true_kappa": kappa,
                         "fit_alpha": fit["persev_alpha"], "fit_beta": fit["persev_beta"],
                         "fit_kappa": fit["kappa"]})
        else:
            rows.append({"true_alpha": alpha, "true_beta": beta,
                         "fit_alpha": fit["alpha"], "fit_beta": fit["beta"]})
    return pd.DataFrame(rows)
