"""Figures: learning curves, fitted parameters, parameter recovery, CRT scores."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PALETTE = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def learning_curves(data: dict[str, pd.DataFrame], path: str | Path) -> None:
    """Share of choices of the better machine on each trial, averaged over games."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, df) in enumerate(data.items()):
        curve = (df["choice"] == df["best_arm"]).groupby(df["trial"]).mean()
        sem = (df["choice"] == df["best_arm"]).groupby(df["trial"]).sem()
        ax.plot(curve.index + 1, curve.values, color=PALETTE[i % 6], label=name, lw=2)
        ax.fill_between(curve.index + 1, curve - sem, curve + sem, color=PALETTE[i % 6],
                        alpha=0.15, lw=0)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(choose better machine)")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Learning curves")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parameters(fits: pd.DataFrame, path: str | Path) -> None:
    """Fitted learning rate vs inverse temperature, one dot per participant."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (name, d) in enumerate(fits.groupby("group", sort=False)):
        ax.scatter(d["alpha"], d["beta"], s=28, alpha=0.75, color=PALETTE[i % 6], label=name,
                   edgecolor="none")
    ax.set_xlabel("Learning rate (alpha)")
    ax.set_ylabel("Inverse temperature (beta), higher = less random")
    ax.set_title("Fitted Q-learning parameters")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def recovery(df: pd.DataFrame, path: str | Path) -> None:
    params = [c.removeprefix("true_") for c in df.columns if c.startswith("true_")]
    fig, axes = plt.subplots(1, len(params), figsize=(4.5 * len(params), 4))
    for ax, p in zip(axes, params, strict=True):
        t, f = df[f"true_{p}"], df[f"fit_{p}"]
        r = spearmanr(t, f)[0]
        ax.scatter(t, f, s=22, alpha=0.7, color=PALETTE[0], edgecolor="none")
        lim = [min(t.min(), f.min()), max(t.max(), f.max())]
        ax.plot(lim, lim, color="grey", ls="--", lw=1)
        ax.set_xlabel(f"True {p}")
        ax.set_ylabel(f"Fitted {p}")
        ax.set_title(f"{p}: Spearman r = {r:.2f}")
        _style(ax)
    fig.suptitle("Parameter recovery on simulated learners")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def crt_scores(df: pd.DataFrame, path: str | Path) -> None:
    """Share of correct / intuitive / other answers, original vs new items, per model."""
    table = (df.groupby(["model", "kind"])["score"].value_counts(normalize=True)
             .unstack(fill_value=0).reindex(columns=["correct", "intuitive", "other"],
                                            fill_value=0))
    fig, ax = plt.subplots(figsize=(1.4 * len(table) + 3, 4))
    bottom = np.zeros(len(table))
    colors = {"correct": "#55A868", "intuitive": "#C44E52", "other": "#BBBBBB"}
    labels = [f"{m}\n{k}" for m, k in table.index]
    for col in table.columns:
        ax.bar(labels, table[col], bottom=bottom, color=colors[col], label=col)
        bottom += table[col].to_numpy()
    ax.set_ylabel("Share of answers")
    ax.set_title("Cognitive Reflection Test")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
