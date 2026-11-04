"""Command-line interface.

    cogllm recovery                         check the model fitting works
    cogllm simulate --agent qlearning       simulated learners (no LLM needed)
    cogllm bandit --model llama3.2:3b       an LLM plays the bandit (needs Ollama)
    cogllm crt --model llama3.2:3b          an LLM takes the CRT (needs Ollama)
    cogllm analyze results/bandit_*.csv     fit models, compare groups, make figures
    cogllm psych101-peek                    inspect the human dataset's format
    cogllm psych101-convert                 human bandit data in the common format
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu, spearmanr

from . import plots
from .agents import LLMAgent, QLearningAgent, RandomAgent
from .bandit import behaviour_summary, make_games, play
from .crt import HUMAN_REFERENCE, run_crt
from .llm import OllamaClient
from .models import fit_all, parameter_recovery


def _slug(name: str) -> str:
    return name.replace(":", "-").replace("/", "-")


def _out(cfg) -> Path:
    out = Path(cfg.get("output_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _games(cfg):
    """Reward probabilities per game: fixed in the config, or drawn at random."""
    b = cfg["bandit"]
    if "games" in b:
        return [tuple(g) for g in b["games"]]
    return make_games(b["n_games"], cfg["seed"])


def cmd_recovery(cfg, args):
    b = cfg["bandit"]
    df = parameter_recovery(args.n, b["n_games"], b["n_trials"], cfg["seed"],
                            persev=args.persev)
    out = _out(cfg)
    name = "recovery_persev" if args.persev else "recovery"
    df.to_csv(out / f"{name}.csv", index=False)
    plots.recovery(df, out / f"{name}.png")
    for p in ("alpha", "beta", "kappa") if args.persev else ("alpha", "beta"):
        r = spearmanr(df[f"true_{p}"], df[f"fit_{p}"])[0]
        verdict = "good" if r >= 0.8 else ("acceptable" if r >= 0.7 else "too weak to interpret")
        print(f"  {p}: Spearman r (true vs fitted) = {r:.2f} -> {verdict}")
    print(f"Saved {out}/{name}.csv and {name}.png")


def cmd_simulate(cfg, args):
    b = cfg["bandit"]
    games = _games(cfg)
    rng = np.random.default_rng(cfg["seed"])
    frames = []
    for p in range(b["n_participants"]):
        if args.agent == "random":
            agent = RandomAgent(seed=cfg["seed"] + p)
        else:
            agent = QLearningAgent(rng.uniform(0.1, 0.6), rng.uniform(3, 10), seed=cfg["seed"] + p)
        frames.append(play(agent, f"{args.agent}_{p}", games, b["n_trials"], cfg["seed"] + p))
    path = _out(cfg) / f"bandit_sim-{args.agent}.csv"
    pd.concat(frames).to_csv(path, index=False)
    print(f"Saved {path}")


def cmd_bandit(cfg, args):
    b = cfg["bandit"]
    client = OllamaClient(args.model, temperature=b["temperature"])
    client.check()
    games = _games(cfg)
    frames, invalid, calls = [], 0, 0
    for p in range(b["n_participants"]):
        agent = LLMAgent(client, b["n_trials"], seed=cfg["seed"] + p)
        frames.append(play(agent, f"{_slug(args.model)}_{p}", games, b["n_trials"],
                           cfg["seed"] + p))
        invalid += agent.invalid
        calls += agent.calls
        acc = (frames[-1]["choice"] == frames[-1]["best_arm"]).mean()
        print(f"  participant {p + 1}/{b['n_participants']}: accuracy {acc:.2f}, "
              f"unparseable answers so far {invalid}", flush=True)
    path = _out(cfg) / f"bandit_{_slug(args.model)}.csv"
    pd.concat(frames).to_csv(path, index=False)
    total = b["n_participants"] * b["n_games"] * b["n_trials"]
    print(f"Saved {path}. {invalid}/{total} choices were random fallbacks "
          f"({calls} model calls).")


def cmd_crt(cfg, args):
    c = cfg["crt"]
    client = OllamaClient(args.model, temperature=c["temperature"])
    client.check()
    df = pd.DataFrame(run_crt(client, c["n_samples"], cfg["seed"]))
    df.insert(0, "model", args.model)
    path = _out(cfg) / f"crt_{_slug(args.model)}.csv"
    df.to_csv(path, index=False)
    summary = df.groupby("kind")["score"].value_counts(normalize=True).unstack(fill_value=0)
    print(summary.round(2).to_string())
    orig = df[df["kind"] == "original"]
    per_sample = orig.groupby("sample")["score"].apply(lambda s: (s == "correct").sum())
    print(f"\nOriginal 3 items: mean {per_sample.mean():.2f} of 3 correct "
          f"(humans: {HUMAN_REFERENCE['mean_correct_of_3']}, {HUMAN_REFERENCE['source']})")
    all_crt = pd.concat(pd.read_csv(f) for f in sorted(_out(cfg).glob("crt_*.csv")))
    plots.crt_scores(all_crt, _out(cfg) / "crt_scores.png")
    print(f"Saved {path} and crt_scores.png")


def cmd_analyze(cfg, args):
    data = {Path(f).stem.removeprefix("bandit_"): pd.read_csv(f) for f in args.files}
    out = _out(cfg)
    fits, behaviour = [], []
    for name, df in data.items():
        print(f"Fitting {name} ...", flush=True)
        f = fit_all(df, cfg["fitting"]["n_starts"], cfg["seed"])
        f.insert(0, "group", name)
        fits.append(f)
        bsum = behaviour_summary(df)
        bsum.insert(0, "group", name)
        behaviour.append(bsum)
    fits, behaviour = pd.concat(fits), pd.concat(behaviour)
    fits.to_csv(out / "fits.csv", index=False)
    behaviour.to_csv(out / "behaviour.csv", index=False)

    cols = ["accuracy", "switch_rate", "win_stay", "lose_shift"]
    # alpha and beta mean something only where Q-learning is the best model
    q_best = fits[fits["best_model"] == "q_learning"]
    table = behaviour.groupby("group", sort=False)[cols].median().join(
        q_best.groupby("group", sort=False)[["alpha", "beta"]].median()).join(
        fits.groupby("group", sort=False)["kappa"].median())
    table["n_q_best"] = q_best.groupby("group", sort=False).size()
    table["n_q_best"] = table["n_q_best"].fillna(0).astype(int)
    table["best_model"] = fits.groupby("group", sort=False)["best_model"].agg(
        lambda s: s.value_counts().idxmax())
    print("\nMedians per group:\n" + table.round(3).to_string())

    tests = {}
    if args.reference and args.reference in data:
        for name in data:
            if name == args.reference:
                continue
            for p, pool in (("alpha", q_best), ("beta", q_best), ("kappa", fits)):
                ref = pool.loc[pool["group"] == args.reference, p]
                other = pool.loc[pool["group"] == name, p]
                if len(ref) < 2 or len(other) < 2:
                    continue
                stat, pval = mannwhitneyu(ref, other)
                tests[f"{name} vs {args.reference}: {p}"] = {"U": float(stat), "p": float(pval)}
        print("\nMann-Whitney U tests (alpha, beta: Q-learning-best participants only; "
              "kappa: all):")
        for k, v in tests.items():
            print(f"  {k}: U = {v['U']:.1f}, p = {v['p']:.4f}")
    (out / "analysis.json").write_text(json.dumps(
        {"medians": json.loads(table.to_json(orient="index")), "tests": tests}, indent=2))

    plots.learning_curves(data, out / "learning_curves.png")
    plots.parameters(fits, out / "parameters.png")
    print(f"\nSaved fits.csv, behaviour.csv, analysis.json and figures to {out}/")


def cmd_peek(cfg, args):
    from .psych101 import peek

    path = peek(_out(cfg) / "psych101_peek.txt")
    print(f"Saved {path}. Send this file's first part to continue with the human data.")


def cmd_convert(cfg, args):
    from .psych101 import convert_lefebvre

    path = _out(cfg) / "bandit_human.csv"
    df = convert_lefebvre(path)
    print(f"Saved {path}: {df['participant'].nunique()} participants, {len(df)} choices.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cogllm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("recovery", help="parameter recovery on simulated learners")
    p.add_argument("--n", type=int, default=40, help="number of simulated learners")
    p.add_argument("--persev", action="store_true",
                   help="learners with stickiness kappa, fitted with the perseveration model")
    p.set_defaults(func=cmd_recovery)

    p = sub.add_parser("simulate", help="simulated agents play the bandit")
    p.add_argument("--agent", choices=["qlearning", "random"], default="qlearning")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("bandit", help="an LLM plays the bandit through Ollama")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_bandit)

    p = sub.add_parser("crt", help="an LLM takes the Cognitive Reflection Test")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_crt)

    p = sub.add_parser("analyze", help="fit models and compare groups")
    p.add_argument("files", nargs="+", help="bandit CSV files")
    p.add_argument("--reference", help="group name to test the others against, e.g. human")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("psych101-peek", help="inspect the Psych-101 human dataset")
    p.set_defaults(func=cmd_peek)

    p = sub.add_parser("psych101-convert", help="convert the human bandit data (Lefebvre 2017)")
    p.set_defaults(func=cmd_convert)

    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.func(cfg, args)


if __name__ == "__main__":
    main()
