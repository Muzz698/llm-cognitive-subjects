"""Tests run without Ollama or internet: LLMs are replaced by fake clients."""
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import spearmanr

from cogllm import cli
from cogllm.agents import LLMAgent, QLearningAgent, RandomAgent
from cogllm.bandit import COLUMNS, TwoArmedBandit, behaviour_summary, make_games, play
from cogllm.crt import ITEMS, score
from cogllm.models import fit_participant, nll_q, nll_q_persev, parameter_recovery
from cogllm.psych101 import parse_lefebvre


class FakeClient:
    """Stands in for OllamaClient. Returns the queued answers in a loop."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def generate(self, prompt, seed=0, max_tokens=None):
        self.prompts.append(prompt)
        return self.answers[(len(self.prompts) - 1) % len(self.answers)]

    def check(self):
        pass


# ---------- bandit task ----------

def test_games_have_a_learnable_gap_and_are_reproducible():
    games = make_games(20, seed=1)
    assert games == make_games(20, seed=1)
    assert all(abs(a - b) >= 0.2 for a, b in games)


def test_bandit_rewards_follow_probabilities():
    env = TwoArmedBandit((0.2, 0.8), np.random.default_rng(0))
    rewards = [env.pull(1) for _ in range(5000)]
    assert abs(np.mean(rewards) - 0.8) < 0.03
    assert env.best_arm == 1
    with pytest.raises(ValueError):
        env.pull(2)


def test_q_learner_beats_random_agent():
    games = make_games(10, seed=0)
    q = play(QLearningAgent(0.3, 8, seed=0), "q", games, 30, seed=0)
    r = play(RandomAgent(seed=0), "r", games, 30, seed=0)
    acc = lambda d: (d["choice"] == d["best_arm"]).mean()  # noqa: E731
    assert list(q.columns) == COLUMNS
    assert acc(q) > acc(r) + 0.15


def test_behaviour_summary_counts_stays_and_switches():
    # choices 0,0,1,1 with rewards 1,0,0,1 -> win-stay once, lose-shift once, lose-stay once
    df = pd.DataFrame({"participant": "p", "game": 0, "trial": range(4),
                       "choice": [0, 0, 1, 1], "reward": [1, 0, 0, 1], "best_arm": 1})
    s = behaviour_summary(df).iloc[0]
    assert s["accuracy"] == 0.5
    assert s["switch_rate"] == pytest.approx(1 / 3)
    assert s["win_stay"] == 1.0
    assert s["lose_shift"] == 0.5


# ---------- cognitive models ----------

def test_q_likelihood_matches_hand_calculation():
    # One game, two trials. Trial 1: Q = [0.5, 0.5] -> P(choice) = 0.5.
    # Choose 0, reward 1, alpha 0.5 -> Q = [0.75, 0.5]; beta 2 -> P(0) = 1/(1+e^-0.5).
    df = pd.DataFrame({"game": [0, 0], "trial": [0, 1], "choice": [0, 0], "reward": [1, 0]})
    expected = -np.log(0.5) - np.log(1 / (1 + np.exp(-0.5)))
    assert nll_q([0.5, 2.0], df) == pytest.approx(expected)


def test_persev_likelihood_adds_bonus_to_previous_choice():
    # As above, but trial 2 also gets kappa = 1 for repeating choice 0 -> logit gap 1.5.
    df = pd.DataFrame({"game": [0, 0], "trial": [0, 1], "choice": [0, 0], "reward": [1, 0]})
    expected = -np.log(0.5) - np.log(1 / (1 + np.exp(-1.5)))
    assert nll_q_persev([0.5, 2.0, 1.0], df) == pytest.approx(expected)
    assert nll_q_persev([0.5, 2.0, 0.0], df) == pytest.approx(nll_q([0.5, 2.0], df))


def test_sticky_learner_is_best_explained_by_perseveration_model():
    agent = QLearningAgent(0.3, 3.0, seed=5, kappa=3.0)
    df = play(agent, "s", make_games(10, seed=5), 30, seed=5)
    fit = fit_participant(df, n_starts=3)
    assert fit["best_model"] == "q_persev"
    assert 1.5 < fit["kappa"] < 5.0


def test_random_agent_is_best_explained_by_random_model():
    df = play(RandomAgent(seed=3), "r", make_games(4, seed=3), 20, seed=3)
    assert fit_participant(df, n_starts=3)["best_model"] == "random"


def test_parameter_recovery_works_with_default_design():
    """Fitted parameters must track the true ones before we interpret them.
    With the default 10 games x 30 trials: alpha r ~ 0.94, beta r ~ 0.8-0.9 across
    seeds. With only 80 trials beta was not reliable, which is why 300 is the default."""
    cfg = yaml.safe_load(open("configs/default.yaml"))["bandit"]
    df = parameter_recovery(n_agents=40, n_games=cfg["n_games"], n_trials=cfg["n_trials"],
                            seed=0, n_starts=3)
    assert spearmanr(df["true_alpha"], df["fit_alpha"])[0] > 0.85
    assert spearmanr(df["true_beta"], df["fit_beta"])[0] > 0.7


def test_kappa_is_recovered_with_default_design():
    cfg = yaml.safe_load(open("configs/default.yaml"))["bandit"]
    df = parameter_recovery(n_agents=20, n_games=cfg["n_games"], n_trials=cfg["n_trials"],
                            seed=1, n_starts=3, persev=True)
    assert spearmanr(df["true_kappa"], df["fit_kappa"])[0] > 0.8


# ---------- LLM agent ----------

def test_llm_agent_parses_letters_and_hides_arm_identity():
    agent = LLMAgent(FakeClient([" F"]), n_trials=5, seed=0)
    choices = set()
    for _ in range(20):  # letters are reassigned at random each game
        agent.reset()
        choices.add(agent.choose([]))
    assert choices == {0, 1}  # always "F", but F maps to both arms across games
    assert agent.parse("j.") == agent.label_of.index("J")
    assert agent.parse("I would choose Machine F") == agent.label_of.index("F")
    assert agent.parse("no idea") is None


def test_llm_agent_retries_then_falls_back_and_counts_it():
    client = FakeClient(["hmm", "not sure", "maybe"])
    agent = LLMAgent(client, n_trials=5, seed=0, max_retries=3)
    assert agent.choose([]) in (0, 1)
    assert agent.invalid == 1 and agent.calls == 3


def test_llm_prompt_contains_history():
    agent = LLMAgent(FakeClient(["F"]), n_trials=10, seed=0)
    text = agent.prompt([(0, 1), (1, 0)])
    assert f"Round 1: you played Machine {agent.label_of[0]} and won 1 point." in text
    assert f"Round 2: you played Machine {agent.label_of[1]} and won 0 points." in text
    assert text.rstrip().endswith("Answer: Machine")


# ---------- CRT ----------

def _item(item_id):
    return next(i for i in ITEMS if i.id == item_id)


@pytest.mark.parametrize("answer,expected", [
    ("5", "correct"), ("$0.05", "correct"), ("5 cents", "correct"),
    ("10", "intuitive"), ("0.10", "intuitive"), ("7", "other"), ("I don't know", "other"),
])
def test_crt_scoring_bat_ball(answer, expected):
    assert score(_item("bat_ball"), answer) == expected


def test_crt_scoring_lily_pads():
    item = _item("lily_pads")
    assert score(item, "The answer is 47 days.") == "correct"
    assert score(item, "24") == "intuitive"


def test_crt_answer_key_is_right():
    """Recompute every answer from the puzzle's structure."""
    # "costs X in total, one costs D more": smaller = (X - D) / 2
    for item_id, total, diff in [("bat_ball", 110, 100), ("notebook_pen", 220, 200),
                                 ("pizza_drink", 1300, 1200)]:
        assert _item(item_id).correct == (total - diff) / 2
        assert _item(item_id).intuitive == total - diff
    # "N workers take T to make N things": always T
    for item_id, t in [("widgets", 5), ("printers", 3), ("cooks", 8)]:
        assert _item(item_id).correct == t
    # "doubles every period, full after D": half after D - 1
    for item_id, d in [("lily_pads", 48), ("mold", 20), ("followers", 30)]:
        assert _item(item_id).correct == d - 1
        assert _item(item_id).intuitive == d / 2


# ---------- human data ----------

def test_parse_lefebvre_keeps_asymmetric_casinos_in_visit_order():
    text = (
        "You go to casino 3. You can choose between machines B and C. "
        "You press <<C>> and receive 0.5 points.\n"
        "You go to casino 1. You can choose between machines F and I. "
        "You press <<I>> and receive 0.0 points.\n"
        "You go to casino 2. You can choose between machines L and J. "
        "You press <<L>> and receive 0.0 points.\n"
        "You go to casino 3. You can choose between machines B and C. "
        "You press <<B>> and receive 0.0 points.\n"
    )
    df = parse_lefebvre(text, "h1")
    assert list(df.columns) == COLUMNS
    assert df.values.tolist() == [["h1", 3, 0, 1, 1, 1], ["h1", 2, 0, 0, 0, 0],
                                  ["h1", 3, 1, 0, 0, 1]]


# ---------- command line, end to end ----------

@pytest.fixture
def small_config(tmp_path):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["output_dir"] = str(tmp_path)
    cfg["bandit"].update(n_participants=4, n_games=2, n_trials=10)
    cfg["crt"]["n_samples"] = 2
    cfg["fitting"]["n_starts"] = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, tmp_path


def test_cli_bandit_and_analyze(small_config, monkeypatch):
    config, out = small_config
    monkeypatch.setattr(cli, "OllamaClient", lambda model, **kw: FakeClient([" F", " J"]))
    cli.main(["--config", str(config), "bandit", "--model", "fake:1b"])
    cli.main(["--config", str(config), "simulate", "--agent", "qlearning"])
    cli.main(["--config", str(config), "analyze", str(out / "bandit_fake-1b.csv"),
              str(out / "bandit_sim-qlearning.csv"), "--reference", "sim-qlearning"])
    fits = pd.read_csv(out / "fits.csv")
    assert set(fits["group"]) == {"fake-1b", "sim-qlearning"}
    for name in ["learning_curves.png", "parameters.png", "analysis.json", "behaviour.csv"]:
        assert (out / name).exists()


def test_cli_uses_fixed_games_from_config(small_config):
    config, out = small_config
    cfg = yaml.safe_load(config.read_text())
    cfg["bandit"].update(n_games=2, games=[[0.75, 0.25], [0.25, 0.75]])
    config.write_text(yaml.safe_dump(cfg))
    cli.main(["--config", str(config), "simulate", "--agent", "random"])
    df = pd.read_csv(out / "bandit_sim-random.csv")
    assert df.groupby("game")["best_arm"].first().tolist() == [0, 1]


def test_cli_crt(small_config, monkeypatch):
    config, out = small_config
    monkeypatch.setattr(cli, "OllamaClient", lambda model, **kw: FakeClient(["5"]))
    cli.main(["--config", str(config), "crt", "--model", "fake:1b"])
    df = pd.read_csv(out / "crt_fake-1b.csv")
    assert len(df) == len(ITEMS) * 2
    assert (df.loc[df["item"] == "bat_ball", "score"] == "correct").all()
    assert (out / "crt_scores.png").exists()
