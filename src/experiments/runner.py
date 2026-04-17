"""Experiment runner: trains agents, runs episodes, saves results."""

import csv
import numpy as np
from pathlib import Path

from src.env.perturbed_kuhn import (
    KuhnPokerEnv, PerturbedKuhnPoker, PASS, BET, CARD_NAMES,
)
from src.agents.cfr_agent import CFRTrainer, CFRAgent
from src.agents.q_learning_agent import QLearningAgent
from src.utils.metrics import summarize_results
from src.utils.plotting import plot_results

ACTION_MAP = {"pass": PASS, "bet": BET}


def play_episode(env, agent, rng):
    """Play one self-play episode. Returns (reward_p0, trajectory)."""
    env.reset()
    trajectory = []

    while not env.is_terminal:
        player = env.current_player
        info = env.info_state_str(player)
        legal = env.legal_actions()
        action = agent.select_action(info, legal, rng)
        trajectory.append((player, info, action))
        env.step(action)

    return env.returns[0], trajectory


def run_experiment(config):
    """Run a full experiment from a config dict.

    Returns (csv_path, plot_path).
    """
    name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]
    num_episodes = config["experiment"]["num_episodes"]
    perturbation_ep = config["experiment"]["perturbation_episode"]
    cfr_iters = config["cfr"]["iterations"]
    alpha = config["q_learning"]["alpha"]
    epsilon = config["q_learning"]["epsilon"]
    removed = ACTION_MAP[config["perturbation"]["removed_action"]]
    affected = config["perturbation"]["affected_player"]
    root_only = config["perturbation"]["root_only"]

    rng = np.random.default_rng(seed)

    # --- Train CFR on the unperturbed game ---
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {name}")
    print(f"{'=' * 60}")
    print(f"Training CFR ({cfr_iters:,} iterations)...")

    trainer = CFRTrainer()
    trainer.train(cfr_iters)
    policy = trainer.get_average_strategy()

    print("Nash equilibrium strategy:")
    for info_set in sorted(policy):
        card = CARD_NAMES[int(info_set[0])]
        history = info_set[1:] or "(root)"
        p = policy[info_set]
        print(f"  {card} {history:>6s}:  pass={p[0]:.3f}  bet={p[1]:.3f}")

    # --- Create agents and environments ---
    cfr_agent = CFRAgent(policy)
    ql_agent = QLearningAgent(alpha=alpha, epsilon=epsilon)

    cfr_env = PerturbedKuhnPoker(
        KuhnPokerEnv(np.random.default_rng(rng.integers(1 << 31))),
        removed_action=removed, affected_player=affected, root_only=root_only,
    )
    ql_env = PerturbedKuhnPoker(
        KuhnPokerEnv(np.random.default_rng(rng.integers(1 << 31))),
        removed_action=removed, affected_player=affected, root_only=root_only,
    )
    cfr_rng = np.random.default_rng(rng.integers(1 << 31))
    ql_rng = np.random.default_rng(rng.integers(1 << 31))

    # --- Run episodes ---
    mode = "root only" if root_only else "all P0 nodes"
    print(f"\nRunning {num_episodes:,} episodes "
          f"(perturbation at {perturbation_ep:,}, {mode})\n")

    results = []

    for ep in range(num_episodes):
        if ep == perturbation_ep:
            print(f"  >>> Perturbation applied at episode {ep:,}")
            cfr_env.set_perturbed(True)
            ql_env.set_perturbed(True)

        cfr_reward, _ = play_episode(cfr_env, cfr_agent, cfr_rng)
        results.append((ep, cfr_reward, "CFR"))

        ql_reward, ql_traj = play_episode(ql_env, ql_agent, ql_rng)
        ql_agent.update(ql_traj, ql_reward)
        results.append((ep, ql_reward, "Q-Learning"))

        if (ep + 1) % 5000 == 0:
            recent_cfr = np.mean(
                [r[1] for r in results[-1000:] if r[2] == "CFR"])
            recent_ql = np.mean(
                [r[1] for r in results[-1000:] if r[2] == "Q-Learning"])
            print(f"  Episode {ep + 1:>6,} | "
                  f"CFR: {recent_cfr:+.3f} | QL: {recent_ql:+.3f}")

    # --- Summary ---
    summary = summarize_results(results, perturbation_ep, num_episodes)
    print("\n--- Summary (mean P0 reward) ---")
    for agent, s in summary.items():
        print(f"  {agent:12s}  pre={s['pre']:+.4f}  "
              f"post={s['post']:+.4f}  delta={s['delta']:+.4f}")

    # --- Save CSV ---
    csv_path = Path("results/raw") / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "agent"])
        writer.writerows(results)
    print(f"\nResults -> {csv_path}")

    # --- Plot ---
    plot_path = Path("results/plots") / f"{name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(str(csv_path), config, str(plot_path))
    print(f"Plot   -> {plot_path}")

    return str(csv_path), str(plot_path)
