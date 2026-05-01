"""Generate exploitability curves and DQN neural analysis figures.

Usage:
    python generate_analysis_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.env.perturbed_kuhn import KuhnPokerEnv, PerturbedKuhnPoker, PASS, BET
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.metrics import compute_exploitability

FIGURES_DIR = Path("report/latex/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _q_to_policy(agent, num_actions=2):
    """Convert Q-table to a policy dict for exploitability computation."""
    policy = {}
    for info_state, q_vals in agent.q.items():
        probs = np.zeros(num_actions)
        best = np.argmax(q_vals)
        eps = agent.epsilon
        for a in range(num_actions):
            probs[a] = eps / num_actions
        probs[best] += 1.0 - eps
        policy[info_state] = probs
    return policy


def _dqn_to_policy(agent):
    """Convert DQN to a policy dict by evaluating all known info states."""
    import torch
    policy = {}
    kuhn_states = []
    for card in [0, 1, 2]:
        for h in ["", "p", "b", "pb"]:
            kuhn_states.append(f"{card}{h}")

    for info_state in kuhn_states:
        features = agent.encoder(info_state)
        with torch.no_grad():
            q_vals = agent.q_net(torch.FloatTensor(features).unsqueeze(0))[0]
        probs = np.zeros(2)
        best = int(q_vals.argmax().item())
        eps = agent._get_epsilon()
        for a in range(2):
            probs[a] = eps / 2
        probs[best] += 1.0 - eps
        policy[info_state] = probs

    return policy


def generate_exploitability_curves():
    """Exploitability over time for Q-Learning under full removal (Kuhn)."""
    print("  Generating exploitability curves...")

    seeds = [42, 123, 456]
    num_episodes = 20000
    perturbation_ep = 10000
    measure_every = 100

    all_exploit = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        agent = QLearningAgent(alpha=0.1, epsilon=0.15, num_actions=2)
        env = PerturbedKuhnPoker(KuhnPokerEnv(), removed_action=BET,
                                 affected_player=0, root_only=False)

        exploit_curve = []

        for ep in range(num_episodes):
            if ep == perturbation_ep:
                env.set_perturbed(True)

            deck = np.array([0, 1, 2])
            rng.shuffle(deck)
            cards = (int(deck[0]), int(deck[1]))
            env.reset(cards=cards)
            traj = []
            while not env.is_terminal:
                player = env.current_player
                info = env.info_state_str(player)
                legal = env.legal_actions()
                action = agent.select_action(info, legal, rng)
                traj.append((player, info, action))
                env.step(action)
            agent.update(traj, env.returns[0])

            if ep % measure_every == 0:
                policy = _q_to_policy(agent)
                exploit = compute_exploitability(policy, "kuhn")
                exploit_curve.append((ep, exploit))

        all_exploit.append(exploit_curve)

    fig, ax = plt.subplots(figsize=(10, 5))
    for curve in all_exploit:
        eps_arr = [c[0] for c in curve]
        vals = [c[1] for c in curve]
        ax.plot(eps_arr, vals, alpha=0.3, color="#FF9800")

    mean_vals = np.array([[c[1] for c in curve] for curve in all_exploit]).mean(axis=0)
    eps_arr = [c[0] for c in all_exploit[0]]
    ax.plot(eps_arr, mean_vals, color="#FF9800", linewidth=2, label="Q-Learning")

    ax.axvline(x=perturbation_ep, color="red", linestyle="--", alpha=0.7,
               label="Perturbation")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploitability")
    ax.set_title("Exploitability Over Time (Kuhn, Full Removal)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "exploitability_curve.png"), dpi=150)
    plt.close()
    print("    done")


def generate_dqn_analysis():
    """DQN entropy and Q-value gap over time."""
    print("  Generating DQN analysis plots...")
    import torch

    seed = 42
    rng = np.random.default_rng(seed)
    agent = DQNAgent(num_actions=2, game="kuhn", epsilon_start=0.15,
                     epsilon_end=0.01, epsilon_decay=50000)
    env = PerturbedKuhnPoker(KuhnPokerEnv(), removed_action=BET,
                             affected_player=0, root_only=False)

    num_episodes = 50000
    perturbation_ep = 25000
    measure_every = 200

    entropy_curve = []
    qgap_curve = []

    for ep in range(num_episodes):
        if ep == perturbation_ep:
            env.set_perturbed(True)

        deck = np.array([0, 1, 2])
        rng.shuffle(deck)
        cards = (int(deck[0]), int(deck[1]))
        env.reset(cards=cards)
        traj = []
        while not env.is_terminal:
            player = env.current_player
            info = env.info_state_str(player)
            legal = env.legal_actions()
            action = agent.select_action(info, legal, rng)
            traj.append((player, info, action))
            env.step(action)
        agent.update(traj, env.returns[0])

        if ep % measure_every == 0:
            entropies = []
            gaps = []
            for card in [0, 1, 2]:
                for h in ["", "p", "b", "pb"]:
                    info = f"{card}{h}"
                    features = agent.encoder(info)
                    with torch.no_grad():
                        q_vals = agent.q_net(
                            torch.FloatTensor(features).unsqueeze(0))[0]
                    probs = torch.softmax(q_vals, dim=0).numpy()
                    ent = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(ent)
                    gaps.append(abs(float(q_vals[0] - q_vals[1])))

            entropy_curve.append((ep, np.mean(entropies)))
            qgap_curve.append((ep, np.mean(gaps)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    eps_e = [c[0] for c in entropy_curve]
    vals_e = [c[1] for c in entropy_curve]
    ax1.plot(eps_e, vals_e, color="#9C27B0", linewidth=1.5)
    ax1.axvline(x=perturbation_ep, color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean Policy Entropy")
    ax1.set_title("DQN Policy Entropy Over Time")
    ax1.grid(True, alpha=0.3)

    eps_g = [c[0] for c in qgap_curve]
    vals_g = [c[1] for c in qgap_curve]
    ax2.plot(eps_g, vals_g, color="#009688", linewidth=1.5)
    ax2.axvline(x=perturbation_ep, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean |Q(a0) - Q(a1)|")
    ax2.set_title("DQN Q-Value Gap Over Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "dqn_analysis.png"), dpi=150)
    plt.close()
    print("    done")


def main():
    print("Generating analysis figures...")
    generate_exploitability_curves()
    generate_dqn_analysis()
    print(f"\nAll analysis figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
