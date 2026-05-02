"""Experiment runner: trains agents, runs multi-seed episodes, saves results.

Supports multiple games (via registry), agent types (Q-Learning, SARSA,
REINFORCE), opponent regimes (self-play, fixed, mixed), perturbation families
(action removal, biased constraint, action noise), and recovery experiments.
"""

import csv
import numpy as np
from pathlib import Path

from src.env.game_registry import get_game
from src.agents.cfr_agent import CFRAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.q_learning_frozen_agent import QLearningFrozenAgent
from src.agents.sarsa_agent import SarsaAgent
from src.agents.reinforce_agent import ReinforceAgent
from src.agents.fixed_opponents import RandomAgent, ExploitativeAgent, NashAgent
from src.utils.metrics import (
    summarize_seed, statistical_summary, format_stat_table,
)
from src.utils.plotting import plot_results


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _create_learning_agent(config, num_actions):
    """Create the primary learning agent based on config."""
    agent_type = config["experiment"].get("agent_type", "q_learning")
    alpha = config["q_learning"]["alpha"]
    epsilon = config["q_learning"]["epsilon"]

    if agent_type == "sarsa":
        return SarsaAgent(alpha=alpha, epsilon=epsilon,
                          num_actions=num_actions), "SARSA"
    elif agent_type == "reinforce":
        return ReinforceAgent(alpha=alpha * 0.1, num_actions=num_actions), "REINFORCE"
    elif agent_type == "ppo":
        from src.agents.ppo_agent import PPOAgent
        return PPOAgent(num_actions=num_actions), "PPO"
    elif agent_type == "dqn":
        from src.agents.dqn_agent import DQNAgent
        game = config["experiment"].get("game", "kuhn")
        return DQNAgent(num_actions=num_actions, game=game), "DQN"
    elif agent_type == "nfsp":
        from src.agents.nfsp_agent import NFSPAgent
        eta = config.get("nfsp", {}).get("eta", 0.1)
        return NFSPAgent(alpha=alpha, epsilon=epsilon,
                         num_actions=num_actions, eta=eta), "NFSP"
    else:
        return QLearningAgent(alpha=alpha, epsilon=epsilon,
                              num_actions=num_actions), "Q-Learning"


def _create_frozen_agent(config, num_actions):
    """Create a frozen variant of the learning agent."""
    agent_type = config["experiment"].get("agent_type", "q_learning")
    alpha = config["q_learning"]["alpha"]
    epsilon = config["q_learning"]["epsilon"]
    frozen_cfg = config.get("q_learning_frozen", {})
    frozen_eps = frozen_cfg.get("frozen_epsilon", 0.0)

    if agent_type == "sarsa":
        agent = SarsaAgent(alpha=alpha, epsilon=epsilon,
                           num_actions=num_actions)
    elif agent_type == "reinforce":
        agent = ReinforceAgent(alpha=alpha * 0.1, num_actions=num_actions)
    elif agent_type == "ppo":
        from src.agents.ppo_agent import PPOAgent
        agent = PPOAgent(num_actions=num_actions)
    elif agent_type == "dqn":
        from src.agents.dqn_frozen_agent import DQNFrozenAgent
        game = config["experiment"].get("game", "kuhn")
        return DQNFrozenAgent(num_actions=num_actions, game=game)
    else:
        return QLearningFrozenAgent(
            alpha=alpha, epsilon=epsilon,
            frozen_epsilon=frozen_eps, num_actions=num_actions)
    return agent


# ---------------------------------------------------------------------------
# Episode play (with action noise support)
# ---------------------------------------------------------------------------

def play_episode(env, agent, action_rng, cards, mask_active=None,
                 noise_prob=0.0, bias_action=None, bias_prob=0.0,
                 affected_player=0, perturbed=False, opponent_agent=None,
                 opponent_rng=None):
    """Play one episode. Supports self-play and opponent-regime modes.

    If opponent_agent is provided, P1 uses that agent instead of the
    primary agent (enabling fixed-opponent and mixed-population regimes).
    """
    env.reset(cards=cards, mask_active=mask_active)
    trajectory = []

    while not env.is_terminal:
        player = env.current_player
        info = env.info_state_str(player)
        legal = env.legal_actions()

        if player == 1 and opponent_agent is not None:
            action = opponent_agent.select_action(info, legal, opponent_rng)
        else:
            action = agent.select_action(info, legal, action_rng)

        if perturbed and player == affected_player:
            if noise_prob > 0 and action_rng.random() < noise_prob:
                action = int(action_rng.choice(legal))
            elif bias_action is not None and bias_prob > 0:
                if action_rng.random() < bias_prob and bias_action in legal:
                    action = bias_action

        trajectory.append((player, info, action))
        env.step(action)

    return env.returns[0], trajectory


# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------

def _parse_perturbation(config, action_map):
    """Extract perturbation parameters from config, handling all formats."""
    pert = config["perturbation"]
    disabled = pert.get("disabled", False)
    affected = pert.get("affected_player", 0)
    mask_prob = pert.get("mask_prob", 1.0)
    noise_prob = pert.get("noise_prob", 0.0)
    bias_prob = pert.get("bias_prob", 0.0)
    bias_action_str = pert.get("biased_action")
    bias_action = action_map.get(bias_action_str) if bias_action_str else None

    node_masks_raw = pert.get("node_masks")
    if node_masks_raw is not None:
        node_masks = {}
        for node, actions in node_masks_raw.items():
            node_masks[node] = [action_map[a] for a in actions]
        return (disabled, affected, None, False, node_masks, mask_prob,
                noise_prob, bias_action, bias_prob)

    removed_str = pert.get("removed_action")
    if removed_str and removed_str in action_map:
        removed = action_map[removed_str]
    elif removed_str:
        removed = removed_str
    else:
        default_actions = list(action_map.values())
        removed = default_actions[-1] if default_actions else 1
    root_only = pert.get("root_only", False)
    return (disabled, affected, removed, root_only, None, mask_prob,
            noise_prob, bias_action, bias_prob)


def _make_env(wrapper_class, env_class, removed, affected, root_only,
              node_masks, mask_prob):
    """Create a perturbation-wrapped environment."""
    if node_masks is not None:
        return wrapper_class(
            env_class(), affected_player=affected,
            node_masks=node_masks, mask_prob=mask_prob)
    return wrapper_class(
        env_class(), removed_action=removed,
        affected_player=affected, root_only=root_only,
        mask_prob=mask_prob)


def _create_opponent(config, policy, num_actions, game_info):
    """Create opponent agent based on regime config."""
    regime = config["experiment"].get("opponent_regime", "self_play")
    if regime == "self_play":
        return None
    elif regime == "fixed_opponent":
        return CFRAgent(policy, num_actions=num_actions)
    elif regime == "mixed_population":
        return [
            CFRAgent(policy, num_actions=num_actions),
            RandomAgent(num_actions=num_actions),
            ExploitativeAgent(num_actions=num_actions),
        ]
    return None


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_single_seed(seed, policy, config, game_info, include_frozen=False,
                    override_env_seed=None, override_policy_seed=None,
                    csv_suffix="", quiet=False):
    """Run one seed. Returns (results_list, csv_path, seed_summary)."""
    name = config["experiment"]["name"]
    num_episodes = config["experiment"]["num_episodes"]
    perturbation_ep = config["experiment"]["perturbation_episode"]
    recovery_ep = config["experiment"].get("recovery_episode")
    num_actions = game_info["constants"]["num_actions"]
    action_map = game_info["constants"]["action_map"]

    (disabled, affected, removed, root_only, node_masks, mask_prob,
     noise_prob, bias_action, bias_prob) = _parse_perturbation(config, action_map)

    # ------------------------------------------------------------------
    # RNG decomposition
    # ------------------------------------------------------------------
    master_rng = np.random.default_rng(seed)
    env_sub_seed = int(master_rng.integers(1 << 63))
    cfr_sub_seed = int(master_rng.integers(1 << 63))
    ql_sub_seed = int(master_rng.integers(1 << 63))
    qlf_sub_seed = int(master_rng.integers(1 << 63))
    opp_sub_seed = int(master_rng.integers(1 << 63))

    env_seed = override_env_seed if override_env_seed is not None else env_sub_seed
    env_rng = np.random.default_rng(env_seed)
    card_rng = np.random.default_rng(env_rng.integers(1 << 63))
    mask_rng = np.random.default_rng(env_rng.integers(1 << 63))

    if override_policy_seed is not None:
        cfr_action_rng = np.random.default_rng(override_policy_seed)
        ql_action_rng = np.random.default_rng(override_policy_seed + 1)
        qlf_action_rng = np.random.default_rng(override_policy_seed + 2)
    else:
        cfr_action_rng = np.random.default_rng(cfr_sub_seed)
        ql_action_rng = np.random.default_rng(ql_sub_seed)
        qlf_action_rng = np.random.default_rng(qlf_sub_seed)

    opp_rng = np.random.default_rng(opp_sub_seed)

    # ------------------------------------------------------------------
    # Create agents
    # ------------------------------------------------------------------
    cfr_agent = CFRAgent(policy, num_actions=num_actions)
    ql_agent, ql_label = _create_learning_agent(config, num_actions)

    qlf_agent = None
    qlf_label = f"{ql_label}-Frozen"
    if include_frozen:
        qlf_agent = _create_frozen_agent(config, num_actions)

    opponent = _create_opponent(config, policy, num_actions, game_info)

    # ------------------------------------------------------------------
    # Create environments
    # ------------------------------------------------------------------
    env_cls = game_info["env_class"]
    wrapper_cls = game_info["wrapper_class"]
    deal_fn = game_info["deal_fn"]

    cfr_env = _make_env(wrapper_cls, env_cls, removed, affected,
                        root_only, node_masks, mask_prob)
    ql_env = _make_env(wrapper_cls, env_cls, removed, affected,
                       root_only, node_masks, mask_prob)
    qlf_env = _make_env(wrapper_cls, env_cls, removed, affected,
                        root_only, node_masks, mask_prob) \
        if include_frozen else None

    if not quiet:
        mode = "disabled" if disabled else (
            "root only" if root_only else "all P0 nodes")
        print(f"\n  Seed {seed}: {num_episodes:,} episodes "
              f"(perturbation at {perturbation_ep:,}, {mode})")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    results = []
    is_perturbed = False

    for ep in range(num_episodes):
        if ep == perturbation_ep and not disabled:
            cfr_env.set_perturbed(True)
            ql_env.set_perturbed(True)
            if qlf_env is not None:
                qlf_env.set_perturbed(True)
                if hasattr(qlf_agent, 'freeze'):
                    qlf_agent.freeze()
            is_perturbed = True

        if recovery_ep is not None and ep == recovery_ep:
            cfr_env.set_perturbed(False)
            ql_env.set_perturbed(False)
            if qlf_env is not None:
                qlf_env.set_perturbed(False)
            is_perturbed = False

        cards = deal_fn(card_rng)

        ep_mask_active = None
        if mask_prob < 1.0 and is_perturbed:
            ep_mask_active = bool(mask_rng.random() < mask_prob)

        cur_opponent = None
        if isinstance(opponent, list):
            cur_opponent = opponent[ep % len(opponent)]
        elif opponent is not None:
            cur_opponent = opponent

        cfr_reward, _ = play_episode(
            cfr_env, cfr_agent, cfr_action_rng, cards, ep_mask_active,
            noise_prob=noise_prob, bias_action=bias_action,
            bias_prob=bias_prob, affected_player=affected,
            perturbed=is_perturbed,
            opponent_agent=cur_opponent, opponent_rng=opp_rng)
        results.append((ep, cfr_reward, "CFR"))

        ql_reward, ql_traj = play_episode(
            ql_env, ql_agent, ql_action_rng, cards, ep_mask_active,
            noise_prob=noise_prob, bias_action=bias_action,
            bias_prob=bias_prob, affected_player=affected,
            perturbed=is_perturbed,
            opponent_agent=cur_opponent, opponent_rng=opp_rng)
        ql_agent.update(ql_traj, ql_reward)
        results.append((ep, ql_reward, ql_label))

        if qlf_agent is not None:
            qlf_reward, qlf_traj = play_episode(
                qlf_env, qlf_agent, qlf_action_rng, cards, ep_mask_active,
                noise_prob=noise_prob, bias_action=bias_action,
                bias_prob=bias_prob, affected_player=affected,
                perturbed=is_perturbed,
                opponent_agent=cur_opponent, opponent_rng=opp_rng)
            qlf_agent.update(qlf_traj, qlf_reward)
            results.append((ep, qlf_reward, qlf_label))

    # ------------------------------------------------------------------
    # Summary and CSV
    # ------------------------------------------------------------------
    seed_summary = summarize_seed(results, perturbation_ep, num_episodes,
                                  recovery_ep=recovery_ep)

    if not quiet:
        agents_present = sorted(set(r[2] for r in results))
        finals = {}
        for a in agents_present:
            finals[a] = np.mean([r[1] for r in results[-2000:] if r[2] == a])
        parts = "  ".join(f"{a}: {v:+.3f}" for a, v in finals.items())
        print(f"    done  |  {parts}")

    csv_path = Path("results/raw") / f"{name}{csv_suffix}_seed{seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "agent"])
        writer.writerows(results)

    capacity = cfr_env.decision_capacity if not disabled else \
        game_info["constants"]["num_actions"]
    seed_summary["_meta"] = {"capacity": capacity, "seed": seed}

    return results, str(csv_path), seed_summary


# ---------------------------------------------------------------------------
# Full experiment (multi-seed)
# ---------------------------------------------------------------------------

def run_experiment(config):
    """Run a full multi-seed experiment.

    Dispatches on config for game type, agent type, opponent regime,
    perturbation family, and recovery.

    Returns (csv_paths, plot_path, stat_summary).
    """
    name = config["experiment"]["name"]
    seeds = config["experiment"]["seeds"]
    cfr_iters = config["cfr"]["iterations"]
    game_name = config["experiment"].get("game", "kuhn")
    include_frozen = "q_learning_frozen" in config

    game_info = get_game(game_name)
    constants = game_info["constants"]

    # --- CFR/Nash training ---
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {name}  ({len(seeds)} seeds, game={game_name})")
    print(f"{'=' * 60}")
    print(f"Training CFR ({cfr_iters:,} iterations)...")

    trainer = game_info["cfr_trainer_class"]()
    trainer.train(cfr_iters)
    policy = trainer.get_average_strategy()

    nash_val = constants.get("nash_value_p0")
    if nash_val is None and hasattr(trainer, "nash_value_p0"):
        nash_val = trainer.nash_value_p0()
        constants["nash_value_p0"] = nash_val

    n_info = len(policy)
    print(f"  Strategy computed: {n_info} information sets")
    if nash_val is not None:
        print(f"  Nash value (P0): {nash_val:+.6f}")

    agent_type = config["experiment"].get("agent_type", "q_learning")
    regime = config["experiment"].get("opponent_regime", "self_play")
    print(f"  Agent: {agent_type} | Regime: {regime}")

    if include_frozen:
        frozen_eps = config["q_learning_frozen"].get("frozen_epsilon", 0.0)
        print(f"  Frozen baseline enabled (frozen_epsilon={frozen_eps})")

    # --- Per-seed runs ---
    csv_paths = []
    seed_summaries = []
    all_results = []

    for seed in seeds:
        results, csv_path, seed_summary = run_single_seed(
            seed, policy, config, game_info,
            include_frozen=include_frozen)
        csv_paths.append(csv_path)
        seed_summaries.append(seed_summary)
        all_results.append(results)

    # --- Aggregated statistics ---
    clean_summaries = [{k: v for k, v in s.items() if k != "_meta"}
                       for s in seed_summaries]
    stat_summary = statistical_summary(clean_summaries)

    meta = seed_summaries[0].get("_meta", {})
    stat_summary["_meta"] = {
        "capacity": meta.get("capacity", -1),
        "game": game_name,
        "constants": constants,
    }

    print(f"\n--- Statistical Summary ({len(seeds)} seeds) ---")
    print(format_stat_table(stat_summary))

    # --- Plot ---
    plot_path = Path("results/plots") / f"{name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(csv_paths, config, str(plot_path))
    print(f"\nPlot -> {plot_path}")

    return csv_paths, str(plot_path), stat_summary
