"""Reach sweep: how post-perturbation reward varies with epsilon
(which controls the reach of the retained 'pb' node in Kuhn root-only
perturbation).

Higher epsilon → more random P1 → different effective reach of pb.
"""

import numpy as np
from src.config_loader import load_config
from src.env.game_registry import get_game
from src.experiments.runner import (
    _create_learning_agent, play_episode, _parse_perturbation, _make_env,
)
from src.agents.cfr_agent import CFRTrainer, CFRAgent


def run_sweep():
    epsilons = [0.05, 0.15, 0.30, 0.50]
    base_config = load_config("configs/root_only.yaml")
    game_info = get_game("kuhn")
    constants = game_info["constants"]
    num_actions = constants["num_actions"]
    action_map = constants["action_map"]

    print("Training CFR (10000 iters)...")
    trainer = CFRTrainer()
    trainer.train(10000)
    policy = trainer.get_average_strategy()

    (disabled, affected, removed, root_only, node_masks, mask_prob,
     noise_prob, bias_action, bias_prob) = _parse_perturbation(base_config, action_map)

    seed = 42
    num_episodes = 20000
    perturbation_ep = 10000

    print(f"\n{'epsilon':>8s}  {'reach(pb)':>10s}  {'QL_post':>10s}")
    print("-" * 34)

    for eps in epsilons:
        rng = np.random.default_rng(seed)
        card_rng = np.random.default_rng(rng.integers(1 << 63))
        action_rng = np.random.default_rng(rng.integers(1 << 63))
        cfr_action_rng = np.random.default_rng(rng.integers(1 << 63))

        from src.agents.q_learning_agent import QLearningAgent
        agent = QLearningAgent(alpha=0.1, epsilon=eps, num_actions=num_actions)

        env_cls = game_info["env_class"]
        wrapper_cls = game_info["wrapper_class"]
        deal_fn = game_info["deal_fn"]
        env = _make_env(wrapper_cls, env_cls, removed, affected,
                        root_only, node_masks, mask_prob)

        post_rewards = []
        pb_reached_count = 0
        post_episode_count = 0

        for ep in range(num_episodes):
            if ep == perturbation_ep:
                env.set_perturbed(True)

            cards = deal_fn(card_rng)
            env.reset(cards=cards)
            trajectory = []

            while not env.is_terminal:
                player = env.current_player
                info = env.info_state_str(player)
                legal = env.legal_actions()
                action = agent.select_action(info, legal, action_rng)
                trajectory.append((player, info, action))
                env.step(action)

            reward_p0 = env.returns[0]
            agent.update(trajectory, reward_p0)

            if ep >= perturbation_ep:
                post_rewards.append(reward_p0)
                post_episode_count += 1
                for _, info, _ in trajectory:
                    if info.endswith("pb"):
                        pb_reached_count += 1
                        break

        reach_pb = pb_reached_count / post_episode_count if post_episode_count else 0
        ql_post = np.mean(post_rewards[-5000:]) if post_rewards else 0

        print(f"{eps:8.2f}  {reach_pb:10.4f}  {ql_post:+10.4f}")


if __name__ == "__main__":
    run_sweep()
