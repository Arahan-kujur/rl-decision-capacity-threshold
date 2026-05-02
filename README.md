# A Structural Threshold in Decision Capacity Governs Collapse in Self-Play RL

When self-play RL agents lose access to actions mid-training, do they adapt
or collapse? We show that a sharp threshold in decision capacity determines
the outcome -- and the finding holds across games, algorithms, and
representations.

**Paper:** [report/paper.md](report/paper.md) | **LaTeX:** [report/latex/](report/latex/)

## Key Results

- **Zero contingency** (all decisions forced): Q-Learning collapses to
  near-maximal exploitation. DQN collapses even harder (-0.994).
- **Residual contingency** (one decision preserved): agents stabilise near
  Nash equilibrium.
- **Co-adaptation is the mechanism**: a frozen baseline and fixed-opponent
  control prove it.
- **Tested across**: 8 games (1--24,576 info sets, collapse confirmed at scale),
  5 algorithms (Q-Learning, SARSA, REINFORCE, PPO, DQN),
  3 perturbation schedules.

## Quick Start

```bash
pip install -r requirements.txt
python run_experiments.py
```

## All Experiments

| Script | What it runs |
|---|---|
| `python run_experiments.py` | Base Kuhn (full removal + root-only, 20 seeds) |
| `python run_leduc_experiments.py` | Leduc Poker experiments |
| `python run_matrix_experiments.py` | IPD + Matching Pennies |
| `python run_deep_experiments.py` | DQN experiments |
| `python run_capacity_sweep.py` | Decision capacity sweep (0, 1, 2) |
| `python run_severity_sweep.py` | Timing x severity (3x2 grid) |
| `python run_regime_comparison.py` | Self-play vs fixed-opponent |
| `python run_recovery.py` | Collapse + recovery experiment |
| `python run_algorithm_comparison.py` | Q-Learning vs SARSA vs REINFORCE |
| `python run_cross_game.py` | Kuhn vs Leduc comparison |
| `python run_variance_decomposition.py` | Environment vs policy variance |
| `python run_hyperparam_grid.py` | 3x3 hyperparameter sensitivity |
| `python run_separate_selfplay.py` | Separate P0/P1 verification |
| `python run_perturbation_families.py` | Bias + noise perturbation types |
| `python run_scaling_analysis.py` | Collapse across game complexities |
| `python generate_paper_figures.py` | Publication-quality figures |

Single experiment:

```bash
python run_experiments.py --config configs/liars_dice/ld_full_removal.yaml
```

## Games

| Game | Info Sets | Actions | Type |
|---|---|---|---|
| Matching Pennies | 1 | 2 | Matrix |
| Kuhn Poker | 12 | 2 | Poker |
| Leduc Poker | 288 | 3 | Poker |
| Leduc-4 Poker | 504 | 3 | Poker (4 ranks) |
| Liar's Dice (1d) | 24,576 | 13 | Dice |
| Liar's Dice (2d) | 200,000+ | 25 | Dice (DQN) |
| Coordination | -- | 3 | Cooperative |
| Negotiation | -- | 11 | Bargaining |
| IPD | ~32 | 2 | Matrix |

All implemented from scratch. No external game libraries.

## Agents

- **CFR** -- Nash equilibrium via counterfactual regret minimisation
- **Q-Learning** -- Tabular, epsilon-greedy, self-play
- **SARSA** -- On-policy tabular
- **REINFORCE** -- Tabular policy gradient
- **DQN** -- 2-layer MLP, experience replay, target network
- **QL-Frozen** -- Q-Learning frozen at perturbation point
- **PPO** -- Tabular proximal policy optimization with entropy bonus

## Requirements

Python 3.10+. Dependencies: `numpy`, `matplotlib`, `pyyaml`, `scipy`, `torch`.

## Project Structure

```
configs/                  Experiment configs (YAML)
  full_removal.yaml       Base Kuhn (20 seeds)
  root_only.yaml
  capacity/               Decision capacity sweep
  frozen/                 Frozen Q-learning comparison
  leduc/                  Leduc Poker
  leduc4/                 Extended Leduc (4 ranks)
  liars_dice/             Liar's Dice 1-die (24k info sets)
  liars_dice2/            Liar's Dice 2-dice (200k+ info sets, DQN)
  negotiation/            Ultimatum bargaining game
  coordination/           Cooperative target-matching game
  matrix/                 IPD + Matching Pennies
  severity/               Timing x severity sweep
  stochastic/             Stochastic masking
  algorithms/             SARSA + REINFORCE
  regimes/                Opponent regime comparison
  recovery/               Collapse + recovery
  deep/                   DQN experiments
  hyperparam/             Hyperparameter sensitivity
  perturbation_families/  Bias + noise perturbation
src/
  env/                    Game environments + perturbation wrappers
  agents/                 All agent implementations
  experiments/runner.py   Multi-seed experiment runner
  utils/                  Metrics, plotting, variance decomposition
results/
  plots/                  Generated figures
  raw/                    Per-seed CSVs (gitignored)
report/
  paper.md                Full paper (Markdown)
  latex/                  LaTeX project (NeurIPS preprint style)
```
