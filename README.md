# Kuhn Poker Perturbation Experiments

Studying how planning (CFR) and reinforcement learning (Q-learning) agents adapt
when Player 0 loses the ability to bet mid-game in Kuhn Poker. We compare
catastrophic collapse under full action removal against stable behaviour when
partial decision-making capacity is preserved.

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.10+. No external game libraries needed -- Kuhn Poker is
implemented from scratch.

## Running experiments

Run both experiments:

```
python run_experiments.py
```

Run a single experiment:

```
python run_experiments.py configs/full_removal.yaml
python run_experiments.py configs/root_only.yaml
```

Results are saved to `results/raw/` (CSV) and `results/plots/` (PNG).

## Experiments

Each experiment runs 20,000 self-play episodes. At episode 10,000 a rule
perturbation is applied to Player 0 only.

| Config | Perturbation | P0 decisions remaining |
|---|---|---|
| `full_removal` | Remove bet from P0 at **all** nodes | None (forced check + fold) |
| `root_only` | Remove bet from P0 at **root** only | Call/fold at "pb" node |

### Agents

- **CFR**: Trained to Nash equilibrium before perturbation, then frozen.
- **Q-Learning**: Tabular, epsilon-greedy, self-play. Continues learning after
  perturbation.

## Key finding

The difference between zero and one remaining decision is not incremental --
it determines whether self-play RL collapses or stabilises:

| Perturbation | CFR post | Q-Learning post |
|---|---|---|
| Full removal  | -0.23 | **-0.91** |
| Root-only     | -0.07 | **-0.05** |

Minimal decision capacity prevents catastrophic exploitation in self-play RL.

## Project structure

```
configs/             YAML experiment definitions
src/agents/          CFR and Q-learning agent implementations
src/env/             Kuhn Poker environment + perturbation wrapper
src/experiments/     Experiment runner
src/utils/           Metrics and plotting
results/             Output CSVs and plots
report/              Analysis draft
```
