# Minimal Decision Capacity Prevents Catastrophic Exploitation in Self-Play RL

## Abstract

We study how planning (CFR) and reinforcement learning (Q-learning) agents
adapt to asymmetric rule perturbations in Kuhn Poker. After 10,000 episodes
of normal play, Player 0 loses the ability to bet -- either at all decision
nodes or only at the opening move. We find that complete action removal causes
Q-learning to collapse to -0.91 reward as its self-play opponent learns total
exploitation, while retaining a single call/fold decision keeps Q-learning
stable at -0.05. The gap between zero and one remaining decision is
catastrophic, not incremental.

## Background

Kuhn Poker is a two-player zero-sum game with three cards (J < Q < K) and two
actions (pass, bet). Its Nash equilibrium is analytically known, making it an
ideal testbed for studying agent adaptation under rule changes.

We compare two agent types:

- **CFR (Counterfactual Regret Minimization)**: a planning agent that solves
  for the Nash equilibrium offline and uses the frozen strategy during play.
- **Tabular Q-learning**: an RL agent that learns through self-play with
  epsilon-greedy exploration.

## Experiment 1: Full Action Removal

**Config**: `configs/full_removal.yaml`

After episode 10,000, bet (action 1) is removed from Player 0 at all decision
nodes. This means P0 cannot bet at the root *and* cannot call when facing a
bet (since "call" uses the same action index as "bet"). P0 is left with zero
meaningful decisions: forced check, forced fold.

### Results

| Agent | Pre-perturbation | Post-perturbation | Delta |
|---|---|---|---|
| CFR | -0.055 | -0.228 | -0.173 |
| Q-Learning | +0.003 | -0.908 | -0.911 |

**CFR** drops from Nash value (-1/18) to approximately -2/9. The frozen P1
strategy still plays Nash, limiting exploitation. The damage is bounded.

**Q-learning** collapses toward -1.0. In self-play, P1 learns that P0 always
folds and converges to betting every hand regardless of card. Every game
becomes check-bet-fold and P0 loses its ante. The epsilon-greedy exploration
floor keeps the average at -0.91 rather than exactly -1.0.

## Experiment 2: Root-Only Removal

**Config**: `configs/root_only.yaml`

After episode 10,000, bet is removed from P0 only at the root (opening move).
P0 can still call or fold when facing P1's bet at the "pb" node.

### Results

| Agent | Pre-perturbation | Post-perturbation | Delta |
|---|---|---|---|
| CFR | -0.055 | -0.065 | -0.010 |
| Q-Learning | +0.003 | -0.049 | -0.052 |

**CFR** barely changes. At Nash equilibrium, P0 is indifferent between bet and
check at the root for every card. Removing one branch of an indifferent
mixed strategy does not change the game value. The frozen strategy at "pb"
nodes remains near-optimal.

**Q-learning** shows a brief dip during transition as Q-values recalibrate,
then stabilises near the Nash value. P1 adjusts its bluffing frequency, and
P0 re-learns optimal call/fold thresholds. The retained decision point gives
P0 a lever to prevent exploitation.

## Comparison

| Metric | Full removal | Root-only |
|---|---|---|
| P0 decisions after perturbation | 0 | 1 (call/fold) |
| CFR post-perturbation reward | -0.23 | -0.07 |
| Q-Learning post-perturbation reward | **-0.91** | **-0.05** |
| Q-Learning delta | -0.91 | -0.05 |

The 18x difference in Q-learning degradation (0.91 vs 0.05) between the two
conditions demonstrates that the transition from zero to one decision is not
smooth -- it is a phase transition in exploitability.

## Key Insight

**Minimal decision capacity prevents catastrophic exploitation in self-play
RL.** When an agent retains even a single meaningful decision point, its
opponent cannot achieve total exploitation. The call/fold choice at the "pb"
node forces P1 to maintain honest betting (bluff less with J, avoid
value-betting Q into a potential call), which bounds P0's losses near the
Nash equilibrium value. Remove that one decision, and P1 converges to
unconditional betting with a guaranteed win.

This has implications for safe deployment of RL agents: preserving minimal
agency under constraint changes is qualitatively more important than the
magnitude of the constraint.
