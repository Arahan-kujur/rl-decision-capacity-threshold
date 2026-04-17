# Minimal Decision Capacity Prevents Catastrophic Exploitation in Self-Play RL

## Abstract

We compare how CFR (planning) and Q-learning (RL) agents adapt to asymmetric
rule perturbations in Kuhn Poker. After 10,000 episodes of normal play,
Player 0 loses the ability to bet -- either at all decision nodes or only at
the opening move. Complete removal causes Q-learning to collapse to -0.91
reward as its opponent learns total exploitation. Retaining a single call/fold
decision keeps Q-learning stable at -0.05. The gap is catastrophic, not
incremental.

## Background

Kuhn Poker is a two-player zero-sum game with three cards (J < Q < K) and two
actions (pass, bet). Its Nash equilibrium is analytically known, making it an
ideal testbed for studying adaptation under rule changes.

- **CFR**: Solves for Nash equilibrium offline, then plays a frozen strategy.
- **Q-learning**: Learns through self-play with epsilon-greedy exploration.

## Experiment 1: Full Action Removal

**Config**: `configs/full_removal.yaml`

After episode 10,000, bet (action 1) is removed from Player 0 at all nodes.
P0 cannot bet at the root *and* cannot call when facing a bet -- "call" shares
the same action index as "bet." P0 has zero remaining decisions: forced check,
forced fold.

| Agent | Pre | Post | Delta |
|---|---|---|---|
| CFR | -0.055 | -0.228 | -0.173 |
| Q-Learning | +0.003 | -0.908 | -0.911 |

**CFR** drops from Nash value (-1/18) to roughly -2/9. P1 still plays Nash,
so exploitation is bounded.

**Q-learning** collapses toward -1.0. P1 learns that P0 always folds and
converges to betting every hand. Every game becomes check-bet-fold. The
epsilon exploration floor holds the average at -0.91 rather than exactly -1.0.

## Experiment 2: Root-Only Removal

**Config**: `configs/root_only.yaml`

After episode 10,000, bet is removed from P0 only at the root. P0 can still
call or fold at the "pb" node.

| Agent | Pre | Post | Delta |
|---|---|---|---|
| CFR | -0.055 | -0.065 | -0.010 |
| Q-Learning | +0.003 | -0.049 | -0.052 |

**CFR** barely changes. At Nash equilibrium P0 is indifferent between bet and
check at the root, so removing one branch of an indifferent mixed strategy
does not shift the game value.

**Q-learning** dips briefly as Q-values recalibrate, then stabilises near
Nash. The retained decision point gives P0 a lever to prevent exploitation.

## Comparison

| | Full removal | Root-only |
|---|---|---|
| P0 decisions remaining | 0 | 1 (call/fold) |
| CFR post reward | -0.23 | -0.07 |
| QL post reward | **-0.91** | **-0.05** |
| QL delta | -0.91 | -0.05 |

The 18x difference in Q-learning degradation between the two conditions shows
that the transition from zero to one decision is a phase transition in
exploitability, not a gradual decline.

## Key Insight

**Minimal decision capacity prevents catastrophic exploitation in self-play
RL.** A single call/fold choice forces the opponent to maintain honest betting
-- bluff less with weak hands, avoid value-betting mediocre hands into a
potential call -- which bounds losses near the Nash value. Remove that one
decision and the opponent converges to unconditional betting with a guaranteed
win.

For safe deployment of RL agents, preserving minimal agency under constraint
changes is qualitatively more important than the magnitude of the constraint.
