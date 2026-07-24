# Option-Critic v6: Progressive Eight-Slot Capacity

This post-install hotfix applies to the integrated v5 two-option stability tree.
It restores eight option slots without cloning two Tactical Mixture modes into
six immediately independent policies.

## Design

Options use the layout:

```text
0 = source group 0, anchor slot
1 = source group 1, anchor slot
2/3 = first child slots, unlock at 150k
4/5 = second child slots, unlock at 350k
6/7 = third child slots, unlock at 550k
```

The manager factorizes as:

```text
P(option=(group,slot)|state)
  = P(source tactical group|state) * P(slot|group,state)
```

At step zero only options 0 and 1 are selectable. Their group manager and worker
are migrated exactly from Tactical Mixture v1.2, preserving the source policy.
Child slots are progressively gated in. Each child adds a bounded delta around
its source group, with zero causal effect at unlock and a gradual 200k learning
ramp. Anchor slots never receive child deltas.

The run must start from the Tactical Mixture v1.2 best checkpoint with macro
validation win rate 0.375, not from any Option-Critic checkpoint.

## Training phase

- 1,000,000 fresh environment steps
- startup validation and evaluation every 200k
- fresh uniform replay
- frozen JEPA world model and inherited primitive actor
- no task-agnostic option-diversity pressure
- Exp45 forecast pipeline attempted after RL, even when RL fails
