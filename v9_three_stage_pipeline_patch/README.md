# v9 three-stage pipeline correction

This supplemental patch assumes Option-Critic v9 is already installed and audited.
It adds, without replacing v9:

1. Forecast JEPA install/train/ordinary evaluation/hidden evaluation.
2. A fresh ordinary Tactical Mixture v1.2 actor-critic run with imagination horizon 15 for exactly 800k new environment steps.
3. A fresh Option-Critic v9 run with imagination horizon 15 for exactly 800k new environment steps.

Both RL runs resume independently from the exact same Tactical-v1.2 best checkpoint. The Option-Critic run does not inherit the baseline's additional 800k updates, preserving a fair architecture comparison.
