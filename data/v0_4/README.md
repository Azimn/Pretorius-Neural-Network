# v0.4 public experimental payload

This directory is the public, non-terminal data payload required for recurrent-only v0.4 experiments.

The training split contains 100 axis-grounded Pretorius phenotype situations, five for each of 20 phenotype axes. Validation contains 40 items and adversarial evaluation contains 20 items.

The large source files were partitioned only to make repository review and transport easier. `persona_net.v04_data.load_v04_split` reconstructs the original split in memory and checks the expected item count.

The sealed terminal inputs and terminal scoring key are intentionally absent. Their precommitted SHA-256 hashes remain in `manifest.json` and `SEALED_TERMINAL.md`.

The partitioning operation does not change item contents, targets, or experimental roles. New experiments should treat actual neural update calls as the compute budget and report that budget explicitly.
