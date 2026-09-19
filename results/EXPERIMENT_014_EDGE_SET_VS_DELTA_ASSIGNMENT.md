# Experiment 014: edge-set selection vs delta assignment

Status: COMPLETE AND SCIENTIFICALLY FROZEN

Governing preregistration: Issue #17. Implementation branch: `v0.4-edge-set-vs-delta-assignment`. Scientific execution head: `f5170ac2678effebabebb07b681525e74dd36969`. Workflow run: `35409536717`.

## Frozen protocol

Experiment 014 inherited Experiment 013 without redesign: 1,024 neurons, topology seeds 1842/101/202/303/404/505, cyclic donor-recipient pairing, 24,000 actual neural updates per independent topology, 16x recurrent plasticity, recurrent-only readout, 1% magnitude-defined causal core, frozen response-fingerprint cosine mapping, hard Dale compatibility, existing clipping, 40/60 evaluation, validation/adversarial separation, and sealed terminal exclusion. Six independent topologies consumed 144,000 training updates. The permutation-isomorph positive control consumed 24,000 additional updates, for 168,000 total.

For each donor-recipient pair the same donor delta multiset was used in A-D. A used the aligned recipient edge set with exact donor-delta assignment. B used the same aligned set with Dale-class-preserving shuffled deltas. C used the preregistered non-aligned random recipient set with deterministic donor-core delta order. D reused exactly C's recipient edge set with Dale-class-preserving shuffled deltas. Randomization namespace was frozen as `experiment014-v1-preregistered`.

Controls retained recipient virgin, recipient mature, within-topology 1% core, permutation-isomorph positive control, and zero-delta bookkeeping. No raw recurrent edge indices were compared across independent topologies and the terminal battery was not touched.

## Execution and immutable provenance

GitHub Actions run `35409536717` completed successfully. Dependency installation, runtime/data/protocol preflight, unit/protocol tests, scientific execution, report verification, SHA-256 recording, and immutable artifact upload all passed.

Artifact ID: `10574505033`. Artifact digest: `sha256:fe88e3bca9c3e37d8ecf58769cb0a41c0cdb493113a9d5b18dd647c5b73f80ad`. Report path: `results/experiment_014_edge_set_vs_assignment/report_n1024_steps24000.json`. Report SHA-256: `ba72c49b98b0d8d5fe8af5110cb83869c352f6ad39ee404d801a69a7815ef174`.

The permutation-isomorph control passed with neuron-mapping accuracy 1.0, mapped-core correspondence 1.0, and zero validation/adversarial top-1 gap. Zero-delta bookkeeping produced exactly zero recovery relative to virgin.

## Pair-level preregistered contrasts

Values below are recovery-fraction differences, shown as validation JS / validation top-1 / adversarial JS / adversarial top-1.

| donor -> recipient | B-D edge set | A-B exact assignment | C-D negative assignment |
|---|---|---|---|
| 1842 -> 101 | 0.299406 / 1.055556 / 0.306152 / 1.000000 | 0.001694 / 0.000000 / -0.002068 / 0.000000 | 0.002870 / 0.000000 / 0.001369 / -0.250000 |
| 101 -> 202 | 0.261659 / 1.000000 / 0.253884 / 0.333333 | 0.009067 / 0.000000 / 0.009308 / 0.000000 | -0.007971 / 0.250000 / -0.007087 / 0.166667 |
| 202 -> 303 | 0.301299 / 0.176471 / 0.296997 / 0.200000 | -0.003132 / 0.117647 / -0.002197 / 0.000000 | 0.009928 / 0.058824 / 0.008854 / 0.000000 |
| 303 -> 404 | 0.244088 / 0.214286 / 0.247107 / -0.200000 | -0.004042 / 0.000000 / -0.006386 / -0.200000 | 0.005233 / 0.071429 / 0.008255 / 0.000000 |
| 404 -> 505 | 0.237515 / 1.000000 / 0.257972 / 0.857143 | -0.000477 / 0.000000 / -0.000533 / 0.000000 | 0.001933 / 0.153846 / 0.001678 / 0.000000 |
| 505 -> 1842 | 0.280506 / 0.058824 / 0.269916 / 0.000000 | 0.019373 / 0.411765 / 0.036502 / 0.333333 | 0.012867 / 0.000000 / 0.012724 / 0.000000 |

## Aggregate results

The preregistered B-D recipient-edge-set contrast was large and positive: validation JS recovery +0.270746 and top-1 recovery +0.584189; adversarial JS +0.272005 and top-1 +0.365079.

The A-B exact-assignment contrast was small: validation JS +0.003747 and top-1 +0.088235; adversarial JS +0.005771 and top-1 +0.022222. The C-D negative-control assignment contrast was of similar JS magnitude: validation +0.004143 and adversarial +0.004299, with top-1 +0.089016 validation and -0.013889 adversarial.

Mean A recovery was 0.285373 validation JS, 0.640231 validation top-1, 0.284141 adversarial JS, and 0.353968 adversarial top-1. Mean B recovery was 0.281625, 0.551996, 0.278370, and 0.331746 respectively. Mean C and D recovery remained near zero on JS and generally near zero on top-1.

Clipping was sparse. Across the six pairs, A/B/C/D clipping counts were respectively 5/3/4/6 total; within-topology core clipped 4 weights; zero-delta clipped none.

## Post-result interpretation audit

The result supports the preregistered edge-set hypothesis. Selecting the functionally homologous recipient recurrent edge set accounts for the dominant tested component of cross-topology phenotype transfer. Exact donor-delta-to-edge assignment adds little beyond selection of that set under this protocol.

The validation top-1 A-B difference must not be treated as clean evidence for exact assignment because the C-D negative-control top-1 contrast is nearly identical in aggregate. JS, the continuous metric, shows A-B and C-D effects of similarly small magnitude. The result therefore does not justify claiming a robust exact-assignment effect.

This experiment does not show that delta values are irrelevant, nor that edge-set selection alone is sufficient. A and B both retain the learned donor delta multiset, its Dale-class structure, and clipping constraints. The supported claim is narrower: within this frozen graft protocol, where that learned multiset is held constant, recipient functional edge-set selection is much more consequential than exact within-set delta assignment.

No metric, mapping rule, control, hypothesis, inclusion criterion, or analysis choice was changed after observing the result. Any further methodological change belongs to a new experiment.
