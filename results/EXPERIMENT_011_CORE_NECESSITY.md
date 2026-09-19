# Experiment 011: Core Necessity

## Question

Are the high-change recurrent synapses that were sufficient to transfer Pretorius in Experiment 010 also disproportionately necessary for maintaining the mature phenotype?

## Protocol

Experiment 011 used three 1,024-neuron seeds (1842, 101, 202), an explicit 24,000-neural-update training budget, recurrent-only readout, 40 settle steps and 60 probe steps, and the existing public validation and adversarial batteries. The sealed terminal battery was not touched.

For each mature network, recurrent edges were ranked by absolute founder-to-mature weight change. At 0.25%, 1%, 5%, and 10% of recurrent edges, selected mature weights were restored to their exact matched-founder values. The high-change intervention was compared with equal-size random-location and lowest-change controls. Restoring founder values rather than zeroing edges avoids the structural zero-hole artifact isolated in Experiment 009.

## Results

The mature baseline averaged 0.475 validation top-1 agreement and 0.350 adversarial top-1 agreement across the three seeds, compared with founder baselines of 0.117 and 0.083 respectively.

Restoring only the highest-change 0.25% of recurrent edges to founder values reduced mean top-1 agreement to 0.400 validation and 0.250 adversarial. At 1%, agreement fell to 0.183 validation and 0.100 adversarial. At 5%, it fell to 0.092 validation and 0.083 adversarial. At 10%, it was 0.083 validation and 0.100 adversarial, effectively returning the discrete phenotype measure to the founder regime.

The equal-size random and lowest-change controls did not show this collapse. At every tested fraction, both retained 0.475 validation top-1 agreement and 0.350 adversarial top-1 agreement. Random restoration caused only small changes in JS similarity, while lowest-change restoration was essentially indistinguishable from the mature baseline.

The continuous similarity metric shows the same ordering. Mature mean validation JS similarity was 0.783664. High-change restoration reduced it monotonically to 0.782663, 0.781201, 0.778807, and 0.778426 across the four fractions. The founder validation value was 0.778408. On the adversarial set, mature mean JS similarity was 0.785285; high-change restoration reduced it to 0.784244, 0.782876, 0.780964, and 0.780610, compared with founder 0.780610.

## Interpretation

Experiment 011 supports a localized causal-core interpretation under this model and protocol. The recurrent weights that changed most during biography development are not merely correlated with the mature Pretorius phenotype and not merely sufficient for partial phenotype transfer. Reverting a small ranked subset toward the founder state disproportionately removes the learned phenotype, while reverting the same number of random or minimally changed weights does not.

The strongest result is the convergence at larger fractions. Restoring the top 10% produces mean JS similarity almost exactly equal to the founder baseline on both validation and adversarial batteries, while the random and bottom controls remain near mature performance. The discrete top-1 measure collapses much earlier: restoring the top 1% already removes most of the mature advantage, and 5% places the phenotype at approximately founder-level agreement.

This does not establish that the high-change set is a unique minimal representation of identity. Absolute weight change may identify a larger causal pathway containing redundant, interacting, or downstream-supporting edges. Experiment 011 establishes necessity in the intervention sense tested here, not semantic localization to individual synapses.

## Reproducibility

Successful GitHub Actions run: `35319470706` at commit `57227ba2ac27f7d7c6cb09bf194c6064afe14f70`.

Immutable artifact: `experiment-011-report-57227ba2ac27f7d7c6cb09bf194c6064afe14f70`, artifact ID `10536098135`, SHA-256 `a21aa79b4c422ccfc94902151ae2f2fe1cdd5a2aee60027bc3d6e6caa7d5ab27`.

The raw machine-readable report remains preserved as the workflow artifact. This document records the interpretation without modifying the experiment protocol or terminal battery.

## Next experimental question

The appropriate next phase is to test whether this causal core is stable across independently developed Pretorius instances. Experiment 011 identifies high-change edges within each matched founder-to-mature trajectory. A cross-seed overlap and cross-seed causal-transfer experiment can distinguish a genuinely recurrent shared identity substructure from seed-specific implementations that merely produce the same phenotype.