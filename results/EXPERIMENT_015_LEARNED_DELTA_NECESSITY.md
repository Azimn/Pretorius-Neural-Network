# Experiment 015: learned delta multiset necessity

Experiment 015 is complete and scientifically frozen. It is governed by Issue #19 and was executed from reviewed head `de781db190ebb0127444aa91c9a35b6311bdd8d5` on branch `v0.4-learned-delta-necessity` / draft PR #20.

## Frozen protocol

The experiment retained the 1,024-neuron architecture, topology seeds `1842/101/202/303/404/505`, cyclic donor-recipient pairings, exactly 24,000 actual neural updates per independent topology, 16x recurrent plasticity, recurrent-only readout, the frozen Experiment 013 response-fingerprint/cosine alignment, hard Dale compatibility, 1% magnitude-defined causal core, existing clipping, 40/60 evaluation, validation/adversarial separation, permutation-isomorph positive control, recipient virgin/mature controls, within-topology core control, and sealed terminal exclusion. Raw recurrent edge indices were not compared across independent topologies.

The independent topology development budget was 144,000 neural updates. The permutation-isomorph positive control used 24,000 updates, for 168,000 total training neural updates.

All L/M/R/Z conditions used the identical frozen functionally aligned recipient edge set and assignment order. L used the learned donor delta multiset. M independently and deterministically permuted learned absolute magnitudes and sign labels within Dale class while preserving the exact magnitude multiset and exact positive/negative/zero counts. R independently sampled magnitudes and signs with replacement from the learned per-Dale empirical distributions. Z applied zero deltas. Randomization used the preregistered `experiment015-v1-preregistered` SHA-256 seed namespace. No phenotype-dependent redraw, normalization, tuning, or selection occurred.

## Execution provenance

GitHub Actions run `35418529441` completed successfully. Dependency installation, imports, `PYTHONPATH`, the public `100/40/20` train/validation/adversarial split, protocol constants, artifact path, and all selected preflight tests passed. The preflight suite reported 19 passing tests. The expected report was produced at `results/experiment_015_learned_delta_necessity/report_n1024_steps24000.json`.

Report SHA-256: `3d6dff187ce6db4adb83029aa16b88bbe7f1208b4051373082d18120520131db`.

Immutable Actions artifact ID: `10576658782`.

Artifact ZIP SHA-256: `f1efb5bfc51a5d4a1f9ca864cdfb7f6eda73ea5c51d59783ec1f9d08eab97a9e`.

The permutation-isomorph positive control passed: mean cosine 1.0, hidden-permutation neuron accuracy 1.0, mapped-core correspondence 1.0, and zero validation/adversarial top-1 gap between mature and mapped graft.

## Pair-level results

Values below are recovery fractions relative to each recipient's virgin-to-mature change, reported as validation JS / top-1 and adversarial JS / top-1. Recovery fractions can exceed 1 or become negative when the corresponding virgin-to-mature denominator is small; they are retained exactly rather than clipped.

| donor -> recipient | L learned | M matched-permuted | R matched-resampled | within-topology core | clipping L/M/R/Z/within |
| --- | --- | --- | --- | --- | --- |
| 1842 -> 101 | V 0.297550 / 1.000000; A 0.312147 / 1.000000 | V 0.302688 / 1.000000; A 0.317354 / 1.000000 | V 0.297208 / 0.944444; A 0.307521 / 0.750000 | V 0.493009 / 1.000000; A 0.553194 / 1.000000 | 2 / 1 / 1 / 0 / 0 |
| 101 -> 202 | V 0.305207 / 0.875000; A 0.291198 / 0.333333 | V 0.296462 / 0.875000; A 0.283986 / 0.333333 | V 0.293894 / 0.750000; A 0.281234 / 0.333333 | V 0.436084 / 1.000000; A 0.473784 / 1.000000 | 0 / 0 / 0 / 0 / 0 |
| 202 -> 303 | V 0.326151 / 0.352941; A 0.303637 / 0.000000 | V 0.322442 / 0.411765; A 0.306065 / 0.000000 | V 0.331035 / 0.411765; A 0.315177 / 0.000000 | V 0.462026 / 1.117647; A 0.517858 / 1.000000 | 0 / 0 / 0 / 0 / 0 |
| 303 -> 404 | V 0.239636 / 0.142857; A 0.226554 / -0.400000 | V 0.234782 / 0.071429; A 0.227651 / -0.200000 | V 0.231972 / 0.214286; A 0.220360 / -0.400000 | V 0.439187 / 1.000000; A 0.486903 / 1.000000 | 0 / 0 / 0 / 0 / 0 |
| 404 -> 505 | V 0.244016 / 1.000000; A 0.264561 / 0.857143 | V 0.242494 / 1.000000; A 0.266662 / 0.857143 | V 0.239858 / 1.000000; A 0.263148 / 0.857143 | V 0.418816 / 1.000000; A 0.481368 / 1.000000 | 0 / 0 / 0 / 0 / 2 |
| 505 -> 1842 | V 0.299510 / 0.470588; A 0.306590 / 0.333333 | V 0.289674 / 0.058824; A 0.291895 / 0.000000 | V 0.292825 / 0.117647; A 0.294463 / 0.000000 | V 0.424196 / 1.000000; A 0.462257 / 1.000000 | 3 / 5 / 1 / 0 / 2 |

Z was exactly zero recovery for every pair and metric, confirming bookkeeping against the virgin recipient.

## Preregistered primary contrasts

Mean recovery difference L minus M was validation JS `0.0039212941`, validation top-1 `0.0707282913`, adversarial JS `0.0018456264`, and adversarial top-1 `0.0222222222`.

Mean recovery difference L minus R was validation JS `0.0042128814`, validation top-1 `0.0672074385`, adversarial JS `0.0037973920`, and adversarial top-1 `0.0972222222`.

The pair-level contrasts are heterogeneous. The largest apparent learned advantage occurs for pair 505 -> 1842, while several contrasts are zero or reverse sign. The preregistered aggregate contrasts are therefore small relative to the transfer recovered by L, M, and R themselves.

## Descriptor and clipping audit

The report records realized L/M/R count, per-Dale count, positive/negative/zero counts, mean and median absolute delta, L1, L2, deterministic seed provenance, and clipping counts before phenotype interpretation. M exactly preserved the learned per-Dale magnitude multiset and sign counts. R was not redrawn based on realized scale or phenotype. Clipping remained sparse and was reported rather than normalized away.

A notable structural limitation is visible in the realized descriptors: the selected 1% magnitude-defined donor core in these tested grafts consists of positive deltas from a single Dale class. Consequently, the M sign-label permutation has no effective sign-identity variation in those cores. This does not violate the preregistered null construction, but it limits what Experiment 015 can say about fine-grained sign arrangement. The experiment principally tests exact learned magnitude assignment/multiset information on the already selected homologous edge set under the realized core composition.

## Post-result interpretation audit

The result does not support a claim that the exact learned donor delta multiset is necessary for the bulk of cross-topology transfer on the frozen aligned recipient edge set. Both preregistered matched synthetic nulls retained nearly the same recovery as L on average, while Z recovered none. This strengthens the Experiment 014 conclusion that functional recipient edge-set selection is the dominant identified factor in this regime.

The result also does not establish that learned delta statistics are irrelevant. M deliberately preserves the exact learned magnitude distribution and sign counts, while R samples from those learned empirical distributions. Both nulls therefore inherit low-order information from learning. Experiment 015 distinguishes exact realized learned delta identity/assignment from matched learned-distribution surrogates; it does not compare learned statistics against an unrelated or parametric distribution.

No metric, mapping, null construction, inclusion rule, or control was changed after observing the phenotype. The small L-minus-null aggregate differences should not be promoted into a positive necessity claim without a separately preregistered experiment designed to test that new question. The 505 -> 1842 pair should likewise be treated as heterogeneity to be explained, not as a post hoc success criterion.

Experiment 015 is frozen at this result. Any methodological extension belongs to Experiment 016 or later.