# Experiment 016: Delta-distribution necessity

Status: complete and scientifically frozen.

## Frozen question

With the Experiment 014 functionally homologous recipient edge set fixed, does learned empirical delta-distribution shape carry transferable information beyond a shape-independent scale-controlled amplitude null?

The preregistered primary comparison is L versus S. L is the learned donor delta realization under the inherited assignment convention. S is the preregistered per-Dale half-normal amplitude null, deterministically generated under `experiment016-v1-preregistered` and exactly RMS/L2 matched to L before inherited network clipping. R, the Experiment 015 empirical-resampling condition, is the secondary anchor. Z is zero-delta bookkeeping.

No mapping, metric, seed, inclusion rule, null, or clipping rule was changed after phenotype observation.

## Execution provenance

Scientific head: `8a69ef26a8c8c789eba45568c59c3790743c2c9d`

GitHub Actions run: `35426641235`

Immutable artifact ID: `10579336290`

Report path: `results/experiment_016_delta_distribution_necessity/report_n1024_steps24000.json`

Report SHA-256: `845079466d8db9aaf9e190dcd7f27d0a2aed0e5f6aea9b1454424b0de66e7d50`

Uploaded artifact ZIP SHA-256: `ebb699727afae474b3af6e8fa5823fa67237e62e82efe5d2489340f361e0d2fa`

All dependency/import checks, public 100/40/20 split checks, protocol/null preflights, inherited alignment tests, neural-step accounting tests, scientific execution, expected-report-path verification, and artifact upload passed. The selected protocol suite reported 24/24 tests passing.

Training accounting remained frozen: six independent topologies at exactly 24,000 actual neural updates each, plus the 24,000-update permutation-isomorph control, for 168,000 total training neural updates. The 1,024-neuron architecture, 16x recurrent plasticity, recurrent-only readout, six topology seeds and cyclic pairings, 1% core, hard Dale compatibility, 40/60 evaluation, validation/adversarial separation, frozen functional mapping, and sealed terminal exclusion were preserved.

The permutation-isomorph positive control passed with mean cosine 1.0, hidden-permutation neuron accuracy 1.0, mapped-core correspondence 1.0, and zero validation/adversarial top-1 gap. Raw recurrent edge indices were not compared across independent topologies.

## Pair-level preregistered contrasts

| donor -> recipient | L-S validation JS | L-S validation top-1 | L-S adversarial JS | L-S adversarial top-1 | R-S validation JS | R-S adversarial JS |
|---|---:|---:|---:|---:|---:|---:|
| 1842 -> 101 | 0.028705 | 0.000000 | 0.020254 | 0.000000 | 0.028363 | 0.015628 |
| 101 -> 202 | 0.052493 | 0.000000 | 0.040584 | -0.166667 | 0.041180 | 0.030620 |
| 202 -> 303 | 0.039336 | -0.117647 | 0.037814 | 0.000000 | 0.044220 | 0.049354 |
| 303 -> 404 | 0.037966 | 0.000000 | 0.032114 | -0.200000 | 0.030302 | 0.025920 |
| 404 -> 505 | 0.040886 | 0.000000 | 0.050225 | 0.142857 | 0.036727 | 0.048812 |
| 505 -> 1842 | 0.054646 | 0.470588 | 0.057112 | 0.500000 | 0.047961 | 0.044985 |

Pair-level L-S JS recovery is positive for all six topology pairs in both validation and adversarial evaluation.

## Aggregate preregistered contrasts

Mean L-S recovery difference was `0.0423384864` validation JS and `0.0396837306` adversarial JS. Mean L-S top-1 recovery difference was `0.0588235294` validation and `0.0460317460` adversarial.

Mean R-S recovery difference was `0.0381256050` validation JS and `0.0358863387` adversarial JS. Mean R-S top-1 recovery difference was `-0.0083839091` validation and `-0.0511904762` adversarial.

Thus the JS-based phenotype measure gives a consistent pair-level result: both the exact learned realization L and empirical-resampled R outperform the shape-independent RMS-matched half-normal S. The much smaller L-R differences observed in frozen Experiment 015, together with the present similar L-S and R-S JS contrasts, are consistent with transferable information residing in properties of the learned empirical amplitude distribution rather than in exact donor-delta identity.

## Clipping audit

Inherited clipping was retained and reported, not normalized away. Per-pair clipping counts for L/R/S were respectively: 1842->101 = 2/1/2; 101->202 = 0/0/4; 202->303 = 0/0/1; 303->404 = 0/0/2; 404->505 = 0/0/3; 505->1842 = 3/1/4. Z clipped zero times throughout. Within-topology core clipping was 0/0/0/0/2/2 across the six pairs.

S clips somewhat more often than L or R, so clipping is a plausible partial contributor to the size of the observed phenotype difference. It does not by itself explain the direction as a demonstrated causal mechanism, and no post-result normalization or redraw is permitted.

## Post-result interpretation audit

The primary JS result is internally consistent across all six cyclic topology pairs and across validation and adversarial sets. The permutation-isomorph positive control behaved exactly as required, Z remained the bookkeeping baseline, the aligned recipient edge set and assignment order were held fixed, and no execution or artifact failure occurred.

The defensible conclusion is narrow: in this architecture and 1% aligned-core regime, matching the learned delta RMS while replacing the learned empirical amplitude shape with the preregistered half-normal null reduces transfer. Combined with Experiment 015, this supports the hypothesis that learned empirical delta-distribution properties contain transferable information beyond scale alone, while exact individual donor-delta identity is not necessary for most of the observed transfer.

This experiment does not identify which distributional statistic is causal. The half-normal null changes higher-order shape, quantiles, tails, and realized clipping behavior together. It therefore cannot distinguish skew/tail structure, quantile structure, clipping susceptibility, or another empirical-distribution property. Top-1 results are also heterogeneous and substantially less consistent than JS. These are interpretation boundaries, not grounds to alter or rerun Experiment 016.

Any attempt to isolate a particular distributional statistic or clipping mechanism requires a new numbered, preregistered experiment.
