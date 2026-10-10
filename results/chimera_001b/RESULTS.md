# Experiment 001B: 100-row reused-data neural chimera, complete exploratory run

**Status:** six-seed execution COMPLETE; exploratory source-exposed evaluation, **not** original Experiment 001 reproduction.  
**Source:** immutable historical v0.4 commit `7b7eb19ad852dc016ee0d370528d903bab78a5cf`, original SHA-256-authenticated v1 40-item validation and 20-item adversarial files.  
**Full numerical run:** [GitHub Actions 38025333272](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/38025333272); both compute and result-contract validation succeeded; the first attempt to commit results failed because the source branch had advanced concurrently. The already completed run's entire immutable Actions artifact was imported without retraining by [evidence-preservation run 38025494045](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/38025494045).  
**Raw data:** [RUN_REPORT.json](RUN_REPORT.json), [RUN_METADATA.json](RUN_METADATA.json), and separate `seed_1842.json`, `seed_101.json`, `seed_202.json`, `seed_303.json`, `seed_404.json`, `seed_505.json`. These files are authoritative; the precision shown here is rounded.

## Results

Six paired seeds: `1842, 101, 202, 303, 404, 505`. Fixed 1,024-neuron network, 12,000 phenotype training steps and 12,000 additional training steps for **each** fresh-decoder condition, matched initial random structures, fixed evaluation noise, plus a separately trained 12,000-step shuffled-label control. Fresh-decoder fitting freezes the recurrent trainable substrate. Validation and adversarial sets are historically exposed development/diagnostic sets, not newly independent confirmatory holdouts.

| Condition | Validation JS similarity | Adversarial JS similarity |
| --- | ---: | ---: |
| Virgin recurrence + virgin decoder | 0.777380 | 0.780053 |
| Trained recurrence + trained decoder (intact) | **0.854292** | **0.841098** |
| Trained recurrence + virgin decoder | 0.777393 | 0.780060 |
| Virgin recurrence + trained decoder | 0.853628 | 0.840638 |
| Trained recurrence + fresh supervised decoder | 0.853170 | 0.840299 |
| Virgin recurrence + fresh supervised decoder | 0.853672 | 0.840678 |
| Shuffled training target labels + trained decoder | 0.849656 | 0.836615 |
| **Training-label prior, no neural network** | **0.858388** | **0.841921** |
| Uniform 10-action prior, no neural network | 0.778291 | 0.780710 |

Top-1 validation agreement is 0.475 for the intact, virgin recurrence + trained decoder, fresh-decoder and shuffled-label conditions, and approximately 0.125-0.133 for virgin/virgin conditions. The trained direct **action-population-only** validation readout remains near 0.7784, versus roughly 0.7782 for its virgin recurrent counterpart, substantially below the supervised decoder scores.

### Primary paired decoder result

For `fresh decoder trained on mature recurrence MINUS fresh decoder trained on virgin recurrence` the six validation differences were:

| Seed | Validation paired difference |
| --- | ---: |
| 1842 | -0.000572 |
| 101 | -0.000396 |
| 202 | -0.000556 |
| 303 | -0.000653 |
| 404 | -0.000489 |
| 505 | -0.000343 |

Mean **-0.000502** JS-similarity units, all six negative. The range is tiny relative to the 0.0769 gap from a virgin decoder to a trained decoder. The intact-over-virgin-recurrence/trained-decoder gain is just +0.000664 validation units. The intact-over-shuffled-target gain is +0.004635. The no-network training-label prior **exceeds the intact network by +0.004097** validation units.

### Scientific interpretation and explicit limitations

Under this specific network, small authored behavioral dataset, metric, and source-exposed evaluation, the motor decoder and target-action base rates explain much of the absolute apparent competence. **There is no demonstrated advantage from a trained recurrent substrate after matched decoder refitting**; the observed paired effect is slightly negative and extremely small. The direct named action-population readout shows no meaningful spontaneous high-performing action policy.

This does **not** establish that recurrent synapses contain no information or that trained recurrence has no useful causal effect under every decoder. A trained-vs-virgin change also includes homeostatic bias and state differences, not an isolated `W`-only intervention. The six seeds are a deterministic robustness check, not an independent population sample. Likewise, source exposure, label-distribution similarity, semantic overlap and original fixture authorship preclude strong generalization or character-identity claims. No neural data from the 450-memory autobiography and no Hugging Face examples were used in this run.

Experiment 001 remains source-blocked on the complete original 130-item training battery. None of these results satisfy, substitute for or contradict the original 001 four-mean ±0.01 historical reproduction gate; they have their own protocol, manifest, directory, logs and source role tags. The sealed 20-item terminal battery was not loaded.

**Next controlled action:** Construct independently labeled and episode-disjoint behavior cards from the existing 450-event corpus with a reviewer-visible label ledger; use an alternate fictional persona as a matched contrast. Pre-register a genuinely unseen test set and compare a strong non-neural prior/classifier baseline before further neural mechanism claims. Add W-only recurrent lesions with learned biases held fixed to distinguish recurrent synapses from bias plasticity.
