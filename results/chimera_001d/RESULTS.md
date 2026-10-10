# Experiment 001D: Isolated synapse, homeostasis and motor-decoder factorial

**2026-10-10 | COMPLETE | Six paired seeds | Exploratory validation only**

This is a separate mechanistic follow-up to [001B](../chimera_001b/RESULTS.md), **not a reproduction of historical Experiment 001**, and not an independent unseen evaluation. The original 130-item training source remains missing 30 legacy examples. All measured conditions used the pinned historical v0.4 100-example phenotype training subset (commit `7b7eb19ad852dc016ee0d370528d903bab78a5cf`) and the SHA-256-authenticated original v1 40 validation / 20 adversarial records as **previously exposed diagnostic data**.

[Full machine-readable report](RUN_REPORT.json), [source/config fingerprints](RUN_METADATA.json), and all six complete `seed_<seed>.json` with per-item metrics and each component SHA-256 are in this directory. The source-pinned CI run used six seeds `1842,101,202,303,404,505`, 1,024 neurons, and exactly 12,000 neural training steps each. No hyperparameter optimization or terminal-battery reads occurred.

## Outcome 1: synaptic W produces a tiny but consistent positive effect when the trained decoder is fixed

**Validation JS similarity, arithmetic mean of six matched seeds:**

| Recurrent W | Homeostatic bias | Decoder | Validation | Adversarial |
| --- | --- | --- | ---: | ---: |
| Virgin | Virgin | Fully trained | 0.853628 | 0.840638 |
| **Trained** | Virgin | Fully trained | **0.854639** | **0.841351** |
| Virgin | **Trained** | Fully trained | 0.853220 | 0.840338 |
| **Trained** | **Trained** | Fully trained | 0.854292 | 0.841098 |

Paired trained-W/virgin-bias minus virgin-W/virgin-bias validation delta **+0.001011**. Its exact six paired deltas were `+0.000982, +0.001125, +0.001050, +0.000904, +0.001062, +0.000942`. With trained homeostatic bias held fixed, the learned-W effect was **+0.001071**. Trained bias alone was **-0.000408** with virgin W; the learned bias added to already trained W was **-0.000347**.

The learned recurrent W tensor changed measurably (mean L2 displacement from founder **0.261267**) and the learned homeostatic bias changed (L2 **0.030206**). These source-matched interventions separately preserve W connectivity topology and input projections. Every condition has recorded independent component hashes. The r11 intact factorial score exactly matched d11, independently evaluating the same trained network through the decoder-factorial path.

This is evidence for a small **causal influence of learned W on these measured outputs** under a fixed pretrained decoder. It is **not** a causal proof of persistent autobiographical identity, a large recurrent phenotype effect, or a validated mechanism with unseen cases. In 001B, **refitting equally supervised fresh decoders on trained vs virgin recurrent W+bias** gave mean effect -0.000502, not a positive trained-recurrent advantage. Thus the small W benefit depends on the readout intervention and does not establish a reusable recurrent representation.

## Outcome 2: motor decoder weights, not just motor biases, carry major learned effects

**Decoder 2×2 factorial, fully trained W and recurrent bias held fixed:**

| Motor weight matrix | Motor bias vector | Validation | Adversarial |
| --- | --- | ---: | ---: |
| Virgin | Virgin | 0.777393 | 0.780060 |
| **Trained** | Virgin | **0.848820** | **0.837340** |
| Virgin | **Trained** | 0.795619 | 0.795520 |
| **Trained** | **Trained** | 0.854292 | 0.841098 |

Trained motor weights alone minus trained motor bias alone: **+0.053201** validation JS similarity (all six seeds positive), with motor-weight L2 displacement 1.362779 and motor-bias L2 displacement 0.098199. This shows that the output learning effect is not solely a constant motor-bias prior: the trained motor-weight matrix has a much larger contribution. Yet the separately computed **no-network training-label marginal prior** still reaches **0.858388** validation / **0.841921** adversarial, above the intact 0.854292 / 0.841098 on both exposed sets. This is a strong metric-level baseline, not a claim the trained motor weights lack context information.

## Outcome 3: scenario content has a detectable effect

With labels and item identities held fixed, moving intact-model scenario texts and environment scalars to different items produced an average **0.005590** drop in validation JS similarity for global context reassignment and **0.004559** drop for pair swaps within each of the 20 domains. Each effect was positive for all six seeds; global adversarial shuffling is recorded separately in machine-readable results. The context permutations do not change the action targets and are deterministic from the seed.

Therefore describing this network as *completely context-insensitive* would be incorrect. There is measurable conditional signal in its existing fitted model, but it does not suffice to exceed the strong action-prior baseline and its semantic validity is not independently tested.

## Inference boundaries and decisions

The 100 authored training and 40/20 evaluation cards use the same old design; there is no newly authored or blinded independent holdout here. Our six seeds measure sensitivity to random initialization within one algorithm and fixture, not general reproducibility across architectures or persons. The contexts were not adversarially held out from prior design work. Motor matrices can learn weighted features of initially random recurrent activation without needing recurrent plasticity. A small +0.001 W-only causal score gain under a fixed trained motor decoder does not conflict with the -0.000502 fresh decoder paired difference in 001B, since those are distinct interventions.

**Methodological decision:** Stop optimizing on the old 40/20 sets. The next study must use the 450 Pretorius autobiographical records as a separately provenance-controlled developmental source, and independently authored counterfactual situations with two independent blinded behavioral judgments before their ten-action distributions are treated as truth. Source integration must pass 001C's 448-of-450 reference-entanglement and unreviewed annotation gates. This experiment does not justify claims of selfhood or psychological continuity.

No changes were made to historical Experiment 001, its missing-data status, its sealed 20-item terminal, 001B, or the original 450-memory texts.
