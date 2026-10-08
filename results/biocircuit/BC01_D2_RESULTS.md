# BioCircuit BC01-D2: increased plasticity does not rescue behavior

**Date:** 2026-10-08. **Run:** [Actions 37836517245](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37836517245). **Data:** [BC01_D2_SUMMARY.json](BC01_D2_SUMMARY.json) and regenerable raw per-probe JSON in the CI artifact. **Status:** negative. The D2 diagnostic was explicitly specified after seeing the D1 results, so neither comparison is confirmatory.

The intervention changed only the existing recurrent Hebbian and reward eligibility coefficient multiplier from 1.0 to 10.0. The other settings were identical to D1: 450 source events, 16 provisionally action-coded records, three seeds, local/global/generic input conditions, 256 neurons and the same tick/exposure/lesson budgets. The trained motor decoder remained disabled in the recurrent conditions. CI passed 14 unit tests, executed D1 then D2 and successfully saved/reloaded all 18 experimental checkpoints.

## Diagnostic outcome

| Mode | Gain 1 intact mean | Gain 10 intact mean | Gain 10 recurrent-lesion mean | Gain 10 shuffled-label mean |
| --- | ---: | ---: | ---: | ---: |
| Compartment local | 22.92% | **25.00%** | 27.08% | 25.00% |
| Compartment global | 25.00% | **25.00%** | 22.92% | 25.00% |
| Generic recurrent | 27.08% | **25.00%** | 22.92% | 25.00% |

Every intact gain-10 network predicted a single action for *all 16 different cues*. Seed 31 predicted `cooperate` in local, global and generic. Seeds 37 and 43 predicted `challenge` in all three modes. This collapsed class distribution gives exactly four correct predictions and **25% accuracy** for every one of the nine runs.

The weight-delta L1 increased substantially, approximately 4,109 for seed 31, 4,507 for seed 37 and 4,240 for seed 43. Reverting only recurrent weight deltas changed between 9 and 16 of the 16 predicted actions, depending on seed and topology. Nevertheless, those changes were **not advantageous** in consistent decision accuracy; the intact network still collapsed to a seed-dependent constant classifier. All checkpoints reproduced scores exactly.

**Research interpretation:** tenfold plasticity is not a usable remedy. Greater learned recurrent change produces greater dependence on those weights but destroys action differentiation. The correct response is to diagnose representation, recurrent stability, teaching-input coupling, inhibition and readout calibration instead of adding neurons, retraining only the decoder, or further arbitrarily raising learning rates. The fact that external lexical retrieval still scores 100% on these development queries underscores the weakness of treating this cue-matching task as semantic behavior.

## Disposition

BC01 remains incomplete by the established causal-benefit standard. Both dose conditions and their failures are permanently recorded. Future work should test response diversity, per-action activation distributions, training-loss/weight trajectories, physiological saturation and homeostasis, with strict tests distinguishing a synapse-dependent but useless constant policy from a genuinely learned context-sensitive decision. Then design a genuinely new, independently reviewed challenge with source citations, contradiction controls and retrieval baselines.

The canonical Pretorius in The Doctor Lives has not been altered.
