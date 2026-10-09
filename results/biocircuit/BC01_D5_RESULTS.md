# BC01-D5A: localize why recurrent teaching has almost no source-conditioned effect

**2026-10-08** · [Frozen D5 observational protocol](../../research/BIOCIRCUIT_D5_PROTOCOL.md) · [CI run 37862249986](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37862249986) · [18-trial full source-linked telemetry](BC01_D5_FULL_PER_CASE.json).

## Key finding

The existing D4 update is not selective enough for individual recall cues. The current rule gates outgoing presynaptic activity by its firing rate exceeding a constant population target. In the 1× experiment, **100% of the existing edges entering the taught action population were eligible for each event**. At 12× source sensory gain, about 97.1% of candidate incoming edges were eligible. Thus stimulus-independent background activity is sufficient to qualify most synapses for training. This is an empirical mechanism for why shuffled labels or blank-cue learning performed as well as target-label learning in D4, though it is not a proof of all causal failure modes.

No model, sensory encoding, original source record, recurrent W rule, source action label, input matrix or readout was changed by D5. The addition is a *strictly observational callback* in `biocircuit/bc01_d4.py` and a new `bc01_d5.py` audit runner. 37 BioCircuit regressions passed, including exact equality of observed versus unobserved W.data, voltage, rate, tick, motor decoder, policy scores and baseline D4 behavior.

## Measured results

Source blob: `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`, 450 explicitly reconstructed memories, 27 episodes and 16 interpreted first-person source decisions (not independent review). Source-owned deterministic lexical L2 v2: Pretorius-Connectome `7cd631f3b533203e39465ec31c4ee42945610e2f`; no learned semantic representations. Each gain includes 9 development runs (seeds 31,37,43 × local,global,generic), 16 probes and 48 per-card training presentations. D5 tested the frozen D4 sensory gains **1× and 12×**, generating 18 configurations, 864 direct training-event trace entries and 288 source-grounded query counterfactual records. Original D4 accuracy and neural states are reproduced without modifying the model.

| Metric (average across 9 seed/mode configurations) | 1× | 12× |
| --- | ---: | ---: |
| Original D4 decision accuracy | 20.14% | 27.08% |
| Candidate existing recurrent edges entering taught action population per event | 358.92 | 358.92 |
| Eligible edges per event | **358.92 (100%)** | **348.66 (97.1%)** |
| Excitatory eligible edges | 286.58 | 278.39 |
| Inhibitory eligible edges | 72.33 | 70.28 |
| Actual modified edges per event | 358.92 | 348.56 |
| Clipped eligible edges per event | 0 | 0.155 |
| Mean per-event L1 W update | 0.09623 | 0.10404 |
| Mean absolute change in target-action score margin from all newly learned W | **0.0001234** | **0.0013491** |
| Mean absolute margin change from removing the test cue | **0.0069212** | **0.1308750** |
| Learned W-margin effect divided by cue-margin effect | **1.78%** | **1.03%** |
| Policy-choice flips when reverting all card-learned recurrent weights | 1 / 144 | 2 / 144 |
| Policy-choice flips when removing the test cue | 79 / 144 | 99 / 144 |

At 1×, not a single one of the 155,052 eligible edge presentations was clipped to a Dale sign/max-bound limit. At 12×, only **67 of 150,623** eligible edge presentations were clipped (~0.044%), all in the Dale-sign direction, with none at the maximum absolute weight. This argues **against global clipping saturation as the primary blocker**. A single highest-delta recurrent edge's reversion had even less effect on the score margin (mean 0.0000145 at 1×; 0.0000828 at 12×), further indicating a diffuse but weak learned W contribution to this action-population readout.

The existing 16 development questions do not isolate independent generalization, and action margins are neural probabilities, not psychological certainty. Each run's cue-based effects are measured from the identical settled state and externally supplied episode-linked decisions, but source labels never enter neural evaluation input channels. The tested structural sign constraints and blank-cue teaching controls remain unchanged.

## Reproduction and provenance

Run source-pinned CI `biocircuit-bc01-d5.yml` or:

```sh
python -m unittest discover -s tests_biocircuit -v
python scripts/run_biocircuit_bc01_d5.py \
  --corpus upstream-connectome/memories/current/Pretorius_v12_450_Events_Complete.jsonl \
  --cache-root data/derived/shared-l2-cache \
  --connectome-root upstream-connectome \
  --seeds 31,37,43 --modes local,global,generic --gains 1,12 \
  --output results/biocircuit/BC01_D5.json
```

The source-owned L2 v2 seed caches must first be generated using `upstream-connectome/scripts/build_shared_memory_l2.py` and the canonical source commit pinned in CI. The successful [Actions run 37862249986](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37862249986) passed **37 unit tests**, all 18 configurations, exact D4 baseline reproduction, restart and two fresh Python executions under different `PYTHONHASHSEED` values matching every numerical/neuronal datum (only observed elapsed runtime excluded). Full raw event-linked traces are committed as `results/biocircuit/BC01_D5_FULL_PER_CASE.json`, not merely an expiring CI artifact.

## Interpretation and next targeted test

The D5 eligibility condition `max(cue_rate[pre] - target_rate, 0)` does not distinguish **cue-evoked activity** from the same neuron's spontaneous/blank-cue response. Nearly all available action-incoming edges therefore receive eligible credit for nearly any lesson, irrespective of source recall cue. That is an immediate, testable mechanistic bottleneck. The direct Win cue signal also has around 56–97× greater *effect on action margin* than learned W changes, despite active teaching-induced response and substantial W updates.

**Prespecify D5B before trying it:** replace only the *presynaptic* eligibility factor with `max(rate_after_cue[pre] - rate_after_matched_blank[pre], 0)` after a matched blank sensory branch from the identical initial fast state. Keep the postsynaptic teacher-vs-no-teacher counterfactual, label/action-population mask, W sign constraint, network architecture, source L2 cache, data/splits, sensory gains 1/12 and all D4 learning controls unchanged. The additional blank-branch simulation costs extra inference; quantify it and do not claim compute-matched comparisons. Report eligibility sparsity, W delta, accuracy versus shuffled-label/blank/no-training/whole-W-lesion and unchanged original model, across all seeds/modes. If it fails, preserve the negative result rather than further unregistered hyperparameter sweeps. **No The Doctor Lives migration.**

The original BC01 #26 and engineering #29 causal-learning acceptance conditions remain unmet.
