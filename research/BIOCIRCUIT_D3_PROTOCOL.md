# BioCircuit BC01-D3: targeted local eligibility diagnostic

**Protocol frozen for this implementation before inspection of trial metrics, 2026-10-08.** This is a **development-only**, researcher-designed neural-learning challenger against the existing BC01-D1/D2 failures, not a preregistered independent confirmatory benchmark.

## Reason for trying this mechanism

BC01 D1 used uniform reward-modulated Hebbian plasticity across every synapse and achieved approximately four-class chance on 16 source-interpreted decisions. D2 increased plasticity tenfold and collapsed to one action per seed, despite numerous recurrent lesion-induced policy flips. Neither supports useful learning. The working hypothesis for D3 is that eligibility must depend on **both source cue activity and a particular action-population teaching response** rather than distributing an undifferentiated reward to every synapse.

This experiment changes **no recurrent topology**, increases no neural capacity and trains **no motor decoder**. It uses the same 16 development-only source cards, frozen first-person decision strings, action mappings, queries, four balanced action classes, 450 frozen source narratives, source-owned L1/L2 TF-IDF cache and BioCircuit L3 adapter. The original hashed lexical representation is measured as a labeled input control.

## Candidate mechanism

For each decision card, reset the existing `PlasticRecurrentPersonaNet` fast state. Present **cue only** for the first half of 32 ticks; record presynaptic rates. Then run a *no-teacher counterfactual second half* from the saved cue-only network state, then restore that same state and run the **same cue plus the recorded action teaching channel** for the second half; compare the two responses at equal elapsed model time. The additional counterfactual steps cost extra inference CPU, and are not a matched-compute comparison with the older generic learner. Apply one source-label-gated local update to **already existing recurrent edges** arriving at that action population:

```
delta_W[post,pre] = 0.03
                  * max(cue_rate[pre] - target_rate, 0)
                  * max(taught_rate[post] - matched_no_teacher_rate[post], 0)
                  * indicator(post belongs to taught action population)
```

Enforce the original Dale-like presynaptic sign, weight bounds, sparse CSR indices and row pointers. No action, source ID or evaluator truth label enters inference inputs. No action population's output weights or bias is supervised. The teaching channel is discarded before policy evaluation. This is a deliberately simplified three-factor **engineering surrogate**, not proof of fly neurotransmitter physiology.

## Fixed comparisons and verdict

Retain the exact same 450-event background curriculum, 16 source cards, three epochs, 32 ticks per card, 32 probe-settle ticks, 256 units, three seeds `31,37,43`, and three input competition modes `local,global,generic`. The generic Hebbian control receives label and cue simultaneously for 32 exposure ticks. The candidate executes 32 exposure ticks plus 16 additional matched-cue counterfactual ticks per card. **Compute is not equal**; the scientific comparator is equal data/teaching exposure, and runtime overhead must be disclosed. The original first implementation mistakenly compared different timepoints and yielded zero update on a fixture; CI caught it and this protocol explicitly preserves the correction. Targeted controls start from the **identical pre-card network** and include (i) shuffled teacher action assignments, (ii) no sensory cue during teaching, (iii) no further training, and (iv) recurrent-weight lesion restoring only pre-card synaptic weights. Source input uses the cache fitted on the matching training episode partition for each seed, with exact source and encoder hashes checked at load.

Archive per-event action scores, predicted label, target, confusion matrix, action-count histogram, per-population firing means, low/high activity saturation, baseline and trained bias statistics, cue-response RMS variation, recurrent synaptic delta L1, trained checkpoint and exact restart replay.

**No single-seed result will be called proof.** A development-positive direction requires action diversity (at least two outputs), accuracy above balanced 25% chance, AND improvement over each of pre-card recurrent lesion, shuffled teaching, blank-cue teaching and legacy global Hebbian exposure, across all three seeds. A large synaptic update or many lesion flips without correct source-dependent choices is a failure. These reused cue prompts cannot establish semantic generalization. No selected configuration should be moved to The Doctor Lives without a new independent challenge and production gate.

## Reproduction

```sh
python -m unittest discover -s tests_biocircuit -v
python scripts/run_biocircuit_bc01_d3.py --corpus upstream-connectome/memories/current/Pretorius_v12_450_Events_Complete.jsonl --cache-root data/derived/shared-l2-cache --connectome-root upstream-connectome --seeds 31,37,43 --modes local,global,generic --eta 0.03 --output results/biocircuit/BC01_D3.json
```

Use the pinned external shared-memory L2 builder once per independent seed before the second command; the full CI workflow performs it and archives the machine-readable traces. This implementation is independent of the separate FlyWire biological CSR experiment. No user-facing fictional history is regenerated.
