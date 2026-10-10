# Experiment 001G: The Recall Chamber, six-seed outcomes

**Completed 2026-10-10 | exploratory source-editorial cue-to-event discrimination | NO neural recall win.**

[Fixed protocol](../../research/CHIMERA_001G_RECALL_CHAMBER_PROTOCOL.md), [source hashes and run status](RUN_METADATA.json), [six-seed machine-readable report](RUN_REPORT.json) and `seed_1842.json`, `seed_101.json`, `seed_202.json`, `seed_303.json`, `seed_404.json`, `seed_505.json` contain full per-event ranks for all four neural interventions and the direct lexical baseline. [Source-integrity, smoke and complete six-seed CI run 38060772226](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/38060772226) passed. No historical v1 terminal, new 001E reviewer scenario, Hugging Face source or independent correctness labels were used.

## Design and provenance

All six original paired seeds (`1842,101,202,303,404,505`); 1,024 neurons, existing fixed signed 512-dimensional lexical sensory encoder, original 100 authored phenotype training examples (12,000 steps), and 450 v12 reconstructed Pretorius autobiographical events in each matched 26-epoch / 11,700-step frozen-vs-plastic source exposure. Original source corpus, git blob, SHA-256 and all historical training digests were verified, with the existing `biocircuit.bc01.load_corpus` importer reused. All target events have editorial `recall_cues`; the first two were joined into a query without supplying event identifiers. The neural source vector uses the same unchanged six text fields as 001F, not the cue array.

**Critical evaluation limitation:** The 450 expected cue-to-event matches are **editorial associations already present in the corpus**, not independently reviewed natural-language memory questions or verified human autobiographical facts. Every memory candidate is encoded and supplied at evaluation time. This is a *source association similarity* comparison, not evidence of autonomous, stored recollection or that the neural substrate contains source facts. The outcome is transductive and 001G has no 001E behavioral labels.

Every 450×450 similarity matrix is evaluated by exact source rank and the harder same-episode rank. Neural readouts use mean recurrent rate minus matched blank response, no motor decoder; trained W-only and trained homeostatic bias-only lesions are exact copied-parameter interventions, not re-fitted models. Controls include the original sensory vectors without recurrence, within-episode derangement of expected event labels, a single repeated neutral key, and BC01's independently existing **cue-metadata-based** lexical candidate retriever.

## Result 1: Recurrent plasticity did not improve cue-to-source discrimination

| Representation or external lookup | Full 450 top-1 | Top-5 | Mean reciprocal rank | Within-episode top-1 | Within-episode MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen-replay neural activity | **0.054815** | 0.143704 | 0.108221 | 0.282593 | 0.454631 |
| Plastic autobiographical replay | **0.054815** | 0.143333 | 0.108211 | 0.282593 | 0.454591 |
| Plastic replay + exact W-only lesion (trained bias preserved) | 0.054444 | 0.143333 | 0.107939 | 0.282593 | 0.454573 |
| Plastic replay + exact homeostatic bias-only lesion (trained W preserved) | 0.054815 | 0.143333 | 0.108196 | 0.282593 | 0.454634 |
| **No-neural raw lexical encoder cosine** | **0.100000** | **0.244444** | **0.176761** | **0.413333** | **0.581677** |
| Identical neutral key for every event | 0.002222 | 0.011111 | 0.014861 | 0.060000 | 0.203361 |

Results are six-seed arithmetic means. The neural pair `plastic top-1 MINUS frozen top-1` was **exactly 0 for each of the six seeds**, while the W-lesion effect was on average **+0.000370**, amounting to one changed correct candidate among 2,700 source queries across seeds. This is insufficient for a practical learned synaptic-source-discrimination claim. The direct lexical control beats the plastic neural representation by **0.045185 absolute full top-1** and by **0.130741 absolute same-episode top-1**. Neural input projection and recurrent rate dynamics reduce discriminability here; they have no measured source-identity benefit on this source-editorial benchmark.

Plastic-minus-frozen MRR averaged **−0.0000103** across seeds. The positive plastic-minus-W-lesion mean MRR was **+0.0002723**, uneven across seeds and far below the preregistered strong-benefit requirement. Full per-seed deltas and per-event ranks are preserved in the run JSON.

## Result 2: Null-label and lexical index controls reveal the source of apparent success

Each neural matrix evaluated against a deterministic **wrong source assignment permuted within the actual source episode** gave mean top-1 of **0.002222**, a strong fall compared with actual editor-associated source IDs; it confirms that the cue-to-record metric detects some *static input* association, not that post-replay synaptic learning improves that association. The neutral equal-key exact rate was **0.002222**, the expected 1/450 under identical queries and deterministic tie-breaks.

The **existing BC01 lexical retriever** proposed the editor-associated source for **447 of 450 cue queries**: 99.33% at 100% candidate coverage, while continuing to report semantic `verdict=unknown`. **These numbers must not be compared as a fair architecture contest with the 10.00% raw sensory or 5.48% neural baselines:** BC01 looks directly at the `recall_cues` metadata used to construct each query, whereas neural and raw sensory similarity use the six narrative text fields and are not given that cue metadata within the source vector. The BC01 result is essentially a cue-index roundtrip / source metadata integrity test, not semantic understanding or an independently collected recall ability.

## Decision

**001G's strong neural-specific improvement gate fails on all three prerequisites:** no +0.05 top-1 gain over frozen, no +0.05 gain over the W-lesioned model, and no gain over the raw lexical baseline. Source specificity in immediate responses reflects existing lexical input and stored external cue metadata, not demonstrated recollection from changed recurrent weights.

Do not increase Hebbian rate, add memory replay epochs or teach a cue-to-ID motor decoder using these exact 450 source/editorial pairs and relabel its in-sample accuracy as memory acquisition. Useful next development is a **provenance-gated source retrieval → inspectable subjective-access transition**, distinct from the engineering audit channel, and then a genuinely new independently reviewed, source-heldout counterfactual assay. Reuse the Connectome's published L1/L2 feature caches and BC01 interface rather than duplicate the index. Compare neutral-key and wrong-provenance controls and test neural causal contributions only after the source identity and external retrieval claims are separated.

001F's earlier change to recurrent W is real but has now been tested against a more targeted source-discrimination readout with a negative functional result. 001E remains pending independent behavior review, and historical Experiment 001 is still not reproduced.
