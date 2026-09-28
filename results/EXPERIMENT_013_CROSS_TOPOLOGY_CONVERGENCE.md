# Experiment 013: Cross-topology functional core convergence

## Status

Experiment 013 is complete through Stage D on draft PR #16 (`v0.4-cross-topology-convergence`). Stage B passed the preregistered permutation-isomorph gate, Stage C found topology-independent causal-core functional similarity above Dale-sign-matched random controls for every tested topology pair and core fraction, and Stage D completed the preregistered cyclic cross-topology learned-delta graft.

The governing protocol is GitHub Issue #15. No raw recurrent edge indices were treated as homologous across independently randomized founder topologies. Terminal material remained sealed.

## Fixed protocol

Six independent 1,024-neuron founder topologies used seeds `1842, 101, 202, 303, 404, 505`. Each independent topology received exactly 24,000 actual neural training updates under the established architecture, 16x recurrent plasticity, recurrent-only behavioral readout, and canonical public curriculum. Evaluation retained the established 40-settle / 60-probe protocol and validation/adversarial separation.

Stage D used the preregistered cyclic donor-recipient pairs `1842->101`, `101->202`, `202->303`, `303->404`, `404->505`, and `505->1842`. Functional alignment used the Stage C representation and cosine metric derived only from unlabeled public training situations, with hard Dale compatibility. Donor mature-minus-founder recurrent deltas were assigned one-to-one to compatible recipient edges and added to virgin recipient founder weights, followed only by the existing Dale-sign and maximum-weight clipping rules.

Controls included recipient virgin founder, independently mature recipient, recipient within-topology 1% learned-delta core, Dale-sign-matched random recipient edges, donor deltas shuffled within Dale class across the same aligned recipient edge set, and a fresh permutation-isomorph positive control.

## Stage B gate

The permutation-isomorph positive control recovered 100% of the hidden neuron permutation and 100% mapped 1% causal-core correspondence. Its mapped graft reproduced mature validation and adversarial top-1 with zero percentage-point gap. The preregistered alignment gate therefore passed before independent-topology phenotype interpretation.

## Stage C result

Across all 15 independent-topology pairs and every tested causal-core fraction (0.25%, 1%, 5%, and 10%), causal-core endpoint functional similarity exceeded the Dale-sign-matched random-edge control. This established the descriptive prerequisite for Stage D without relying on raw edge indices.

Immutable Stage C provenance: workflow run `35385810828`, artifact ID `10563508320`, SHA-256 `e1aec8991611e58636b31b846c489bf40685cbd1c2ab4530f13a695a48615f35`.

## Stage D result

Stage D completed successfully at commit `2068d61159a1d7c6d840a9483d3a96c90d3ca204` in workflow run `35391466789`.

Cross-topology aligned 1% core grafts recovered approximately 64.0% of the virgin-to-mature validation top-1 change and 35.4% of the adversarial top-1 change on average across the six fixed cyclic pairs. Mean JS recovery was approximately 28.5% on validation and 28.4% on adversarial evaluation.

The exact aligned-graft validation/adversarial top-1 outcomes were: `1842->101: 47.5% / 35%`; `101->202: 45% / 15%`; `202->303: 15% / 10%`; `303->404: 17.5% / 0%`; `404->505: 47.5% / 30%`; `505->1842: 25% / 15%`.

The within-topology 1% core control was stronger, averaging essentially full top-1 recovery and approximately 44.6% validation / 49.6% adversarial JS recovery. Dale-sign-matched random-edge grafts produced approximately zero mean recovery.

The shuffled-delta control was close to the aligned condition. Shuffling donor deltas within Dale class across the same functionally selected recipient edge set retained approximately 58.2% validation and 33.2% adversarial top-1 recovery, with approximately 28.6% / 29.0% JS recovery.

Immutable Stage D provenance: artifact ID `10565518028`, SHA-256 `bb04869a879262832d6f208fa149143271b948ccadc3696e8ea61055f4c489f3`.

## Post-result interpretation audit

Experiment 013 supports topology-independent functional homology of a recurrent causal core and demonstrates partial causal phenotype transfer between independently randomized neural topologies. It does not establish identical internal representations, a literal biological engram, consciousness, or a substrate-independent identity object.

The stronger claim that precise donor-delta-to-recipient-edge correspondence is the principal cause of transfer is not supported by Stage D. The shuffled-delta control retained most of the aligned condition's aggregate effect. A substantial component of transfer can therefore be explained by selection of the functionally aligned recipient edge set together with the learned delta distribution and Dale-sign structure, rather than exact one-to-one synaptic delta placement. Pair-level heterogeneity also remains substantial.

No Stage D metric, mapping rule, control, inclusion criterion, or hypothesis was changed after observing the result, and Stage D was not rerun to improve the outcome.

## Next experimental question

A separately preregistered Experiment 014 should isolate the causal contribution of recipient edge-set selection from the causal contribution of exact learned-delta assignment. That follow-up must preserve the validated Experiment 013 alignment method and treat the Stage D shuffled-delta result as the motivating observation, not as a reason to retrospectively alter Experiment 013.