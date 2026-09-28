# Experiment 012: Causal Core Convergence

## Status

Completed successfully on GitHub Actions from commit `5a938ff22465a8365dc2be76d2f17297afc168e9`. The immutable workflow artifact is `experiment-012-core-convergence-dd9d6337f23a4a1588f6112ecaa066101972c9d3`, artifact ID `10547557335`, SHA-256 `55463433056062dee90b0024e8f9a830d55c40e9e82d8db8f688ed3449cf0cad`.

The workflow passed dependency installation, runtime/data preflight, scientific execution, and artifact upload. The sealed terminal battery was not touched.

## Question

Does the causal recurrent Pretorius core converge to the same locations and values when the same founder topology develops under different curriculum orders?

## Protocol

Three 1,024-neuron seeds (`1842`, `101`, `202`) were developed from matched founder topology under canonical, reversed, and deterministically shuffled curriculum order. Every condition received exactly 24,000 neural updates. Plasticity multiplier remained 16.0. Evaluation retained the established 40-step settling and 60-step probe windows, separate validation and adversarial batteries, recurrent-only readout, and sealed terminal exclusion.

For each developed network, recurrent edges were ranked by absolute change from founder. Causal-core overlap and learned-delta correlation were measured at 0.25%, 1%, 5%, and 10%. Cross-grafts then placed values learned under one curriculum ordering into a matched virgin founder, either at the canonical core locations or at the donor's independently selected core locations.

## Results

All three developmental orderings converged on the same measured phenotype. Across seeds, canonical, reversed, and shuffled development each produced mean validation top-1 agreement of 47.5% and adversarial top-1 agreement of 35.0%. The founder baseline was 11.67% validation and 8.33% adversarial. Mean JS similarity was likewise nearly identical across the three mature conditions: 0.783664, 0.783689, and 0.783679 on validation, and 0.785285, 0.785317, and 0.785312 on adversarial for canonical, reversed, and shuffled respectively.

The learned causal cores showed very high location convergence. At 0.25%, mean Jaccard overlap was 0.9221 for canonical versus reversed, 0.9221 for canonical versus shuffled, and 0.9841 for reversed versus shuffled. At 1%, the corresponding overlaps were 0.9600, 0.9679, and 0.9718. At 5%, they were 0.9528, 0.9693, and 0.9741. At 10%, they were 0.9717, 0.9805, and 0.9781.

Value convergence was even stronger. Mean learned-delta correlations on the selected core were already 0.9917 to 0.9975 at 0.25%, rose to 0.9962 to 0.9985 at 1%, and reached 0.9983 to 0.9993 at 10%.

The graft intervention confirms that this overlap is functionally meaningful rather than merely descriptive. At the 1%, 5%, and 10% core sizes, every tested graft condition, including reversed values placed at canonical core locations, shuffled values placed at canonical core locations, each donor's own independently selected core, and the canonical core itself, reproduced the full mature top-1 scores of 47.5% validation and 35.0% adversarial on average. At 0.25%, the core was near but below full transfer, with validation averaging roughly 44.2% to 45.8% and adversarial 23.3% to 25.0% depending on graft condition.

## Interpretation

Under a fixed founder topology and fixed developmental content, curriculum order has remarkably little effect on the final recurrent implementation. The three developmental histories converge not only on the same observable phenotype, but on nearly the same high-change recurrent locations and almost identical learned weight deltas. Cross-grafting independently learned values into independently selected or canonical locations reproduces the mature phenotype once approximately 1% of recurrent edges is transferred.

Together with Experiments 010 and 011, the evidence now supports a stronger statement than compact sufficiency or necessity alone: on a fixed neural topology, this Pretorius training regime exhibits a highly reproducible causal attractor. The relevant recurrent core is compact, disproportionately necessary, sufficient for phenotype transfer, and largely invariant to substantial reordering of the same developmental experiences.

This result should not be generalized to independently randomized neural topologies. Raw recurrent edge indices are homologous here because each comparison shares the same founder topology. Whether independent topologies converge on a functionally equivalent but structurally non-homologous implementation remains unresolved.

## Next Experiment

The next scientifically useful question is cross-topology convergence. Experiment 013 should test whether independently randomized founder topologies develop functionally interchangeable Pretorius cores despite lacking edge-index homology. The experiment should compare phenotype convergence first, then use a topology-independent representation of causal-core function rather than raw edge-index overlap. A precommitted mapping or functional-signature method is required before any cross-topology graft is interpreted.