# BC01-D2: plasticity-gain diagnostic, explicitly post-hoc development

This diagnostic is proposed **after observing the D1 negative result**. It is not a preregistered confirmatory test. It is not permitted to relabel D1 outcomes as successful or select only the most favorable trial.

Retain the same frozen 450-event source, same 16 labelled cards, original probes, topology settings local/global/generic, seeds 31/37/43, neuron count 256, background and card training budgets and evaluation controls. Increase only the existing generic donor's Hebbian and reward eligibility learning coefficients by **10.0x** during source background exposure and card exposure. Do not change the motor output decoder in recurrent conditions, neuron topology, source data, action annotation, or probe text.

The primary diagnostic remains per-seed action accuracy and lesion-minus-intact difference. A meaningful positive direction requires intact policy outperforming restored-recurrent, shuffled-label and no-plasticity outcomes across multiple seeds. A dramatic accuracy change without that three-way control separation is not a successful synaptic memory mechanism. Retain deterioration, chance, and instability. Use the same GitHub Actions runner with `--plasticity-gain 10.0` and a separate raw JSON artifact. Future confirmatory data must be independently reviewed, cluster-disjoint and finalized before further model selection.

If this diagnostic fails, prioritize investigating the recurrent teaching/readout mechanism and report action-collapse confusion statistics, not further scale-up or arbitrary parameter searches.
