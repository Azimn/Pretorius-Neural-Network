# v0.4 Recurrent Persona Status

The v0.4 line makes the recurrent substrate, rather than the trainable motor decoder, the primary object of study. The terminal battery remains sealed.

Experiment 005 established a topology-specific recurrent trace. Learned synaptic deltas transfer phenotype behavior dose-dependently into a matched founder, lose their effect when scrambled, reverse the phenotype when inverted, survive substantial random lesions, and are disproportionately disrupted by lesions aimed at the most changed synapses.

Experiment 006 showed partial composability. Two networks trained on disjoint halves of the phenotype can be combined by adding learned synaptic deltas, producing a stronger phenotype than either half alone. Joint learning remains stronger than post hoc addition, and each half generalizes into domains it never directly experienced.

Experiment 007 tested developmental ordering. Interleaved learning, A-then-B, B-then-A, and short alternating epochs all converged to nearly identical behavior. Across three 1,024-neuron seeds, validation JS similarity was 0.797610 for interleaved training, 0.797715 for A-then-B, 0.797632 for B-then-A, and 0.797689 for short alternating epochs. Learned synaptic-delta cosine similarity between schedules was greater than 0.9989. The joint-learning advantage therefore appears to arise from state-dependent learning inside an already modified substrate, not from a privileged temporal order.

Experiment 008 tested relearning after targeted identity damage. Removing the 10% of synapses with the largest learned changes reduced mean validation JS similarity across three seeds from 0.783664 in the mature networks to 0.777552. Contrary to a savings hypothesis, damaged networks relearned more slowly than virgin founders at every measured checkpoint. After 24,000 additional steps, damaged networks reached 0.782584 versus 0.783664 for newly trained founders. The lesion therefore leaves a maladaptive residual configuration rather than a latent intact phenotype that rapidly re-emerges.

These are exploratory results. They constrain the architecture and motivate stronger controls, but they are not evidence of consciousness, biological memory, or human-equivalent identity.
