# Experimental Protocol

The purpose of this repository is not to maximize a character imitation score by any available means. It is to determine where character-relevant behavioral structure is represented and how it changes through development.

The following rules apply to the v0.4 line.

1. Terminal evaluation remains sealed until architecture, plasticity regime, evaluation readout, and seed policy are frozen.
2. Actual calls to the neural update function are the budget unit. Labels such as tick, event, or presentation may be reported additionally but may not substitute for neural-step accounting.
3. The recurrent-only readout is the primary persona endpoint. A trainable motor decoder may be retained only as an explicit control condition.
4. Training, validation, adversarial, and terminal sets remain disjoint in function. Validation and adversarial sets may guide experimental development. Terminal data may not.
5. Every destructive manipulation must include an intact matched control and a virgin-founder control.
6. Multi-seed replication is required before interpreting a manipulation as a property of the architecture.
7. Raw machine-readable results should be preserved before narrative interpretation.
8. Pretorius-specific semantic variables must not be added to the organism merely to improve recognizability.
9. Language-model rendering is excluded from neural phenotype scoring.
10. Null results, confounds, and failed hypotheses are first-class results and should remain in the repository history.
