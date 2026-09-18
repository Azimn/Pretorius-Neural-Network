# Experiment Log

## Terminal integrity

Experiments 001 through 004 used only training, validation, adversarial, and historical holdout data. The sealed terminal battery has not been scored. Its public repository representation should remain limited to hashes until the experimental protocol is frozen.

## Experiment 001: Identity Surgery / Neural Chimera

Question: where does the measurable phenotype learned by Persona Inverse Synthesis reside?

Method: train one member of an identical pair of networks, then exchange the mature recurrent substrate and learned motor decoder with those from the untouched twin. Six seeds were run with 1,024 neurons and 12,000 phenotype-training ticks.

Mean validation Jensen-Shannon similarity across the six seeds was 0.7791 for the virgin twin, 0.8545 for the intact trained network, 0.7791 for trained recurrent core plus virgin motor decoder, and 0.8546 for virgin recurrent core plus trained motor decoder.

Interpretation: under the v0.3 architecture and training procedure, almost all measured inverse-synthesized phenotype performance is localized in the trainable motor decoder. v0.3 therefore demonstrates phenotype emulation but does not by itself demonstrate a distributed neural persona.

## Experiment 002: Decoder Amputation

Question: does a Pretorius signal remain if the learned motor decoder is bypassed entirely?

Method: ignore the motor decoder and read behavior directly from the ten designated neural action populations.

Result: default phenotype training changed recurrent weights, but direct action-population performance remained effectively unchanged across the tested seeds.

Interpretation: the default recurrent plasticity regime is too weak, poorly targeted, or both, to make the phenotype behaviorally recoverable without the decoder.

## Experiment 003: Biography Surgery

Question: does biography development distribute identity more deeply than inverse synthesis?

Method: repeat neural-chimera transplantation after biography-based development.

Result: the same qualitative localization appeared. Most measurable behavioral change followed the learned motor decoder. A budget confound was also discovered: 12,000 reported biography-development ticks produced approximately 29,042 neural network step calls because active developmental ticks can execute both a context step and a teaching/action step.

Interpretation: future biography versus PIS comparisons must use matched actual neural-update budgets.

## Experiment 004: Recurrent Embedding Sweep

Question: can stronger recurrent plasticity embed the phenotype so it becomes recoverable directly from neural action populations?

Method: bypass the learned decoder, evaluate direct action-population activity, and jointly scale recurrent Hebbian and reward-modulated learning rates.

At 1,024 neurons, a 16x plasticity multiplier produced a repeatable recurrent-only signal. Across five replication seeds, mean validation JS similarity rose from 0.7782 to 0.7831 and adversarial JS from 0.7806 to 0.7849. Mean validation top-1 agreement reached 0.465 and adversarial top-1 agreement 0.350.

A 64x condition degraded top-1 discrimination, indicating an overplastic regime.

At 4,096 neurons and 24,000 recurrent-only neural steps, seed 1842 improved validation JS similarity from 0.7784 to 0.8009 and adversarial JS from 0.7808 to 0.8012, with validation top-1 0.475 and adversarial top-1 0.350. Seed 101 independently completed at validation JS 0.7960, adversarial JS 0.7970, validation top-1 0.475, and adversarial top-1 0.350.

Interpretation: Pretorius-relevant behavioral structure can be embedded in the recurrent substrate. The architecture is therefore not incapable of distributed representation. The default learning regime and trainable decoder obscure the question.

## Next experimental series

v0.4 will freeze or bypass the trainable decoder, count actual neural steps, and test the recurrent phenotype using persistence, lesion, graft, and convergence experiments before any sealed terminal evaluation.
