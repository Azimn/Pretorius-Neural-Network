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

## 2026-10-08: Experiment 001 reconstruction attempt (no reproduction claim)

Status: **source-blocked, not reproduced**. The original Experiment 001 narrative remains unchanged above. A new runner, `scripts/run_chimera_001.py`, and pinned reconstruction file, `config/chimera_001.json`, now specify the six historical seeds, 1,024 neurons, 12,000 phenotype-training neural steps, four exact component-copy chimeras, per-seed validation/adversarial Jensen-Shannon scoring, and the ±0.01 historical validation-mean acceptance gate. The runner halts Phase 2 on any failed Phase 1 condition, without score-guided tuning.

The original 100 axis-grounded + 30 legacy training inputs and 40 validation inputs are absent from this repository's main branch. Their SHA-256 identities are recorded in the historical Phenotype Battery v1 manifest; canonical adversarial and phenotype-profile inputs were recovered unchanged from the historical v0.4 branch. Missing hash-matched sources block execution. Default v0.3.1 network settings and the concatenated training presentation order remain explicitly identified as reconstruction assumptions, since the historical narrative does not pin their exact Experiment 001 values. **No per-seed Experiment 001 scores or successful reproduction are asserted here.** The preflight and unresolved details are preserved at `results/chimera_001/`.

Phase 2 is implemented but **not executed**. It initializes a fresh decoder on the frozen mature recurrent substrate using the same target schedule and neural-step count. A matched virgin recurrent + freshly trained decoder arm is included because a supervised decoder may learn phenotype targets from generic fixed recurrent features. Only a comparative six-seed paired difference can support a mature recurrent substrate-specific interpretation; reaching intact score alone cannot establish that distinction. The sealed terminal battery is neither read nor evaluated.

Journal evidence entry CC-E01 must continue to describe the original result as **narrative-only** until canonical sources are recovered, six-seed per-condition JSON is committed, the reproduction gate passes, and a pinned result commit hash can be cited. Preserve any eventual mismatch as a measured outcome.

## 2026-10-08: Correction to reconstruction source inventory

A subsequent CI SHA-256 verification caught a source mismatch in the v0.4 preservation branch. That branch's file named `pretorius_adversarial_v1.json` hashes to `20f87fd29798be3540a67daec93d71175d019e5611e806c36f4263e196eb6a87`, not the original v1 manifest's `e1510589bbf468d4c4d94a9587ac5dba9cd3284370efe9dae6779d043ccbc5cf`. The nonmatching adversarial file was removed from this reconstruction branch. **Only the phenotype profile is currently original-hash-verified and recoverable.** The original 100+30 train, 40 validation and 20 adversarial data files are missing. The earlier reconstruction entry's suggestion that canonical adversarial inputs had been recovered is superseded by this correction. No experimental scores have been produced; neither Experiment 001 nor the fresh-decoder control has been executed.

## 2026-10-09: Recovery of the original Experiment 001 adversarial battery bytes

A historical source-integrity audit established that the later v0.4 preservation branches contain the same 20 adversarial *records* even though their raw JSON file hash did not match the original Phenotype Battery v1 manifest. The records were reserialized using an independently checked ordinary JSON representation (version, split and items fields; two-space indentation, UTF-8 with non-ASCII preserved, one final LF). The result exactly reproduces the **precommitted original v1 SHA-256** `e1510589bbf468d4c4d94a9587ac5dba9cd3284370efe9dae6779d043ccbc5cf`. The hash-matching file has been restored to `data/phenotype_battery/pretorius_adversarial_v1.json`, with CI checking its integrity. The earlier reconstruction status reporting this file unavailable is now superseded. The 100 axis-grounded training, 30 legacy training and 40 validation items still lack authenticated original v1 source files. **No Phase 1 seed was trained, no Phase 2 fresh decoder was trained, the four-condition reproduction criterion has not been evaluated, and the terminal battery remains sealed.** See `research/CHIMERA_001_SOURCE_RECOVERY.md` and `results/chimera_001/STATUS.json`.

## 2026-10-09: Original v1 validation battery recovered; Experiment 001 remains input-blocked

The four-file public v0.4 payload was created by distributing successive source rows across four stride partitions. Reversing that transformation yields the original 40 validation rows in their domain-grouped order. Reserializing the unchanged rows in the v1 JSON format produces the **exact precommitted original v1 validation SHA-256**, `54aeab2e4ab0bde7f9a4c1acdaad062731657f2a664b5e3cdf5409636b5d1246`. The canonical `data/phenotype_battery/pretorius_validation_v1.json` is restored, independently verified and regression-tested. The 20 adversarial rows were already recovered with their original v1 SHA-256. The only unresolved canonical input sources now are the 100 axis-grounded train rows' original v1 file representation and the missing 30 legacy train rows. No six-seed numerical reproduction, fresh-decoder training or terminal evaluation was performed. This supersedes the earlier status of validation being unavailable; it does not supersede the original Experiment 001 results or upgrade journal CC-E01 to 'reproduced'.

## 2026-10-09: Experiment 001B reusable 100-item phenotype pilot, six-seed result

**Distinct experiment:** `chimera-001b-100row-exploratory-pilot-v1`, not historical Experiment 001. Source train is the preserved 100-item v0.4 four-way stride-split phenotype battery pinned to commit `7b7eb19ad852dc016ee0d370528d903bab78a5cf`; 30 original legacy train items remain missing. The original SHA-256-authenticated 40 validation and 20 adversarial sets were used only as previously exposed development diagnostics. Six seeds `1842, 101, 202, 303, 404, 505` each received 12,000 main neural training steps, 12,000 steps per matched freshly supervised decoder (trained recurrent and virgin recurrent), and a 12,000-step shuffled-label control. Detailed per-item and per-seed JSON is committed under `results/chimera_001b/`; CI run `38025333272` executed and source-verified all seeds. Its concurrent branch commit race was corrected by replay-free archival CI run `38025494045`.

**Outcomes, validation JS similarity, arithmetic mean of six seeds:** intact 0.854292; virgin recurrent + trained decoder 0.853628; trained recurrent + virgin decoder 0.777393; virgin + virgin 0.777380; trained recurrent + freshly fit decoder 0.853170; virgin recurrent + equally fitted fresh decoder 0.853672; shuffled-target trained model 0.849656. No-network training-label prior **0.858388**; uniform predictor 0.778291. Paired fresh decoder (trained minus virgin recurrence) mean **-0.000502**, all six seeds negative. Adversarial intact 0.841098, no-network prior 0.841921. Direct named-action-population readout without motor decoder remains near uniform on validation.

**Decision:** no demonstrated positive matched-fresh-decoder effect attributable to this recurrent training in the current method; motor supervision and class-frequency priors are major competing explanations. Small negative difference does not prove missing recurrence information. No biography or Hugging Face data entered the run. No fresh confirmatory holdout exists; no parameter tuning is authorized against this exposed evaluation. See `results/chimera_001b/RESULTS.md`, `results/chimera_001b/RUN_REPORT.json`, `research/CHIMERA_001B_REUSED_DATA_PROTOCOL.md`. No terminal battery was opened; historical Experiment 001 remains **not reproduced**.

## 2026-10-10: Experiment 001D, controlled W/homeostatic and motor-decoder factorial

Completed all six paired seeds at 1,024 neurons and 12,000 training steps on the *separate* 100-row v0.4 exploratory training source, with original SHA-256-verified previously exposed validation/adversarial diagnostics. This is **001D, not Experiment 001 reproduction**, and does not use the sealed terminal or the 450-memory autobiography. Raw machine-readable outputs, source hashes, per-item metrics, tests and interpretation are at `results/chimera_001d/`, executed on [CI run 38028214002](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/38028214002).

**Six-seed mean validation JS similarity, recurrent W × homeostatic bias, trained motor decoder held fixed:** virgin W/virgin bias 0.853628; trained W/virgin bias **0.854639**; virgin W/trained bias 0.853220; trained W/trained bias 0.854292. Paired W-only effect **+0.001011**, positive for all six seeds. Homeostatic bias-only effect **-0.000408**. This identifies a small causal W contribution with an unchanged already-trained decoder, while the separate 001B matched freshly refitted decoder contrast remained negative (-0.000502): a reusable recurrent code is not established.

**Motor matrix versus motor bias, mature recurrent W+bias fixed:** virgin motor W/virgin motor bias 0.777393; trained motor W/virgin motor bias 0.848820; virgin motor W/trained motor bias 0.795619; trained/trained 0.854292. Motor-weight-only minus motor-bias-only contrast **+0.053201**. Context reassignment while preserving ground-truth labels reduced intact validation similarity by +0.005590 for global shuffle and +0.004559 for within-domain pair swaps, demonstrating some context sensitivity. Nonetheless the no-neural training-target prior scored **0.858388**, outperforming intact network 0.854292 on exposed validation.

**Conclusion:** do not describe the network as wholly context-insensitive, do not misattribute decoder weights to trivial motor biases, and do not mistake a tiny W-only effect for validated identity continuity. No fresh generalization claim is warranted. Develop independent new labeled counterfactual situations and isolate graph entanglement before future phenotype claims. See `results/chimera_001d/RESULTS.md` and source-pinned `config/chimera_001d.json`. Historical Experiment 001 remains not reproduced.

## 2026-10-10: 001F first actual Pretorius autobiographical neural exposure

**Distinct research protocol** `chimera-001f-autobiographical-unsupervised-replay-v1`. Completed and CI verified [six-seed Actions run 38058087497](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/38058087497). The immutable 450-event `Pretorius-Connectome` v12 reconstructed first-person archive (commit `5364f43dfe3c192b6c13b7bf373405ee7a41b420`, SHA-256 `becdf4ca72b85365c35940ccd68f3b6a9bbb1093b7b3cc89c33b113bbe5608ec`) was supplied as **unlabeled narrative input** through the unchanged signed lexical `ExperienceEncoder`, with reward=0, no action teaching, and motor decoder byte-frozen after 12,000 original 100-row phenotype foundation steps. Six seeds; exactly 26 passes of 450 narratives, 11,700 unsupervised recurrent steps in each exposure arm. Controls include no replay, plasticity-disabled chronological replay, plastic chronological, plastic reversed, and plastic epoch-shuffled, all sharing the exact same event exposure budget where replay occurred.

**Results (previously exposed original 40-item validation JS similarity):** no replay **0.854291551**; frozen chronology **0.854291551** (identical source weight hashes and exact scores per seed); plastic chronology **0.854055558**; plastic reverse **0.854055529**; epoch-shuffled plastic **0.854055499**. Paired chronology-minus-frozen **−0.000235993** (all seeds negative); chronology-minus-reverse **+2.85e−8**; chronology-minus-shuffled **+5.94e−8**. Old adversarial similarity also fell slightly from 0.841098451 (frozen) to 0.840922775 (plastic chronology). Although plastic W and homeostatic bias changed in every replay (mean L2 **0.140921** and **0.010838**), there is no demonstrated order-sensitive autobiographical functional improvement on this old phenotype metric. Generic Hebbian textual exposure alone is **not yet an operational episodic-memory mechanism**, and scoring old exposed behavior does not test actual recall or continuity.

Raw per-item results, paired seeds and exact SHA are preserved under `results/chimera_001f/` with interpretation in `results/chimera_001f/RESULTS.md`. Do not describe this as Experiment 001 historical reproduction or a failed persona, or silently tune these already exposed diagnostics. The 001E twelve new unreviewed dilemmas were not exposed to the neural model, all remain without approved labels, and the sealed terminal was not touched.

## 2026-10-10: 001G Recall Chamber, source-specific neural discrimination vs external index

**Completed six-seed result, source and mechanics CI green** at [run 38060772226](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/38060772226). Distinct protocol `chimera-001g-cue-source-neural-discrimination-v1`, not historical Experiment 001. Source-pinned 450 reconstructed Pretorius-Connectome v12 memories and editor-authored first-two `recall_cues` per event were evaluated as a **transductive cue-to-record correspondence benchmark**, not independent semantic truth. The six previous seeds, 1,024 neurons, 12,000 phenotype steps and 11,700 zero-reward unsupervised replay steps per frozen/plastic comparison were preserved. The existing BioCircuit BC01 source importer and candidate retriever, 001F encoder/plasticity functions, exact W and homeostatic bias lesions, source-level wrong-label permutations, repeated neutral key and non-neural sensory cosine baseline were reused rather than a new memory database.

**Results, 450 events × six seeds:** exact source top-1 frozen replay **5.4815%**; plastic replay **5.4815%**; W-only lesion **5.4444%**; bias-only lesion **5.4815%**; raw lexical-sensory cosine **10.00%**. All six paired plastic-minus-frozen top-1 effects were **zero**. Hard same-episode exact source top-1 was **28.26%** for all four neural conditions, vs **41.33%** raw lexical. Neutral-key and within-episode wrong-assignment top-1 both **0.2222%**, or 1/450. External BC01 lexical candidate retrieval returned the editor-associated source **447/450 (99.33%)** with all verdicts semantically `unknown`, but BC01 searches the very `recall_cues` metadata used to generate the questions; that is essentially a source-metadata roundtrip, **not a fair head-to-head semantic or neural-recall win**.

**Scientific decision:** source-specific neural advantage gate fails. Learned recurrent W from the 450-event replay does not yield a useful observed improvement under this cue-to-record rate readout; unmodified lexical features discriminate better. The small W-lesion difference corresponds to only one extra correct assignment across 2,700 trials. This does not disprove hidden recurrent information under all possible readouts and is not an autonomous memory probe, since all 450 candidate source representations are supplied during test. Further repeated Hebbian exposure is deprioritized in favor of architecturally explicit source retrieval, provenance ownership, first-person subject-access gating and genuinely new independently adjudicated examples. Raw detailed `results/chimera_001g/RUN_REPORT.json`, six `seed_*.json` and [interpretation](results/chimera_001g/RESULTS.md) are preserved. Source-original Experiment 001 remains incomplete; no terminal battery opened; 001E reviewer packets remain answer-free.
