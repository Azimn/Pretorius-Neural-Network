# BioCircuit BC01: first autobiographical preview results

**Date:** 2026-10-08. **Research status:** exploratory implementation, **BC01 causal-behavior acceptance gate NOT met**. No independent outcome or semantic-verification benchmark has been completed.

**Code tested:** `feature/bc01-autobiographical-demo`, commit `3e68e2c7d00ca1f372af81f99014a69246899249`. **CI:** [run 37834764525](https://github.com/Azimn/Pretorius-Neural-Network/actions/runs/37834764525), completed successfully. A subsequent minor fix to generic-only reporting will require a separate green check before review/merge.

## Reproduction

```sh
python -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2"
python -m unittest discover -s tests_biocircuit -v
python scripts/demo_biocircuit_bc01.py --neurons 256 --seed 1842 --exposures 8
```

The CI workflow repeats this with the exact upstream `Pretorius-Connectome` Git commit `597fb23473a60eecf2e1b50f79c22bfbea816be5` and original 450-record source blob `718dcc2d5ba4feccdef1690d447edfcebaa9bfb5`. It uses the byte-exact bundled three-record fixture for the offline smoke. The full-corpus source is **not** being copied into or edited in this repository.

## Verified engineering outcomes

- Ten tests passed (BC00 five; BC01 five); the smoke and pinned full-source CI jobs passed.
- 450 reconstructed memories imported with strict Git-blob validation. All present original event IDs and provenance remain external to the neural feature input.
- Exact checkpoint/restart probabilities reproduced on all three scripted queries in both smoke and full modes.
- The negative spacecraft question previously returned an irrelevant full-corpus source candidate, caused by a single shared adjective. The retriever now requires stronger independent lexical overlap and returns **UNKNOWN with no candidate** for that question. This is a lexical abstention heuristic, not semantic contradiction handling.

## First measured neural outcome at 256 units, one seed (1842)

The script exposed each first-person narrative for eight steps, holding externally supplied reward and motor-decoder training at zero. It compared trained recurrence with replacement of **only** the recurrent weight delta; the motor decoder and adapted homeostatic bias remained untouched. It also compared an otherwise matched blank-input exposure control.

| Full 450-memory query | Recurrent weight lesion, maximum policy-score difference | Content vs blank exposure, maximum policy-score difference | Selected action changed after recurrent lesion | Exact restart |
| --- | ---: | ---: | --- | --- |
| Millstream map and glove | 0.000007043 | 0.000002226 | **No** | Yes |
| Beetle and specimen drawer | 0.000007698 | 0.000002227 | **No** | Yes |
| Impossible spacecraft query | 0.000006609 | 0.000002255 | **No** | Yes |

The underlying sparse recurrent matrix reports 7,714 changed stored weight entries after both three-event and 450-event exposure. This **count is not evidence of memory encoding**: weight decay and common update dynamics can change coefficients even without autobiographical content. The blank-exposure comparator helps constrain that interpretation, but both effects are still extremely small.

The 3-event smoke showed even smaller synaptic-lesion score effects: about `4.2e-8` to `6.0e-8`. No tested action flipped in either set. The full-condition measured content-vs-blank score differences were approximately `2.2e-6`.

**Interpretation:** source-backed retrieval and checkpoint persistence are real and reproducible. The neural state changes, but the available evidence does not show meaningful autobiography-dependent policy behavior. BC01 has NOT passed the declared synaptic-necessity behavioral gate. This result should be retained rather than characterized as successful persona imprinting.

## Methodological limitations and required next tests

These three demonstration queries were chosen during development, not as a blinded/independent holdout. The fixed hashed-lexical encoder provides neither verified paraphrase understanding nor entailment/refutation. The output policy is an untrained generic motor decoder with no independently justified Pretorius decision target, making both score changes and unchanged choices difficult to interpret. The generic recurrent control sees the raw hashed input, while compartment conditions receive a modified fixed bridge; feature statistics and compute are not yet completely matched. Blank exposure is not a scrambled-autobiography control, and a one-seed result cannot characterize robustness.

Next experimental design should define a blinded behavioral endpoint from source consequences and choices **without feeding labels or event IDs during inference**, then compare local/global compartments, generic recurrence, shuffled memories, blank exposure, no-plasticity, decoder-only and targeted recurrent-delta lesion under identical training and evaluation budgets. Challenge data must be cluster-disjoint from development-exposed Pilot 04/05/06. Do not update The Doctor Lives production state from this preview.
