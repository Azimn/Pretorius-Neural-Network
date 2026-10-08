# BioCircuit BC00: executable deliverable

**What ships now:** a runnable local/CI sparse neural associative circuit prototype, checkpoint persistence, matched competition control, five deterministic unit tests and full multi-seed JSON output. This is software users can run today, not another architecture-only proposal.

**Run with Python 3.11 or newer:**

```sh
python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2'
python -m unittest discover -s tests_biocircuit -v
python scripts/run_biocircuit_bc00.py --neurons 4096 --items 128 --epochs 4 --seeds 31,37,43 --output results/biocircuit/BC00.json
```

The runner prints neural action-prediction accuracy after plastic learning, and after removal of learned synaptic weights, and writes an auditable JSON report. To inspect the next iteration quickly: `python scripts/run_biocircuit_bc00.py --neurons 1024 --items 64 --epochs 3`. CPU only; no GPU, hosted API or external data download. The new files use the same NumPy/SciPy stack already required by `persona_net`; they do not fork the character engine.

**Bio-inspired kernel:** an explicitly sparse input projection, four compartment populations, and a local winner-take-all competition operation approximate the *functional ideas* of sparse Kenyon-cell coding and feedback inhibition. A learned mapping from activated neurons to four action populations is updated using an explicit outcome/reward gate. The local model and its global-competition comparator have exactly identical sensory projection matrices, total activation count, output parameter count, outcome signal, seeds and presentation budgets. Only the competition policy differs.

**Experimental task:** 128 synthetic events are each assigned one of four random outcomes, independently of the event's input. The network sees each event four times. No architecture should be expected to generalize random outcomes to unseen event identities. The meaningful metrics are retention of these learned associations when parts of their cues are replaced, and the change when learned output synapses are destroyed. Also report the other architecture's result even if it wins. The synthetic task provides a deliberately minimal causal demonstration and is **not** evidence of autobiographical semantic understanding.

**What this does NOT include:** the full 65,536-unit proposed circuit, true recurrent neural dynamics, real fly topology, biologically accurate dopamine neurons, structural rewiring, Pretorius's 450 stored events, human-reviewed semantic challenges, long-term live-agent cognition or evidence of persona continuity. Its neural state is a one-pass sparse expansion plus trained action pathways. The learned output weights are themselves an action readout, so a positive lesion test only establishes that those synapses affect predictions; it does **not** resolve the earlier research question about causally necessary *recurrent* identity storage. BC01 must add recurrence with recurrent-weight lesion and matched decoder controls before making that claim.

**Why BC00 first:** it verifies that the experimental execution, exact matched-wiring controls, sparse recruitment, local competition, plasticity, replay and synaptic lesions work end to end before spending resources on high neuron counts or copying FlyWire anatomical wiring. These are the smallest testable elements specific to the newly proposed organization, beyond the existing generic sparse recurrent network.

**BC01 gate:** integrate recurrent context with the existing `persona_net.network.PlasticRecurrentPersonaNet`/The Doctor Lives donor interfaces without breaking their checkpoints, test memory interference and reversal learning, then integrate frozen v12 autobiographical event features with a shared semantic encoder. A future confirmatory human-reviewed challenge must be frozen before selecting the encoder or topology; the previous Pilot 04/05 cases are already exposed.

**Source-of-truth:** the architecture specification is [BioCircuit RFC v0.1](BIOCIRCUIT_RFC_V0_1.md). The direct biological-topology research remains in `Pretorius-Connectome`, and the production Pretorius identity remains in `The-Doctor-Lives`.
