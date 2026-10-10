"""Small mechanical tests. No phenotype battery and no terminal access required."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from persona_net.phenotype_training import PhenotypeCurriculum
from scripts.run_chimera_001 import (
    CONDITIONS, FRESH_CONDITIONS, PreflightError, aggregate, fingerprint,
    graft, preflight, train_fresh_decoder,
)


def small_cfg():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "config/chimera_001.json").read_text())
    cfg["network"].update(neurons=64, sensory_dim=32, avg_recurrent_degree=4,
                          input_degree=3, action_population_size=4, plasticity_interval=2,
                          seed=7)
    cfg["phenotype_training_ticks"] = 16
    cfg["phase2"]["decoder_training_ticks"] = 16
    return cfg


def item(name, domain):
    return {
        "id": name,
        "domain": domain,
        "scenario": "A trusted stranger offers cooperation without control.",
        "scalars": {"social": 0.5, "autonomy": 0.3},
        "target_actions": {"cooperate": 0.65, "approach": 0.35},
    }


class ChimeraMechanics(unittest.TestCase):
    def setUp(self):
        self.cfg = small_cfg()
        self.enc = ExperienceEncoder(32)
        self.founder = PlasticRecurrentPersonaNet(self.cfg["network"], self.enc)
        self.mature = copy.deepcopy(self.founder)
        self.train = [item("train1", "a"), item("train2", "b")]
        PhenotypeCurriculum(self.train, self.enc).run(
            self.mature, total_ticks=16, progress_every=0
        )

    def test_exact_graft_and_no_shared_memory(self):
        self.assertEqual(len(CONDITIONS), 4)
        for name in CONDITIONS:
            r, d = name.split("_")
            net = graft(self.founder, self.mature, r, d)
            source_r = self.mature if r == "trained" else self.founder
            source_d = self.mature if d == "trained" else self.founder
            self.assertEqual(fingerprint(net, "recurrent"), fingerprint(source_r, "recurrent"))
            self.assertEqual(fingerprint(net, "decoder"), fingerprint(source_d, "decoder"))
            net.motor_w[0, 0] += 9
            self.assertFalse(np.array_equal(net.motor_w, source_d.motor_w))
            self.assertEqual(fingerprint(source_d, "decoder"),
                             fingerprint(self.mature if d == "trained" else self.founder, "decoder"))

    def test_fresh_decoder_fits_both_substrates_without_recurrence_change(self):
        self.assertEqual(len(FRESH_CONDITIONS), 2)
        for recurrent in ("trained", "virgin"):
            net = graft(self.founder, self.mature, recurrent, "virgin")
            before_r = fingerprint(net, "recurrent")
            before_d = fingerprint(net, "decoder")
            fit = train_fresh_decoder(net, self.enc, self.train, ticks=16)
            self.assertEqual(fit["actual_neural_steps"], 16)
            self.assertEqual(fit["presentations"], 8)
            self.assertEqual(fingerprint(net, "recurrent"), before_r)
            self.assertNotEqual(fingerprint(net, "decoder"), before_d)
            self.assertEqual(fingerprint(self.founder, "decoder"), before_d)

    def test_battery_hashes_counts_and_terminal_never_opened(self):
        cfg = small_cfg()
        cfg["battery"].update(dimensions=2, train=3, axis_train=2,
                              legacy_train=1, validation=4, adversarial=2)
        sets = {
            "axis_train": [item("ax1", "a"), item("ax2", "b")],
            "legacy_train": [item("lg1", "a")],
            "validation": [item("v1", "a"), item("v2", "a"),
                           item("v3", "b"), item("v4", "b")],
            "adversarial": [item("x1", "a"), item("x2", "b")],
            "profile": {"domains": [{"id": "a"}, {"id": "b"}]},
        }
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for key, value in sets.items():
                name = cfg["battery"]["files"][key]["name"]
                payload = value if key == "profile" else {"items": value}
                path = root / name
                path.write_text(json.dumps(payload))
                cfg["battery"]["files"][key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            # Neither path is part of preflight's inputs, even if present.
            (root / "pretorius_terminal_inputs_v1.json").write_text('{"secret":"unopened"}')
            rows, hashes = preflight(cfg, root)
            self.assertEqual(len(rows["train"]), 3)
            self.assertEqual(len(rows["validation"]), 4)
            self.assertEqual(len(rows["adversarial"]), 2)
            self.assertEqual(len(hashes), 5)
            (root / cfg["battery"]["files"]["legacy_train"]["name"]).write_text("{}")
            with self.assertRaisesRegex(PreflightError, "Hash mismatch"):
                preflight(cfg, root)

    def test_recovered_original_files_match_sha256_manifest(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "config/chimera_001.json").read_text())
        root = Path(__file__).resolve().parents[1] / "data/phenotype_battery"
        # The historical profile matches the original v1 manifest exactly.
        source = cfg["battery"]["files"]["profile"]
        data = (root / source["name"]).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), source["sha256"])
        # The audit recovered the adversarial source from content-equivalent
        # records, and CI proved the restored bytes match the original v1 SHA-256.
        adversarial = cfg["battery"]["files"]["adversarial"]
        recovered = root / adversarial["name"]
        self.assertTrue(recovered.is_file(), "Original SHA-256-authenticated adversarial source is required")
        self.assertEqual(hashlib.sha256(recovered.read_bytes()).hexdigest(), adversarial["sha256"])

    def test_criterion_is_four_condition_gate_not_seed_pool(self):
        base = {"phase1": {k: {"validation": {"js_similarity": v}}
                           for k, v in zip(CONDITIONS, [0.779, 0.854, 0.779, 0.855])}}
        targets = dict(zip(CONDITIONS, [0.7791, 0.8545, 0.7791, 0.8546]))
        report = aggregate([base] * 6, targets, 0.01)
        self.assertTrue(report["all_pass"])
        self.assertEqual(len(report["seed_distributions"]["trained_trained"]), 6)
        bad = copy.deepcopy(base)
        bad["phase1"]["trained_virgin"]["validation"]["js_similarity"] = 0.30
        report = aggregate([bad] * 6, targets, 0.01)
        self.assertFalse(report["all_pass"])
        self.assertFalse(report["per_condition_pass"]["trained_virgin"])


if __name__ == "__main__":
    unittest.main()
