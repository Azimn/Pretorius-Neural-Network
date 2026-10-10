"""Experiment 001B mechanics, deterministic source ordering and frozen controls."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_chimera_001b import (
    EXPLORATORY_CONDITIONS, load_data, round_robin, shuffle_targets,
    source_priors, summarize, train_one_seed,
)
from tests_chimera.test_chimera_001 import item, small_cfg


class Chimera001BTests(unittest.TestCase):
    def test_stride_interleave_restores_original_order(self):
        original = [{"id": f"event-{i}"} for i in range(12)]
        partitions = [original[i::4] for i in range(4)]
        self.assertEqual(round_robin(partitions), original)
        self.assertNotEqual([x for p in partitions for x in p], original)

    def test_permutation_preserves_target_multiset_but_not_mapping(self):
        rows = [dict(item(f"e{i}", "x"), target_actions={"approach": i + 1})
                for i in range(20)]
        shuffled = shuffle_targets(rows, 1842)
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in shuffled])
        self.assertEqual(sorted(r["target_actions"]["approach"] for r in rows),
                         sorted(r["target_actions"]["approach"] for r in shuffled))
        self.assertNotEqual([r["target_actions"] for r in rows],
                            [r["target_actions"] for r in shuffled])
        self.assertEqual(shuffled, shuffle_targets(rows, 1842))

    def test_in_memory_ablation_produces_all_controls_without_terminal(self):
        cfg = small_cfg()
        cfg["protocol_id"] = "chimera-001b-100row-exploratory-pilot-v1"
        cfg["fresh_decoder_training_ticks"] = 16
        cfg["eval"].update(settle_ticks=2, probe_ticks=3)
        train = [
            dict(item(f"train{i}", "x"), target_actions={
                "cooperate": 0.8 if i % 2 else 0.2,
                "challenge": 0.2 if i % 2 else 0.8,
            })
            for i in range(10)
        ]
        val = [item("val1", "x"), item("val2", "y")]
        adv = [item("adv1", "x"), item("adv2", "y")]
        splits = {"train": train, "validation": val, "adversarial": adv}
        report = train_one_seed(7, cfg, splits, {"toy": "source"})
        self.assertEqual(set(report["condition_results"]), set(EXPLORATORY_CONDITIONS))
        self.assertEqual(report["main_neural_training_steps"], 16)
        self.assertTrue(report["trained_recurrent_differs_from_virgin"])
        for condition in EXPLORATORY_CONDITIONS:
            self.assertEqual(report["condition_results"][condition]["validation"]["n"], 2)
        for key in ("trained_recurrent_fresh_decoder", "virgin_recurrent_fresh_decoder"):
            fit = report["condition_results"][key]["fresh_decoder_fitting"]
            self.assertEqual(fit["actual_neural_steps"], 16)
            self.assertEqual(fit["recurrent_sha256_before_after"],
                             report["condition_results"][key]["recurrent_sha256"])
        for readout in ("virgin", "trained"):
            self.assertEqual(report["population_readout_without_motor_decoder"][readout]["validation"]["n"], 2)
        self.assertFalse(report["terminal_battery_touched"])
        self.assertFalse(report["historical_experiment_001_reproduced"])
        first = summarize([report], cfg, {"validation": source_priors(train, val)})
        self.assertEqual(len(first["paired_fresh_trained_minus_virgin_recurrent"]), 1)
        self.assertFalse(first["historical_experiment_001_reproduced"])

    def test_source_gate_checksum_and_duplicate_check(self):
        # Use a tiny mock 20-domain split to test that source sha256 must match.
        # No dependency on the inaccessible original 001 training battery.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "history/data/v0_4"
            src.mkdir(parents=True)
            current = root / "current/data/phenotype_battery"
            current.mkdir(parents=True)
            rows = []
            for i in range(20):
                for j in range(5):
                    rows.append(dict(item(f"{i}-{j}", f"d{i}")))
            val = [dict(item(f"v{i}-{j}", f"d{i}"))
                   for i in range(20) for j in range(2)]
            adv = [dict(item(f"a{i}", f"d{i}")) for i in range(20)]
            filenames = [f"train_part{i}.json" for i in range(1, 5)]
            cfg = {
                "train_source_dir": "data/v0_4",
                "train_source_files": filenames,
                "train_source_sha256": {},
                "train_rows": 100,
                "validation_file": "pretorius_validation_v1.json",
                "adversarial_file": "pretorius_adversarial_v1.json",
            }
            for i, name in enumerate(filenames):
                p = src / name
                p.write_text(json.dumps({"items": rows[i::4]}))
                cfg["train_source_sha256"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = {
                "train_files": filenames,
                "counts": {"train": 100, "validation": 40, "adversarial": 20},
            }
            manifest_path = src / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            cfg["train_source_sha256"]["manifest.json"] = hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()
            for split, values in (("validation", val), ("adversarial", adv)):
                p = current / cfg[split + "_file"]
                p.write_text(json.dumps({"items": values}))
                cfg[split + "_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            data, checksums = load_data(cfg, root / "history", root / "current")
            self.assertEqual([r["id"] for r in data["train"]], [r["id"] for r in rows])
            self.assertEqual(len(checksums), 7)
            p = src / filenames[0]
            p.write_text(p.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                load_data(cfg, root / "history", root / "current")


if __name__ == "__main__":
    unittest.main()
