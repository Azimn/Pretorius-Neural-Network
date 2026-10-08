"""D4 regression: sensory amplitude as sole intervention and D3 1x identity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from biocircuit.bc01 import FIXTURE_PATH, load_corpus, neural_config, represent
from biocircuit.bc01_decisions import CARD_PATH, EVAL_ACTIONS
from biocircuit.bc01_d3 import experiment as d3_experiment
from biocircuit.bc01_d4 import experiment as d4_experiment
from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet


class BC01D4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(FIXTURE_PATH)
        cards = json.loads(CARD_PATH.read_text(encoding="utf-8"))["cards"]
        ids = set(cls.corpus.by_id())
        cls.cards = tuple(c for c in cards if c["event_id"] in ids and
                          c["action"] in ("challenge", "create"))
        if len(cls.cards) != 2:
            raise AssertionError("Two source-anchored smoke decisions expected")

    def _run(self, sensory_gain, *, mode="generic", checkpoint_dir=None):
        return d4_experiment(
            self.corpus, self.cards, sensory_gain=sensory_gain,
            mode=mode, neurons=256, seed=31, epochs=1,
            background_ticks=2, card_ticks=8, settle_ticks=8,
            checkpoint_dir=checkpoint_dir
        )

    def test_one_x_reproduces_previous_d3_choice_scores_and_physiology(self):
        legacy = d3_experiment(
            self.corpus, self.cards, mode="generic", neurons=256,
            seed=31, epochs=1, background_ticks=2,
            card_ticks=8, settle_ticks=8
        )
        versioned = self._run(1.0)
        for field in ("rows", "result", "recurrent_delta_l1", "physiology",
                      "gates", "input_drive", "neural_cue_rms_spread"):
            self.assertEqual(legacy[field], versioned[field], field)
        self.assertEqual(versioned["background_sensory_gain"], 1.0)
        self.assertTrue(versioned["checkpoint_restart_exact"])

    def test_gain_affects_only_sensory_channels(self):
        encoder = ExperienceEncoder(sensory_dim=256)
        x = represent(self.cards[0]["probe"], encoder, None)
        four = x.copy()
        four[:encoder.sensory_dim] *= 4.0
        np.testing.assert_array_equal(
            four[encoder.sensory_dim:], x[encoder.sensory_dim:]
        )
        np.testing.assert_allclose(
            four[:encoder.sensory_dim], 4.0 * x[:encoder.sensory_dim], rtol=0, atol=0
        )
        self.assertTrue(np.all(four[encoder.action_offset:] == 0))
        trial_1 = self._run(1.0)
        trial_4 = self._run(4.0)
        self.assertAlmostEqual(
            trial_4["input_drive"]["sensory_to_teacher_ratio_mean"],
            trial_1["input_drive"]["sensory_to_teacher_ratio_mean"] * 4,
            places=6
        )
        self.assertEqual(trial_4["result"]["targeted"]["confusion"].keys(),
                         set(EVAL_ACTIONS))
        self.assertEqual(trial_4["annotated_cards"],2)

    def test_gate_explicitly_rejects_no_source_conditioning(self):
        for gain in (1., 4., 12.):
            t = self._run(gain)
            self.assertGreaterEqual(
                t["result"]["targeted"]["distinct_choices"], 1
            )
            self.assertEqual(t["sensory_gain"],gain)
            self.assertEqual(t["shared_feature_manifest_sha256"], None)
            self.assertTrue(t["checkpoint_restart_exact"])
            for row in t["rows"]:
                for condition in row["conditions"].values():
                    self.assertAlmostEqual(
                        sum(condition["scores"].values()),1.0,places=8
                    )
            if t["passes_all_development_gates"]:
                self.assertTrue(all(t["gates"].values()))

    def test_incompatible_gain_and_checkpoint_guard(self):
        for gain in (0, -1, float("nan"), float("inf"),13):
            with self.subTest(gain=gain),self.assertRaises(ValueError):
                self._run(gain)
        with tempfile.TemporaryDirectory() as folder:
            trial=self._run(12.,mode="local",checkpoint_dir=Path(folder))
            self.assertTrue(trial["checkpoint_restart_exact"])
            self.assertTrue(
                (Path(folder)/"bc01_d4_gain12_local_31.npz").is_file()
            )


if __name__=="__main__":
    unittest.main()
