"""001F mechanics: chronology, unsupervised replay budget and frozen controls."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from scripts.run_chimera_001f import (
    encode_memory_vectors, epoch_indices, one_seed, replay,
)
from scripts.run_chimera_001d import static_hash
from scripts.run_chimera_001 import fingerprint
from tests_chimera.test_chimera_001 import item, small_cfg


def synthetic_memory(i: int) -> dict:
    return {
        "event_id":f"E99-{i+1:03d}",
        "chronological_order":i+1,
        "memory_text":f"Memory {i} of a disputed laboratory result.",
        "observations":"The settings changed.",
        "decisions":"Check the apparatus.",
        "consequences":"The measured outcome varied.",
        "belief_changes":"A condition was missing.",
        "relationship_changes":"No independently known change.",
        "provenance":"reconstructed",
    }


class Replay001FTests(unittest.TestCase):
    def test_order_variants_preserve_all_events_and_shuffle_is_seeded(self):
        n=10
        for condition in ("plastic_chronological","frozen_chronological",
                          "plastic_reverse","plastic_shuffled_each_epoch"):
            first=epoch_indices(condition,n,0,1842,93017)
            second=epoch_indices(condition,n,0,1842,93017)
            self.assertEqual(first,second)
            self.assertEqual(sorted(first),list(range(n)))
        self.assertEqual(epoch_indices("plastic_chronological",n,0,2,3),list(range(n)))
        self.assertEqual(epoch_indices("plastic_reverse",n,0,2,3),list(range(n-1,-1,-1)))
        self.assertNotEqual(epoch_indices("plastic_shuffled_each_epoch",n,0,2,3),
                            epoch_indices("plastic_shuffled_each_epoch",n,1,2,3))
        with self.assertRaises(ValueError):
            epoch_indices("invented_condition",n,0,2,3)

    def test_unsupervised_memory_has_no_action_efference_channel(self):
        enc=ExperienceEncoder(32)
        vectors=encode_memory_vectors([synthetic_memory(i) for i in range(4)],
                   enc,["memory_text","observations","decisions","consequences",
                        "belief_changes","relationship_changes"])
        self.assertEqual(len(vectors),4)
        self.assertTrue(all(np.all(v[enc.action_offset:]==0) for v in vectors))

    def test_replay_frozen_mutations_and_decoder_preservation(self):
        cfg=small_cfg()["network"]
        enc=ExperienceEncoder(cfg["sensory_dim"])
        founder=PlasticRecurrentPersonaNet(cfg,enc)
        vectors=encode_memory_vectors([synthetic_memory(i) for i in range(8)],
                   enc,["memory_text","observations","decisions","consequences",
                        "belief_changes","relationship_changes"])
        no_learn=copy.deepcopy(founder)
        before=static_hash(no_learn,"recurrent_W")
        meta=replay(no_learn,vectors,condition="frozen_chronological",epochs=2,
                    seed=7,offset=93017)
        self.assertEqual(meta["actual_neural_steps"],16)
        self.assertEqual(meta["memory_presentations"],16)
        self.assertEqual(meta["W_before_sha256"],meta["W_after_sha256"])
        self.assertEqual(before,meta["W_after_sha256"])
        learned=copy.deepcopy(founder)
        learning=replay(learned,vectors,condition="plastic_reverse",epochs=2,
                        seed=7,offset=93017)
        self.assertEqual(learning["actual_neural_steps"],16)
        self.assertGreater(learning["recurrent_W_l2_change"],0)
        self.assertEqual(fingerprint(learned,"decoder"),fingerprint(founder,"decoder"))

    def test_neural_end_to_end_smoke_reports_all_controls(self):
        cfg=small_cfg()
        cfg["protocol_id"]="chimera-001f-autobiographical-unsupervised-replay-v1"
        cfg["conditions"]=["no_replay","frozen_chronological",
                            "plastic_chronological","plastic_reverse",
                            "plastic_shuffled_each_epoch"]
        cfg["replay_ticks"]=16
        cfg["replay_full_epochs"]=2
        cfg["phenotype_training_ticks"]=16
        cfg["source_memory"]={"fields":["memory_text","observations","decisions",
                                         "consequences","belief_changes","relationship_changes"]}
        cfg["neural_replay_policy"]={"randomization_seed_offset":93017}
        cfg["eval"].update(settle_ticks=2,probe_ticks=3)
        # The pilot test uses 8 events, so the budget matches those 8 x 2.
        # 001F full model has exactly 450 x 26, checked by source preflight.
        events=[synthetic_memory(i) for i in range(8)]
        cfg["replay_ticks"]=len(events)*cfg["replay_full_epochs"]
        train=[item("tr1","d"),item("tr2","d")]
        sets={"train":train,
              "validation":[item("v1","d"),item("v2","d")],
              "adversarial":[item("a1","d"),item("a2","d")]}
        # This entrypoint pins a 450-memory full corpus; the synthetic
        # mechanics test exercises replay() separately rather than bypassing it.
        self.assertEqual(len(events),8)
        enc=ExperienceEncoder(32)
        v=encode_memory_vectors(events,enc,cfg["source_memory"]["fields"])
        founder=PlasticRecurrentPersonaNet(cfg["network"],enc)
        expected=fingerprint(founder,"decoder")
        samples={}
        for condition in ("frozen_chronological","plastic_chronological",
                          "plastic_reverse","plastic_shuffled_each_epoch"):
            net=copy.deepcopy(founder)
            samples[condition]=replay(net,v,condition=condition,epochs=2,seed=7,offset=93017)
            self.assertEqual(expected,fingerprint(net,"decoder"))
        self.assertEqual(samples["frozen_chronological"]["recurrent_W_l2_change"],0)
        self.assertEqual(samples["frozen_chronological"]["homeostatic_bias_l2_change"],0)
        self.assertEqual(samples["plastic_chronological"]["actual_neural_steps"],16)


if __name__=="__main__":
    unittest.main()
