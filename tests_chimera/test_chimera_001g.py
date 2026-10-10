"""001G neural cued source-association mechanics, leakage and controls."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from scripts.run_chimera_001g import (
    candidate_cues,deranged_within_episode,cosine_table,rank_assay,activation_states
)
from tests_chimera.test_chimera_001 import small_cfg


def events(n:int=8)->list[dict]:
    return [{
        "event_id":f"E01-{i+1:03d}",
        "episode_id":"E01" if i<n//2 else "E02",
        "recall_cues":[f"unique object {i}",f"unique place {i}","memory"],
        "provenance":"reconstructed",
    } for i in range(n)]


class SourceSelectivity001GTests(unittest.TestCase):
    def test_target_labels_not_leaked_into_cues(self):
        source=events()
        cues=candidate_cues(source,2)
        self.assertEqual(len(cues),len(source))
        self.assertFalse(any("E01-00" in q for q in cues))
        bad=copy.deepcopy(source)
        bad[0]["recall_cues"][0]="E01-001 secret archive"
        with self.assertRaisesRegex(ValueError,"Event ID leakage"):
            candidate_cues(bad,2)
        bad=copy.deepcopy(source)
        bad[0]["recall_cues"]=[]
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            candidate_cues(bad,2)

    def test_deranged_same_episode_permutation_and_repeatability(self):
        source=events()
        a=deranged_within_episode(source,1842)
        self.assertEqual(a,deranged_within_episode(source,1842))
        self.assertEqual(sorted(a),list(range(len(source))))
        self.assertTrue(all(a[i]!=i for i in range(len(source))))
        self.assertTrue(all(source[a[i]]["episode_id"]==source[i]["episode_id"]
                            for i in range(len(source))))

    def test_rank_baselines_neutral_key_and_oracle(self):
        source=events()
        identity=np.eye(8,dtype=np.float64)
        good=rank_assay(identity,source)
        self.assertEqual(good["top1"],1)
        self.assertEqual(good["mean_reciprocal_rank"],1)
        self.assertEqual(good["same_episode_top1"],1)
        self.assertEqual(rank_assay(identity,source,deranged_within_episode(source,42),
                                    retain_per_case=False)["top1"],0)
        neutral=rank_assay(np.zeros((8,8),dtype=np.float64),source)
        self.assertEqual(neutral["top1"],1/8)
        self.assertEqual(neutral["same_episode_top1"],2/8)
        with self.assertRaisesRegex(ValueError,"permutation"):
            rank_assay(identity,source,[0]*8)

    def test_neural_activity_projection_has_no_supervised_decoder_dependency(self):
        cfg=small_cfg()["network"]
        enc=ExperienceEncoder(cfg["sensory_dim"])
        net=PlasticRecurrentPersonaNet(cfg,enc)
        prompts=[
            enc.encode(s,action=None,outcome=0).vector
            for s in ("cold room apparatus","wet coil and lamp","yellow notebook")
        ]
        inp=np.stack(prompts)
        a=activation_states(net,inp,blank_ticks=2,stimulus_ticks=4)
        b=activation_states(copy.deepcopy(net),inp,blank_ticks=2,stimulus_ticks=4)
        self.assertEqual(a.shape,(3,net.n))
        np.testing.assert_array_equal(a,b)
        matrix=cosine_table(a,a)
        self.assertEqual(matrix.shape,(3,3))
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(np.allclose(matrix,matrix.T,atol=1e-6))
        neutral=np.zeros_like(a)
        self.assertTrue(np.all(cosine_table(neutral,a)==0))

    def test_mismatched_embedding_shape_refused(self):
        with self.assertRaisesRegex(ValueError,"dimension mismatch"):
            cosine_table(np.zeros((3,4)),np.zeros((2,8)))


if __name__=="__main__":
    unittest.main()
