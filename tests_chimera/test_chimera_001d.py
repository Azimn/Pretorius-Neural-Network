"""001D mechanical tests: exact component isolation and context-label preservation."""
import copy
import unittest

import numpy as np

from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from persona_net.phenotype_training import PhenotypeCurriculum
from scripts.run_chimera_001 import graft, score
from scripts.run_chimera_001d import (
    REC_CONDS, MOTOR_CONDS, factorial_graft, measure_one,
    permute_context, static_hash, summarize,
)
from tests_chimera.test_chimera_001 import item, small_cfg


class Chimera001DMechanics(unittest.TestCase):
    def setUp(self):
        self.config = small_cfg()
        self.config["protocol_id"] = "chimera-001d-isolated-synapse-bias-decoder-components-v1"
        self.config["phenotype_training_ticks"] = 16
        self.config["eval"].update(settle_ticks=2, probe_ticks=3)
        self.config["evaluation_roles"] = {"validation":"exposed","adversarial":"exposed"}
        self.encoder = ExperienceEncoder(self.config["network"]["sensory_dim"])
        self.virgin = PlasticRecurrentPersonaNet(self.config["network"], self.encoder)
        self.trained = copy.deepcopy(self.virgin)
        self.train = [
            dict(item("domain_train_%02d" % i, "domain"),
                 target_actions={"cooperate":0.2+0.06*(i%10),
                                 "challenge":0.8-0.06*(i%10)})
            for i in range(10)
        ]
        PhenotypeCurriculum(self.train,self.encoder).run(self.trained,16,progress_every=0)

    def test_exact_four_component_graft_and_source_not_aliased(self):
        self.assertEqual(len(REC_CONDS),4)
        self.assertEqual(len(MOTOR_CONDS),4)
        for w in (False,True):
            for bias in (False,True):
                for mw in (False,True):
                    for mb in (False,True):
                        out=factorial_graft(self.virgin,self.trained,
                             trained_W=w,trained_bias=bias,
                             trained_motor_w=mw,trained_motor_b=mb)
                        for name,flag in [
                            ("recurrent_W",w),("homeostatic_bias",bias),
                            ("motor_w",mw),("motor_b",mb)]:
                            source=self.trained if flag else self.virgin
                            self.assertEqual(static_hash(out,name),static_hash(source,name))
                        out.W.data[0]+=0.5
                        out.bias[0]+=0.5
                        out.motor_w[0,0]+=0.5
                        out.motor_b[0]+=0.5
                        self.assertFalse(np.array_equal(out.W.data,self.trained.W.data))
                        self.assertFalse(np.array_equal(out.bias,self.trained.bias))

    def test_unchanged_legacy_intact_graft_score(self):
        d=copy.deepcopy(self.virgin)
        a=factorial_graft(self.virgin,self.trained,trained_W=True,trained_bias=True,
                           trained_motor_w=True,trained_motor_b=True)
        b=graft(self.virgin,self.trained,"trained","trained")
        probes=[item("v1","domain"),item("v2","other")]
        self.assertEqual(score(a,self.encoder,probes,self.config,7,0),
                         score(b,self.encoder,probes,self.config,7,0))

    def test_counterfactual_changes_context_but_not_labels(self):
        rows=[dict(item(f"v{i}","d"+str(i//2)),scalars={"social":i/20},
                   scenario=f"Scenario with token {i} and independent wording.")
              for i in range(10)]
        for within in (True,False):
            changed=permute_context(rows,1842,within_domain=within)
            self.assertEqual(changed,permute_context(rows,1842,within_domain=within))
            self.assertEqual([x["id"] for x in changed],[x["id"] for x in rows])
            self.assertEqual([x["target_actions"] for x in changed],
                             [x["target_actions"] for x in rows])
            self.assertTrue(any(x["scenario"] != y["scenario"]
                                for x,y in zip(changed,rows)))
            if within:
                for i in range(0,len(rows),2):
                    self.assertEqual(changed[i]["scenario"],rows[i+1]["scenario"])
                    self.assertEqual(changed[i+1]["scenario"],rows[i]["scenario"])

    def test_full_factorial_smoke_all_measures_and_no_terminal(self):
        rows={"train":self.train,
              "validation":[item("v1","d"),item("v2","d")],
              "adversarial":[item("a1","d"),item("a2","d")]}
        result=measure_one(7,self.config,rows,{"source":"synthetic"})
        self.assertEqual(len(result["conditions"]),8)
        self.assertEqual(result["neural_training_steps"],16)
        self.assertTrue(all(result["conditions"][name]["validation"]["n"]==2
                            for name in result["conditions"]))
        self.assertTrue(all(k in result["paired_deltas"]["validation"] for k in (
            "W_only_vs_virgin_W","bias_only_vs_virgin_bias",
            "intact_minus_context_global_shuffled")))
        self.assertFalse(result["terminal_battery_touched"])
        self.assertFalse(result["historical_001_reproduced"])
        data=summarize([result],self.config,{"source":"synthetic"})
        self.assertEqual(data["seed_count"],1)


if __name__=="__main__":
    unittest.main()
