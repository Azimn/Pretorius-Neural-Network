from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json, math
import numpy as np

from .encoding import ACTIONS, ExperienceEncoder, SCALAR_KEYS
from .network import PlasticRecurrentPersonaNet


def load_items(path: str | Path) -> List[Dict]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return payload.get('items', payload.get('training_items', []))


def run_items(net: PlasticRecurrentPersonaNet, encoder: ExperienceEncoder, items: List[Dict], settle_ticks: int = 40, probe_ticks: int = 60) -> List[Dict]:
    zero = np.zeros(encoder.input_dim, dtype=np.float32)
    results=[]
    for probe in items:
        net.reset_fast_state(noise=0.01)
        for _ in range(settle_ticks):
            net.step(zero, reward=0.0, learn=False)
        scalars={k:float(probe.get('scalars',{}).get(k,0.0)) for k in SCALAR_KEYS}
        x=encoder.encode(probe['scenario'], scalars=scalars, action=None, outcome=0.0).vector
        accum={a:0.0 for a in ACTIONS}
        for _ in range(probe_ticks):
            net.step(x,reward=0.0,learn=False)
            for a,v in net.action_scores().items(): accum[a]+=v
        avg={a:v/probe_ticks for a,v in accum.items()}
        ranked=sorted(avg.items(), key=lambda kv:kv[1], reverse=True)
        row={k:probe.get(k) for k in ('id','domain','split','variant','scenario','contrast_group') if k in probe}
        row.update({'action_probabilities':avg,'top_actions':ranked[:3]})
        if 'target_actions' in probe: row['target_actions']=probe['target_actions']
        if 'anti_actions' in probe: row['anti_actions']=probe['anti_actions']
        results.append(row)
    return results


def _norm(d: Dict[str,float]) -> Dict[str,float]:
    vals={a:max(float(d.get(a,0.0)),0.0) for a in ACTIONS}
    s=sum(vals.values()) or 1.0
    return {a:v/s for a,v in vals.items()}


def js_divergence(p: Dict[str,float], q: Dict[str,float]) -> float:
    p=_norm(p); q=_norm(q)
    m={a:(p[a]+q[a])/2 for a in ACTIONS}
    def kl(a,b):
        total=0.0
        for k in ACTIONS:
            if a[k] > 0: total += a[k]*math.log(a[k]/max(b[k],1e-12),2)
        return total
    return 0.5*kl(p,m)+0.5*kl(q,m)


def score_results(results: List[Dict], keys: Dict[str,Dict] | None = None) -> Dict:
    rows=[]
    by_domain={}
    for r in results:
        target=r.get('target_actions')
        anti=r.get('anti_actions',[])
        if target is None and keys is not None:
            key=keys.get(r['id'],{})
            target=key.get('target_actions')
            anti=key.get('anti_actions',[])
        if not target: continue
        target=_norm(target)
        pred=_norm(r['action_probabilities'])
        target_support=[a for a,v in target.items() if v >= 0.08]
        top1=max(pred,key=pred.get)
        target_top=max(target,key=target.get)
        top3=[a for a,_ in sorted(pred.items(),key=lambda kv:kv[1],reverse=True)[:3]]
        expected_mass=sum(pred[a] for a in target_support)
        coverage=len(set(top3)&set(target_support))/max(len(set(target_support)),1)
        anti_mass=sum(pred.get(a,0.0) for a in anti)
        js=js_divergence(pred,target)
        row={'id':r['id'],'domain':r.get('domain','unknown'),'js_divergence':js,'js_similarity':1.0-js,'target_mass':expected_mass,'top1_agreement':1.0 if top1==target_top else 0.0,'top3_coverage':coverage,'anti_mass':anti_mass}
        rows.append(row)
        by_domain.setdefault(row['domain'],[]).append(row)
    if not rows: return {'n':0}
    def avg(seq,k): return sum(x[k] for x in seq)/len(seq)
    summary={'n':len(rows)}
    for k in ('js_similarity','target_mass','top1_agreement','top3_coverage','anti_mass'):
        summary[k]=avg(rows,k)
    summary['domain_macro_js_similarity']=sum(avg(v,'js_similarity') for v in by_domain.values())/len(by_domain)
    summary['domains']={d:{'n':len(v),'js_similarity':avg(v,'js_similarity'),'target_mass':avg(v,'target_mass'),'top1_agreement':avg(v,'top1_agreement'),'top3_coverage':avg(v,'top3_coverage')} for d,v in sorted(by_domain.items())}
    summary['items']=rows
    return summary
