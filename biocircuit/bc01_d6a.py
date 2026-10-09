"""BC01-D6A: honest transductive, label-card holdout for the existing decoder.

Four stratified folds: three development cards per action used for supervised
training; fourth card for each action is not used to train either motor decoder
or recurrent learning. The 450-story background and source L2 v2 may STILL
contain the held-out events; DO NOT describe this as inductive or independent.
Paraphrases are candidate assistant-authored probes, not third-party reviewed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from biocircuit.bc01 import Corpus, EVENT_ID, make_circuit, neural_config, represent, words
from biocircuit.bc01_decisions import EVAL_ACTIONS, motor_decoder_scores, validate_cards
from biocircuit.bc01_d4 import _curriculum, _scores, _train_targeted
from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet

PARAPHRASES = Path(__file__).resolve().parents[1] / "resources/biocircuit/bc01_d6a_paraphrase_candidates_v1.json"
GAIN = 12.0
FOLDS = 4


def load_candidate_challenge(cards, path=PARAPHRASES):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (payload.get("schema") != "BC01-D6A-candidate-paraphrases-v1" or
            payload.get("source_git_blob") != "718dcc2d5ba4feccdef1690d447edfcebaa9bfb5" or
            "NOT independently reviewed" not in payload.get("status", "")):
        raise ValueError("Unregistered or misleading D6A candidate challenge")
    texts = payload["paraphrases"]
    if set(texts) != {c["event_id"] for c in cards}:
        raise ValueError("D6A candidate events must exactly match frozen 16 development cards")
    for card in cards:
        query = texts[card["event_id"]]
        if (not query or query == card["probe"] or
                EVENT_ID.search(query) or
                any(action in words(query) for action in EVAL_ACTIONS)):
            raise ValueError("D6A candidate paraphrase leaks direct source identifiers/actions")
    unknown = payload["unknown_controls"]
    if len(unknown) != 4 or len(set(unknown)) != 4 or any(
        EVENT_ID.search(text) or any(a in words(text) for a in EVAL_ACTIONS)
        for text in unknown
    ):
        raise ValueError("Invalid fixed non-source unknown controls")
    return payload


def strata(cards):
    groups = {action: [card for card in cards if card["action"] == action]
              for action in EVAL_ACTIONS}
    if any(len(group) != FOLDS for group in groups.values()):
        raise ValueError("D6A requires exactly 4 source decision cards per label")
    folds = []
    for fold in range(FOLDS):
        held = tuple(groups[action][fold] for action in EVAL_ACTIONS)
        held_ids = {card["event_id"] for card in held}
        training = tuple(card for card in cards if card["event_id"] not in held_ids)
        if len(training) != 12 or len(held) != 4 or len(held_ids) != 4:
            raise AssertionError("D6A train-test label leakage")
        folds.append((training,held))
    if {c["event_id"] for _, held in folds for c in held} != {
        c["event_id"] for c in cards
    }:
        raise AssertionError("D6A must test each source card exactly once per seed/mode")
    return tuple(folds)


def stimulus(text, encoder, circuit, shared, gain=GAIN):
    vector=represent(text,encoder,circuit,shared=shared)
    vector[:encoder.sensory_dim] *= gain
    if np.count_nonzero(vector[encoder.action_offset:]):
        raise AssertionError("D6A neural probe leaked action-target channel")
    return vector


def _motor_train(net, training, encoder, circuit, shared, labels, epochs, ticks):
    old_w=net.W.data.copy()
    for _ in range(epochs):
        for card,label in zip(training,labels):
            x=stimulus(card["training_cues"],encoder,circuit,shared)
            net.reset_fast_state()
            for _ in range(ticks):
                net.step(x, learn=False)
            net.learn_motor(label)
    if not np.array_equal(old_w,net.W.data):
        raise AssertionError("D6A decoder-only secretly trained recurrent weights")


def _decision(model,vector,ticks,use_motor=False):
    if use_motor:
        probs=motor_decoder_scores(model,vector,ticks)
    else:
        probs,_,_=_scores(model,vector,ticks)
    choice=max(EVAL_ACTIONS,key=lambda action: probs[action])
    return {"choice":choice,"scores":probs,"confidence":float(probs[choice])}


def _retrieval(text,training):
    query=words(text)
    matches=sorted(((len(query & words(c["training_cues"])),c["event_id"],c["action"])
                    for c in training), key=lambda row:(-row[0],row[1]))
    if not matches or matches[0][0]==0:
        return {"choice":None,"abstained":True,"lexical_overlap":0}
    return {"choice":matches[0][2],"abstained":False,
            "lexical_overlap":matches[0][0]}


def _metrics(rows,variant,condition):
    selected=[r for r in rows if r["variant"]==variant]
    if not selected:
        raise ValueError("No cases for evaluation")
    counts={a:{b:0 for b in EVAL_ACTIONS} for a in EVAL_ACTIONS}
    correct=0
    predicted=0
    abstained=0
    for r in selected:
        result=r["conditions"][condition]
        p=result["choice"]
        if p is None:
            abstained+=1
            continue
        predicted+=1
        counts[r["target"]][p]+=1
        correct+=int(p==r["target"])
    f1s=[]
    for action in EVAL_ACTIONS:
        tp=counts[action][action]
        false_pos=sum(counts[a][action] for a in EVAL_ACTIONS if a!=action)
        false_neg=sum(counts[action][a] for a in EVAL_ACTIONS if a!=action)
        denom=2*tp+false_pos+false_neg
        f1s.append((2*tp/denom) if denom else 0.0)
    return {
        "accuracy_with_abstain_counted_wrong":correct/len(selected),
        "accuracy_when_answering":correct/predicted if predicted else None,
        "macro_f1":float(np.mean(f1s)),
        "coverage":predicted/len(selected),
        "abstentions":abstained,
        "confusion":counts,
    }


def experiment(corpus: Corpus, cards: tuple[dict,...], *, seed=31,
               mode="generic", shared=None, sensory_gain=GAIN,
               epochs=3, card_ticks=32, background_ticks=8,
               settle_ticks=32, neurons=256, checkpoint_dir=None,
               challenge_path=PARAPHRASES):
    cards=validate_cards(corpus,cards)
    if shared is not None:
        shared.verify_corpus(corpus)
    if mode not in ("local","global","generic") or sensory_gain!=GAIN:
        raise ValueError("D6A prespecifies one high-gain input and three existing modes")
    if min(epochs,card_ticks,background_ticks,settle_ticks)<1:
        raise ValueError("D6A requires nonzero exposures")
    challenge=load_candidate_challenge(cards,challenge_path)
    encoder=ExperienceEncoder(sensory_dim=256)
    circuit=make_circuit(neurons,seed,mode)
    cfg=neural_config(neurons,seed)
    background=PlasticRecurrentPersonaNet(cfg,encoder)
    _curriculum(background,corpus,encoder,circuit,shared,background_ticks)
    old_w=background.W.data.copy()
    folds=[]
    for fold,(training,held) in enumerate(strata(cards)):
        train_ids={c["event_id"] for c in training}
        held_ids={c["event_id"] for c in held}
        assert train_ids.isdisjoint(held_ids)
        decoder=copy.deepcopy(background)
        labels=tuple(c["action"] for c in training)
        _motor_train(decoder,training,encoder,circuit,shared,labels,epochs,card_ticks)
        shuffled=copy.deepcopy(background)
        cycle={a:EVAL_ACTIONS[(i+1)%len(EVAL_ACTIONS)] for i,a in enumerate(EVAL_ACTIONS)}
        _motor_train(shuffled,training,encoder,circuit,shared,
                     tuple(cycle[a] for a in labels),epochs,card_ticks)
        recurrent=copy.deepcopy(decoder)
        assert np.array_equal(recurrent.W.data,old_w)
        _train_targeted(recurrent,training,encoder,circuit,shared,epochs,card_ticks,
                        labels,eta=.03,sensory_gain=sensory_gain,
                        presynaptic_mode="cue_minus_blank")
        if not (np.array_equal(recurrent.motor_w,decoder.motor_w) and
                np.array_equal(recurrent.motor_b,decoder.motor_b)):
            raise AssertionError("Recurrent learner modified frozen supervised output decoder")
        lesion=copy.deepcopy(recurrent)
        lesion.W.data[:]=old_w
        if checkpoint_dir is not None:
            path=Path(checkpoint_dir)/f"d6a_{mode}_{seed}_fold{fold}.npz"
            path.parent.mkdir(parents=True,exist_ok=True)
            decoder.save(path)
            restored=PlasticRecurrentPersonaNet.load(path,cfg,encoder)
        else:
            path=None
            restored=copy.deepcopy(decoder)
        rows=[]
        training_confidence=[]
        for c in training:
            v=stimulus(c["training_cues"],encoder,circuit,shared)
            training_confidence.append(_decision(decoder,v,settle_ticks,True)["confidence"])
        # Confidence threshold fitted on the TRAINING source cards ONLY,
        # not on the unseen episode/card or on unknown controls.
        threshold=float(np.quantile(training_confidence,0.10))
        for card in held:
            for variant,text in (
                ("original_source_probe",card["probe"]),
                ("assistant_paraphrase",challenge["paraphrases"][card["event_id"]]),
            ):
                v=stimulus(text,encoder,circuit,shared)
                motor=_decision(decoder,v,settle_ticks,True)
                conditions={
                    "decoder_frozen_recurrence":motor,
                    "decoder_shuffled_labels":_decision(shuffled,v,settle_ticks,True),
                    "decoder_with_cue_contrast_W":_decision(recurrent,v,settle_ticks,True),
                    "decoder_with_W_only_lesion":_decision(lesion,v,settle_ticks,True),
                    "fixed_population_readout":_decision(background,v,settle_ticks,False),
                    "train_card_lexical_retrieval":_retrieval(text,training),
                }
                conditions["decoder_training_confidence_abstain"]={
                    "choice":motor["choice"] if motor["confidence"]>=threshold else None,
                    "threshold_from_train_only":threshold,
                    "confidence":motor["confidence"],
                }
                if _decision(restored,v,settle_ticks,True)["scores"]!=motor["scores"]:
                    raise AssertionError("D6A source holdout checkpoint changed decoder score")
                rows.append({"event_id":card["event_id"],"episode_id":
                             corpus.by_id()[card["event_id"]]["episode_id"],
                             "target":card["action"],"variant":variant,
                             "query":text,"conditions":conditions})
        # Unknown examples remain label-free; report behavior only, not
        # correctness, as the classifier has no 'unknown' output action.
        unknown=[]
        for text in challenge["unknown_controls"]:
            vec=stimulus(text,encoder,circuit,shared)
            prediction=_decision(decoder,vec,settle_ticks,True)
            unknown.append({"query":text,"prediction":prediction,
                            "train_only_abstains":prediction["confidence"]<threshold,
                            "lexical_retrieval":_retrieval(text,training)})
        conditions=(
            "decoder_frozen_recurrence","decoder_shuffled_labels",
            "decoder_with_cue_contrast_W","decoder_with_W_only_lesion",
            "fixed_population_readout","train_card_lexical_retrieval",
            "decoder_training_confidence_abstain"
        )
        metrics={v:{cond:_metrics(rows,v,cond) for cond in conditions}
                 for v in ("original_source_probe","assistant_paraphrase")}
        folds.append({
            "fold":fold,"train_event_ids":[x["event_id"] for x in training],
            "heldout_event_ids":[x["event_id"] for x in held],
            "heldout_episodes":sorted({corpus.by_id()[x["event_id"]]["episode_id"]
                                       for x in held}),
            "heldout_episode_present_in_train_labels":any(
                corpus.by_id()[x["event_id"]]["episode_id"] in
                {corpus.by_id()[c["event_id"]]["episode_id"] for c in training}
                for x in held),
            "source_background_contains_all_heldout_events":True,
            "l2_encoder_train_fit_excludes_test_episodes":False,
            "train_only_decoder_confidence_threshold":threshold,
            "decoder_recurrent_W_preserved":bool(np.array_equal(decoder.W.data,old_w)),
            "shuffled_recurrent_W_preserved":bool(np.array_equal(shuffled.W.data,old_w)),
            "frozen_decoder_parameters_exact":bool(
                np.array_equal(decoder.motor_w,recurrent.motor_w) and
                np.array_equal(decoder.motor_b,recurrent.motor_b)),
            "trained_recurrent_delta_l1":float(np.sum(np.abs(recurrent.W.data-old_w))),
            "metrics":metrics,"heldout_rows":rows,"unknown_rows":unknown,
            "checkpoint_name":path.name if path else None,
        })
    return {
        "schema":"BC01-D6A-stratified-four-fold-transductive-v1",
        "source_git_blob":corpus.blob_sha,"source_records":len(corpus.records),
        "shared_l2_manifest_sha256":shared.cache_file_sha if shared else None,
        "seed":seed,"mode":mode,"sensory_gain":sensory_gain,
        "label_card_count":len(cards),"heldout_per_fold":4,
        "train_per_fold":12,"folds":folds,
        "evidence_scope":"unreviewed candidate paraphrases; labels held out but canonical autobiography & owner L2 fit were not held out; transductive developmental CV ONLY",
        "never_independent_or_sealed":True,
    }
