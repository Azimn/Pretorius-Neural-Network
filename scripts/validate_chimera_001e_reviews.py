#!/usr/bin/env python3
"""001E blind-rating intake for NEW source-linked counterfactual situations.

Never infer ground-truth action probabilities from biography prose or scenario
metadata. No actual review data currently exist. The --preflight-only path
produces a machine-readable pending-status report and no scored data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from persona_net.encoding import ACTIONS
PINNED_SOURCE = "5364f43dfe3c192b6c13b7bf373405ee7a41b420"
DRAFTS = ROOT / "research/chimera_001e/SCENARIO_DRAFTS.jsonl"
SOURCE_QUEUE = ROOT / "results/chimera_001c/REVIEW_QUEUE.jsonl"
EXPOSED_VALIDATION = ROOT / "data/phenotype_battery/pretorius_validation_v1.json"
EXPOSED_ADVERSARIAL = ROOT / "data/phenotype_battery/pretorius_adversarial_v1.json"
KNOWN_ACTIONS = set(ACTIONS)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip()]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def check_targets(target: dict) -> None:
    if not isinstance(target, dict):
        raise ValueError("A ten-action distribution is required")
    if not set(target).issubset(KNOWN_ACTIONS) or not target:
        raise ValueError("Unknown action in independently assessed target")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0 or x > 1
           for x in target.values()):
        raise ValueError("Out-of-range target action probability")
    if abs(sum(target.values())-1.0)>1e-6:
        raise ValueError("Target probabilities must sum to one")


def check_drafts(drafts: list[dict], queue: list[dict], previously_seen: list[dict],
                 expected: int | None = 12) -> dict:
    sources = {x["source_event_id"]:x for x in queue}
    if len(sources)!=450 or len(queue)!=450:
        raise ValueError("Source inventory must contain all 450 unique events")
    if expected is not None and len(drafts)!=expected:
        raise ValueError(f"Expected {expected} source-linked scenario drafts")
    ids=[x.get("id") for x in drafts]
    if any(not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError("Duplicate or missing draft IDs")
    original_scenarios = {normalize(x["scenario"]) for x in previously_seen}
    seen_scenarios=set()
    for card in drafts:
        event=card.get("source_event_id")
        if event not in sources:
            raise ValueError(f"Unknown source event {event}")
        if card.get("source_commit")!=PINNED_SOURCE:
            raise ValueError("Source commit mismatch")
        if card.get("target_actions") is not None:
            raise ValueError("Draft cannot contain an inferred answer label")
        if card.get("evaluation_role")!="unassigned":
            raise ValueError("Cannot mark unaudited draft as a holdout")
        if card.get("review_status")!="unreviewed_draft":
            raise ValueError("Expected explicitly unreviewed scenario")
        if card.get("draft_source")!="assistant_authored_counterfactual_not_independent_validation":
            raise ValueError("Scenario authorship independence is not established")
        scenario=card.get("scenario","")
        if not isinstance(scenario,str) or len(scenario.strip())<80:
            raise ValueError("Scenario too short to be an interpretable situation")
        compact=normalize(scenario)
        if compact in original_scenarios:
            raise ValueError("Scenario exactly duplicates an exposed v1 evaluation item")
        if compact in seen_scenarios:
            raise ValueError("Duplicate draft scenario")
        seen_scenarios.add(compact)
    return {
        "draft_count":len(drafts),
        "distinct_source_events":len({x["source_event_id"] for x in drafts}),
        "draft_sha256": {x["id"]:digest(x["scenario"]) for x in drafts},
    }


def audit_reviews(drafts: list[dict], reviews: list[dict],
                  adjudications: list[dict]) -> tuple[list[dict],dict]:
    """Gate human/self-declared independent raters and explicit adjudication.

    Cryptographic hashes establish text agreement but do NOT establish that two
    purported ratings were genuinely independently authored. Both are required
    but neither is epistemic proof of behavioral truth.
    """
    candidates={x["id"]:x for x in drafts}
    by_id=defaultdict(list)
    for r in reviews:
        cid=r.get("candidate_id")
        if cid not in candidates:
            raise ValueError("Review targets missing draft")
        if r.get("scenario_sha256")!=digest(candidates[cid]["scenario"]):
            raise ValueError("Review is for changed counterfactual scenario")
        reviewer=r.get("reviewer_id")
        if not isinstance(reviewer,str) or len(reviewer.strip())<2:
            raise ValueError("Reviewer identity required")
        if not r.get("attests_independent_read") or r.get("saw_model_predictions") is not False:
            raise ValueError("Independent, model-blind reading must be explicitly attested")
        if not isinstance(r.get("review_note"),str) or len(r["review_note"].strip())<20:
            raise ValueError("Reviewer must provide a contextual evidence note")
        check_targets(r.get("target_actions"))
        if reviewer in {x["reviewer_id"] for x in by_id[cid]}:
            raise ValueError("Duplicate rating from the same reviewer on a card")
        by_id[cid].append(r)
    by_adjudication={}
    for a in adjudications:
        cid=a.get("candidate_id")
        if cid not in candidates or cid in by_adjudication:
            raise ValueError("Missing or duplicated adjudication card")
        if a.get("scenario_sha256")!=digest(candidates[cid]["scenario"]):
            raise ValueError("Adjudication was made against a changed scenario")
        if a.get("decision")!="approve":
            raise ValueError("This release workflow accepts explicitly approved cards only")
        if len(by_id[cid])<2:
            raise ValueError("Two independent reviewer ratings required first")
        if a.get("adjudicator_id") in {r["reviewer_id"] for r in by_id[cid]}:
            raise ValueError("Adjudicator must be separate from source label reviewers")
        if not isinstance(a.get("adjudicator_id"),str) or len(a["adjudicator_id"].strip())<2:
            raise ValueError("Adjudicator identity required")
        if not isinstance(a.get("adjudication_note"),str) or len(a["adjudication_note"].strip())<20:
            raise ValueError("Adjudication explanation required")
        check_targets(a.get("target_actions"))
        by_adjudication[cid]=a
    approved=[]
    for cid,card in candidates.items():
        if cid not in by_adjudication:
            continue
        approved.append({
            "id":cid,
            "source_event_id":card["source_event_id"],
            "scenario":card["scenario"],
            "scalars":{},
            "target_actions":by_adjudication[cid]["target_actions"],
            "reviewer_ids":[x["reviewer_id"] for x in by_id[cid]],
            "adjudicator_id":by_adjudication[cid]["adjudicator_id"],
            "scenario_sha256":digest(card["scenario"]),
            "evaluation_role":"candidate_pending_holdout_freeze",
            "independent_validity_claim":"reviewer_attestation_only",
        })
    state={
        "reviewed_count":len(by_id),
        "two_review_count":sum(len(v)>=2 for v in by_id.values()),
        "approved_count":len(approved),
        "unapproved_count":len(drafts)-len(approved),
        "no_model_score_created":True,
        "no_confirmatory_holdout_released":True,
        "terminal_battery_touched":False,
        "historical_001_reproduced":False,
    }
    return approved,state


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts",type=Path,default=DRAFTS)
    parser.add_argument("--source-queue",type=Path,default=SOURCE_QUEUE)
    parser.add_argument("--reviews",type=Path,help="Human reviews JSONL, optional for preflight only")
    parser.add_argument("--adjudications",type=Path,help="Explicit independent adjudications JSONL")
    parser.add_argument("--out",type=Path,default=ROOT/"results/chimera_001e")
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    drafts=load_rows(args.drafts)
    queue=load_rows(args.source_queue)
    exposed=[]
    for p in (EXPOSED_VALIDATION,EXPOSED_ADVERSARIAL):
        exposed.extend(json.loads(p.read_text(encoding="utf-8"))["items"])
    meta=check_drafts(drafts,queue,exposed)
    if args.preflight_only and (args.reviews or args.adjudications):
        raise ValueError("Preflight may not silently import model/user-supplied rating files")
    reviews=load_rows(args.reviews) if args.reviews else []
    adjudications=load_rows(args.adjudications) if args.adjudications else []
    approved,ratings=audit_reviews(drafts,reviews,adjudications)
    if not args.preflight_only and (not args.reviews or not args.adjudications or not approved):
        raise ValueError("No adjudicated data available. Rejecting scored/frozen data export.")
    args.out.mkdir(parents=True,exist_ok=True)
    status={
        "protocol_id":"chimera-001e-counterfactual-independent-label-gate-v1",
        "source_commit":PINNED_SOURCE,
        "draft_file_sha256":exact_hash(args.drafts),
        "source_queue_sha256":exact_hash(args.source_queue),
        "draft_audit":meta,
        "review_status":ratings,
        "status":"unreviewed_pending_adjudication" if args.preflight_only else "provisional_adjudication_not_confirmatory",
        "source_role":"new_situations_derived_from_existing_autobiography",
        "external_holdout_verified":False,
    }
    (args.out/"STATUS.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    if not args.preflight_only:
        (args.out/"PROVISIONALLY_APPROVED_ITEMS.jsonl").write_text(
            "".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in approved),
            encoding="utf-8",
        )
    print("001E status:",status["status"],"scenario drafts:",len(drafts),
          "review-approved:",len(approved),"model runs:",0)


if __name__=="__main__":
    main()
