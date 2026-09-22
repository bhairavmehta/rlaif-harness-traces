"""The RLAIF cycle: replay -> score -> credit-assign -> emit artifacts.

For a batch of synthesized episodes:
  1. Baseline replay (no levers)  ->  the 'rejected' candidates
  2. Single-lever ablations       ->  per-lever credit (this is why RL: the
     judge delta per lever on pinned episodes IS the reward/credit signal)
  3. Full replay (all levers)     ->  the 'chosen' candidates
  4. Emit preference pairs + failure-derived recommendation JSONs, all with
     MEASURED evidence and lift; nothing is asserted that wasn't replayed.
"""
from __future__ import annotations
import json, statistics as st
from collections import Counter
from .registry import Registry, LEVERS, PATCHES
from .runtime import Runtime
from .judge import score
from .llm import MockLLM
from .policies import CATALOG

# production RCA bucket -> synthetic policies whose replay exercises that failure
REPLAY_COVERAGE = {"raw_json_leakage": ["GEN-02", "GEN-08"], "generic_close": ["BEX-08"],
                   "intent_missed": ["BEX-02", "GEN-06"], "waiver_not_pursued": ["BEX-01", "BEX-04"],
                   "bill_copy_channel": ["BEX-07"], "scope_routing": ["GEN-05"], "containment": ["GEN-07"],
                   "wrong_tool_fidelity": ["GEN-09"], "unclassified": ["GEN-07"]}


def _avg(vs): return round(st.mean(vs), 3) if vs else 0.0


def replay(episodes, levers: set[str]):
    rt = Runtime(Registry(levers), MockLLM())
    out = []
    for ep in episodes:
        tr = rt.run(ep)
        out.append((ep, tr, score(ep, tr)))
    return out


def policy_fail_counts(run) -> dict:
    c = Counter(r["policy_id"] for _, _, v in run for r in v.policy_verdicts if r["verdict"] == "FAIL")
    return {pid: c[pid] for pid in CATALOG if pid in c}


def summarize(run):
    ws = [v.weighted for _, _, v in run]
    return {
        "avg_compliance_rate": _avg([v.compliance["overall_compliance_rate"] for _, _, v in run]),
        "action_completion_rate": round(sum(v.action_completion["action_completion"] for _, _, v in run) / len(run), 3),
        "action_categories": dict(Counter(v.action_completion["category"] for _, _, v in run)),
        "policy_fail_counts": policy_fail_counts(run),
        "avg_weighted": _avg(ws),
        "chosen_rate": round(sum(v.recommendation == "Chosen" for _, _, v in run) / len(run), 3),
        "resolved_rate": round(sum(v.outcome == "resolved" for _, _, v in run) / len(run), 3),
        "route_accuracy": round(sum(v.dims["intent_recognition"] >= 3 for _, _, v in run) / len(run), 3),
        "arg_error_calls": sum(v.reasoning["path_delta"]["argument_errors"] for _, _, v in run),
        "output_leaks": sum(v.reasoning["path_delta"]["output_leaks"] for _, _, v in run),
        "gate_violations": sum(bool(v.gate_violations) for _, _, v in run),
        "avg_dims": {k: _avg([v.dims[k] for _, _, v in run])
                     for k in run[0][2].dims},
    }


def _recommendations(baseline, full, credit):
    """Derive recommendation artifacts from MEASURED baseline failures."""
    n = len(baseline)
    missed = Counter(); arg_err = Counter(); leaks = 0; misroutes = Counter()
    for ep, tr, v in baseline:
        called_ok = {c["tool"] for c in tr.tool_calls if c["ok"]}
        for b in [b.split(":")[0] for b in ep.blueprint]:
            if b not in called_ok:
                missed[b] += 1
        for c in tr.tool_calls:
            if not c["ok"]:
                arg_err[c["tool"]] += 1
        leaks += tr.leaks
        if tr.route != ep.expected_route:
            misroutes[(ep.journey, tr.route)] += 1
    b_sum, f_sum = summarize(baseline), summarize(full)
    recs = []
    if misroutes:
        recs.append({
            "id": "R-TAX-01", "category": "taxonomy_routing", "lever": "taxonomy",
            "patch_ref": PATCHES["taxonomy"]["id"],
            "finding": {f"{j}->{r}": c for (j, r), c in misroutes.items()},
            "evidence_episodes": sum(misroutes.values()),
            "measured_lift": {"route_accuracy": f"{b_sum['route_accuracy']} -> {f_sum['route_accuracy']}",
                              "credit_weighted_delta": credit["taxonomy"]},
            "status": "pending_approval"})
    for tool, cnt in missed.most_common():
        recs.append({
            "id": f"R-META-{tool[:12]}", "category": "tool_metadata", "lever": "metadata",
            "classification": "missed" if arg_err.get(tool, 0) == 0 else "poorly_used",
            "tool": tool, "patch_ref": PATCHES["metadata"]["id"],
            "evidence": {"blueprint_episodes_not_served": cnt, "of_total": n,
                         "argument_errors": arg_err.get(tool, 0)},
            "measured_lift": {"credit_weighted_delta": credit["metadata"]},
            "status": "pending_approval"})
    if sum(arg_err.values()):
        recs.append({
            "id": "R-SEQ-01", "category": "sequencing_preconditions", "lever": "sequencing",
            "patch_ref": PATCHES["sequencing"]["id"],
            "evidence": {"failed_calls_baseline": sum(arg_err.values()),
                         "failed_calls_full": f_sum["arg_error_calls"]},
            "measured_lift": {"credit_weighted_delta": credit["sequencing"]},
            "status": "pending_approval"})
    if leaks:
        recs.append({
            "id": "R-FMT-01", "category": "output_contract", "lever": "formatter",
            "patch_ref": PATCHES["formatter"]["id"],
            "evidence": {"raw_json_leaks_baseline": leaks, "leaks_full": f_sum["output_leaks"]},
            "measured_lift": {"credit_weighted_delta": credit["formatter"]},
            "status": "pending_approval"})
    recs.append({
        "id": "R-PROMPT-01", "category": "instruction_layer", "lever": "prompt_hat",
        "patch_ref": PATCHES["prompt_hat"]["id"],
        "evidence": {"consent_gate_violations_baseline": b_sum["gate_violations"],
                     "dropped_intents_visible_in": "path_delta.dropped_intents"},
        "measured_lift": {"credit_weighted_delta": credit["prompt_hat"]},
        "status": "pending_approval"})
    return recs


def _attach_production(recs, b_sum, f_sum, production) -> list:
    """Production evidence per lever + policy-level lift; uncovered RCA buckets become coverage requests."""
    buckets = production["failure_buckets"]
    for r in recs:
        owned = [pid for pid, p in CATALOG.items() if p.lever == r["lever"]]
        r["policy_lift"] = {pid: f"{b_sum['policy_fail_counts'].get(pid, 0)} -> {f_sum['policy_fail_counts'].get(pid, 0)} fails"
                            for pid in owned if b_sum["policy_fail_counts"].get(pid)}
        mine = [b for b in buckets if b["lever"] == r["lever"]]
        r["production_evidence"] = {
            "fail_verdicts": sum(b["fail_verdicts"] for b in mine),
            "calls": sum(b["calls"] for b in mine),
            "buckets": [{k: b[k] for k in ("bucket", "fail_verdicts", "calls", "policies", "examples")} for b in mine],
        }
    for b in buckets:
        covered = [pid for pid in REPLAY_COVERAGE.get(b["bucket"], []) if b_sum["policy_fail_counts"].get(pid)]
        if covered:
            continue
        recs.append({
            "id": f"R-PROD-{b['bucket'][:18]}", "category": b["category"], "lever": b["lever"],
            "finding": f"{b['fail_verdicts']} production FAIL verdict(s) across {b['calls']} call(s): {b['policies']}",
            "production_evidence": {k: b[k] for k in ("fail_verdicts", "calls", "policies", "examples")},
            "measured_lift": None,
            "status": "needs_replay_coverage",
            "next_step": ("synthesize episodes that exercise this failure, then replay before approval"
                          if b["lever"] else "outside the five-lever action space - needs a new guard (e.g. grounding check)"),
        })
    return recs


def run_cycle(episodes, out_dir: str | None = None, production: dict | None = None,
              deployed: set | None = None) -> dict:
    """`deployed` = lever set the "full" side runs with (default: all levers)."""
    deployed = set(LEVERS) if deployed is None else set(deployed)
    baseline = replay(episodes, set())
    full = replay(episodes, deployed)
    b_sum = summarize(baseline)

    # per-lever ablation: baseline + one lever -> credit assignment
    credit = {}
    for lv in LEVERS:
        single = replay(episodes, {lv})
        credit[lv] = round(summarize(single)["avg_weighted"] - b_sum["avg_weighted"], 3)

    pairs = []
    for (ep, tb, vb), (_, tf, vf) in zip(baseline, full):
        if vf.weighted > vb.weighted:
            pairs.append({
                "pair_id": f"pp-{ep.id}", "episode": ep.id, "journey": ep.journey,
                "utterance": ep.utterance,
                "rejected": {"levers": vb.levers, "weighted": vb.weighted,
                             "verdict": vb.recommendation, "gates": vb.gate_violations,
                             "path_delta": vb.reasoning["path_delta"]},
                "chosen": {"levers": vf.levers, "weighted": vf.weighted,
                           "verdict": vf.recommendation,
                           "path_delta": vf.reasoning["path_delta"]},
                "rationale": "same pinned episode; only harness levers differ"})

    f_sum = summarize(full)
    recs = _recommendations(baseline, full, credit)
    if production:
        recs = _attach_production(recs, b_sum, f_sum, production)
    result = {
        "episodes": len(episodes),
        "deployed_levers": [lv for lv in LEVERS if lv in deployed],
        "baseline": b_sum, "full": f_sum,
        "credit_assignment_weighted_delta_per_lever": credit,
        "preference_pairs": len(pairs),
        "recommendations": recs,
    }
    if production:
        result["production"] = {k: production[k] for k in ("records", "journeys", "verdicts_parsed", "judge_health")}
    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
        json.dump(result, open(f"{out_dir}/cycle_summary.json", "w"), indent=2)
        json.dump(pairs, open(f"{out_dir}/preference_pairs.json", "w"), indent=2)
        json.dump(recs, open(f"{out_dir}/recommendations.json", "w"), indent=2)
        json.dump([v.to_json() for _, _, v in baseline],
                  open(f"{out_dir}/verdicts_baseline.json", "w"), indent=2)
        json.dump([v.to_json() for _, _, v in full],
                  open(f"{out_dir}/verdicts_full.json", "w"), indent=2)
        if production:
            json.dump(production, open(f"{out_dir}/trace_rca.json", "w"), indent=2)
    return result
