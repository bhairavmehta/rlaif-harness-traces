"""Agent runtime: route -> (phased) tool selection -> execution -> response.

Every decision is asked of the LLM backend using metadata from the Registry,
so applying a patch changes behavior through exactly one path: the metadata.
The Backend simulates the billing systems (backend-is-truth). The Trace is the
audit artifact the judge scores.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .registry import Registry
from .llm import LLMBackend

MAX_STEPS = 6


@dataclass
class Step:
    kind: str                 # route | phase | select | call | result | error | respond | leak | reverify
    detail: dict


@dataclass
class Trace:
    episode_id: str
    levers: list[str]
    route: str = ""
    steps: list[Step] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    leaks: int = 0
    reverified: bool = False
    consent_given: bool = False
    terms_read: bool = False
    disclosures_made: bool = False
    intents_handled: list[str] = field(default_factory=list)
    outcome: str = "unresolved"      # resolved | handoff | unresolved

    def log(self, kind: str, **detail):
        self.steps.append(Step(kind, detail))


class Backend:
    """Simulated billing systems — the source of truth tools answer from."""

    def __init__(self, account: dict):
        self.acc = dict(account)

    def execute(self, tool: str, args: dict) -> tuple[bool, dict]:
        a = self.acc
        if tool == "get_account_balance":
            return True, {"balance": a["balance"]}
        if tool == "explain_bill":
            if args.get("detail_level") == "full":
                return True, {"line_items": a["line_items"]}
            return True, {"summary": {"total": a["balance"], "categories": ["equipment", "plan"]}}
        if tool == "submit_fee_waiver":
            cid = args.get("charge_id")
            if not cid:
                return False, {"error": "MISSING_REQUIRED", "param": "charge_id"}
            if cid in a["waiver_eligible"]:
                return True, {"status": "CONFIRMED", "ref": "W-77231"}
            return False, {"error": "NOT_ELIGIBLE"}
        if tool == "manage_bill_preferences":
            return True, {"sms_sent": True, "statusMessage": "SUCCESS"}
        if tool == "check_paper_free_eligibility":
            return True, {"eligible": a["pfb_eligible"], "enrolled": a["pfb_enrolled"]}
        if tool == "enroll_paper_free_billing":
            if not args.get("terms_accepted"):
                return False, {"error": "TERMS_NOT_ACCEPTED"}
            a["pfb_enrolled"] = True
            return True, {"status": "ENROLLED", "ref": "P-33417"}
        if tool == "check_ddc_eligibility":
            ok = (not a["pending_order"]) and a["days_since_date_change"] > 31
            return True, {"eligible": ok, "valid_range": [3, 25]}
        if tool == "change_due_date":
            nd = args.get("new_date")
            if nd is None or not (3 <= nd <= 25):
                return False, {"error": "INVALID_DATE"}
            a["due_date"] = nd
            return True, {"status": "CONFIRMED", "ref": "D-58204", "new_date": nd}
        if tool == "payment_arrangement":
            return True, {"status": "ARRANGED", "installments": 2}
        if tool == "transfer_to_agent":
            return True, {"result": {"status": "Transfer initiated", "hangup": True}}
        return False, {"error": "UNKNOWN_TOOL"}


class Runtime:
    def __init__(self, registry: Registry, llm: LLMBackend):
        self.R, self.llm = registry, llm

    # ------------------------------------------------------------------ #
    def route(self, utterance: str, trace: Trace) -> str:
        scores = self.llm.score_options(utterance, self.R.agent_options())
        best = max(scores, key=lambda k: (scores[k], k))
        if scores[best] == 0:
            best = "general_care"                       # fallback
        trace.route = best
        trace.log("route", scores=scores, target=best)
        return best

    def _emit(self, trace: Trace, tool: str, result: dict):
        """Response formatting: verbatim tool JSON (leak) or templated."""
        contract = self.R.tools[tool].output_contract
        if contract == "verbatim" and tool in ("transfer_to_agent", "manage_bill_preferences"):
            trace.leaks += 1
            trace.log("leak", tool=tool, rendered=str(result))
        else:
            trace.log("respond", tool=tool, template=True)

    def _call(self, trace: Trace, backend: Backend, tool: str, args: dict, ctx: dict) -> tuple[bool, dict]:
        meta = self.R.tools[tool]
        # precondition gate (sequencing lever)
        for pc in meta.preconditions:
            if "explain_bill" in pc and ctx.get("detail_level") != "full":
                trace.log("gate", tool=tool, precondition=pc, action="blocked; forcing explain_bill(full)")
                ok, res = self._call(trace, backend, "explain_bill",
                                     {"account_id": ctx["account_id"], "detail_level": "full"}, ctx)
                ctx["detail_level"] = "full"
            if "check_paper_free" in pc and "pfb_check" not in ctx:
                trace.log("gate", tool=tool, precondition=pc, action="blocked; forcing eligibility check")
                self._call(trace, backend, "check_paper_free_eligibility",
                           {"account_id": ctx["account_id"]}, ctx)
            if "terms_accepted" in pc and not trace.terms_read:
                trace.terms_read = True                  # harness forces the T&C step
                trace.log("gate", tool=tool, precondition=pc, action="terms read + acceptance captured")
            if "check_ddc" in pc and "ddc_check" not in ctx:
                trace.log("gate", tool=tool, precondition=pc, action="blocked; forcing eligibility gates")
                self._call(trace, backend, "check_ddc_eligibility", {"account_id": ctx["account_id"]}, ctx)
            if "disclosures" in pc and not trace.disclosures_made:
                trace.disclosures_made = True
                trace.consent_given = True
                trace.log("gate", tool=tool, precondition=pc, action="disclosures + explicit consent captured")
        ok, res = backend.execute(tool, args)
        trace.tool_calls.append({"tool": tool, "args": args, "ok": ok, "result": res})
        trace.log("call" if ok else "error", tool=tool, args=args, result=res)
        if ok:
            if tool == "explain_bill" and "line_items" in res:
                ctx["charge_id"] = res["line_items"][0]["charge_id"]
            if tool == "check_paper_free_eligibility":
                ctx["pfb_check"] = res
            if tool == "check_ddc_eligibility":
                ctx["ddc_check"] = res
        return ok, res

    def _select(self, utterance: str, allowed: list[str], trace: Trace) -> str:
        scores = self.llm.score_options(utterance, self.R.tool_options(allowed))
        best = max(scores, key=lambda k: (scores[k], k))
        trace.log("select", allowed=allowed, scores=scores, chosen=best)
        return best

    # ------------------------------------------------------------------ #
    def run(self, ep) -> Trace:
        trace = Trace(ep.id, sorted(self.R.levers))
        backend = Backend(ep.account)
        ctx = {"account_id": ep.account["account_id"]}
        B = self.R.behaviors

        agent = self.route(ep.utterance, trace)
        if "confirm_intent_first" in B:
            trace.log("respond", confirm_intent=True)
        if "carry_ivr_verification" not in B:
            trace.reverified = True
            trace.log("reverify", note="asked SSN last-four again despite IVR verification")

        phases = self.R.phases(ep.journey)
        agent_tools = self.R.agents[agent].tools
        phase_idx, steps = 0, 0
        goal_done = False

        while steps < MAX_STEPS and not goal_done:
            steps += 1
            allowed = (phases[min(phase_idx, len(phases) - 1)] if phases else agent_tools)
            allowed = [t for t in allowed if t in self.R.tools]
            if (ep.journey == "gen" and "hold_intent_queue" in B
                    and "explain_bill" in allowed
                    and not any(c["tool"] == "explain_bill" and c["ok"] for c in trace.tool_calls)
                    and len(allowed) > 1):
                allowed = [t for t in allowed if t != "transfer_to_agent"]
            trace.log("phase", index=phase_idx if phases else None, allowed=allowed,
                      mode="ANY" if phases else "AUTO(all)")
            tool = self._select(ep.utterance, allowed, trace)

            # argument construction from documented metadata + ctx (provenance)
            args = {"account_id": ctx["account_id"]}
            if tool == "explain_bill":
                args["detail_level"] = "full" if "detail_level" in self.R.tools[tool].required else "summary"
            if tool == "submit_fee_waiver":
                if "charge_id" in ctx:
                    args["charge_id"] = ctx["charge_id"]
                elif "COMES FROM" in self.R.tools[tool].description and phases is None:
                    # metadata documents provenance: fetch it first even without hard gate
                    self._call(trace, backend, "explain_bill",
                               {"account_id": ctx["account_id"], "detail_level": "full"}, ctx)
                    ctx["detail_level"] = "full"
                    args["charge_id"] = ctx.get("charge_id")
            if tool == "explain_bill":
                ctx["detail_level"] = args["detail_level"]
            if tool == "enroll_paper_free_billing":
                if "read_terms_get_consent" in B and not trace.terms_read:
                    trace.terms_read = True
                    trace.log("respond", terms_and_conditions=True, acceptance="explicit")
                args["terms_accepted"] = trace.terms_read
            if tool == "change_due_date":
                args["new_date"] = 15
                if "make_disclosures" in B and not trace.disclosures_made:
                    trace.disclosures_made = True; trace.consent_given = True
                    trace.log("respond", disclosures=["extra_interim_bill", "applies_all_lines"],
                              consent="explicit")
                args["consent"] = trace.consent_given
            if tool == "transfer_to_agent":
                args["target"] = "specialist"

            ok, res = self._call(trace, backend, tool, args, ctx)
            self._emit(trace, tool, res)

            # journey progress / termination
            bp_done = {c["tool"] for c in trace.tool_calls if c["ok"]}
            if ep.journey == "bex":
                goal_done = "submit_fee_waiver" in bp_done
                if tool == "get_account_balance":
                    trace.log("respond", text="Your balance is $500.47. Which charge would you like me to review?")
                if not ok and tool == "submit_fee_waiver" and steps >= 3:
                    self._call(trace, backend, "transfer_to_agent",
                               {"target": "specialist"}, ctx)
                    self._emit(trace, "transfer_to_agent", {"result": {"status": "Transfer initiated"}})
                    trace.outcome = "handoff"; break
            elif ep.journey == "pfb":
                goal_done = "enroll_paper_free_billing" in bp_done
                if not ok and tool == "enroll_paper_free_billing" and steps >= 3:
                    trace.outcome = "unresolved"; break
            elif ep.journey == "ddc":
                goal_done = "change_due_date" in bp_done
            else:  # gen
                goal_done = "transfer_to_agent" in bp_done and "explain_bill" in bp_done
                if steps >= 3 and "transfer_to_agent" not in bp_done:
                    self._call(trace, backend, "transfer_to_agent", {"target": "specialist"}, ctx)
                    self._emit(trace, "transfer_to_agent", {"result": {"status": "Transfer initiated"}})
                    bp_done.add("transfer_to_agent")
                    goal_done = "explain_bill" in bp_done
            if phases and ok:
                phase_idx = min(phase_idx + 1, len(phases) - 1)

        # intent handling accounting
        trace.intents_handled = list(ep.intents) if goal_done else ep.intents[:1]
        if ep.journey == "gen":
            trace.intents_handled = (ep.intents if ("hold_intent_queue" in B and goal_done)
                                     else ep.intents[:1])
        if ep.journey == "pfb":
            # discount cross-sell handled only with accurate-pitch behavior
            if "accurate_combined_discount" in B and goal_done:
                trace.log("respond", discount="requires BOTH paper-free AND AutoPay; routing AutoPay to specialist")
            else:
                trace.intents_handled = [i for i in trace.intents_handled if i != "discount"]
        if goal_done and trace.outcome == "unresolved":
            trace.outcome = "resolved"
        if "contextual_next_step" in B and trace.outcome == "resolved":
            trace.log("respond", next_step="contextual")
        return trace
