"""Agent runtime: route -> (phased) tool selection -> execution -> response.

Every decision is asked of the LLM backend using metadata from the Registry,
so applying a patch changes behavior through exactly one path: the metadata.
The Backend simulates the billing systems (backend-is-truth). The Trace is the
audit artifact the judge scores.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .registry import Registry, SELECTION_THRESHOLD
from .llm import LLMBackend

MAX_STEPS = 6

# Simulated per-call latency (ms) — the substrate for the tab's parallel-execution row
# ("estimated latency reduction of about 160-200 ms" when the three reads run in parallel).
LATENCY_MS = {"explain_bill": 95, "get_prior_tickets": 88, "get_usage_guidance": 102,
              "get_account_balance": 60, "submit_fee_waiver": 70, "manage_bill_preferences": 65,
              "check_paper_free_eligibility": 55, "enroll_paper_free_billing": 75,
              "check_ddc_eligibility": 55, "change_due_date": 75, "payment_arrangement": 70,
              "submit_plan_upgrade": 80, "transfer_to_agent": 40}

# AP-17 security output gate — IMMUTABLE hard gate (tab: "Hard gates ... unchanged").
# It auto-redacts internal identifiers that reach a customer-facing draft; the
# prompt_hat behavior 'forbid_internal_identifiers' prevents the draft from ever
# containing one, which is what stops the REPEATED auto-redactions.
import re as _re
_INTERNAL_ID = _re.compile(r"\bTKT-\d{4,}\b")   # internal ticket ids; bill-facing CHG- refs are legitimate


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
    context_attrs: dict = field(default_factory=dict)   # explicit versioned attrs (context lever)
    recurrence_recognized: bool = False
    latency_ms: int = 0
    latency_saved_ms: int = 0
    ap17_redactions: int = 0
    grounded_actions: int = 0
    savings_claim: str = ""          # '' | 'generic' | 'labeled_estimate'
    selection_confidence: dict = field(default_factory=dict)   # tool -> confidence at selection

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
        if tool == "get_prior_tickets":
            lb = args.get("lookback_days", 90)
            hits = [t for t in a.get("prior_tickets", []) if t["days_ago"] <= lb]
            return True, {"tickets": hits, "recurrence_flag": len(hits) >= 1,
                          "lookback_days": lb, "window_config": args.get("window_config", "hardcoded-90d")}
        if tool == "get_usage_guidance":
            return True, {"actions": a.get("kb_actions", []),
                          "savings_estimate": a.get("savings_estimate")}
        if tool == "submit_plan_upgrade":
            if not args.get("customer_confirmed"):
                return False, {"error": "CONFIRMATION_REQUIRED",
                               "note": "explicit customer confirmation must precede any plan change"}
            return True, {"status": "UPGRADED", "ref": "U-90112"}
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
        """Response formatting: verbatim tool JSON (leak) or templated. AP-17 always runs."""
        contract = self.R.tools[tool].output_contract
        if contract == "verbatim" and tool in ("transfer_to_agent", "manage_bill_preferences",
                                               "get_prior_tickets"):
            rendered = str(result)
            rendered = self._ap17(trace, rendered)          # immutable security gate
            trace.leaks += 1
            trace.log("leak", tool=tool, rendered=rendered)
        else:
            trace.log("respond", tool=tool, template=True)

    def _ap17(self, trace: Trace, text: str) -> str:
        """AP-17 output gate: redact internal identifiers in customer-facing text.
        Immutable in every configuration; 'forbid_internal_identifiers' removes the need."""
        hits = _INTERNAL_ID.findall(text)
        if hits:
            trace.ap17_redactions += len(hits)
            text = _INTERNAL_ID.sub("[REDACTED-AP17]", text)
            trace.log("redact", gate="AP-17", count=len(hits),
                      note="internal identifier reached a customer-facing draft; auto-redacted")
        return text

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
        trace.latency_ms += LATENCY_MS.get(tool, 50)
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
        # Selection confidence vs the FROZEN 0.61 threshold (tab: "do not immediately lower
        # threshold; improve metadata first"). Confidence = best / (best + runner-up).
        ranked = sorted(scores.values(), reverse=True)
        conf = round(ranked[0] / (ranked[0] + (ranked[1] if len(ranked) > 1 else 0)), 3) if ranked and ranked[0] else 0.0
        trace.selection_confidence[best] = conf
        trace.log("select", allowed=allowed, scores=scores, chosen=best,
                  confidence=conf, threshold=SELECTION_THRESHOLD, threshold_changed=False)
        return best

    # ------------------------------------------------------------------ #
    # rbc (repeat-billing-complaint) helpers — Harness-Optimization tab rows
    # ------------------------------------------------------------------ #
    def _rbc_args(self, tool: str, ctx: dict) -> dict:
        args = {"account_id": ctx["account_id"]}
        if tool == "explain_bill":
            args["detail_level"] = "full"
        if tool == "get_prior_tickets":
            args["lookback_days"] = self.R.lookback_days
            args["window_config"] = "LOOKBACK-90d-v1" if self.R.lookback_versioned else "hardcoded-90d"
        if tool == "get_usage_guidance":
            args["topic"] = "high_bill_data_usage"
        return args

    def _post_rbc(self, trace: Trace, tool: str, ok: bool, res: dict, B: set):
        if not ok:
            return
        if tool == "get_prior_tickets" and res.get("recurrence_flag"):
            trace.recurrence_recognized = True
            if "root_cause_summary" in B:
                trace.log("respond", recurrence_ack=True, root_cause=True)
        if tool == "get_usage_guidance":
            actions, est = res.get("actions", []), res.get("savings_estimate") or {}
            if self.R.evidence:   # evidence lever: <=3 KB-sourced actions, labeled estimates
                n = min(self.R.evidence["max_grounded_actions"], len(actions))
                trace.grounded_actions = n
                trace.savings_claim = "labeled_estimate"
                trace.log("respond", quick_wins=[a["action"] for a in actions[:n]],
                          sources=[a["source"] for a in actions[:n]],
                          savings=f"~${est.get('amount', 0)}/mo — an ESTIMATE, per {est.get('source')}",
                          labeled_estimate=True)
            else:
                trace.savings_claim = "generic"
                trace.log("respond", generic_savings=True)

    def _rbc_goal(self, trace: Trace) -> bool:
        full = any(c["tool"] == "explain_bill" and c["ok"] and c["args"].get("detail_level") == "full"
                   for c in trace.tool_calls)
        return full and trace.recurrence_recognized and trace.grounded_actions >= 1

    # ------------------------------------------------------------------ #
    def run(self, ep) -> Trace:
        trace = Trace(ep.id, sorted(self.R.levers))
        backend = Backend(ep.account)
        ctx = {"account_id": ep.account["account_id"]}
        B = self.R.behaviors

        agent = self.route(ep.utterance, trace)
        if ep.journey == "rbc":
            recent = [t for t in ep.account.get("prior_tickets", [])
                      if t["days_ago"] <= self.R.lookback_days]
            if self.R.context_attrs and recent:
                # context lever: soft repeat_contact_context PROMOTED to an explicit,
                # versioned attribute the planner reliably consumes.
                trace.context_attrs["RECURRING_CONTACT"] = {
                    "value": True, "version": self.R.context_attrs["RECURRING_CONTACT"]["version"],
                    "tickets_in_window": len(recent),
                    "lookback": {"days": self.R.lookback_days, "config_id": "LOOKBACK-90d-v1",
                                 "configurable": True}}
                trace.recurrence_recognized = True
                trace.log("context", attribute="RECURRING_CONTACT",
                          promoted_from="repeat_contact_context", version="ctx-attr-v1",
                          tickets_in_window=len(recent))
            else:
                trace.log("context", attribute="repeat_contact_context", soft_flag=True,
                          note="soft flag present but not consumed by the planner")
            if "empathy_first_opening" in B:
                trace.log("respond", empathy=True)
        if "confirm_intent_first" in B:
            trace.log("respond", confirm_intent=True)
        if "carry_ivr_verification" not in B:
            trace.reverified = True
            trace.log("reverify", note="asked SSN last-four again despite IVR verification")

        phases = self.R.phases(ep.journey)
        agent_tools = self.R.agents[agent].tools
        phase_idx, steps = 0, 0
        goal_done = False

        pgroups = self.R.parallel_groups(ep.journey)
        while steps < MAX_STEPS and not goal_done:
            steps += 1
            allowed = (phases[min(phase_idx, len(phases) - 1)] if phases else agent_tools)
            allowed = [t for t in allowed if t in self.R.tools]
            if ep.journey == "rbc":   # STM caches idempotent reads; don't re-call succeeded ones
                dedup = [t for t in allowed if not (t in ("get_prior_tickets", "get_usage_guidance")
                         and any(c["tool"] == t and c["ok"] for c in trace.tool_calls))]
                allowed = dedup or allowed
            group = next((g for g in pgroups if set(g) <= set(allowed)
                          and not any(c["tool"] in g and c["ok"] for c in trace.tool_calls)), None)
            if group:
                # Tab row "Tool execution order": three independent reads run in PARALLEL
                # after dependency validation (none consumes another's output).
                trace.log("parallel", tools=list(group), dependency_validation="passed",
                          note="independent reads; wall-clock = max, not sum")
                seq_ms = sum(LATENCY_MS[t] for t in group)
                before_ms = trace.latency_ms
                for t2 in group:
                    ok, res = self._call(trace, backend, t2, self._rbc_args(t2, ctx), ctx)
                    self._emit(trace, t2, res)
                    self._post_rbc(trace, t2, ok, res, B)
                trace.latency_ms = before_ms + max(LATENCY_MS[t] for t in group)
                trace.latency_saved_ms += seq_ms - max(LATENCY_MS[t] for t in group)
                phase_idx = min(phase_idx + 1, len(phases) - 1)
                goal_done = self._rbc_goal(trace)
                continue
            if (ep.journey == "gen" and "hold_intent_queue" in B
                    and "explain_bill" in allowed
                    and not any(c["tool"] == "explain_bill" and c["ok"] for c in trace.tool_calls)
                    and len(allowed) > 1):
                allowed = [t for t in allowed if t != "transfer_to_agent"]
            trace.log("phase", index=phase_idx if phases else None, allowed=allowed,
                      mode="ANY" if phases else "AUTO(all)")
            if (ep.journey == "rbc" and trace.context_attrs.get("RECURRING_CONTACT")
                    and not phases):
                # Tab row "Repeat-contact signal": the planner CONSUMES the explicit
                # RECURRING_CONTACT attribute - history and guidance reads come first.
                pri = [t for t in ("get_prior_tickets", "get_usage_guidance")
                       if t in allowed and not any(c["tool"] == t and c["ok"] for c in trace.tool_calls)]
                if pri:
                    tool = pri[0]
                    trace.log("select", allowed=allowed, chosen=tool,
                              planner_signal="RECURRING_CONTACT (ctx-attr-v1)",
                              note="explicit context attribute consumed by the planner")
                else:
                    tool = self._select(ep.utterance, allowed, trace)
            else:
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
            if tool == "get_prior_tickets":
                args["lookback_days"] = self.R.lookback_days
                args["window_config"] = "LOOKBACK-90d-v1" if self.R.lookback_versioned else "hardcoded-90d"
            if tool == "get_usage_guidance":
                args["topic"] = "high_bill_data_usage"
            if tool == "submit_plan_upgrade":
                # Control PRESERVED (tab row "Customer confirmation"): the upgrade is
                # deferred in every configuration until explicit confirmation exists.
                trace.log("respond", plan_option=True, deferred=True,
                          note="explicit customer confirmation required before any plan change")
                if ep.journey == "rbc":
                    goal_done = self._rbc_goal(trace)
                continue
            if tool == "transfer_to_agent":
                args["target"] = "specialist"

            ok, res = self._call(trace, backend, tool, args, ctx)
            self._emit(trace, tool, res)
            if ep.journey == "rbc":
                self._post_rbc(trace, tool, ok, res, B)

            # journey progress / termination
            bp_done = {c["tool"] for c in trace.tool_calls if c["ok"]}
            if ep.journey == "rbc":
                goal_done = self._rbc_goal(trace)
                if tool in ("get_account_balance", "explain_bill") and ok and steps == 1:
                    text = f"Your balance is ${ep.account['balance']:.2f} this cycle."
                    if "forbid_internal_identifiers" not in B and ep.account.get("prior_tickets"):
                        # Tab row "Internal ticket IDs": the vanilla draft references the raw
                        # internal id from the soft flag; the immutable AP-17 gate redacts it.
                        draft = text + (f" I also see your earlier ticket "
                                        f"{ep.account['prior_tickets'][0]['ticket_id']} about this issue.")
                        text = self._ap17(trace, draft)
                    trace.log("respond", text=text)
                if not goal_done and not trace.savings_claim and steps >= 2:
                    trace.savings_claim = "generic"   # tab Before: generic savings language
                    trace.log("respond", generic_savings=True)
            elif ep.journey == "bex":
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
        if ep.journey == "rbc" and goal_done and "structured_rbc_response" in B:
            # Tab row "Response structure": ... plan option -> customer choice (deferred upgrade).
            trace.log("respond", plan_option=True, customer_choice=True)
        if goal_done and trace.outcome == "unresolved":
            trace.outcome = "resolved"
        if "contextual_next_step" in B and trace.outcome == "resolved":
            trace.log("respond", next_step="contextual")
        return trace
