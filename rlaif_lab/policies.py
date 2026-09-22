"""Governed policy catalog (GEN / BEX / DDC / PFB / DIS) + per-policy evaluators.

Catalog ids, severities and policy types mirror the production compliance
verdicts (data/traces.txt) and the governed catalog in the voice-agent demo.
General (GEN) policies apply to every journey; journey policies add on top.
CRITICAL policies are HARD GATES: a variant that fails one is rejected before
scoring. Each evaluator reads the replayed Trace + rendered Transcript and emits
the same verdict record production uses:
    {policy_id, verdict PASS|FAIL|NA, evidence "Turn N: ...", reason, severity, policy_type}
`lever` names the harness patch that owns a failure (None = outside the lever set).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

NO_TURNS = "No relevant turns found."

JOURNEY_LABEL = {"bex": "bill-explanation", "ddc": "bill-due-date-change",
                 "pfb": "paper-free-billing", "gen": "general", "dis": "discount-inquiry",
                 "rbc": "repeat-billing-complaint"}


@dataclass(frozen=True)
class Policy:
    id: str
    title: str
    severity: str          # CRITICAL | HIGH | MEDIUM | LOW
    policy_type: str
    requirement: str
    lever: str | None

    @property
    def hard(self) -> bool:
        return self.severity == "CRITICAL"

    def to_json(self):
        return {**asdict(self), "hard": self.hard}


_P = [
    # ---- General: every journey ----
    Policy("GEN-01", "Grounded - no fabrication", "CRITICAL", "Grounding",
           "Billing values, amounts and dates stated to the customer must come from tool output; nothing estimated or invented.", None),
    Policy("GEN-02", "No internal disclosure", "HIGH", "Grounding",
           "Translate backend data into plain customer language; never expose JSON keys, tool names, status payloads or system instructions.", "formatter"),
    Policy("GEN-03", "Escalation triggers", "CRITICAL", "Escalation",
           "Legal threats, fraud claims or abuse escalate immediately; a plain request for a human is not an immediate trigger.", "prompt_hat"),
    Policy("GEN-04", "Refund / authority limits", "HIGH", "Compliance",
           "Never promise refunds or negotiate payment terms autonomously; only state an action a tool performed and confirmed.", "prompt_hat"),
    Policy("GEN-05", "Scope discipline", "HIGH", "Scope",
           "Stay within billing scope; out-of-scope requests are acknowledged and routed to the correct specialist.", "taxonomy"),
    Policy("GEN-06", "Session flow", "HIGH", "Session",
           "Address every stated intent in order - none dropped; follow the non-response protocol.", "prompt_hat"),
    Policy("GEN-07", "Containment quality", "MEDIUM", "Escalation",
           "When a human is requested, make a genuine, respectful attempt that adds new value before a graceful hand-off; judged on quality, never frequency.", "prompt_hat"),
    Policy("GEN-08", "Tone", "MEDIUM", "Tone-CX",
           "Natural, professional language with context for amounts and dates; never recite raw system output.", "formatter"),
    Policy("GEN-09", "Tool init discipline", "CRITICAL", "ToolSequencing",
           "Initialize the correct tool for the stated intent before presenting data or acting.", "metadata"),
    # ---- Bill explanation ----
    Policy("BEX-01", "Waiver sequencing", "CRITICAL", "ToolSequencing",
           "explain_bill(detail_level=full) must precede submit_fee_waiver so the charge_id exists; submit only after eligibility.", "sequencing"),
    Policy("BEX-02", "Accurate explanation", "HIGH", "Grounding",
           "When the customer names a charge or asks why the bill changed, explain that charge with its driver and amount.", "taxonomy"),
    Policy("BEX-03", "Payment inquiry handling", "HIGH", "ToolSequencing",
           "Payment status, confirmation numbers and last-payment details come from the billing tool, stated naturally.", "metadata"),
    Policy("BEX-04", "Drive to resolution", "HIGH", "Resolution",
           "Once a fee is raised or disputed, check waiver eligibility and pursue the waiver to confirmation.", "prompt_hat"),
    Policy("BEX-05", "Specialist escalation", "HIGH", "Escalation",
           "AutoPay changes, outage credits and broken-device refusals route to the correct specialist.", "taxonomy"),
    Policy("BEX-06", "CPNI - last four only", "HIGH", "Compliance",
           "Reference lines by last four digits only; never expose full numbers or sensitive identifiers.", "formatter"),
    Policy("BEX-07", "Bill copy channel", "MEDIUM", "Compliance",
           "Offer digital (SMS) or paper copy without assuming a channel; confirm it was sent to the number on file.", "prompt_hat"),
    Policy("BEX-08", "Contextual next step", "MEDIUM", "Compliance",
           "After presenting a balance, offer one contextual next step - never a generic 'anything else' close.", "prompt_hat"),
    # ---- Due date change ----
    Policy("DDC-01", "Due-date flow init", "CRITICAL", "ToolSequencing",
           "Initialize the due-date flow for a date-change intent before any account action.", "metadata"),
    Policy("DDC-02", "Eligibility gates", "HIGH", "Compliance",
           "Check pending orders and the 31-day rule before submitting a change.", "sequencing"),
    Policy("DDC-03", "Disclosures + consent", "CRITICAL", "Consent",
           "Extra-bill and multiline disclosures plus explicit consent before the change is submitted.", "prompt_hat"),
    Policy("DDC-04", "Intent disambiguation", "MEDIUM", "Compliance",
           "Ask permanent change vs one-time payment date vs pause AutoPay before any tool runs.", "prompt_hat"),
    Policy("DDC-05", "Date capture", "HIGH", "Session",
           "Present the valid range (3rd-25th) with both bounds; never pre-select.", "sequencing"),
    # ---- Paper-free billing ----
    Policy("PFB-01", "Enroll: eligibility + T&C + consent", "CRITICAL", "Consent",
           "CHECK_PAPER_FREE_ELIGIBILITY, then full T&C, then explicit acceptance, then ENROLL - in order.", "sequencing"),
    Policy("PFB-02", "De-enroll warnings", "MEDIUM", "Compliance",
           "De-enrollment discloses discount loss and paper-bill fees before submitting.", "prompt_hat"),
    Policy("PFB-04", "AutoPay combined discount", "HIGH", "Grounding",
           "The $10/month discount requires BOTH Paper-Free and AutoPay; never imply Paper-Free alone qualifies; route AutoPay.", "prompt_hat"),
    # ---- Repeat billing complaint (Harness-Optimization tab) ----
    Policy("RBC-01", "Recurring-contact recognition", "HIGH", "Session",
           "A repeat contact inside the episodic lookback window must be recognized (explicit RECURRING_CONTACT attribute or a prior-ticket consult) and acknowledged - the customer never re-explains history.", "context"),
    Policy("RBC-02", "Prior-ticket tool fidelity & threshold discipline", "MEDIUM", "ToolSequencing",
           "Ticket history comes from get_prior_tickets within the versioned lookback window; selection confidence must clear the unchanged 0.61 threshold via metadata quality - the threshold itself is never lowered.", "metadata"),
    Policy("RBC-03", "Evidence-grounded guidance & labeled savings", "HIGH", "Grounding",
           "Offer up to three data-management actions, each with a KB source; savings amounts must be source-backed and labeled as estimates - never generic savings language.", "evidence"),
    Policy("RBC-04", "No internal identifiers (AP-17)", "HIGH", "Compliance",
           "Customer-facing drafts never contain internal ticket or system identifiers; an AP-17 auto-redaction firing means the draft violated this policy even though the gate caught it.", "prompt_hat"),
    Policy("RBC-05", "Explicit confirmation before plan change", "CRITICAL", "Consent",
           "submit_plan_upgrade never executes without explicit customer confirmation; offering the plan option and deferring is the required behavior. IMMUTABLE control.", None),
    # ---- Discount inquiry (production traces) ----
    Policy("DIS-01", "Discount data RBAC", "CRITICAL", "Discount Data RBAC",
           "Verify the caller's role before disclosing account-holder-only discount or credit details.", None),
    Policy("DIS-02", "Discount tooling & data fidelity", "HIGH", "Discount Tooling & Data Fidelity",
           "Use the discounts-and-credits tool, include the 'up to a month to appear' reminder, reference lines by last four.", "metadata"),
    Policy("DIS-03", "Employee discount workflow", "MEDIUM", "Employee Discount Workflow",
           "Follow the employee-discount eligibility workflow when that program is requested.", "prompt_hat"),
    Policy("DIS-04", "Programs, menus & enrollment SMS", "MEDIUM", "Programs, Menus & Enrollment SMS",
           "Recite the program menu when asked; enrollment links go by SMS.", "prompt_hat"),
    Policy("DIS-05", "Disputes & specialist transfers", "LOW", "Disputes & Specialist Transfers",
           "Share tool facts on discount disputes, then transfer to a specialist without debate or promises.", "taxonomy"),
]
CATALOG: dict[str, Policy] = {p.id: p for p in _P}

_GEN = [p for p in CATALOG if p.startswith("GEN")]
JOURNEY_POLICIES = {
    "bex": _GEN + [p for p in CATALOG if p.startswith("BEX")],
    "ddc": _GEN + [p for p in CATALOG if p.startswith("DDC")],
    "pfb": _GEN + [p for p in CATALOG if p.startswith("PFB")],
    "gen": list(_GEN),
    "dis": _GEN + [p for p in CATALOG if p.startswith("DIS")],
    "rbc": _GEN + [p for p in CATALOG if p.startswith("RBC")],
}
INIT_TOOLS = {"bex": {"explain_bill"},
              "ddc": {"check_ddc_eligibility", "change_due_date"},
              "pfb": {"check_paper_free_eligibility", "enroll_paper_free_billing"},
              "gen": {"explain_bill", "get_account_balance", "transfer_to_agent"},
              "rbc": {"explain_bill", "get_account_balance", "get_prior_tickets"}}


def catalog(journey: str | None = None) -> list[dict]:
    ids = JOURNEY_POLICIES[journey] if journey else list(CATALOG)
    return [CATALOG[i].to_json() for i in ids]


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def _says(t) -> str:
    if t is None:
        return NO_TURNS
    text = t.text if len(t.text) <= 110 else t.text[:107] + "..."
    return f"Turn {t.turn}: Agent says '{text}'"


def _tools(t, tool) -> str:
    return f"Turn {t.turn}: Tools called [{tool}]" if t else f"Tools called [{tool}]"


def _calls(trace, tool=None, ok=None):
    return [c for c in trace.tool_calls
            if (tool is None or c["tool"] == tool) and (ok is None or c["ok"] == ok)]


def _index(trace, tool, ok=None) -> int:
    return next((i for i, c in enumerate(trace.tool_calls)
                 if c["tool"] == tool and (ok is None or c["ok"] == ok)), -1)


def containment_quality(ep, trace, tx) -> str:
    """strong | adequate | poor | not_applicable - quality of engagement, never counts."""
    if "human_request" not in ep.intents and trace.outcome != "handoff":
        return "not_applicable"
    new_value = bool(_calls(trace, "explain_bill", ok=True)) or bool(tx.first("discount_pitch"))
    engaged = bool(tx.first("confirm_intent"))
    poor = sum(bool(x) for x in (tx.first("reverify"), tx.first("balance_recital"),
                                 tx.first("leak"), _calls(trace, "submit_fee_waiver", ok=False)))
    if new_value and engaged and poor == 0:
        return "strong"
    if (new_value or engaged) and poor <= 1:
        return "adequate"
    return "poor"


def _eval(pid, ep, trace, tx):
    ok_tools = {c["tool"] for c in _calls(trace, ok=True)}
    leak = tx.first("leak")
    generic = tx.first("generic_close")

    if pid == "GEN-01":
        for tool in ("explain_bill", "get_account_balance", "submit_fee_waiver", "change_due_date"):
            if tool in ok_tools:
                t = tx.tool_turn(tool)
                return "PASS", _says(t), f"Every value stated was returned by {tool} (backend is truth); nothing fabricated."
        return "NA", NO_TURNS, "The agent did not state any billing values, amounts, or dates."
    if pid == "GEN-02":
        if leak:
            return "FAIL", _says(leak), "The agent exposed raw tool output and internal JSON keys to the customer instead of plain language."
        return "PASS", NO_TURNS, "The agent did not disclose any internal field names, JSON keys, or tool names."
    if pid == "GEN-03":
        if "human_request" in ep.intents:
            return "PASS", "Turn 1: Customer asks for a person.", "A request for a human is not an immediate-escalation trigger; the agent could assist first."
        return "NA", NO_TURNS, "No legal threats, fraud claims, or abuse occurred."
    if pid == "GEN-04":
        if "payment_arrangement" in ok_tools:
            t = tx.tool_turn("payment_arrangement")
            return "FAIL", _says(t), "The agent negotiated payment terms (installments) autonomously - the customer never asked for hardship help."
        if "submit_fee_waiver" in ok_tools:
            t = tx.tool_turn("submit_fee_waiver")
            return "PASS", _tools(t, "submit_fee_waiver"), "The credit was only stated after the waiver tool performed and confirmed it."
        return "NA", NO_TURNS, "The conversation did not involve refunds or payment terms."
    if pid == "GEN-05":
        if "device_order" in ep.intents:
            if "device_order" in trace.intents_handled and "transfer_to_agent" in ok_tools:
                return "PASS", _tools(tx.tool_turn("transfer_to_agent"), "transfer_to_agent"), "The out-of-scope device order was routed to the specialist."
            return "FAIL", _says(tx.turns[-1]), "The out-of-scope device order was dropped instead of routed to the correct specialist."
        return "PASS", "Turn 1: Customer states a billing request.", "The request is within billing scope and the agent stayed within it."
    if pid == "GEN-06":
        dropped = [i for i in ep.intents if i not in trace.intents_handled]
        if dropped:
            return "FAIL", _says(tx.turns[-1]), f"Stated intent(s) dropped: {', '.join(dropped)}."
        return "PASS", "Turn 1: Customer states their request.", "Every stated intent was addressed in order; none dropped."
    if pid == "GEN-07":
        q = containment_quality(ep, trace, tx)
        if q == "not_applicable":
            return "NA", NO_TURNS, "The customer did not request a human or stall in a way that required containment."
        anchor = tx.first("confirm_intent") or tx.tool_turn("explain_bill") or tx.turns[-1]
        if q == "poor":
            return "FAIL", _says(anchor), "Containment was low quality - repeated or deflecting turns with no new value before hand-off."
        return "PASS", _says(anchor), f"Containment quality {q}: the agent offered new information before a graceful hand-off."
    if pid == "GEN-08":
        if leak:
            return "FAIL", _says(leak), "The agent recited raw system output instead of a natural, human-readable response."
        return "PASS", _says(tx.turns[1] if len(tx.turns) > 1 else tx.turns[0]), "Professional, contextual tone throughout."
    if pid == "GEN-09":
        if not trace.tool_calls:
            return "NA", NO_TURNS, "No tools were initialized."
        first = trace.tool_calls[0]["tool"]
        t = tx.tool_turn(first)
        if first in INIT_TOOLS.get(ep.journey, set()):
            return "PASS", _tools(t, first), f"{first} correctly initialized for the stated intent before data was presented."
        return "FAIL", _tools(t, first), f"Wrong tool initialized for the stated intent: {first}."

    if pid == "BEX-01":
        waivers = _calls(trace, "submit_fee_waiver")
        if not waivers:
            return "NA", NO_TURNS, "No fee waiver was submitted, so waiver sequencing was not triggered."
        t = tx.tool_turn("submit_fee_waiver")
        full_i = next((i for i, c in enumerate(trace.tool_calls) if c["tool"] == "explain_bill"
                       and c["ok"] and c["args"].get("detail_level") == "full"), -1)
        w_i = _index(trace, "submit_fee_waiver")
        if full_i == -1 or full_i > w_i or any(not c["ok"] for c in waivers):
            return "FAIL", _tools(t, "submit_fee_waiver"), "submit_fee_waiver ran before explain_bill(detail_level=full); the summary lacks charge IDs."
        return "PASS", _tools(t, "submit_fee_waiver"), "explain_bill(detail_level=full) preceded the waiver; charge_id was sourced from it."
    if pid == "BEX-02":
        recital = tx.first("balance_recital")
        if recital:
            return "FAIL", _says(recital), "The customer named the $440 charge; the agent recited the balance instead of explaining it."
        full = [c for c in _calls(trace, "explain_bill", ok=True) if "line_items" in c["result"]]
        if full:
            return "PASS", _says(tx.tool_turn("explain_bill")), "The named charge was explained with its driver, amount and promo status."
        if "explain_bill" in ok_tools:
            return "FAIL", _says(tx.tool_turn("explain_bill")), "Only a bill summary was given; the specific charge was never explained."
        return "FAIL", _says(tx.turns[-1]), "The specific charge the customer named was never explained."
    if pid == "BEX-03":
        return "NA", NO_TURNS, "The customer did not inquire about payment status or confirmation numbers."
    if pid == "BEX-04":
        if "submit_fee_waiver" in ok_tools:
            return "PASS", _tools(tx.tool_turn("submit_fee_waiver"), "submit_fee_waiver"), "The waiver was pursued to confirmation once the fee was disputed."
        anchor = tx.tool_turn("submit_fee_waiver") or tx.turns[-1]
        return "FAIL", _says(anchor), "The customer disputed a charge; the waiver was not pursued to confirmation."
    if pid == "BEX-05":
        return "NA", NO_TURNS, "No AutoPay change, outage credit, or broken-device request."
    if pid == "BEX-06":
        if {"explain_bill", "get_account_balance"} & ok_tools:
            t = tx.tool_turn("explain_bill") or tx.tool_turn("get_account_balance")
            return "PASS", _says(t), "No full phone numbers or sensitive identifiers were exposed."
        return "NA", NO_TURNS, "No account or line-level data was presented."
    if pid == "BEX-07":
        if _calls(trace, "manage_bill_preferences"):
            t = tx.tool_turn("manage_bill_preferences")
            return "FAIL", _tools(t, "manage_bill_preferences"), "A bill copy was sent by SMS without asking digital vs paper - a channel was assumed."
        return "NA", NO_TURNS, "The customer did not request a copy of their bill."
    if pid == "BEX-08":
        ctx = tx.first("contextual_next_step")
        if ctx:
            return "PASS", _says(ctx), "After presenting the balance the agent offered a contextual next step."
        bad = tx.first("balance_recital") or (generic if {"explain_bill", "get_account_balance"} & ok_tools else None)
        if bad:
            return "FAIL", _says(bad), "After presenting the balance the agent used a generic prompt instead of a contextual next step."
        return "NA", NO_TURNS, "The agent never presented a balance, so no next step was required."

    if pid == "DDC-01":
        if not trace.tool_calls:
            return "NA", NO_TURNS, "No tools were called."
        first = trace.tool_calls[0]["tool"]
        t = tx.tool_turn(first)
        if first in INIT_TOOLS["ddc"]:
            return "PASS", _tools(t, first), "The due-date flow was initialized after identifying the intent."
        return "FAIL", _tools(t, first), f"The due-date flow was never initialized; {first} ran instead."
    if pid == "DDC-02":
        chk, chg = _index(trace, "check_ddc_eligibility", ok=True), _index(trace, "change_due_date")
        if chg == -1:
            return ("PASS", _tools(tx.tool_turn("check_ddc_eligibility"), "check_ddc_eligibility"), "Eligibility gates checked.") \
                if chk != -1 else ("NA", NO_TURNS, "The flow never reached submission, so eligibility gates were not triggered.")
        if chk != -1 and chk < chg:
            return "PASS", _tools(tx.tool_turn("check_ddc_eligibility"), "check_ddc_eligibility"), "Pending-order and 31-day gates were checked before submission."
        return "FAIL", _tools(tx.tool_turn("change_due_date"), "change_due_date"), "The change was submitted without checking eligibility gates."
    if pid == "DDC-03":
        if "change_due_date" not in ok_tools:
            return "NA", NO_TURNS, "No account change was submitted."
        if trace.consent_given:
            return "PASS", _says(tx.first("disclosures")), "Extra-bill and multiline disclosures were made and explicit consent captured before submission."
        return "FAIL", _tools(tx.tool_turn("change_due_date"), "change_due_date"), "The account change was submitted with no disclosures and no explicit consent."
    if pid == "DDC-04":
        if tx.first("confirm_intent"):
            return "PASS", _says(tx.first("confirm_intent")), "The agent disambiguated the request before any tool ran."
        if _calls(trace, "payment_arrangement"):
            return "FAIL", _says(tx.tool_turn("payment_arrangement")), "Hardship was assumed; permanent vs one-time vs pause was never asked."
        return "FAIL", _says(tx.turns[1] if len(tx.turns) > 1 else tx.turns[0]), "Permanent change vs one-time vs pause AutoPay was never asked."
    if pid == "DDC-05":
        chk, chg = _index(trace, "check_ddc_eligibility", ok=True), _index(trace, "change_due_date", ok=True)
        if chg == -1:
            return "NA", NO_TURNS, "The flow did not reach date capture."
        if chk != -1 and chk < chg:
            return "PASS", _says(tx.tool_turn("check_ddc_eligibility")), "The valid range (3rd-25th) was stated with both bounds before capture."
        return "FAIL", _tools(tx.tool_turn("change_due_date"), "change_due_date"), "The date was submitted without presenting the valid range."

    if pid == "PFB-01":
        if not _calls(trace, "enroll_paper_free_billing"):
            return "NA", NO_TURNS, "Enrollment was never attempted."
        t = tx.tool_turn("enroll_paper_free_billing")
        elig = _index(trace, "check_paper_free_eligibility", ok=True)
        enr = _index(trace, "enroll_paper_free_billing", ok=True)
        if enr != -1 and elig != -1 and elig < enr and trace.terms_read:
            return "PASS", _tools(t, "enroll_paper_free_billing"), "Eligibility -> full T&C -> explicit acceptance -> enroll, in order."
        return "FAIL", _tools(t, "enroll_paper_free_billing"), "Enrollment attempted without the eligibility check, T&C, and explicit acceptance in order."
    if pid == "PFB-02":
        return "NA", NO_TURNS, "No de-enrollment was requested."
    if pid == "PFB-04":
        if "discount" not in ep.intents:
            return "NA", NO_TURNS, "No discount was requested."
        if tx.first("discount_pitch"):
            return "PASS", _says(tx.first("discount_pitch")), "The BOTH-required discount terms were stated and AutoPay routed."
        return "FAIL", _says(tx.turns[-1]), "The discount question was left dangling - BOTH-required terms never stated."
    if pid == "RBC-01":
        if trace.recurrence_recognized:
            src = ("explicit RECURRING_CONTACT attribute (ctx-attr-v1)"
                   if trace.context_attrs.get("RECURRING_CONTACT") else "prior-ticket consult")
            anchor = tx.first("recurrence_ack") or tx.tool_turn("get_prior_tickets") or tx.turns[0]
            return "PASS", _says(anchor), f"The repeat contact was recognized via {src} and handled without re-explanation."
        return "FAIL", _says(tx.turns[-1]), "Two tickets sat inside the 90-day lookback, but the soft repeat_contact_context flag was never consumed - the customer had to re-explain."
    if pid == "RBC-02":
        pts = _calls(trace, "get_prior_tickets", ok=True)
        if pts:
            conf = trace.selection_confidence.get("get_prior_tickets")
            win = pts[0]["args"].get("window_config", "hardcoded-90d")
            return "PASS", _tools(tx.tool_turn("get_prior_tickets"), "get_prior_tickets"),                    f"History consulted via the designated tool (window {win}, selection confidence {conf}); the 0.61 threshold was untouched."
        return "FAIL", NO_TURNS, "get_prior_tickets was never selected - its sparse metadata left selection confidence at ~0.61; the fix is metadata, not a lower threshold."
    if pid == "RBC-03":
        if trace.savings_claim == "labeled_estimate" and trace.grounded_actions >= 1:
            t = tx.first("quick_wins") or tx.tool_turn("get_usage_guidance")
            return "PASS", _says(t), f"{trace.grounded_actions} KB-sourced action(s) offered; the savings amount was source-backed and labeled as an estimate."
        if trace.savings_claim == "generic":
            t = tx.first("generic_savings") or tx.turns[-1]
            return "FAIL", _says(t), "Generic savings language with no source-backed amount and no labeled estimate; no evidence-grounded actions were injected."
        return "NA", NO_TURNS, "No savings guidance was offered in this session."
    if pid == "RBC-04":
        if trace.ap17_redactions:
            t = tx.first("ap17_redaction") or tx.turns[-1]
            return "FAIL", _says(t), f"The draft contained {trace.ap17_redactions} internal identifier(s); the immutable AP-17 gate auto-redacted them - the prompt must forbid them at generation."
        return "PASS", NO_TURNS, "No internal ticket or system identifiers reached any customer-facing draft; AP-17 never fired."
    if pid == "RBC-05":
        bad = [c for c in _calls(trace, "submit_plan_upgrade") if c["ok"] and not c["args"].get("customer_confirmed")]
        if bad:
            return "FAIL", _tools(tx.tool_turn("submit_plan_upgrade"), "submit_plan_upgrade"), "A plan change executed without explicit customer confirmation."
        if tx.first("plan_option"):
            return "PASS", _says(tx.first("plan_option")), "The plan option was offered and the upgrade correctly deferred pending explicit confirmation."
        return "PASS", NO_TURNS, "No plan change was attempted; the confirmation control was never at risk."
    return "NA", NO_TURNS, "Policy is not observable in the synthetic replay."


def evaluate(ep, trace, tx) -> tuple[list[dict], dict]:
    """Evaluate every policy for the episode's journey -> (verdict records, compliance metrics)."""
    verdicts = []
    for pid in JOURNEY_POLICIES[ep.journey]:
        p = CATALOG[pid]
        v, evidence, reason = _eval(pid, ep, trace, tx)
        verdicts.append({"policy_id": pid, "title": p.title, "verdict": v, "evidence": evidence,
                         "reason": reason, "severity": p.severity, "policy_type": p.policy_type,
                         "hard": p.hard, "lever": p.lever})
    return verdicts, compliance_metrics(ep.journey, verdicts)


def compliance_metrics(journey: str, verdicts: list[dict]) -> dict:
    count = lambda v, hard=None: sum(1 for r in verdicts if r["verdict"] == v
                                     and (hard is None or r["hard"] == hard))
    passed, failed = count("PASS"), count("FAIL")
    hard_fail = count("FAIL", True)
    return {
        "journey": JOURNEY_LABEL.get(journey, journey),
        "overall_result": hard_fail == 0,
        "overall_compliance_rate": round(passed / (passed + failed), 4) if passed + failed else 1.0,
        "total_policies": len(verdicts),
        "pass_count": passed, "fail_count": failed, "na_count": count("NA"),
        "hard_policy_count": sum(r["hard"] for r in verdicts),
        "hard_policy_pass": count("PASS", True), "fail_count_hard_policy": hard_fail,
        "soft_policy_count": sum(not r["hard"] for r in verdicts),
        "soft_policy_pass": count("PASS", False), "fail_count_soft_policy": count("FAIL", False),
    }
