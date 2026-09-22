"""Render a runtime Trace as the customer-facing transcript + tool evidence.

This is the view the action-completion judge prompt and the policy catalog
both reason over ("Turn 5: Agent says ..."). Every agent turn is derived from a
logged Step, so each policy's evidence can cite the exact turn that produced it,
and `step_turn` maps a step index back to its turn number.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

GENERIC_CLOSE = "Is there anything else I can help you with today?"


@dataclass
class Turn:
    turn: int
    speaker: str                       # CUSTOMER | AGENT
    text: str
    tools: list[dict] = field(default_factory=list)   # tool evidence surfaced in this turn
    tags: list[str] = field(default_factory=list)     # leak | reverify | generic_close | ...


@dataclass
class Transcript:
    turns: list[Turn]
    step_turn: dict[int, int]

    def first(self, tag: str) -> Turn | None:
        return next((t for t in self.turns if tag in t.tags), None)

    def tool_turn(self, tool: str) -> Turn | None:
        return next((t for t in self.turns if any(c["tool"] == tool for c in t.tools)), None)

    def to_json(self):
        return [{"turn": t.turn, "speaker": t.speaker, "text": t.text, "tools": t.tools,
                 "tags": t.tags, "repeat": getattr(t, "repeat", 1)} for t in self.turns]

    def as_text(self) -> str:
        out = []
        for t in self.turns:
            rep = getattr(t, "repeat", 1)
            out.append(f"Turn {t.turn} - {t.speaker}: {t.text}" + (f"  [repeated x{rep}]" if rep > 1 else ""))
            for c in t.tools:
                out.append(f"    [tool] {c['tool']}({json.dumps(c['args'])}) -> "
                           f"{'ok' if c['ok'] else 'ERROR'} {json.dumps(c['result'])}")
        return "\n".join(out)


def _template(tool: str, ok: bool, res: dict) -> str:
    if not ok:
        if tool == "submit_fee_waiver" and res.get("param") == "charge_id":
            return "I attempted to submit a fee waiver, but I'm unable to locate the required charge ID."
        return f"I wasn't able to complete that request ({res.get('error', 'error')})."
    if tool == "explain_bill":
        if "line_items" in res:
            li = res["line_items"][0]
            return (f"The ${li['amount']:.2f} charge ({li['charge_id']}) is for {li['desc'].split(' - ')[-1]}; "
                    f"the promotion credit was not applied this cycle.")
        return f"Your bill total is ${res['summary']['total']:.2f} across equipment and plan charges."
    return {
        "get_account_balance": f"Your account balance is ${res.get('balance', 0):.2f}.",
        "submit_fee_waiver": f"Done - the waiver is submitted and confirmed, reference {res.get('ref')}.",
        "check_paper_free_eligibility": "You're eligible for Paper-Free Billing and not yet enrolled.",
        "enroll_paper_free_billing": f"You're enrolled in Paper-Free Billing - confirmation {res.get('ref')}.",
        "check_ddc_eligibility": "No pending orders and no date change in the last 31 days - your new date can be the 3rd to the 25th.",
        "change_due_date": f"Done - your due date is now the {res.get('new_date')}th, confirmation {res.get('ref')}.",
        "payment_arrangement": "I can split this month's balance into two installments.",
        "manage_bill_preferences": "I've texted you a copy of your latest bill.",
        "transfer_to_agent": "I'm connecting you with a specialist who can complete this - your details carry over.",
        "get_prior_tickets": "I can see you've contacted us about this before - let me pick up where we left off rather than start over.",
        "get_usage_guidance": "I've pulled up what's driving the usage and some concrete steps that can bring it down.",
        "submit_plan_upgrade": "Your plan is updated - confirmation on its way.",
    }.get(tool, "Done.")


def render(ep, trace) -> Transcript:
    turns: list[Turn] = [Turn(1, "CUSTOMER", ep.utterance)]
    step_turn: dict[int, int] = {}
    pending: list[dict] = []           # tool calls not yet surfaced in an agent turn
    last: dict[str, tuple[bool, dict]] = {}

    def agent(text, *tags):
        prev = turns[-1]
        if prev.speaker == "AGENT" and prev.text == text:      # a loop: fold into one turn
            prev.tools += pending
            prev.tags = sorted(set(prev.tags) | set(tags) | {"repeated"})
            prev.repeat = getattr(prev, "repeat", 1) + 1
            prev.text = text
            pending.clear()
            return prev
        t = Turn(len(turns) + 1, "AGENT", text, list(pending), list(tags))
        pending.clear()
        turns.append(t)
        return t

    def terms():
        agent("Before I switch you: no more paper bills, your bill arrives by email and text, "
              "effective next cycle. Do you accept these terms?", "terms")
        customer("I accept.")

    def disclosures():
        agent("Before I submit: you'll get one extra interim bill, and the new date applies to all "
              "lines. Do I have your permission to make this change?", "disclosures")
        customer("Yes, go ahead.")

    def customer(text):
        turns.append(Turn(len(turns) + 1, "CUSTOMER", text))

    for i, s in enumerate(trace.steps):
        d = s.detail
        if s.kind in ("call", "error"):
            pending.append({"tool": d["tool"], "args": d["args"], "ok": s.kind == "call", "result": d["result"]})
            last[d["tool"]] = (s.kind == "call", d["result"])
        elif s.kind == "gate":
            pending.append({"tool": d["tool"], "args": {"harness_gate": d["precondition"]},
                            "ok": True, "result": {"action": d["action"]}})
            if "terms_accepted" in d["precondition"]:
                terms()                        # the harness forces the T&C step the customer hears
            elif "disclosures" in d["precondition"]:
                disclosures()
        elif s.kind == "reverify":
            agent("For security, could you confirm the last four digits of your SSN?", "reverify")
            customer("I already verified at the start of the call.")
        elif s.kind == "leak":
            agent(d["rendered"], "leak")
        elif s.kind == "redact":
            pass   # the redacted draft itself is surfaced by the respond step that follows
        elif s.kind == "context":
            if d.get("attribute") == "RECURRING_CONTACT":
                agent("I can see this is a repeat issue - you've reached out twice in the last 90 days "
                      "about your bill, and I'm sorry it's still not resolved. Let me get to the bottom of it.",
                      "recurrence_ack")
        elif s.kind == "parallel":
            pending.append({"tool": "parallel_group", "args": {"tools": d["tools"]},
                            "ok": True, "result": {"dependency_validation": d["dependency_validation"]}})
        elif s.kind == "respond":
            if d.get("empathy"):
                agent("I'm really sorry you're having to contact us about this again - that's frustrating, "
                      "and I want to get it sorted properly this time.", "empathy_first")
            elif d.get("recurrence_ack"):
                agent("Here's the short version of what's going on: your data usage crossed the plan "
                      "allowance again, the same driver as your last two contacts.", "recurrence_ack", "root_cause")
            elif d.get("quick_wins") is not None:
                acts = "; ".join(f"{a} [{src}]" for a, src in zip(d["quick_wins"], d["sources"]))
                agent(f"Three quick wins, each from our knowledge base: {acts}. "
                      f"Together these typically save {d['savings']}.", "quick_wins", "labeled_estimate")
            elif d.get("generic_savings"):
                agent("You could probably save a lot by cutting back on your usage.", "generic_savings")
            elif d.get("plan_option") and d.get("deferred"):
                agent("There's also a plan that fits your usage for $10/month more - I won't change "
                      "anything unless you explicitly confirm you want it.", "plan_option", "deferred_upgrade")
            elif d.get("plan_option"):
                agent("If you'd rather not manage usage, a better-fitting plan is $10/month more. "
                      "Your choice - try the quick wins first, or I can tee up the plan change for "
                      "your explicit confirmation.", "plan_option", "customer_choice")
            elif d.get("confirm_intent"):
                agent("Just to confirm what you need - " + ", ".join(i.replace("_", " ") for i in ep.intents)
                      + ". Let me pull that up.", "confirm_intent")
            elif "text" in d:
                agent(d["text"], "balance_recital",
                      *(["ap17_redaction"] if "[REDACTED-AP17]" in d["text"] else []))
            elif d.get("terms_and_conditions"):
                terms()
            elif "disclosures" in d:
                disclosures()
            elif "discount" in d:
                agent("The $10 monthly discount requires BOTH Paper-Free Billing and AutoPay - I can connect "
                      "you to our payments specialist to activate AutoPay.", "discount_pitch")
            elif d.get("next_step") == "contextual":
                agent("Since your balance is due on the 28th, would you like a payment reminder by text?",
                      "contextual_next_step")
            elif d.get("template"):
                ok, res = last.get(d["tool"], (True, {}))
                nxt = trace.steps[i + 1] if i + 1 < len(trace.steps) else None
                if not (nxt and nxt.kind == "respond" and "text" in nxt.detail):   # the spoken text covers it
                    agent(_template(d["tool"], ok, res), "tool_response")
        step_turn[i] = len(turns)

    if pending:
        agent("One moment.", "tool_response")
    if trace.outcome != "handoff" and not any("contextual_next_step" in t.tags for t in turns):
        agent(GENERIC_CLOSE, "generic_close")
    return Transcript(turns, step_turn)
