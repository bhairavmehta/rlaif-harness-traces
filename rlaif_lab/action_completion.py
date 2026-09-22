"""Action-completion evaluator — the production judge prompt, runnable offline.

prompts/action_completion.txt is the production LLM-judge prompt verbatim. It
decides one boolean (was the customer's intended action completed?) with a
single best-matching category and a containment-quality judgment made on
QUALITY, never frequency. Two ways to use it here:

  render_request(ep, trace, tx) -> the exact prompt + transcript + tool evidence
                                   to send to the calibrated LLM judge (Gemini).
  evaluate(ep, trace, tx)       -> a rule-anchored stand-in that follows the
                                   prompt's <reasoning_steps> on the replayed
                                   trace, emitting the same output fields.
"""
from __future__ import annotations
from pathlib import Path
from .policies import containment_quality, _calls

PROMPT_PATH = Path(__file__).parent / "prompts" / "action_completion.txt"

TRUE_CATEGORIES = ["RESOLVED", "INFORMED_REFUSAL", "CUSTOMER_DEPARTED",
                   "LIVE_AGENT_HANDOFF", "SELF_SERVICE_REDIRECT"]
FALSE_CATEGORIES = ["POOR_CONTAINMENT", "TRUNCATED_RESPONSE", "REPETITIVE_LOOP",
                    "DATA_ACCESS_FAILURE", "INTENT_MISSED", "WRONG_INFORMATION",
                    "PROCESS_SKIP", "LANGUAGE_FAILURE", "CAPABILITY_GAP",
                    "SESSION_ENDED_UNRESOLVED"]
CONTAINMENT_LEVELS = ["strong", "adequate", "poor", "not_applicable"]


def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")


def render_request(ep, trace, tx) -> str:
    return (prompt_text().rstrip() + "\n\n<conversation>\n" + tx.as_text()
            + "\n</conversation>\n")


def _process_skips(ep, trace) -> list[str]:
    ok = {c["tool"] for c in _calls(trace, ok=True)}
    skips = []
    if "change_due_date" in ok and not trace.consent_given:
        skips.append("due date changed with no disclosures or explicit consent")
    if "enroll_paper_free_billing" in ok and not trace.terms_read:
        skips.append("paper-free enrollment without T&C acceptance")
    return skips


def evaluate(ep, trace, tx, declined_at: int | None = None) -> dict:
    ok = {c["tool"] for c in _calls(trace, ok=True)}
    errors = _calls(trace, ok=False)
    quality = containment_quality(ep, trace, tx)
    skips = _process_skips(ep, trace)
    misrouted = trace.route != ep.expected_route
    transferred = "transfer_to_agent" in {c["tool"] for c in trace.tool_calls}
    leak = tx.first("leak")
    risk = (["PROCESS_SKIP"] if skips else []) + (["OUTPUT_LEAK"] if leak else [])

    # (1) primary intent + turn-by-turn handling
    intent = ", ".join(i.replace("_", " ") for i in ep.intents)
    s1 = (f"The customer's primary intent was {intent}; the agent routed to {trace.route}"
          + (" (misrouted)" if misrouted else "") + f" and called {len(trace.tool_calls)} tool(s)"
          + (f", {len(errors)} of which failed" if errors else "") + ".")
    moments = []
    if tx.first("reverify"):
        moments.append(f"re-asked verification at Turn {tx.first('reverify').turn}")
    if tx.first("balance_recital"):
        moments.append(f"recited the balance instead of the named charge at Turn {tx.first('balance_recital').turn}")
    if leak:
        moments.append(f"exposed raw tool JSON at Turn {leak.turn}")
    if tx.first("disclosures") or tx.first("terms"):
        t = tx.first("disclosures") or tx.first("terms")
        moments.append(f"made the required disclosures and captured consent at Turn {t.turn}")
    s2 = ("Notable moments: the agent " + "; ".join(moments) + ".") if moments else \
         "The agent advanced the request without redundant turns."
    # (2) containment quality
    s3 = (f"Containment quality: {quality}" + (
        " - the agent offered new information before any hand-off." if quality in ("strong", "adequate")
        else " - turns repeated or deflected without adding value." if quality == "poor"
        else " - the customer never needed or asked for a human.") )

    # (3) final state -> single best-matching category
    if declined_at is not None:
        cat, result = "INFORMED_REFUSAL", True
        why = f"the customer declined at Turn {declined_at} after complete, correct information"
    elif trace.outcome == "resolved":
        cat, result = "RESOLVED", True
        why = "the request was fulfilled with a tool-confirmed outcome"
    elif trace.outcome == "handoff" or transferred:
        cat, result = "LIVE_AGENT_HANDOFF", True
        why = "a transfer tool was called - a platform-level or unresolved hand-off still counts as action complete"
    elif "human_request" in ep.intents and quality == "poor":
        cat, result = "POOR_CONTAINMENT", False
        why = "the customer asked for a human, containment was consistently low quality, and no transfer happened"
    elif tx.first("repeated") or sum(1 for c in errors if c["tool"] == "submit_fee_waiver") >= 2:
        cat, result = "REPETITIVE_LOOP", False
        rep = tx.first("repeated")
        why = (f"the agent repeated the same turn x{rep.repeat} from Turn {rep.turn} without advancing"
               if rep else "the agent retried the same malformed call without advancing")
    elif skips:
        cat, result = "PROCESS_SKIP", False
        why = "mandatory steps were skipped: " + "; ".join(skips)
    elif misrouted:
        cat, result = "INTENT_MISSED", False
        why = "the clearly stated request was misidentified and never fulfilled"
    elif errors:
        cat, result = "DATA_ACCESS_FAILURE", False
        why = "tool errors left the request unfulfilled"
    else:
        cat, result = "SESSION_ENDED_UNRESOLVED", False
        why = "the session ended without fulfilling or routing the request"
    s4 = f"Category {cat}: Action Completion is {str(result).upper()} because {why}."
    if risk and result:
        s4 += f" Risk noted: {', '.join(risk)}."
    return {"action_completion": result, "category": cat, "containment_quality": quality,
            "risk_flags": risk, "rationale": " ".join([s1, s2, s3, s4])}
