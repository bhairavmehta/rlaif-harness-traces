"""Live chat over the real engine — the same runtime, judge and policies, one message at a time.

Stateless by design (safe across replicas): the client sends every customer
message so far; the session is replayed deterministically from scratch.

Each new request is labeled by a lever-INDEPENDENT intent oracle (the ground
truth, like the synthetic blueprint) and run through Runtime(Registry(levers)).
The planned trajectory is revealed turn by turn and PAUSES wherever the agent
needs the customer: re-verification, T&C acceptance, disclosure consent. The
customer's real reply resumes it — or declines, which the judge scores as
INFORMED_REFUSAL. Vanilla and RLAIF sessions see identical messages, so the
only difference between the two panes is the harness levers.
"""
from __future__ import annotations
import random, re
from .registry import Registry, LEVERS
from .runtime import Runtime
from .llm import MockLLM
from .judge import score, scorecard
from . import synth

MAX_MESSAGES, MAX_CHARS = 30, 500
WAIT_TAGS = {"reverify": "verification", "terms": "consent", "disclosures": "consent"}

_AFFIRM = re.compile(r"^\s*(yes|yeah|yep|yup|sure|ok(ay)?|i accept|accept(ed)?|go ahead|please do|do it|correct|"
                     r"that'?s right|confirm(ed)?|sounds good|agreed|y)\b", re.I)
_DECLINE = re.compile(r"^\s*(no|nope|nah|don'?t|do not|cancel|stop|never ?mind|i decline|not now|decline)\b", re.I)
_DIGITS = re.compile(r"\b\d[\d\s-]{2,6}\d\b")
_THANKS = re.compile(r"\b(thanks|thank you|bye|goodbye|that'?s (all|everything)|that is all|all set)\b", re.I)
_LEX = {
    "human": r"\b(person|human|representative|rep|agent|operator|someone real|live (agent|person))\b",
    "device": r"\b(new phone|order (a|my|another)|upgrade|buy a (new )?phone|need a (new )?phone)\b",
    "ddc": r"\b(due date|bill date|billing date|paycheck|payment date|move my (bill|payment)|(the )?\d{1,2}(st|nd|rd|th)\b)",
    "pfb": r"\b(paperless|paper[- ]free|paper bills?|stop (getting )?paper|e-?bill(ing)?)\b",
    "bex": r"(\$\s?\d|\bcharge|\bfee\b|dispute|waive|should be free|take it off|remove it|higher|went up|"
           r"\bfactura|explain (my|the|this) bill|why is my bill)",
    "discount": r"\bdiscount",
}


def label(msg: str) -> tuple[str, list[str]] | None:
    """Ground-truth intent oracle (lever-independent) -> (journey, intents) or None."""
    hit = {k: re.search(rx, msg, re.I) for k, rx in _LEX.items()}
    if hit["human"] or hit["device"]:
        intents = (["human_request"] if hit["human"] else []) + (["charge_question"] if hit["bex"] else []) \
                  + (["device_order"] if hit["device"] else [])
        return "gen", intents
    found = sorted((m.start(), j) for j, m in hit.items() if m and j in ("ddc", "pfb", "bex"))
    if not found:
        return ("pfb", ["paperless_enroll", "discount"]) if hit["discount"] else None
    j = found[0][1]
    return j, {"bex": ["charge_dispute"], "ddc": ["due_date_change"],
               "pfb": ["paperless_enroll"] + (["discount"] if hit["discount"] else [])}[j]


def _flow(msg, journey, intents, levers, n, seed):
    ep = synth.Episode(id=f"chat-{n:02d}", journey=journey, persona="live_user", utterance=msg,
                       intents=intents, blueprint=list(synth.BLUEPRINTS[journey]),
                       expected_route=synth.EXPECTED_ROUTE[journey],
                       account=synth._account(random.Random(seed)))
    trace = Runtime(Registry(levers), MockLLM()).run(ep)
    v = score(ep, trace)
    return {"id": ep.id, "ep": ep, "trace": trace, "verdict": v, "pos": 1,
            "waiting": None, "status": "active", "declined_at": None}


def _reveal(flow, out):
    turns = flow["verdict"].transcript
    i = flow["pos"]
    while i < len(turns):
        t = turns[i]
        i += 1
        if t["speaker"] == "CUSTOMER":        # placeholder for a reply the real user gives instead
            continue
        out.append({**t, "flow": flow["id"], "planned_turn": t["turn"]})
        tag = next((g for g in t["tags"] if g in WAIT_TAGS), None)
        if tag:
            flow["waiting"], flow["pos"] = WAIT_TAGS[tag], i
            return
    flow["waiting"], flow["pos"], flow["status"] = None, i, "done"


def run_session(messages: list[str], levers: set[str], seed: int = 7) -> dict:
    R = Registry(levers)
    helpful = bool({"taxonomy", "prompt_hat"} & R.levers)
    out: list[dict] = []
    flows: list[dict] = []
    active = None
    say = lambda text, *tags: out.append({"speaker": "AGENT", "text": text, "tags": list(tags),
                                          "tools": [], "flow": active["id"] if active else None})

    for n, raw in enumerate(messages[:MAX_MESSAGES]):
        msg = raw.strip()[:MAX_CHARS]
        out.append({"speaker": "CUSTOMER", "text": msg, "tags": [], "tools": [], "flow": None})
        labeled = label(msg)
        is_reply = bool(_AFFIRM.match(msg) or _DECLINE.match(msg) or _DIGITS.search(msg))

        if active and active["waiting"] and not (labeled and not is_reply):
            if _DECLINE.match(msg):
                turn_no = len(out)
                if active["waiting"] == "consent":
                    say("Understood - I haven't made any changes. " +
                        ("If you'd like, I can explain the terms again or help with something else." if helpful
                         else GENERIC), "declined")
                    active["declined_at"] = turn_no
                    active["verdict"] = score(active["ep"], active["trace"], declined_at=turn_no)
                else:
                    say("I can't open account details without verification. " + GENERIC, "declined")
                active.update(status="declined", waiting=None)
                active = None
            elif _AFFIRM.match(msg) or (active["waiting"] == "verification" and _DIGITS.search(msg)):
                _reveal(active, out)
                if active["status"] == "done":
                    active = None
            else:
                say("Sorry, I need a quick answer to continue - " +
                    ("yes or no?" if active["waiting"] == "consent" else "please confirm the last four digits of your SSN."),
                    "reprompt")
            continue

        if labeled and not (is_reply and not re.search(r"[a-z]{4,}", _AFFIRM.sub("", msg), re.I)):
            if active:
                active["status"] = "abandoned"
            active = _flow(msg, labeled[0], labeled[1], R.levers, len(flows), seed)
            flows.append(active)
            _reveal(active, out)
            if active["status"] == "done":
                active = None
        elif _THANKS.match(msg) or _THANKS.search(msg) or _AFFIRM.match(msg):
            say("You're all set. Would you like a payment reminder before your bill is due on the 28th?"
                if "prompt_hat" in R.levers else GENERIC, "close")
        else:
            say("I can explain a specific charge, change your due date, set up paperless billing, or connect "
                "you with a specialist - what would you like to do?" if helpful
                else "I can help with billing. Your account balance is $500.47.", "unrecognized")

    return {
        "levers": sorted(R.levers),
        "turns": [{**t, "n": i + 1} for i, t in enumerate(out)],
        "waiting": active["waiting"] if active else None,
        "flows": [_flow_json(f) for f in flows],
    }


GENERIC = "Is there anything else I can help you with today?"


def _flow_json(f) -> dict:
    ep, tr, v = f["ep"], f["trace"], f["verdict"]
    return {"id": f["id"], "journey": ep.journey, "intents": ep.intents, "utterance": ep.utterance,
            "status": f["status"], "waiting": f["waiting"], "declined_at": f["declined_at"],
            "expected_route": ep.expected_route, "route": tr.route,
            "steps": [{"kind": s.kind, "detail": s.detail} for s in tr.steps],
            "verdict": v.to_json(transcript=True), "scorecard": scorecard(ep, v)}


def delta(a: dict, b: dict) -> dict:
    """What the levers changed for the same request: vanilla flow a -> improved flow b."""
    va, vb = a["verdict"], b["verdict"]
    pa = {r["policy_id"]: r["verdict"] for r in va["policy_verdicts"]}
    flips = [{"policy_id": r["policy_id"], "title": r["title"], "severity": r["severity"],
              "from": pa.get(r["policy_id"]), "to": r["verdict"]}
             for r in vb["policy_verdicts"] if pa.get(r["policy_id"]) != r["verdict"]]
    return {
        "weighted": round(vb["weighted"] - va["weighted"], 2),
        "dims": {k: vb["dims"][k] - va["dims"][k] for k in va["dims"]},
        "compliance": round(vb["compliance"]["overall_compliance_rate"] - va["compliance"]["overall_compliance_rate"], 4),
        "recommendation": [va["recommendation"], vb["recommendation"]],
        "action_completion": [va["action_completion"]["category"], vb["action_completion"]["category"]],
        "route": [a["route"], b["route"]],
        "tool_calls": [len([s for s in a["steps"] if s["kind"] in ("call", "error")]),
                       len([s for s in b["steps"] if s["kind"] in ("call", "error")])],
        "gates_cleared": [g.split(":")[0] for g in va["gate_violations"]
                          if g.split(":")[0] not in {x.split(":")[0] for x in vb["gate_violations"]}],
        "policy_flips": flips,
        "preference_pair": {"chosen": "improved" if vb["weighted"] >= va["weighted"] else "vanilla",
                            "margin": round(abs(vb["weighted"] - va["weighted"]), 2)},
    }


def compare(messages: list[str], levers: set[str] | None = None, seed: int = 7) -> dict:
    levers = set(LEVERS) if levers is None else levers
    van, imp = run_session(messages, set(), seed), run_session(messages, levers, seed)
    pairs = [delta(a, b) for a, b in zip(van["flows"], imp["flows"])]
    return {"vanilla": van, "improved": imp, "deltas": pairs}
