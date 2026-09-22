"""Blueprint-first synthetic episode generator.

APIGen-MT pattern: decide the ground-truth tool trajectory (blueprint) against
the REAL tool schemas first, then render the utterance with persona + noise
(paraphrase, es-US code-switching, multi-intent). Backend-is-truth: every
episode carries the account state the tools will answer from.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import random

PERSONAS = ["frustrated_repeat_caller", "polite_elder", "busy_multi_intent", "es_US_bilingual", "neutral"]

UTTER = {
    "bex": [
        "There's a $440 equipment charge on my bill for a device that was supposed to be free - explain it and take it off.",
        "Why is my bill higher this month? That equipment charge should be free. Remove it.",
        "You charged me $440 for a device that came with a promo. I dispute that - waive it.",
    ],
    "bex_es": ["Mi factura subio este mes - ese cargo de equipo should be free. Take it off."],
    "ddc": [
        "My paycheck moved - can I get my bill date changed to the 15th?",
        "I need my due date moved to the 15th, permanently.",
        "Change my bill date to the 15th please, my paycheck moved.",
    ],
    "pfb": [
        "I want to stop getting paper bills - go paperless - and get whatever discount comes with that.",
        "Go paperless on my account and give me the paperless discount.",
        "Stop the paper bills. And I heard there's a discount for paperless?",
    ],
    "gen": [
        "I want to talk to a person. Also what is this $9.99 charge, and I need a new phone.",
        "Get me a human. And explain the $9.99 charge. And I need to order a new phone.",
    ],
}

BLUEPRINTS = {
    "bex": ["explain_bill:full", "submit_fee_waiver"],
    "ddc": ["check_ddc_eligibility", "change_due_date"],
    "pfb": ["check_paper_free_eligibility", "enroll_paper_free_billing"],
    "gen": ["explain_bill:full", "transfer_to_agent"],
}
EXPECTED_ROUTE = {"bex": "billing_explainer", "ddc": "due_date_changer",
                  "pfb": "paperfree_manager", "gen": "general_care"}


@dataclass
class Episode:
    id: str
    journey: str
    persona: str
    utterance: str
    intents: list[str]
    blueprint: list[str]
    expected_route: str
    account: dict = field(default_factory=dict)


def _account(rng: random.Random) -> dict:
    return {
        "account_id": f"ACC-{rng.randint(1000, 9999)}",
        "balance": 500.47,
        "line_items": [{"charge_id": "CHG-88213", "desc": "EQUIP - Galaxy device",
                        "amount": 440.00, "promo_credit_applied": False},
                       {"charge_id": "CHG-11007", "desc": "Plan", "amount": 60.47,
                        "promo_credit_applied": True}],
        "waiver_eligible": ["CHG-88213"],
        "pfb_enrolled": False, "pfb_eligible": True, "autopay": False,
        "due_date": 28, "pending_order": False, "days_since_date_change": 90,
    }


def generate(n: int = 60, seed: int = 7) -> list[Episode]:
    rng = random.Random(seed)
    eps: list[Episode] = []
    journeys = ["bex", "ddc", "pfb", "gen"]
    for i in range(n):
        j = journeys[i % 4]
        persona = rng.choice(PERSONAS)
        if j == "bex" and (persona == "es_US_bilingual" or rng.random() < 0.12):
            utt = rng.choice(UTTER["bex_es"]); persona = "es_US_bilingual"
        else:
            utt = rng.choice(UTTER[j])
        intents = {"bex": ["charge_dispute"], "ddc": ["due_date_change"],
                   "pfb": ["paperless_enroll", "discount"],
                   "gen": ["human_request", "charge_question", "device_order"]}[j]
        eps.append(Episode(
            id=f"ep-{i:04d}", journey=j, persona=persona, utterance=utt,
            intents=intents, blueprint=list(BLUEPRINTS[j]),
            expected_route=EXPECTED_ROUTE[j], account=_account(rng)))
    return eps
