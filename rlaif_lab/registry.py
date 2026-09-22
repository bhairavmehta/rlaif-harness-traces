"""Versioned agent + tool registry.

Everything the model 'knows' about agents and tools lives here as data.
The five PATCHES are the harness levers: each is a named, reversible config
change. Registry(levers) resolves the effective metadata — the same object
the runtime routes/selects with, the judge audits, and the engine ablates.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import copy

LEVERS = ("taxonomy", "metadata", "sequencing", "formatter", "prompt_hat")


@dataclass
class AgentMeta:
    name: str
    description: str
    triggers: list[str]
    tools: list[str]


@dataclass
class ToolMeta:
    name: str
    description: str
    triggers: list[str]
    params: dict                       # JSON-schema-ish
    required: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)   # active only with 'sequencing'
    output_contract: str = "verbatim"                        # 'template' with 'formatter'


# --------------------------------------------------------------------------- #
# DATE-1 (vanilla) metadata — sparse, generic, the failure surface
# --------------------------------------------------------------------------- #
AGENTS_V1 = {
    "billing_explainer": AgentMeta(
        "billing_explainer", "Handles billing questions.",
        ["bill", "charge", "balance"],
        ["get_account_balance", "explain_bill", "submit_fee_waiver", "manage_bill_preferences", "transfer_to_agent"]),
    "due_date_changer": AgentMeta(
        "due_date_changer", "Handles payment dates.",
        ["due date"],
        ["check_ddc_eligibility", "change_due_date", "transfer_to_agent"]),
    "paperfree_manager": AgentMeta(
        "paperfree_manager", "Handles bill delivery preferences.",
        ["paperless enrollment"],
        ["check_paper_free_eligibility", "enroll_paper_free_billing", "manage_bill_preferences", "transfer_to_agent"]),
    "general_care": AgentMeta(
        "general_care", "Handles everything else, payments and general questions.",
        ["payment", "pay", "help", "paycheck"],
        ["get_account_balance", "payment_arrangement", "transfer_to_agent"]),
}

TOOLS_V1 = {
    "get_account_balance": ToolMeta(
        "get_account_balance", "Retrieves the account balance and charges.",
        ["balance", "charge", "bill", "how much"],
        {"account_id": {"type": "string"}}, ["account_id"]),
    "explain_bill": ToolMeta(
        "explain_bill", "Provides a summary of the customer's bill.",
        ["bill", "summary"],
        {"account_id": {"type": "string"},
         "detail_level": {"type": "string", "enum": ["summary", "full"]}},
        ["account_id"]),
    "submit_fee_waiver": ToolMeta(
        "submit_fee_waiver", "Submits a fee waiver.",
        ["waiver"],
        {"charge_id": {"type": "string"}}, ["charge_id"]),
    "manage_bill_preferences": ToolMeta(
        "manage_bill_preferences", "Manages bill delivery, sends paper bill copy by SMS.",
        ["paper bill", "bill copy", "paper bills", "copy"],
        {"account_id": {"type": "string"}, "action": {"type": "string", "enum": ["SEND_BILL_COPY"]}},
        ["account_id", "action"]),
    "check_paper_free_eligibility": ToolMeta(
        "check_paper_free_eligibility", "Checks eligibility.",
        [],                                               # <- missed tool: nothing to match
        {"account_id": {"type": "string"}}, ["account_id"]),
    "enroll_paper_free_billing": ToolMeta(
        "enroll_paper_free_billing", "Enrolls paper-free billing.",
        ["paperless", "paper-free", "go paperless", "stop getting paper"],
        {"account_id": {"type": "string"}, "terms_accepted": {"type": "boolean"}},
        ["account_id"]),
    "check_ddc_eligibility": ToolMeta(
        "check_ddc_eligibility", "Checks date change eligibility.",
        [],
        {"account_id": {"type": "string"}}, ["account_id"]),
    "change_due_date": ToolMeta(
        "change_due_date", "Updates billing date.",
        ["due date"],
        {"account_id": {"type": "string"}, "new_date": {"type": "integer"},
         "consent": {"type": "boolean"}},
        ["account_id", "new_date"]),
    "payment_arrangement": ToolMeta(
        "payment_arrangement", "Splits a balance into installments for customers who need help paying.",
        ["pay", "paycheck", "payment", "installment", "help paying", "date"],
        {"account_id": {"type": "string"}}, ["account_id"]),
    "transfer_to_agent": ToolMeta(
        "transfer_to_agent", "Transfers the conversation.",
        ["person", "human", "agent", "transfer"],
        {"target": {"type": "string"}}, ["target"]),
}

# --------------------------------------------------------------------------- #
# THE FIVE PATCHES (levers) — each a reversible, versioned config change
# --------------------------------------------------------------------------- #
PATCHES: dict[str, dict] = {
    "taxonomy": {
        "id": "PATCH-TAX-v11", "target": "agent descriptions / routing triggers",
        "agents": {
            "billing_explainer": dict(
                description=("Explains specific bill charges and line items — disputes "
                             "('charge should be free', 'bill went up'), fee waivers, credits. "
                             "Spanish billing inquiries ('mi factura'). NOT for date changes or paperless."),
                triggers=["bill", "charge", "equipment charge", "should be free", "went up",
                          "why is my bill", "dispute", "waive", "take it off", "remove",
                          "factura", "subio", "credit"]),
            "due_date_changer": dict(
                description=("Changes the bill due date — 'move my bill date', 'change my due date', "
                             "'paycheck moved'. Runs eligibility gates and disclosures. NOT hardship."),
                triggers=["due date", "bill date", "date changed", "change my bill", "move my",
                          "paycheck moved", "the 15th", "new date"]),
            "paperfree_manager": dict(
                description=("Enrolls/de-enrolls Paper-Free Billing — 'go paperless', 'stop paper bills', "
                             "'paperless discount'. Eligibility, T&C and consent. Bill COPIES are a "
                             "different task (billing_explainer)."),
                triggers=["paperless", "paper bills", "paper-free", "stop getting paper",
                          "go paperless", "discount"]),
            "general_care": dict(
                description=("Front door for multi-intent requests, human/agent requests, out-of-scope "
                             "routing (device orders → sales, AutoPay → payments). Keeps the intent queue."),
                triggers=["person", "human", "representative", "agent", "new phone", "order",
                          "several things", "and i need", "multiple"]),
        }},
    "metadata": {
        "id": "PATCH-META-v7", "target": "tool descriptions / triggers / params",
        "tools": {
            "explain_bill": dict(
                description=("Bill details. Use when the customer asks about a SPECIFIC charge, disputes "
                             "an amount, or asks why the bill changed. detail_level='full' returns "
                             "charge-level line items incl. charge_id — REQUIRED before any waiver."),
                triggers=["equipment charge", "should be free", "dispute", "take it off",
                          "why is my bill", "went up", "charge", "factura", "subio"],
                required=["account_id", "detail_level"]),
            "submit_fee_waiver": dict(
                description=("Fee waiver. Use when the customer disputes a specific charge and asks for "
                             "removal. charge_id COMES FROM explain_bill(detail_level='full') — never omit."),
                triggers=["waive", "take it off", "remove charge", "should be free", "dispute"]),
            "get_account_balance": dict(
                description=("Total amount owed ONLY. NOT for explaining or disputing a specific "
                             "charge — use explain_bill for that."),
                triggers=["total balance", "amount due", "how much do i owe"]),
            "check_paper_free_eligibility": dict(
                description=("ALWAYS the first step for 'go paperless' / 'stop paper bills' / paperless "
                             "discount. Must precede ENROLL_PAPER_FREE_BILLING."),
                triggers=["paperless", "go paperless", "stop getting paper", "paper bills", "discount"]),
            "manage_bill_preferences": dict(
                description="Sends a COPY of an existing bill by SMS. NOT paperless enrollment.",
                triggers=["bill copy", "send me a copy"]),
            "check_ddc_eligibility": dict(
                description="ALWAYS first for due-date changes: pending order + 31-day gates.",
                triggers=["due date", "bill date", "change my", "move my"]),
            "change_due_date": dict(
                description=("Changes the due date. Requires check_ddc_eligibility PASS, disclosures and "
                             "explicit consent. Valid new_date 3-25."),
                triggers=["due date", "bill date", "the 15th", "move my", "date changed"]),
        }},
    "sequencing": {
        "id": "PATCH-SEQ-v3", "target": "preconditions + per-phase allowed_function_names",
        "tools": {
            "submit_fee_waiver": dict(preconditions=["explain_bill.detail_level=='full'"]),
            "enroll_paper_free_billing": dict(preconditions=["check_paper_free_eligibility==PASS",
                                                             "terms_accepted==true"]),
            "change_due_date": dict(preconditions=["check_ddc_eligibility==PASS",
                                                   "disclosures_made&&consent==true"]),
        },
        "phases": {   # journey -> ordered phases -> allowed tools (mode=ANY analogue)
            "bex": [["explain_bill", "get_account_balance"], ["submit_fee_waiver", "transfer_to_agent"]],
            "pfb": [["check_paper_free_eligibility"], ["enroll_paper_free_billing"], ["transfer_to_agent"]],
            "ddc": [["check_ddc_eligibility"], ["change_due_date"], ["transfer_to_agent"]],
            "gen": [["get_account_balance", "explain_bill", "payment_arrangement", "transfer_to_agent"]],
        }},
    "formatter": {
        "id": "PATCH-FMT-v2", "target": "output contracts",
        "tools": {t: dict(output_contract="template")
                  for t in ("transfer_to_agent", "manage_bill_preferences", "enroll_paper_free_billing",
                            "submit_fee_waiver", "change_due_date")}},
    "prompt_hat": {
        "id": "PATCH-PROMPT-v14", "target": "instruction layer (behavioral)",
        "behaviors": ["confirm_intent_first", "carry_ivr_verification", "read_terms_get_consent",
                      "make_disclosures", "accurate_combined_discount", "route_out_of_scope",
                      "contextual_next_step", "hold_intent_queue"]},
}


class Registry:
    """Resolves effective agent/tool metadata for a set of applied levers."""

    def __init__(self, levers: set[str] | None = None):
        self.levers = set(levers or ())
        assert self.levers <= set(LEVERS), f"unknown levers: {self.levers - set(LEVERS)}"
        self.agents: dict[str, AgentMeta] = copy.deepcopy(AGENTS_V1)
        self.tools: dict[str, ToolMeta] = copy.deepcopy(TOOLS_V1)
        if "taxonomy" in self.levers:
            for a, upd in PATCHES["taxonomy"]["agents"].items():
                self.agents[a] = replace(self.agents[a], **upd)
        if "metadata" in self.levers:
            for t, upd in PATCHES["metadata"]["tools"].items():
                self.tools[t] = replace(self.tools[t], **upd)
        if "sequencing" in self.levers:
            for t, upd in PATCHES["sequencing"]["tools"].items():
                self.tools[t] = replace(self.tools[t], **upd)
        if "formatter" in self.levers:
            for t, upd in PATCHES["formatter"]["tools"].items():
                self.tools[t] = replace(self.tools[t], **upd)
        self.behaviors: set[str] = set(PATCHES["prompt_hat"]["behaviors"]) if "prompt_hat" in self.levers else set()

    def phases(self, journey: str) -> list[list[str]] | None:
        if "sequencing" in self.levers:
            return PATCHES["sequencing"]["phases"].get(journey)
        return None

    def agent_options(self) -> dict[str, list[str]]:
        return {a.name: a.triggers for a in self.agents.values()}

    def tool_options(self, names: list[str]) -> dict[str, list[str]]:
        return {n: self.tools[n].triggers for n in names}
