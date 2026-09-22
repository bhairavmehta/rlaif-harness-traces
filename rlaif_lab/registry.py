"""Versioned agent + tool registry.

Everything the model 'knows' about agents and tools lives here as data.
The five PATCHES are the harness levers: each is a named, reversible config
change. Registry(levers) resolves the effective metadata — the same object
the runtime routes/selects with, the judge audits, and the engine ablates.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import copy

LEVERS = ("taxonomy", "metadata", "sequencing", "formatter", "prompt_hat", "context", "evidence")

# Harness-Optimization controls (Policies.xlsx, "Harness -Optimization" tab) — IMMUTABLE:
# the tool-selection threshold is never lowered ("improve metadata first"), and the
# CPNI/PII/security/regulatory hard gates, SLM weights, LLM model, RAG index and LTM
# schema are all frozen so any measured lift is attributable to the harness alone.
SELECTION_THRESHOLD = 0.61
FROZEN = {"selection_threshold": SELECTION_THRESHOLD, "hard_gates": "immutable",
          "slm_weights": "no_change", "llm_model": "no_change",
          "rag_index": "no_rebuild_query_context_only", "ltm_schema": "no_change"}


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
        ["get_account_balance", "explain_bill", "submit_fee_waiver", "manage_bill_preferences",
         "get_prior_tickets", "get_usage_guidance", "submit_plan_upgrade", "transfer_to_agent"]),
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
    "get_prior_tickets": ToolMeta(
        "get_prior_tickets", "Retrieve support ticket history.",   # <- description only (tab: Before)
        [],                                                        # <- nothing to match: selected at ~0.61
        {"account_id": {"type": "string"}, "lookback_days": {"type": "integer"}},
        ["account_id"]),
    "get_usage_guidance": ToolMeta(
        "get_usage_guidance", "Looks up knowledge base articles.",
        [],                                                        # <- guidance not consistently injected
        {"account_id": {"type": "string"}, "topic": {"type": "string"}},
        ["account_id"]),
    "submit_plan_upgrade": ToolMeta(
        "submit_plan_upgrade", "Changes the customer plan.",
        ["upgrade my plan"],
        {"account_id": {"type": "string"}, "plan_id": {"type": "string"},
         "customer_confirmed": {"type": "boolean"}},
        ["account_id", "plan_id", "customer_confirmed"]),          # <- confirmation requirement PRESERVED
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
        "id": "PATCH-TAX-v12", "target": "agent descriptions / routing triggers",
        # Tab row "Repeat-contact policy mapping": a CANDIDATE rule for recurring contacts —
        # differentiated handling without changing the base route (still billing_explainer).
        "candidate_policy_mapping": {"RBC-01": "recurring-contact handling on the bex route"},
        "agents": {
            "billing_explainer": dict(
                description=("Explains specific bill charges and line items — disputes "
                             "('charge should be free', 'bill went up'), fee waivers, credits, "
                             "and REPEAT billing complaints ('again', 'third time', 'nobody fixed it'). "
                             "Spanish billing inquiries ('mi factura'). NOT for date changes or paperless."),
                triggers=["bill", "charge", "equipment charge", "should be free", "went up",
                          "why is my bill", "dispute", "waive", "take it off", "remove",
                          "factura", "subio", "credit",
                          "again", "third time", "second time", "last time", "still high",
                          "nobody fixed", "keeps going up", "called before"]),
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
            # Tab row "Prior-ticket tool metadata": add signals_produced + use_cases so the
            # selector's confidence rises ABOVE the unchanged 0.61 threshold.
            "get_prior_tickets": dict(
                description=("Retrieve support ticket history within the configured lookback window. "
                             "signals_produced: ['recurrence_flag']. use_cases: ['repeat_billing']. "
                             "Use whenever the customer references a prior contact about the same issue."),
                triggers=["again", "last time", "third time", "second time", "still",
                          "nobody fixed", "called before", "keeps going up"],
                required=["account_id", "lookback_days"]),
            "get_usage_guidance": dict(
                description=("Retrieves up to three evidence-grounded data-management actions from the KB "
                             "(query/context requirements only — NO index rebuild). Use for high-bill root "
                             "causes driven by usage; every action carries its KB source."),
                triggers=["went up", "high", "keeps going up", "data", "usage", "save", "lower my bill"]),
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
        # Tab row "Tool execution order": the three independent reads may run as a parallel
        # group AFTER dependency validation (no read depends on another's output).
        "parallel_groups": {"rbc": [["explain_bill", "get_prior_tickets", "get_usage_guidance"]]},
        "phases": {   # journey -> ordered phases -> allowed tools (mode=ANY analogue)
            "rbc": [["explain_bill", "get_prior_tickets", "get_usage_guidance"],
                    ["submit_plan_upgrade", "transfer_to_agent"]],
            "bex": [["explain_bill", "get_account_balance"], ["submit_fee_waiver", "transfer_to_agent"]],
            "pfb": [["check_paper_free_eligibility"], ["enroll_paper_free_billing"], ["transfer_to_agent"]],
            "ddc": [["check_ddc_eligibility"], ["change_due_date"], ["transfer_to_agent"]],
            "gen": [["get_account_balance", "explain_bill", "payment_arrangement", "transfer_to_agent"]],
        }},
    "formatter": {
        "id": "PATCH-FMT-v2", "target": "output contracts",
        "tools": {t: dict(output_contract="template")
                  for t in ("transfer_to_agent", "manage_bill_preferences", "enroll_paper_free_billing",
                            "submit_fee_waiver", "change_due_date",
                            "get_prior_tickets", "get_usage_guidance", "submit_plan_upgrade")}},
    "prompt_hat": {
        "id": "PATCH-PROMPT-v15", "target": "instruction layer (behavioral)",
        "behaviors": ["confirm_intent_first", "carry_ivr_verification", "read_terms_get_consent",
                      "make_disclosures", "accurate_combined_discount", "route_out_of_scope",
                      "contextual_next_step", "hold_intent_queue",
                      # Tab rows "Response opening" / "Response structure": empathy + concise
                      # root-cause summary, then quick wins -> plan option -> customer choice.
                      "empathy_first_opening", "root_cause_summary", "structured_rbc_response",
                      # Tab row "Internal ticket IDs": the prompt explicitly forbids internal
                      # identifiers so the AP-17 output gate never has to auto-redact.
                      "forbid_internal_identifiers"]},
    # Tab row "Repeat-contact signal": promote the soft repeat_contact_context flag to an
    # explicit, VERSIONED context attribute the planner can consume (STM change only —
    # tab row "STM scratchpad": versioned context attributes; LTM schema untouched).
    "context": {
        "id": "PATCH-CTX-v1", "target": "STM / episodic-memory context attributes",
        "attributes": {"RECURRING_CONTACT": {"promoted_from": "repeat_contact_context",
                                             "version": "ctx-attr-v1"}},
        # Tab row "Lookback window": keep 90 days initially, but versioned + configurable
        # so later sensitivity tests need no code change.
        "episodic_lookback_days": {"value": 90, "config_id": "LOOKBACK-90d-v1", "configurable": True}},
    # Tab rows "Data-management guidance" / "Savings claims": RAG/evidence-fusion
    # requirements — query/context only, NO index rebuild.
    "evidence": {
        "id": "PATCH-EVD-v1", "target": "RAG query/context requirements + evidence fusion",
        "max_grounded_actions": 3,
        "require_source_backed_amounts": True,
        "label_estimates_as_estimates": True},
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
        # context lever: explicit versioned attributes + configurable lookback (default 90d either way)
        self.context_attrs: dict = PATCHES["context"]["attributes"] if "context" in self.levers else {}
        self.lookback_days: int = PATCHES["context"]["episodic_lookback_days"]["value"]
        self.lookback_versioned: bool = "context" in self.levers
        # evidence lever: RAG/evidence-fusion requirements (no index rebuild)
        self.evidence: dict | None = ({k: PATCHES["evidence"][k] for k in
                                       ("max_grounded_actions", "require_source_backed_amounts",
                                        "label_estimates_as_estimates")}
                                      if "evidence" in self.levers else None)

    def phases(self, journey: str) -> list[list[str]] | None:
        if "sequencing" in self.levers:
            return PATCHES["sequencing"]["phases"].get(journey)
        return None

    def parallel_groups(self, journey: str) -> list[list[str]]:
        if "sequencing" in self.levers:
            return PATCHES["sequencing"].get("parallel_groups", {}).get(journey, [])
        return []

    def agent_options(self) -> dict[str, list[str]]:
        return {a.name: a.triggers for a in self.agents.values()}

    def tool_options(self, names: list[str]) -> dict[str, list[str]]:
        return {n: self.tools[n].triggers for n in names}
