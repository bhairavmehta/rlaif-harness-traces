"""LLM backend abstraction.

The runtime never scores anything itself — it asks a backend to choose among
options described by metadata. MockLLM is a deterministic lexical scorer so the
whole loop runs offline and every decision is inspectable. GeminiBackend shows
exactly where real Vertex/ADK calls plug in (function declarations built from
the same registry, tool_config mode=ANY with allowed_function_names).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


class LLMBackend(Protocol):
    def score_options(self, utterance: str, options: dict[str, list[str]]) -> dict[str, float]:
        """Return a relevance score per option id given its trigger phrases."""
        ...


@dataclass
class MockLLM:
    """Deterministic stand-in: phrase-overlap scoring.

    Directionally faithful to the real mechanism — tool/agent metadata is
    serialized into the model's context and steers selection probability —
    while keeping every score reproducible and explainable in a demo.
    """

    def score_options(self, utterance: str, options: dict[str, list[str]]) -> dict[str, float]:
        u = utterance.lower()
        out: dict[str, float] = {}
        for oid, phrases in options.items():
            s = 0.0
            for p in phrases:
                p = p.lower().strip()
                if p and p in u:
                    s += len(p.split())          # longer matched phrase = stronger signal
            out[oid] = s
        return out


class GeminiBackend:
    """Sketch of the production backend (not executed in the offline demo).

    route():  root LlmAgent with sub_agents -> description-driven
              transfer_to_agent (ADK AutoFlow).
    select(): FunctionDeclarations built from ToolMeta; per-phase hard gating
              via tool_config = {function_calling_config: {mode: "ANY",
              allowed_function_names: [...]}}.
    """

    def __init__(self, model: str = "gemini-2.5-flash", project: str | None = None):
        self.model, self.project = model, project

    def score_options(self, utterance: str, options: dict[str, list[str]]) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError(
            "Wire to Vertex AI: build FunctionDeclarations from the registry and "
            "read the emitted function_call — the registry, judge and engine are "
            "backend-agnostic and unchanged."
        )


class GPT4oBackend:
    """Sketch of the OpenAI-hosted backend used by the RLAIF Plane for VARIANT
    GENERATION (one candidate trajectory per optimization dimension) at
    temperature 0.2 - low enough to stay faithful to the pinned context, high
    enough to explore phrasing. Like GeminiBackend, it is not executed in the
    offline demo; the registry, judge and engine are backend-agnostic."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2):
        self.model, self.temperature = model, temperature

    def score_options(self, utterance: str, options: dict[str, list[str]]) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError(
            "Wire to the OpenAI API: serialize tool/agent metadata as function "
            "definitions and read tool_calls from the response; use "
            "temperature=0.2 for variant generation in the RLAIF Plane."
        )
