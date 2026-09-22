# RLAIF Harness Lab — Runnable Engine

A working implementation of the harness-optimization loop from the demo: a root
agent + 4 sub-agents, a versioned tool/agent metadata registry, a blueprint-first
synthetic episode generator, a replay runtime with phase gating and preconditions,
a judge that **measures** traces on the five weighted dimensions, and the nightly
cycle that does **per-lever credit assignment** and emits **preference pairs** and
**recommendation artifacts**. Pure Python 3.10+, stdlib only — no installs, no keys.

## Run it

```bash
python -m rlaif_lab.cli cycle  --n 60 --seed 7 --out artifacts
python -m rlaif_lab.cli route  "go paperless and give me the discount"
python -m rlaif_lab.cli route  "go paperless and give me the discount" --levers taxonomy
python -m rlaif_lab.cli replay --journey bex --levers all
python -m rlaif_lab.cli replay --journey bex            # vanilla failure path
python -m rlaif_lab.cli patches
```

Typical cycle output (measured, seed-reproducible):

```
[BASELINE] weighted 2.25  chosen 0%   resolved 7%   route-acc 27%  leaks 198  gates 45
[FULL]     weighted 4.35  chosen 100% resolved 100% route-acc 100% leaks 0    gates 0
[CREDIT]   sequencing +0.613 · taxonomy +0.340 · prompt_hat +0.200 · metadata +0.165 · formatter +0.066
[ARTIFACTS] 60 preference pairs · 9 recommendations -> artifacts/
```

## Versioned traces (git)

Every trace improvement is a commit + tag `trace-vN`, with metric deltas in the commit message
and a running log in [TRACE_HISTORY.md](TRACE_HISTORY.md):

```bash
python scripts/snapshot_traces.py -m "formatter: strip raw JSON from agent replies"   # add --push to publish
git log --oneline --decorate                    # improvement history
git diff trace-v1 trace-v2 -- artifacts/        # how the traces changed
```

> **Full technical guide:** [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) — architecture, every component,
> API reference, Azure operations, usage walkthrough, and how to interpret every result.

## Interactive UI & API

```bash
python -m rlaif_lab.server          # http://localhost:8000
```

The UI (`rlaif_lab/web/index.html`, served at `/`) has seven views:

1. **Live chat & internals** — type as the customer; the same message goes to the vanilla harness and the
   RLAIF harness (the levers selected in the top bar). Agents pause for verification / T&C / consent and
   wait for your reply. Toggle *Vanilla · Side by side · RLAIF* to see the conversation next to its RL
   internals: router scores, the decision path (phases, selections, precondition gates, calls, loops),
   five judge dimensions, action completion, every policy verdict, and what the levers changed.
2. **Pinned replay** — one synthesized episode, vanilla vs your lever set: the preference pair.
3. **Learning loop** — GRPO / DPO / UCB1 iterations: π<sub>θ</sub> per lever, rewards, group samples, advantages, update rule.
4. **Nightly cycle** — baseline → ablations → full; credit assignment, policy failures, recommendations with production evidence.
5. **Production traces** — the writeback verdict export, normalized, root-caused and mapped to levers, with judge health.
6. **Policies & judge prompt** — the governed catalog and the action-completion prompt applied to a replay.
7. **What each lever changes** — before/after metadata for every patch.

Deep links: `/?say=<msg>&say=yes&mode=improved&levers=taxonomy,metadata&tab=learn`.
`python -m rlaif_lab.cli chat --levers all` gives the same live chat in a terminal.

API (JSON): `POST /api/chat`, `GET /api/replay|compare|cycle|learn|traces|policies|prompt|levers|route|session`.

**Access control.** Call IDs, quoted production evidence and the verbatim judge prompt are client data.
When `RLAIF_ACCESS_KEY` is set they are returned only with a matching `X-Access-Key` header (the UI's
🔒 button); everyone else sees aggregates. Synthetic replays, chat and RL internals are open.

## Policies, judge prompt and production traces

| Module | Role |
|---|---|
| `policies.py` | Governed catalog — GEN-01..09 on every journey plus BEX-01..08, DDC-01..05, PFB-01/02/04, DIS-01..05 — with the severities and policy types of the production verdicts. Each policy has an evaluator over the replayed trace emitting the production verdict record (`policy_id, verdict, evidence "Turn N: …", reason, severity, policy_type`) and names the lever that owns a failure. **CRITICAL = hard gate.** |
| `transcript.py` | Renders a trace as customer-facing turns + tool evidence — what policies cite and what the LLM judge reads. Loops fold into one turn with a repeat count. |
| `prompts/action_completion.txt` | The production action-completion judge prompt, verbatim. |
| `action_completion.py` | `render_request()` builds the exact prompt + conversation for the LLM judge; `evaluate()` is a rule-anchored stand-in following the prompt's reasoning steps: final-state category (RESOLVED, LIVE_AGENT_HANDOFF, INFORMED_REFUSAL, REPETITIVE_LOOP, PROCESS_SKIP, INTENT_MISSED, …), containment quality judged on quality not frequency, and a 4–6 sentence rationale. |
| `data/traces.txt` | Redacted compliance-verdict export (writeback table). |
| `traces.py` | Tolerant ingest of that export (fragmented JSON, doubled quotes, OCR-mangled ids like `BEX-81`, `BBX-05`), per-policy PASS/FAIL/NA, a root-cause classifier (raw JSON leakage, generic close, intent missed, waiver not pursued, …) mapped to levers, and judge-health checks (FAIL labels whose own reason says "not triggered", severity drift, repaired / unresolvable ids). |
| `chat.py` | Stateless live chat over the real runtime: a lever-independent intent oracle labels each request, the planned trajectory is revealed turn by turn and pauses for the customer's real replies. |

`cycle` now attaches production evidence and policy-level lift to each recommendation. RCA buckets that
production shows but replay never exercises become `needs_replay_coverage` items rather than approvals
(e.g. GEN-01 fabrication, which sits outside the five-lever action space).

```bash
python -m rlaif_lab.cli traces
python -m rlaif_lab.cli policies --journey ddc
python -m rlaif_lab.cli judge-prompt --journey pfb --levers all [--full]
```

## Architecture

| Module        | Role |
|---------------|------|
| `llm.py`      | Backend protocol. `MockLLM` = deterministic phrase-overlap scorer (every decision explainable). `GeminiBackend` = stub showing where Vertex/ADK plugs in (FunctionDeclarations from the same registry; `tool_config mode=ANY` + `allowed_function_names`). |
| `registry.py` | All agent + tool metadata as data, plus the **five patches (levers)**: taxonomy, metadata, sequencing, formatter, prompt_hat. `Registry(levers)` resolves effective metadata — behavior changes through exactly one path. |
| `synth.py`    | Blueprint-first generator (APIGen-MT pattern): ground-truth tool trajectory against the real schemas first, then persona + noise (paraphrase, es-US code-switching, multi-intent). Backend-is-truth account state per episode. |
| `runtime.py`  | Route (description scores) → phased selection (`mode=ANY` analogue) → precondition gates → execution against the simulated billing backend → output contracts. Produces the auditable `Trace`. |
| `judge.py`    | Measures the trace: route correctness, blueprint coverage, argument errors, leaks, consent/T&C, re-verification, dropped intents → five weighted dims, verdict (Chosen ≥ 4.0 / Rejected < 3.0 / Human review), policy verdicts, hard gates, reasoning with path delta. |
| `engine.py`   | The cycle: baseline replay → **single-lever ablations (credit assignment)** → full replay → preference pairs (same pinned episode, only levers differ) → recommendations derived from measured baseline failures with measured lift. |
| `cli.py`      | `cycle`, `route`, `replay`, `patches`. |

## The learning loop — `learn` (RL internals, runnable)

```bash
python -m rlaif_lab.cli learn --algo grpo --iters 20 --group 8   # group-relative, critic-free
python -m rlaif_lab.cli learn --algo dpo  --iters 20 --group 8   # pairwise, implicit reward
python -m rlaif_lab.cli learn --algo ucb  --iters 48             # flat bandit comparator
```

`rl.py` implements a **factored Bernoulli policy over the five levers**
(π(a)=Πσ(θᵢ)ᵃⁱ(1−σ(θᵢ))¹⁻ᵃⁱ, θ initialized at the 50/50 reference policy) and
three updaters whose internals print every iteration:

- **GRPO-style** — sample a GROUP of G lever-configs, replay each on a pinned
  minibatch, reward = judge weighted − gate penalty, advantage Â=(r−mean)/std
  (the group mean IS the baseline — no critic), update
  θ ← θ + lr·mean[Â·(a−σ(θ))] − β_kl·(σ(θ)−σ(θ_ref)) (the KL leash).
- **DPO-style** — best-vs-worst of each group becomes a preference pair;
  update θ ← θ + lr·σ(−s)·β·(a_w−a_l) where s is the implicit-reward margin
  β·log-ratio vs the reference policy — σ(−s) up-weights pairs the policy
  currently ranks wrong. Pairs with margins are emitted as artifacts.
- **UCB1** over all 32 subsets — the structure-free comparator (needs ~32
  pulls just to find the best arm; the factored policy needs none).

Measured on seed 11: group reward climbs **2.73 → 4.12** in 20 GRPO iterations
while σ(θ) rises from 0.50 to 0.91–0.95 per lever — and **sequencing converges
fastest**, independently re-deriving the ablation's credit ranking from reward
alone (see `artifacts/learning_curves.png`). Greedy-policy eval hits the 4.35
all-lever ceiling with gate rate 0.

## Why this earns the word "reinforcement learning"

- **Action space** = the five versioned patches (harness levers).
- **Reward** = the judge's weighted delta on pinned replays; hard policy gates
  reject unsafe variants before scoring.
- **Credit assignment** = single-lever ablations on the same episodes — the
  cycle output ranks which lever earned how much.
- **Learning artifact** = chosen/rejected preference pairs with rationale —
  the dataset that later feeds DPO/preference tuning (Stage 2 of the staged route).
- **Governance** = every recommendation is `pending_approval` with a patch_ref
  and measured evidence; nothing self-applies.

## Wiring it to production (what changes, what doesn't)

1. Replace `MockLLM` with `GeminiBackend` (ADK root agent + sub_agents;
   selection via FunctionDeclarations built from `ToolMeta`). Registry, judge,
   engine, artifacts: unchanged.
2. Replace `synth.generate` episodes with redacted production traces (pinned
   replay) — same `Episode` shape.
3. Replace the rule-anchored judge with the calibrated LLM judge (QWK-gated
   against blind human labels) emitting the same `Verdict` schema.
4. Route `recommendations.json` into the approval workflow → canary → rollback.

## Artifacts written by `cycle`

- `cycle_summary.json` — baseline/full metrics + per-lever credit
- `preference_pairs.json` — chosen/rejected with path deltas and rationale
- `recommendations.json` — approval-ready patches with measured evidence/lift
- `verdicts_baseline.json` / `verdicts_full.json` — full judge output per episode (dims, policy verdicts, compliance metrics, action completion)
- `trace_rca.json` — production trace RCA (also `python -m rlaif_lab.cli traces`)

All data synthetic; policy IDs (BEX-01, PFB-01, DDC-02/03, GEN-02) mirror the
governed catalog used in the demo.
