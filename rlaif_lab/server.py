"""HTTP server for the RLAIF Harness Lab — stdlib only (http.server).

  GET  /                      interactive UI (web/index.html)
  GET  /healthz               liveness
  POST /api/chat              live chat: {vanilla:[msgs], improved:[msgs], levers} -> both sessions + deltas
  GET  /api/replay            ?journey=bex&levers=all&seed=7   one pinned episode, full internals
  GET  /api/compare           ?journey=bex&levers=all&seed=7   vanilla vs levers on the same episode
  GET  /api/cycle             ?n=60&seed=7                     nightly cycle + production evidence
  GET  /api/learn             ?algo=grpo&iters=20&group=8      RL updater internals per iteration
  GET  /api/traces                                             production trace RCA + judge health
  GET  /api/policies          ?journey=bex                     governed policy catalog
  GET  /api/prompt            ?journey=bex&levers=&seed=7      action-completion judge request
  GET  /api/levers                                             what each patch changes (before -> after)
  GET  /api/route             ?utterance=...&levers=taxonomy
  GET  /api/session                                            {evidence_unlocked, key_required}
  GET  /api/history                                            trace-vN versions (metrics, deltas) + commit log

Production evidence (CIDs, quoted verdict evidence) and the verbatim judge
prompt are client data: when RLAIF_ACCESS_KEY is set they are returned only
to requests carrying a matching X-Access-Key header; everyone else gets
aggregates. Synthetic replays, chat and RL internals are always open.
The pre-UI paths (/route, /replay, /cycle, /learn, /patches) remain as aliases.

  python -m rlaif_lab.server            # listens on $PORT (default 8000)
"""
from __future__ import annotations
import contextlib, hmac, io, json, mimetypes, os, traceback
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .registry import Registry, LEVERS, PATCHES
from .runtime import Runtime
from .judge import score, scorecard
from .llm import MockLLM
from . import synth, engine, rl, chat, policies, traces, action_completion, history
from .transcript import render

MAX_N, MAX_ITERS, MAX_GROUP = 200, 50, 16
JOURNEYS = ("bex", "pfb", "ddc", "gen", "rbc")
WEB = Path(__file__).parent / "web"
ACCESS_KEY = os.environ.get("RLAIF_ACCESS_KEY", "")


class BadRequest(ValueError):
    pass


def _levers(s) -> set[str]:
    if not s: return set()
    if isinstance(s, list): picked = set(s)
    elif s == "all": return set(LEVERS)
    else: picked = {x.strip() for x in s.split(",") if x.strip()}
    unknown = picked - set(LEVERS)
    if unknown:
        raise BadRequest(f"unknown levers {sorted(unknown)}; valid: {list(LEVERS)} or 'all'")
    return picked


def _int(p, key, default, lo, hi) -> int:
    try:
        v = int(p.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be an integer")
    if not lo <= v <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}")
    return v


def _float(p, key, default) -> float:
    try:
        return float(p.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number")


def _episode(p):
    journey = p.get("journey", "bex")
    if journey not in JOURNEYS:
        raise BadRequest(f"'journey' must be one of {list(JOURNEYS)}")
    seed = _int(p, "seed", 7, 0, 2**31 - 1)
    eps = [e for e in synth.generate(80, seed) if e.journey == journey]
    idx = _int(p, "episode", 0, 0, len(eps) - 1)
    return eps[idx]


def _replay_json(ep, levers):
    R = Registry(levers)
    tr = Runtime(R, MockLLM()).run(ep)
    v = score(ep, tr)
    return {"episode": {"id": ep.id, "journey": ep.journey, "persona": ep.persona, "utterance": ep.utterance,
                        "intents": ep.intents, "blueprint": ep.blueprint, "expected_route": ep.expected_route},
            "levers": sorted(R.levers), "route": tr.route,
            "steps": [{"kind": s.kind, "detail": s.detail} for s in tr.steps],
            "verdict": v.to_json(transcript=True), "scorecard": scorecard(ep, v)}


@lru_cache(maxsize=2)
def _production(evidence: bool) -> dict:
    return traces.analyze(traces.load(), evidence=evidence)


# ------------------------------------------------------------------ endpoints
def api_route(p, _):
    utt = p.get("utterance")
    if not utt:
        raise BadRequest("'utterance' is required")
    R = Registry(_levers(p.get("levers")))
    scores = MockLLM().score_options(utt, R.agent_options())
    best = max(scores, key=lambda k: (scores[k], k))
    return {"utterance": utt, "levers": sorted(R.levers), "scores": scores,
            "descriptions": {a.name: a.description for a in R.agents.values()},
            "transfer_to_agent": best if scores[best] > 0 else "general_care (fallback)"}


def api_replay(p, _):
    return _replay_json(_episode(p), _levers(p.get("levers")))


def api_compare(p, _):
    ep = _episode(p)
    a, b = _replay_json(ep, set()), _replay_json(ep, _levers(p.get("levers", "all")))
    return {"vanilla": a, "improved": b, "delta": chat.delta(a, b)}


def api_chat(p, _):
    def msgs(key):
        m = p.get(key) or []
        if not isinstance(m, list) or not all(isinstance(x, str) for x in m):
            raise BadRequest(f"'{key}' must be a list of strings")
        if len(m) > chat.MAX_MESSAGES:
            raise BadRequest(f"at most {chat.MAX_MESSAGES} messages per session")
        return m
    van = chat.run_session(msgs("vanilla"), set())
    imp = chat.run_session(msgs("improved"), _levers(p.get("levers", "all")))
    return {"vanilla": van, "improved": imp,
            "deltas": [chat.delta(a, b) for a, b in zip(van["flows"], imp["flows"])]}


def api_cycle(p, authed):
    n = _int(p, "n", 60, 1, MAX_N)
    seed = _int(p, "seed", 7, 0, 2**31 - 1)
    return engine.run_cycle(synth.generate(n, seed), out_dir=None, production=_production(authed))


def api_learn(p, _):
    algo = p.get("algo", "grpo")
    if algo not in ("grpo", "dpo", "ucb"):
        raise BadRequest("'algo' must be one of ['grpo', 'dpo', 'ucb']")
    n = _int(p, "n", 60, 1, MAX_N)
    seed = _int(p, "seed", 11, 0, 2**31 - 1)
    eps = synth.generate(n, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        if algo == "ucb":
            return rl.ucb(eps, iters=_int(p, "iters", 48, 1, MAX_ITERS * 2), seed=seed, verbose=False)
        return rl.learn(eps, algo=algo, iters=_int(p, "iters", 24, 1, MAX_ITERS),
                        group=_int(p, "group", 8, 2, MAX_GROUP),
                        lr=_float(p, "lr", 0.8), beta=_float(p, "beta", 1.0),
                        kl_coef=_float(p, "kl", 0.05), seed=seed, verbose=False)


def api_traces(p, authed):
    out = dict(_production(authed))
    if p.get("records") == "1":
        out["record_list"] = [r.to_json(evidence=authed) for r in traces.load()]
    return out


def api_policies(p, _):
    j = p.get("journey")
    if j and j not in policies.JOURNEY_POLICIES:
        raise BadRequest(f"'journey' must be one of {list(policies.JOURNEY_POLICIES)}")
    return {"journeys": {k: policies.JOURNEY_LABEL[k] for k in policies.JOURNEY_POLICIES},
            "policies": policies.catalog(j)}


def api_prompt(p, authed):
    ep = _episode(p)
    tr = Runtime(Registry(_levers(p.get("levers"))), MockLLM()).run(ep)
    tx = render(ep, tr)
    out = {"true_categories": action_completion.TRUE_CATEGORIES,
           "false_categories": action_completion.FALSE_CATEGORIES,
           "containment_levels": action_completion.CONTAINMENT_LEVELS,
           "conversation": tx.as_text(),
           "rule_based_result": action_completion.evaluate(ep, tr, tx),
           "prompt_unlocked": authed}
    if authed:
        out["request"] = action_completion.render_request(ep, tr, tx)
    return out


def api_levers(_, __):
    base = Registry(set())
    out = {}
    for lv in LEVERS:
        R, changes = Registry({lv}), []
        for name, a in R.agents.items():
            for f in ("description", "triggers"):
                if getattr(a, f) != getattr(base.agents[name], f):
                    changes.append({"kind": "agent", "name": name, "field": f,
                                    "before": getattr(base.agents[name], f), "after": getattr(a, f)})
        for name, t in R.tools.items():
            for f in ("description", "triggers", "required", "preconditions", "output_contract"):
                if getattr(t, f) != getattr(base.tools[name], f):
                    changes.append({"kind": "tool", "name": name, "field": f,
                                    "before": getattr(base.tools[name], f), "after": getattr(t, f)})
        if lv == "sequencing":
            changes += [{"kind": "phases", "name": j, "field": "allowed_function_names",
                         "before": "all agent tools", "after": ph} for j, ph in PATCHES[lv]["phases"].items()]
        if lv == "prompt_hat":
            changes.append({"kind": "behaviors", "name": "instruction layer", "field": "behaviors",
                            "before": [], "after": sorted(R.behaviors)})
        if lv == "context":
            p = PATCHES[lv]
            for name, attr in p["attributes"].items():
                changes.append({"kind": "context attribute", "name": name, "field": "promoted from",
                                "before": attr["promoted_from"] + " (soft flag, not planner-visible)",
                                "after": f"{name} (explicit, versioned {attr['version']})"})
            lb = p["episodic_lookback_days"]
            changes.append({"kind": "context attribute", "name": "episodic lookback",
                            "field": "lookback_days", "before": "hard-coded, unversioned",
                            "after": f"{lb['value']} days · {lb['config_id']} · configurable"})
        if lv == "evidence":
            p = PATCHES[lv]
            changes += [
                {"kind": "retrieval rule", "name": "grounded actions", "field": "max_grounded_actions",
                 "before": "unbounded, ungrounded suggestions", "after": f"at most {p['max_grounded_actions']}, each KB-sourced"},
                {"kind": "retrieval rule", "name": "savings amounts", "field": "require_source_backed_amounts",
                 "before": "generic savings claims, no source", "after": "amounts must cite a KB source"},
                {"kind": "retrieval rule", "name": "estimates", "field": "label_estimates_as_estimates",
                 "before": "stated as fact", "after": "labelled as estimates"}]
        owned = [pid for pid, pol in policies.CATALOG.items() if pol.lever == lv]
        out[lv] = {"id": PATCHES[lv]["id"], "target": PATCHES[lv]["target"], "changes": changes,
                   "owns_policies": owned}
    return out


def api_patches(_, __):
    return {k: {kk: vv for kk, vv in v.items() if kk in ("id", "target")} for k, v in PATCHES.items()}


def api_session(_, authed):
    return {"key_required": bool(ACCESS_KEY), "evidence_unlocked": authed, "levers": list(LEVERS),
            "journeys": {j: policies.JOURNEY_LABEL[j] for j in JOURNEYS}}


def api_history(_, __):
    return history.load()


ROUTES = {"/api/route": api_route, "/api/replay": api_replay, "/api/compare": api_compare,
          "/api/chat": api_chat, "/api/cycle": api_cycle, "/api/learn": api_learn,
          "/api/traces": api_traces, "/api/policies": api_policies, "/api/prompt": api_prompt,
          "/api/levers": api_levers, "/api/patches": api_patches, "/api/session": api_session,
          "/api/history": api_history,
          "/healthz": lambda _, __: {"status": "ok"}, "/api/healthz": lambda _, __: {"status": "ok"}}
for _legacy in ("route", "replay", "cycle", "learn", "patches"):
    ROUTES[f"/{_legacy}"] = ROUTES[f"/api/{_legacy}"]
ROUTES["/api"] = lambda _, __: {"service": "rlaif_lab", "levers": list(LEVERS),
                                "endpoints": sorted(k for k in ROUTES if k.startswith("/api/"))}


class Handler(BaseHTTPRequestHandler):
    server_version = "rlaif_lab"

    def _send(self, code, body, ctype="application/json", cache="no-store"):
        data = body if isinstance(body, bytes) else json.dumps(body, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _authed(self) -> bool:
        if not ACCESS_KEY:
            return True
        return hmac.compare_digest(self.headers.get("X-Access-Key", ""), ACCESS_KEY)

    def _static(self, path) -> bool:
        rel = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        f = (WEB / rel).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            return False
        self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream",
                   cache="no-cache")
        return True

    def _dispatch(self, params):
        path = urlparse(self.path).path.rstrip("/") or "/"
        fn = ROUTES.get(path)
        if fn is None:
            if self.command == "GET" and self._static(path):
                return
            return self._send(404, {"error": f"no such endpoint {path}"})
        try:
            self._send(200, fn(params, self._authed()))
        except BadRequest as e:
            self._send(400, {"error": str(e)})
        except Exception:
            traceback.print_exc()
            self._send(500, {"error": "internal error"})

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        self._dispatch({k: v[-1] for k, v in q.items()})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > 64_000:
            return self._send(413, {"error": "body too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._send(400, {"error": "body must be a JSON object"})
        self._dispatch(body)


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"rlaif_lab listening on :{port}  (evidence key {'set' if ACCESS_KEY else 'not set'})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
