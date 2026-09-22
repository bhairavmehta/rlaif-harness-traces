"""CLI for the RLAIF Harness Lab.

  python -m rlaif_lab.cli cycle   --n 60 --seed 7 --out artifacts
  python -m rlaif_lab.cli route   "go paperless and give me the discount" --levers taxonomy
  python -m rlaif_lab.cli replay  --journey pfb --levers metadata,sequencing,formatter
  python -m rlaif_lab.cli patches
  python -m rlaif_lab.cli traces                       # production trace RCA
  python -m rlaif_lab.cli policies --journey ddc
  python -m rlaif_lab.cli judge-prompt --journey pfb --levers all
  python -m rlaif_lab.cli chat --levers all            # interactive chat in the terminal
"""
from __future__ import annotations
import argparse, json
from .registry import Registry, LEVERS, PATCHES
from .runtime import Runtime
from .judge import score
from .llm import MockLLM
from .transcript import render
from . import synth, engine, rl, traces, policies, action_completion, chat


def _levers(s: str | None) -> set[str]:
    if not s or s == "none": return set()
    if s == "all": return set(LEVERS)
    return {x.strip() for x in s.split(",") if x.strip()}


def cmd_cycle(a):
    eps = synth.generate(a.n, a.seed)
    prod = traces.analyze(traces.load(a.traces)) if a.traces != "none" else None
    res = engine.run_cycle(eps, a.out, production=prod,
                           deployed=None if a.levers is None else _levers(a.levers))
    print(f"\n=== RLAIF CYCLE · {a.n} synthesized episodes (seed {a.seed}) ===")
    for name in ("baseline", "full"):
        s = res[name]
        print(f"\n[{name.upper()}]  weighted {s['avg_weighted']}  chosen {s['chosen_rate']:.0%}  "
              f"resolved {s['resolved_rate']:.0%}  route-acc {s['route_accuracy']:.0%}  "
              f"arg-errors {s['arg_error_calls']}  leaks {s['output_leaks']}  gates {s['gate_violations']}")
        print("   dims:", {k: v for k, v in s["avg_dims"].items()})
    print("\n[CREDIT ASSIGNMENT] avg weighted delta per single lever vs baseline:")
    for lv, d in sorted(res["credit_assignment_weighted_delta_per_lever"].items(),
                        key=lambda kv: -kv[1]):
        print(f"   {lv:<12} {d:+.3f}")
    print(f"\n[ARTIFACTS] preference pairs: {res['preference_pairs']}  "
          f"recommendations: {len(res['recommendations'])}"
          + (f"  -> written to {a.out}/" if a.out else ""))
    for r in res["recommendations"]:
        lift = f"lift={r['measured_lift']['credit_weighted_delta']:+.3f}" if r.get("measured_lift") else "lift=unmeasured"
        print(f"   {r['id']:<24} lever={str(r['lever']):<11} {lift:<16} {r['status']}")


def cmd_route(a):
    R = Registry(_levers(a.levers))
    scores = MockLLM().score_options(a.utterance, R.agent_options())
    best = max(scores, key=lambda k: (scores[k], k))
    print(json.dumps({"utterance": a.utterance, "levers": sorted(R.levers),
                      "scores": scores,
                      "transfer_to_agent": best if scores[best] > 0 else "general_care (fallback)"},
                     indent=2))


def cmd_replay(a):
    eps = [e for e in synth.generate(80, a.seed) if e.journey == a.journey]
    ep = eps[0]
    R = Registry(_levers(a.levers))
    tr = Runtime(R, MockLLM()).run(ep)
    v = score(ep, tr)
    print(f"\nEpisode {ep.id} [{ep.journey}] levers={sorted(R.levers) or 'none'}")
    print(f"utterance: {ep.utterance}\n")
    for s in tr.steps:
        print(f"  {s.kind:<8} {json.dumps(s.detail, default=str)[:150]}")
    print("\nTRANSCRIPT:\n" + render(ep, tr).as_text())
    print("\nVERDICT:", json.dumps(v.to_json(), indent=2))


def cmd_traces(a):
    res = traces.analyze(traces.load(a.path))
    print(f"\n=== PRODUCTION TRACES · {res['records']} calls · {res['verdicts_parsed']} verdicts ===")
    for p in res["policies"]:
        if p["FAIL"]:
            print(f"   {p['policy_id']:<7} fail_rate {p['fail_rate']:>5.0%}  ({p['FAIL']} FAIL / {p['PASS']} PASS)  lever={p['lever']}")
    print("\n[RCA BUCKETS]")
    for b in res["failure_buckets"]:
        print(f"   {b['bucket']:<30} {b['fail_verdicts']:>2} FAIL  {b['calls']:>2} calls  lever={b['lever']}  {b['policies']}")
    jh = res["judge_health"]
    print(f"\n[JUDGE HEALTH] contradictions {jh['verdict_reason_contradictions']} · ids repaired "
          f"{jh['policy_ids_repaired']} · unresolved {len(jh['unresolved_policy_ids'])}")
    if a.out:
        import os; os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(f"{a.out}/trace_rca.json", "w"), indent=2)
        print(f"[ARTIFACT] {a.out}/trace_rca.json")


def cmd_policies(a):
    for p in policies.catalog(a.journey):
        print(f"{p['id']:<7} {p['severity']:<8} {p['policy_type']:<16} lever={str(p['lever']):<11} {p['title']}")


def cmd_judge_prompt(a):
    ep = [e for e in synth.generate(80, a.seed) if e.journey == a.journey][0]
    tr = Runtime(Registry(_levers(a.levers)), MockLLM()).run(ep)
    tx = render(ep, tr)
    print(action_completion.render_request(ep, tr, tx) if a.full else tx.as_text())
    print("\nRULE-ANCHORED RESULT:", json.dumps(action_completion.evaluate(ep, tr, tx), indent=2))


def cmd_chat(a):
    msgs, R = [], _levers(a.levers)
    print(f"Chatting with levers={sorted(R) or 'none (vanilla)'} - blank line to quit.")
    shown = 0
    while True:
        try:
            m = input("\nYOU> ").strip()
        except EOFError:
            break
        if not m:
            break
        msgs.append(m)
        s = chat.run_session(msgs, R)
        for t in s["turns"][shown:]:
            if t["speaker"] == "AGENT":
                print(f"AGENT> {t['text']}" + (f"   [{', '.join(t['tags'])}]" if t["tags"] else ""))
        shown = len(s["turns"])
        if s["flows"]:
            v = s["flows"][-1]["verdict"]
            print(f"   judge {v['weighted']} {v['recommendation']} · {v['action_completion']['category']} · "
                  f"compliance {v['compliance']['overall_compliance_rate']:.0%}" + (f" · waiting for {s['waiting']}" if s["waiting"] else ""))


def cmd_learn(a):
    eps = synth.generate(a.n, a.seed)
    print(f"\n=== RL LEARNING · algo={a.algo} · {a.iters} iterations · group={a.group} ===")
    print("policy: factored Bernoulli over 5 levers, θ init = reference (all 0.50)\n")
    if a.algo == "ucb":
        res = rl.ucb(eps, iters=a.iters, seed=a.seed)
    else:
        res = rl.learn(eps, algo=a.algo, iters=a.iters, group=a.group,
                       lr=a.lr, beta=a.beta, kl_coef=a.kl, seed=a.seed)
        print(f"\n[FINAL POLICY] " + json.dumps(res["final_probs"]))
        print(f"[GREEDY CONFIG] {res['final_greedy']['config']} -> reward {res['final_greedy']['reward']}"
              f"  (avg weighted {res['final_greedy']['avg_weighted']}, gate rate {res['final_greedy']['gate_rate']})")
        if res["preference_pairs"]:
            print(f"[PAIRS EMITTED] {len(res['preference_pairs'])} (chosen vs rejected configs with implicit-reward margins)")
    if a.out:
        import os; os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(f"{a.out}/learning_{a.algo}.json", "w"), indent=2)
        print(f"[ARTIFACT] {a.out}/learning_{a.algo}.json")


def cmd_patches(_):
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("id", "target")}
                      for k, v in PATCHES.items()}, indent=2))


def main():
    ap = argparse.ArgumentParser(prog="rlaif_lab")
    sub = ap.add_subparsers(required=True)
    c = sub.add_parser("cycle"); c.add_argument("--n", type=int, default=60)
    c.add_argument("--seed", type=int, default=7); c.add_argument("--out", default="artifacts")
    c.add_argument("--traces", default=None, help="trace export path; default bundled; 'none' to skip")
    c.add_argument("--levers", default=None, help="deployed lever set for the 'full' side: all (default), none, or a,b,c")
    c.set_defaults(f=cmd_cycle)
    t = sub.add_parser("traces"); t.add_argument("--path"); t.add_argument("--out", default="artifacts")
    t.set_defaults(f=cmd_traces)
    po = sub.add_parser("policies"); po.add_argument("--journey", choices=list(policies.JOURNEY_POLICIES))
    po.set_defaults(f=cmd_policies)
    jp = sub.add_parser("judge-prompt"); jp.add_argument("--journey", default="bex")
    jp.add_argument("--levers"); jp.add_argument("--seed", type=int, default=7)
    jp.add_argument("--full", action="store_true", help="print the full prompt, not just the conversation")
    jp.set_defaults(f=cmd_judge_prompt)
    ch = sub.add_parser("chat"); ch.add_argument("--levers"); ch.set_defaults(f=cmd_chat)
    r = sub.add_parser("route"); r.add_argument("utterance"); r.add_argument("--levers")
    r.set_defaults(f=cmd_route)
    p = sub.add_parser("replay"); p.add_argument("--journey", default="bex")
    p.add_argument("--levers"); p.add_argument("--seed", type=int, default=7)
    p.set_defaults(f=cmd_replay)
    l = sub.add_parser("learn"); l.add_argument("--algo", choices=["grpo", "dpo", "ucb"], default="grpo")
    l.add_argument("--iters", type=int, default=24); l.add_argument("--group", type=int, default=8)
    l.add_argument("--n", type=int, default=60); l.add_argument("--seed", type=int, default=11)
    l.add_argument("--lr", type=float, default=0.8); l.add_argument("--beta", type=float, default=1.0)
    l.add_argument("--kl", type=float, default=0.05); l.add_argument("--out", default="artifacts")
    l.set_defaults(f=cmd_learn)
    q = sub.add_parser("patches"); q.set_defaults(f=cmd_patches)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
