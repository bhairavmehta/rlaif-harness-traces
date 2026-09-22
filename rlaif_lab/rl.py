"""Reinforcement-learning internals: the engine LEARNS which harness changes to apply.

Formulation (mirrors the DPO/GRPO deck, in runnable form):

  State/context : pinned minibatch of synthesized episodes
  Action a      : a lever configuration a ∈ {0,1}^5  (taxonomy, metadata,
                  sequencing, formatter, prompt_hat)
  Policy π_θ    : factored Bernoulli — π_θ(a) = Π σ(θ_i)^a_i (1-σ(θ_i))^(1-a_i)
                  θ starts at the REFERENCE policy (all levers 50/50).
  Reward r(a)   : judge's avg weighted score on the replayed minibatch,
                  MINUS a hard-gate penalty (policy violations shape reward).
  ∇ log π_θ(a)  = a - σ(θ)   (per lever)  — the score function.

Three updaters, all logging their internals every iteration:

  GRPO-style (online, critic-free):
      sample a GROUP of G configs, reward each, Â_g = (r_g - mean r)/std r,
      θ ← θ + lr · mean_g [ Â_g · (a_g - σ(θ)) ]  -  β_kl · (σ(θ) - σ(θ_ref))
      (group mean is the baseline — no value network; β_kl is the KL leash.)

  DPO-style (offline, pairwise):
      from each group take (a_w = best, a_l = worst)  →  preference pair;
      s = β · [ (log π_θ(a_w) - log π_ref(a_w)) - (log π_θ(a_l) - log π_ref(a_l)) ]
      θ ← θ + lr · σ(-s) · β · (a_w - a_l)
      (σ(-s) up-weights exactly the pairs the implicit reward ranks wrong;
       implicit reward r̂(a) = β · log[π_θ(a)/π_ref(a)] — no reward network.)

  UCB1 flat bandit over all 32 configs — the unstructured baseline that shows
  why a factored policy (structure!) learns faster.
"""
from __future__ import annotations
import json, math, random, statistics as st
from dataclasses import dataclass, field
from .registry import LEVERS
from .engine import replay

sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))


def _reward(episodes, config: set[str]) -> tuple[float, dict]:
    run = replay(episodes, config)
    w = st.mean(v.weighted for _, _, v in run)
    gates = sum(bool(v.gate_violations) for _, _, v in run) / len(run)
    leaks = sum(v.reasoning["path_delta"]["output_leaks"] for _, _, v in run)
    return round(w - 1.0 * gates, 4), {"avg_weighted": round(w, 3),
                                       "gate_rate": round(gates, 3), "leaks": leaks}


def _logpi(theta, a):
    lp = 0.0
    for i in range(5):
        p = sigmoid(theta[i])
        lp += math.log(p if a[i] else (1 - p))
    return lp


@dataclass
class IterLog:
    it: int
    group: list = field(default_factory=list)      # per-sample internals
    mean_r: float = 0.0
    std_r: float = 0.0
    theta: list = field(default_factory=list)
    probs: dict = field(default_factory=dict)
    greedy_eval: dict = field(default_factory=dict)
    update_detail: dict = field(default_factory=dict)


def _greedy(theta):
    return {LEVERS[i] for i in range(5) if sigmoid(theta[i]) > 0.5}


def learn(episodes, algo="grpo", iters=24, group=8, minibatch=12, lr=0.8,
          beta=1.0, kl_coef=0.05, seed=11, verbose=True, eval_episodes=None):
    rng = random.Random(seed)
    theta = [0.0] * 5                         # reference policy: 50/50 each lever
    theta_ref = list(theta)
    eval_eps = eval_episodes or episodes
    logs: list[IterLog] = []
    pairs_emitted = []

    for it in range(1, iters + 1):
        mb = rng.sample(episodes, min(minibatch, len(episodes)))
        # ---- sample a GROUP of configs from π_θ ----
        samples = []
        for g in range(group):
            a = [1 if rng.random() < sigmoid(theta[i]) else 0 for i in range(5)]
            cfg = {LEVERS[i] for i in range(5) if a[i]}
            r, stats = _reward(mb, cfg)
            samples.append({"a": a, "config": sorted(cfg), "reward": r, **stats})
        rs = [s["reward"] for s in samples]
        mean_r, std_r = st.mean(rs), (st.pstdev(rs) or 1e-6)

        L = IterLog(it, samples, round(mean_r, 4), round(std_r, 4))

        if algo == "grpo":
            # group-relative advantages; REINFORCE step + KL leash
            grads = [0.0] * 5
            for s in samples:
                adv = (s["reward"] - mean_r) / std_r
                s["advantage"] = round(adv, 3)
                for i in range(5):
                    grads[i] += adv * (s["a"][i] - sigmoid(theta[i]))
            for i in range(5):
                grads[i] = grads[i] / group - kl_coef * (sigmoid(theta[i]) - sigmoid(theta_ref[i]))
                theta[i] += lr * grads[i]
            L.update_detail = {"rule": "θ += lr·mean[Â·(a−σ(θ))] − β_kl·(σ(θ)−σ(θ_ref))",
                               "grad": [round(g, 4) for g in grads]}
        elif algo == "dpo":
            # best vs worst of the group = one preference pair; pairwise update
            best = max(samples, key=lambda s: s["reward"])
            worst = min(samples, key=lambda s: s["reward"])
            aw, al = best["a"], worst["a"]
            s_margin = beta * ((_logpi(theta, aw) - _logpi(theta_ref, aw))
                               - (_logpi(theta, al) - _logpi(theta_ref, al)))
            wgt = sigmoid(-s_margin)          # largest when ranked WRONG
            for i in range(5):
                theta[i] += lr * wgt * beta * (aw[i] - al[i])
            pairs_emitted.append({"iter": it, "chosen": best["config"],
                                  "rejected": worst["config"],
                                  "r_chosen": best["reward"], "r_rejected": worst["reward"],
                                  "implicit_margin": round(s_margin, 4),
                                  "sigma_weight": round(wgt, 4)})
            L.update_detail = {"rule": "θ += lr·σ(−s)·β·(a_w − a_l)",
                               "pair": {"chosen": best["config"], "rejected": worst["config"]},
                               "implicit_reward_margin_s": round(s_margin, 4),
                               "sigma(−s)_weight": round(wgt, 4)}
        else:
            raise ValueError(algo)

        L.theta = [round(t, 3) for t in theta]
        L.probs = {LEVERS[i]: round(sigmoid(theta[i]), 3) for i in range(5)}
        gr, gstats = _reward(eval_eps, _greedy(theta))
        L.greedy_eval = {"config": sorted(_greedy(theta)), "reward": gr, **gstats}
        logs.append(L)
        if verbose:
            print(f"it {it:>2} | group r̄={mean_r:.3f}±{std_r:.3f} | "
                  f"π(lever)= " + " ".join(f"{k[:4]}:{v:.2f}" for k, v in L.probs.items())
                  + f" | greedy {L.greedy_eval['config']} → {gr:.3f}")

    return {"algo": algo, "iters": iters, "group": group,
            "final_probs": logs[-1].probs, "final_greedy": logs[-1].greedy_eval,
            "iterations": [L.__dict__ for L in logs],
            "preference_pairs": pairs_emitted}


def ucb(episodes, iters=48, minibatch=12, seed=11, verbose=True):
    """Flat UCB1 over all 32 lever subsets — the structure-free comparator."""
    rng = random.Random(seed)
    arms = []
    for m in range(32):
        arms.append({LEVERS[i] for i in range(5) if (m >> i) & 1})
    N = [0] * 32; Q = [0.0] * 32; t = 0; hist = []
    for it in range(1, iters + 1):
        t += 1
        ucb_v = [Q[i] + 2.0 * math.sqrt(math.log(t) / N[i]) if N[i] else float("inf")
                 for i in range(32)]
        i = max(range(32), key=lambda k: ucb_v[k])
        mb = rng.sample(episodes, min(minibatch, len(episodes)))
        r, _ = _reward(mb, arms[i])
        N[i] += 1; Q[i] += (r - Q[i]) / N[i]
        best = max(range(32), key=lambda k: Q[k] if N[k] else -9)
        hist.append({"it": it, "pulled": sorted(arms[i]), "r": r,
                     "best_arm": sorted(arms[best]), "best_q": round(Q[best], 3)})
        if verbose and it % 8 == 0:
            print(f"ucb it {it:>2} | best arm {sorted(arms[best])} Q={Q[best]:.3f}")
    return {"algo": "ucb1", "iterations": hist,
            "final_best": hist[-1]["best_arm"], "final_q": hist[-1]["best_q"]}
