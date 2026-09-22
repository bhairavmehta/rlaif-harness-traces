"""Version the trace artifacts in git: one commit + tag per improvement.

  python scripts/snapshot_traces.py -m "formatter: strip raw JSON from agent replies"
  python scripts/snapshot_traces.py -m "..." --learn     # also regenerate GRPO/DPO/UCB curves
  python scripts/snapshot_traces.py -m "..." --push      # push commit + tag to origin
  python scripts/snapshot_traces.py -m "..." --levers sequencing,taxonomy   # deployed lever set

Each run regenerates artifacts/ (cycle + production trace RCA), diffs the headline
metrics against the previous commit, appends an entry to TRACE_HISTORY.md, commits
code + artifacts with the metric deltas in the commit message, and tags trace-vN.
`git log --oneline` then reads as the improvement history; `git diff trace-v1 trace-v2
-- artifacts/` shows exactly how the traces changed. Stdlib only.
"""
from __future__ import annotations
import argparse, datetime, json, os, subprocess, sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rlaif_lab import history  # noqa: E402  (metric definitions shared with the web UI)

ART = history.ART
HISTORY = "TRACE_HISTORY.md"


def sh(*args: str, check: bool = True) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    if check and r.returncode:
        sys.exit(f"$ {' '.join(args)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip() if r.returncode == 0 else ""


def load_current() -> dict:
    return {"cycle": json.load(open(os.path.join(ROOT, ART, "cycle_summary.json"), encoding="utf-8")),
            "rca": json.load(open(os.path.join(ROOT, ART, "trace_rca.json"), encoding="utf-8"))}


def metric_rows(cur: dict, prev: dict | None) -> list[tuple[str, str, str, str]]:
    rows = []
    for c in history.changes(cur, prev):
        fmt = history.FMT[c["unit"]]
        if c["before"] is None:
            rows.append((c["label"], "-", fmt.format(c["after"]), "new"))
            continue
        d = c["delta"]
        mark = "=" if not d else (f"{d:+d}" if c["unit"] == "int" else f"{d:+.3f}") + \
            {True: " ✅", False: " ⚠️", None: ""}[c["good"]]
        rows.append((c["label"], fmt.format(c["before"]), fmt.format(c["after"]), mark))
    return rows


def next_version() -> int:
    tags = sh("git", "tag", "--list", "trace-v*", check=False).split()
    nums = [int(t[len("trace-v"):]) for t in tags if t[len("trace-v"):].isdigit()]
    return max(nums, default=0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--message", required=True, help="what changed in this trace version")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--levers", default=None, help="deployed lever set: all (default), none, or a,b,c")
    ap.add_argument("--traces", default=None, help="trace export path (default: bundled data/traces.txt)")
    ap.add_argument("--learn", action="store_true", help="also regenerate learning_{grpo,dpo,ucb}.json")
    ap.add_argument("--no-run", action="store_true", help="commit artifacts/ as-is, don't regenerate")
    ap.add_argument("--push", action="store_true", help="git push the commit and tag to origin")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not a.no_run:
        py = [sys.executable, "-m", "rlaif_lab.cli"]
        tr = ["--traces", a.traces] if a.traces else []
        lv = ["--levers", a.levers] if a.levers else []
        sh(*py, "cycle", "--n", str(a.n), "--seed", str(a.seed), "--out", ART, *tr, *lv)
        sh(*py, "traces", "--out", ART, *(["--path", a.traces] if a.traces else []))
        if a.learn:
            for algo in ("grpo", "dpo", "ucb"):
                sh(*py, "learn", "--algo", algo, "--out", ART)

    if not sh("git", "status", "--porcelain", "--", ".", f":!{HISTORY}"):
        sys.exit("nothing changed since the last trace version — no commit")
    cur, prev = load_current(), history.snapshot_at("HEAD", Path(ROOT))
    rows = metric_rows(cur, prev)
    pol = [f"{p['policy_id']}: {p['before']} → {p['after']} FAIL" for p in history.policy_changes(cur, prev)]
    ver = next_version()
    tag = f"trace-v{ver}"
    today = datetime.date.today().isoformat()

    table = ["| metric | previous | this version | Δ |", "|---|---|---|---|"]
    table += [f"| {l} | {p} | {c} | {d} |" for l, p, c, d in rows]
    compare = f" Compare: `git diff trace-v{ver - 1} {tag} -- {ART}/`" if ver > 1 else ""
    entry = [f"## {tag} — {today}", "", a.message, "",
             f"Synthetic cycle: n={a.n}, seed={a.seed}, deployed levers: "
             f"{', '.join(cur['cycle'].get('deployed_levers', [])) or 'none (vanilla)'}.{compare}", "", *table, ""]
    if pol:
        entry += ["Production policy FAIL changes: " + "; ".join(pol), ""]

    hist = os.path.join(ROOT, HISTORY)
    old = open(hist, encoding="utf-8").read() if os.path.exists(hist) else \
        "# Trace history\n\nOne entry per trace version (newest first). Generated by `scripts/snapshot_traces.py`.\n\n"
    head, _, rest = old.partition("\n## ")
    open(hist, "w", encoding="utf-8").write(head.rstrip() + "\n\n" + "\n".join(entry) + ("\n## " + rest if rest else ""))

    sh("git", "add", "-A")
    changed = [r for r in rows if r[3] not in ("=", "new")]
    body = "\n".join(f"{l}: {p} -> {c} ({d})" for l, p, c, d in (changed or rows))
    if pol:
        body += "\n\nproduction policy FAIL changes:\n" + "\n".join(pol)
    sh("git", "commit", "-q", "-m", f"{tag}: {a.message}", "-m", body)
    sh("git", "tag", "-a", tag, "-m", f"{tag}: {a.message}")
    print(f"[{tag}] committed {sh('git', 'rev-parse', '--short', 'HEAD')}\n{body}")

    if a.push:
        sh("git", "push", "--follow-tags", "origin", "HEAD")
        print(f"[{tag}] pushed")


if __name__ == "__main__":
    main()
