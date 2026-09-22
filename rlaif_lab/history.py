"""Trace version history, read from git.

Every improvement is committed by scripts/snapshot_traces.py and tagged trace-vN.
This module reads those tags back: for each version the headline metrics from that
version's artifacts/, the deltas against the previous version, and the commit log.
Needs a git checkout; returns {"available": False, ...} otherwise (e.g. in the container).
"""
from __future__ import annotations
import json, re, subprocess
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = "artifacts"
TAG_PREFIX = "trace-v"

# key, label, path into {"cycle": cycle_summary, "rca": trace_rca}, higher_is_better (None = neutral), unit
METRICS = [
    ("full_weighted",   "full weighted",        ("cycle", "full", "avg_weighted"),           True,  "score"),
    ("full_chosen",     "full chosen",          ("cycle", "full", "chosen_rate"),            True,  "pct0"),
    ("full_resolved",   "full resolved",        ("cycle", "full", "resolved_rate"),          True,  "pct0"),
    ("full_route",      "full route-acc",       ("cycle", "full", "route_accuracy"),         True,  "pct0"),
    ("full_compliance", "full compliance",      ("cycle", "full", "avg_compliance_rate"),    True,  "pct1"),
    ("full_action",     "full action-complete", ("cycle", "full", "action_completion_rate"), True,  "pct1"),
    ("full_leaks",      "full leaks",           ("cycle", "full", "output_leaks"),           False, "int"),
    ("full_gates",      "full gate violations", ("cycle", "full", "gate_violations"),        False, "int"),
    ("base_weighted",   "baseline weighted",    ("cycle", "baseline", "avg_weighted"),       True,  "score"),
    ("prod_calls",      "prod calls",           ("rca", "records"),                          None,  "int"),
    ("prod_fail",       "prod overall FAIL",    ("rca", "overall_result_false"),             False, "int"),
    ("prod_verdicts",   "prod verdicts parsed", ("rca", "verdicts_parsed"),                  None,  "int"),
]
FMT = {"score": "{:.3f}", "pct0": "{:.0%}", "pct1": "{:.1%}", "int": "{}"}


def _get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def metric_values(snap: dict) -> dict:
    return {key: _get(snap, path) for key, _, path, _, _ in METRICS}


def fail_counts(snap: dict) -> dict:
    return {p["policy_id"]: p["FAIL"] for p in snap["rca"].get("policies", []) if p.get("FAIL")}


def changes(cur: dict, prev: dict | None) -> list[dict]:
    """Per-metric before/after/delta; `good` is True/False for directional metrics, None otherwise."""
    a, b = metric_values(prev) if prev else {}, metric_values(cur)
    out = []
    for key, label, _, better, unit in METRICS:
        c, p = b.get(key), a.get(key)
        if c is None:
            continue
        d = None if p is None else round(c - p, 6)
        out.append({"key": key, "label": label, "unit": unit, "before": p, "after": c, "delta": d,
                    "good": None if better is None or not d else (d > 0) == better})
    return out


def policy_changes(cur: dict, prev: dict | None) -> list[dict]:
    if not prev:
        return []
    a, b = fail_counts(prev), fail_counts(cur)
    return [{"policy_id": pid, "before": a.get(pid, 0), "after": b.get(pid, 0)}
            for pid in sorted(set(a) | set(b)) if a.get(pid, 0) != b.get(pid, 0)]


def _git(*args: str, root: Path = REPO) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def snapshot_at(ref: str, root: Path = REPO) -> dict | None:
    cyc = _git("show", f"{ref}:{ART}/cycle_summary.json", root=root)
    rca = _git("show", f"{ref}:{ART}/trace_rca.json", root=root)
    return {"cycle": json.loads(cyc), "rca": json.loads(rca)} if cyc and rca else None


def repo_url(root: Path = REPO) -> str | None:
    url = (_git("remote", "get-url", "origin", root=root) or "").strip()
    m = re.match(r"^(?:https://github\.com/|git@github\.com:)([^/]+/[^/]+?)(?:\.git)?/?$", url)
    return f"https://github.com/{m.group(1)}" if m else None


def load(root: Path = REPO) -> dict:
    head = _git("rev-parse", "HEAD", root=root)
    if head is None:
        return {"available": False, "reason": "not a git checkout — run the server from the repo to see trace history",
                "versions": [], "commits": []}
    tags = _git("for-each-ref", f"refs/tags/{TAG_PREFIX}*", "--format=%(refname:short) %(objectname)", root=root) or ""
    return {**_load(str(root), head.strip(), tags), "state": state(root)}


def state(root: Path = REPO) -> dict:
    """Branch, sync with the upstream, and uncommitted changes (the next version's candidates)."""
    branch = (_git("rev-parse", "--abbrev-ref", "HEAD", root=root) or "").strip()
    upstream = (_git("rev-parse", "--abbrev-ref", "@{u}", root=root) or "").strip() or None
    ahead = behind = None
    if upstream:
        lr = (_git("rev-list", "--left-right", "--count", "@{u}...HEAD", root=root) or "").split()
        if len(lr) == 2:
            behind, ahead = int(lr[0]), int(lr[1])
    dirty = [{"status": ln[:2].strip(), "path": ln[3:]}
             for ln in (_git("status", "--porcelain", root=root) or "").splitlines() if ln.strip()]
    return {"branch": branch, "upstream": upstream, "ahead": ahead, "behind": behind, "dirty": dirty}


def _numstat(ref: str, root: Path) -> list[dict]:
    out = []
    for ln in (_git("show", "--numstat", "--format=", ref, root=root) or "").splitlines():
        f = ln.split("\t")
        if len(f) == 3:
            out.append({"path": f[2], "added": None if f[0] == "-" else int(f[0]),
                        "deleted": None if f[1] == "-" else int(f[1])})
    return out


def _diff(a: str | None, b: str, path: str, root: Path, max_lines: int = 160) -> dict:
    if not a:
        return {"path": path, "text": "", "truncated": False}
    lines = (_git("diff", "--unified=1", a, b, "--", path, root=root) or "").splitlines()
    return {"path": path, "text": "\n".join(lines[:max_lines]), "truncated": len(lines) > max_lines}


@lru_cache(maxsize=4)
def _load(root_s: str, head: str, tags: str) -> dict:
    root = Path(root_s)
    US, RS = "\x1f", "\x1e"
    log = _git("log", "--decorate=short", f"--format=%H{US}%h{US}%aI{US}%an{US}%D{US}%s{US}%b{RS}", root=root) or ""
    commits = []
    for rec in log.split(RS):
        f = rec.strip("\n").split(US)
        if len(f) < 7:
            continue
        refs = [r.strip() for r in f[4].split(",") if r.strip()]
        commits.append({"hash": f[0], "short": f[1], "date": f[2], "author": f[3], "subject": f[5],
                        "body": f[6].strip(), "tags": [r[5:] for r in refs if r.startswith("tag: ")]})
    by_hash = {c["hash"]: c for c in commits}

    names = [t.split()[0] for t in tags.splitlines() if t.strip()]
    names = sorted((t for t in names if t[len(TAG_PREFIX):].isdigit()), key=lambda t: int(t[len(TAG_PREFIX):]))
    versions, prev = [], None
    for tag in names:
        sha = (_git("rev-list", "-n", "1", tag, root=root) or "").strip()
        snap = snapshot_at(tag, root)
        c = by_hash.get(sha, {})
        subject = c.get("subject", "")
        versions.append({
            "tag": tag, "n": int(tag[len(TAG_PREFIX):]), "hash": sha, "short": sha[:7],
            "date": c.get("date"), "author": c.get("author"),
            "message": subject[len(tag) + 2:] if subject.startswith(tag + ": ") else subject,
            "deployed_levers": _get(snap, ("cycle", "deployed_levers")) if snap else None,
            "metrics": metric_values(snap) if snap else {},
            "files": _numstat(tag, root),
            "diff": _diff(versions[-1]["tag"] if versions else None, tag, f"{ART}/cycle_summary.json", root),
            "changes": changes(snap, prev) if snap else [],
            "policy_changes": policy_changes(snap, prev) if snap else [],
            "previous": versions[-1]["tag"] if versions else None,
        })
        prev = snap or prev
    return {"available": True, "repo_url": repo_url(root), "head": head[:7],
            "metrics": [{"key": k, "label": l, "better": b, "unit": u} for k, l, _, b, u in METRICS],
            "versions": versions[::-1], "commits": commits}
