"""Trace version history, read from git.

Every improvement is committed by scripts/snapshot_traces.py and tagged trace-vN.
This module reads those tags back: for each version the headline metrics from that
version's artifacts/, the deltas against the previous version, and the commit log.
Needs a git checkout. Where there is none (the container image), it serves the snapshot
baked at build time by `python -m rlaif_lab.history` into data/trace_history.json.
"""
from __future__ import annotations
import difflib, json, re, subprocess
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = "artifacts"
TAG_PREFIX = "trace-v"
BAKED = Path(__file__).resolve().parent / "data" / "trace_history.json"

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
    st = state(root)
    if st is None:
        if BAKED.is_file():
            return {**json.loads(BAKED.read_text(encoding="utf-8")), "source": "baked"}
        return {"available": False, "reason": "not a git checkout — run the server from the repo to see trace history",
                "versions": [], "commits": []}
    tags = _git("for-each-ref", f"refs/tags/{TAG_PREFIX}*",
                "--format=%(refname:short) %(objectname) %(*objectname)", root=root) or ""
    return {**_load(str(root), st.pop("oid"), tags), "state": st}


def state(root: Path = REPO) -> dict | None:
    """Branch, upstream sync and uncommitted changes, from a single `git status` call."""
    out = _git("status", "--porcelain=v2", "--branch", root=root)
    if out is None:
        return None
    st = {"oid": "", "branch": "", "upstream": None, "ahead": None, "behind": None, "dirty": []}
    for ln in out.splitlines():
        if ln.startswith("# branch.oid "):
            st["oid"] = ln[13:]
        elif ln.startswith("# branch.head "):
            st["branch"] = ln[14:]
        elif ln.startswith("# branch.upstream "):
            st["upstream"] = ln[18:]
        elif ln.startswith("# branch.ab "):
            a, b = ln[12:].split()
            st["ahead"], st["behind"] = int(a), -int(b)
        elif ln[:2] in ("1 ", "2 ", "u "):
            f = ln.split(" ", {"1": 8, "2": 9, "u": 10}[ln[0]])
            st["dirty"].append({"status": f[1].replace(".", ""), "path": f[-1].split("\t")[0]})
        elif ln.startswith("? "):
            st["dirty"].append({"status": "??", "path": ln[2:]})
    return st


def _blobs(specs: list[str], root: Path) -> dict[str, str]:
    """Read many `<ref>:<path>` blobs with one `git cat-file --batch` process."""
    try:
        r = subprocess.run(["git", "cat-file", "--batch"], cwd=root, capture_output=True, timeout=30,
                           input="".join(s + "\n" for s in specs).encode())
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out, buf, i = {}, r.stdout, 0
    for spec in specs:
        nl = buf.find(b"\n", i)
        if nl < 0:
            break
        head, i = buf[i:nl].split(), nl + 1
        if len(head) == 3 and head[1] == b"blob":
            size = int(head[2])
            out[spec] = buf[i:i + size].decode("utf-8", "replace")
            i += size + 1
    return out


def _diff(a: str | None, b: str | None, path: str, max_lines: int = 160) -> dict:
    if a is None or b is None:
        return {"path": path, "text": "", "truncated": False}
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), f"a/{path}", f"b/{path}", n=1, lineterm=""))
    return {"path": path, "text": "\n".join(lines[:max_lines]), "truncated": len(lines) > max_lines}


@lru_cache(maxsize=4)
def _load(root_s: str, head: str, tags: str) -> dict:
    root = Path(root_s)
    US, RS = "\x1f", "\x1e"
    log = _git("log", "--decorate=short", "--numstat",
               f"--format={RS}%H{US}%h{US}%aI{US}%an{US}%D{US}%s{US}%b{US}", root=root) or ""
    commits, files = [], {}
    for rec in log.split(RS)[1:]:
        f = rec.split(US)
        if len(f) < 8:
            continue
        refs = [r.strip() for r in f[4].split(",") if r.strip()]
        commits.append({"hash": f[0], "short": f[1], "date": f[2], "author": f[3], "subject": f[5],
                        "body": f[6].strip(), "tags": [r[5:] for r in refs if r.startswith("tag: ")]})
        files[f[0]] = [{"path": n[2], "added": None if n[0] == "-" else int(n[0]),
                        "deleted": None if n[1] == "-" else int(n[1])}
                       for n in (ln.split("\t") for ln in f[7].splitlines()) if len(n) == 3]
    by_hash = {c["hash"]: c for c in commits}

    shas = {}
    for ln in tags.splitlines():
        f = ln.split()
        if len(f) >= 2 and f[0][len(TAG_PREFIX):].isdigit():
            shas[f[0]] = f[2] if len(f) > 2 else f[1]      # peeled commit for annotated tags
    names = sorted(shas, key=lambda t: int(t[len(TAG_PREFIX):]))
    cyc_path, rca_path = f"{ART}/cycle_summary.json", f"{ART}/trace_rca.json"
    blobs = _blobs([f"{t}:{p}" for t in names for p in (cyc_path, rca_path)], root)

    versions, prev, prev_text = [], None, None
    for tag in names:
        sha, c = shas[tag], by_hash.get(shas[tag], {})
        cyc, rca = blobs.get(f"{tag}:{cyc_path}"), blobs.get(f"{tag}:{rca_path}")
        snap = {"cycle": json.loads(cyc), "rca": json.loads(rca)} if cyc and rca else None
        subject = c.get("subject", "")
        versions.append({
            "tag": tag, "n": int(tag[len(TAG_PREFIX):]), "hash": sha, "short": sha[:7],
            "date": c.get("date"), "author": c.get("author"),
            "message": subject[len(tag) + 2:] if subject.startswith(tag + ": ") else subject,
            "deployed_levers": _get(snap, ("cycle", "deployed_levers")) if snap else None,
            "metrics": metric_values(snap) if snap else {},
            "files": files.get(sha, []),
            "diff": _diff(prev_text, cyc, cyc_path),
            "changes": changes(snap, prev) if snap else [],
            "policy_changes": policy_changes(snap, prev) if snap else [],
            "previous": versions[-1]["tag"] if versions else None,
        })
        prev, prev_text = snap or prev, cyc if cyc is not None else prev_text
    return {"available": True, "repo_url": repo_url(root), "head": head[:7],
            "metrics": [{"key": k, "label": l, "better": b, "unit": u} for k, l, _, b, u in METRICS],
            "versions": versions[::-1], "commits": commits}


if __name__ == "__main__":
    h = load()
    if not h["available"]:
        raise SystemExit(h["reason"])
    BAKED.write_text(json.dumps({**h, "source": "git"}, indent=1), encoding="utf-8")
    print(f"baked {len(h['versions'])} versions @ {h['head']} -> {BAKED}")
