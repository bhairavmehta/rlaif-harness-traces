"""Production trace corpus: tolerant ingest -> normalization -> RCA -> lever evidence.

data/traces.txt is the compliance-verdict export (created_at, external_id,
metrics/cm_policy_metric) from the writeback table. The export is not valid
JSON: records are fragmented, keys repeat, quotes are doubled, and OCR noise
mangles policy ids (BEX-81, BBX-05, GEN-89 ...). So ingest is regex-driven and
every repair is recorded, never silent:

  parse()    -> TraceRecord per call (header CID + journey + verdicts)
  analyze()  -> per-policy PASS/FAIL/NA, failure buckets (RCA classifier),
                bucket -> harness lever, judge-health checks (verdict/reason
                contradictions, severity drift, repaired / unresolved ids)

The RCA buckets become production evidence on the engine's recommendations.
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from .policies import CATALOG

DEFAULT_PATH = Path(__file__).parent / "data" / "traces.txt"

_HEADER = re.compile(
    r"(?m)^[\s\"'{]*((?:19|20)\d\d-\d\d-\d\d[ _T]\d\d:\d\d:\d\d[^\n]{0,40}?)[ ,(]*"
    r"(?:\[?C?I?D[:=\- ]*)?\s*([\d][\d.+\- ]{20,})")
_POLICY = re.compile(r'"policy_id"\s*:\s*"+\**([^"*]+?)\**"')
_FIELD = {k: re.compile(r'"%s"\s*:\s*"+(.*?)"+\s*[,}\n]' % k, re.S)
          for k in ("verdict", "evidence", "reason", "severity", "policy_type")}
_JOURNEY = re.compile(r'"journey"\s*:\s*"([^"]+)"')
_RESULT = re.compile(r'"overall_result"\s*:\s*(true|false)')
_RATE = re.compile(r'"overall_compliance_rate"\s*:\s*([\d.]+)')

JOURNEY_ALIASES = {"bill-explanation": "bex", "bill_explanation": "bex", "bill explanation": "bex",
                   "discount-inquiry": "dis", "bill-due-date-change": "ddc", "paper-free-billing": "pfb"}
_PREFIX_FIX = {"BEX": "BEX", "BBX": "BEX", "TBX": "BEX", "PEX": "BEX", "BX": "BEX", "BOP": "BEX",
               "GEN": "GEN", "IGN": "GEN", "GP": "GEN", "GCP": "GEN", "GSP": "GEN", "SM-GEN": "GEN",
               "DIS": "DIS", "DDC": "DDC", "DOC": "DDC", "PFB": "PFB"}
_TYPE_FAMILY = {"grounding": "Grounding", "escalation": "Escalation", "containment": "Escalation",
                "toolsequencing": "ToolSequencing", "tool sequencing": "ToolSequencing",
                "sequencing": "ToolSequencing", "compliance": "Compliance", "scope": "Scope",
                "session": "Session", "tone": "Tone-CX", "resolution": "Resolution", "consent": "Consent"}

# RCA classifier: (bucket, policy-id filter, regex over evidence+reason, lever, demo category)
BUCKETS = [
    ("fabrication", r"GEN-01", r"fabricat|no data supporting|estimated", None, "groundedness"),
    ("internal_tool_name_disclosure", r"GEN-02", r"tool name|internal tool|invoke tool", "formatter", "tool_quality"),
    ("raw_json_leakage", r"GEN-0[25789]", r"json|raw|dictionary|status ?message|sms_?sent|\{|system output|tool response data",
     "formatter", "tool_quality"),
    ("wrong_tool_fidelity", r"DIS-02|GEN-09|BEX-03", r"instead of the required|wrong tool|used '?\w+'? instead", "metadata", "tool_usage"),
    ("generic_close", r"BEX-08", r"anything else|generic|next step", "prompt_hat", "policy_compliance"),
    ("bill_copy_channel", r"BEX-07", r"digital|paper|channel|number on file|without first asking|assum", "prompt_hat", "policy_compliance"),
    ("waiver_not_pursued", r"BEX-0[14]", r"waiver|eligibility|relief|dispute", "sequencing", "agentic_flow_shortest_path"),
    ("intent_missed", r"BEX-02|GEN-06|DDC-04", r"failed to address|specific charge|repeating|ignor|intent|instead", "taxonomy", "intent_recognition"),
    ("scope_routing", r"GEN-05|BEX-05|DIS-05", r"route|specialist|scope", "taxonomy", "routing"),
    ("containment", r"GEN-0[37]", r"escalat|representative|human|contain", "prompt_hat", "context_propagation"),
]
_NOT_TRIGGERED = re.compile(r"not (?:been )?triggered|does not apply|was not applicable|did not (?:request|raise|inquire|ask|involve)"
                            r"|no relevant turns|never (?:requested|raised|reached)|not required", re.I)
_VIOLATION = re.compile(r"\bfailed to\b|\bviolat|\bfabricat|\bexposed\b|\binstead of\b", re.I)


@dataclass
class TraceRecord:
    cid: str
    created_at: str
    journey: str
    overall_result: bool | None
    compliance_rate: float | None
    verdicts: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_json(self, evidence: bool = True):
        vs = self.verdicts if evidence else [{k: v for k, v in r.items() if k not in ("evidence", "reason")}
                                             for r in self.verdicts]
        return {"cid": self.cid if evidence else _mask(self.cid), "created_at": self.created_at,
                "journey": self.journey, "overall_result": self.overall_result,
                "compliance_rate": self.compliance_rate, "verdicts": vs, "issues": self.issues}


def _mask(cid: str) -> str:
    return cid[:8] + "..." + cid[-4:] if len(cid) > 14 else "CID-..."


def _norm_verdict(v: str) -> str:
    v = (v or "").strip().strip('*"').upper()
    if v.startswith("PASS") or v == "TRUE":
        return "PASS"
    if v.startswith("FAIL") or v == "FALSE":
        return "FAIL"
    return "NA"


def _norm_type(t: str) -> str:
    t = re.sub(r"[^a-z ]", "", (t or "").lower()).strip()
    for k, fam in _TYPE_FAMILY.items():
        if t.startswith(k):
            return fam
    return (t or "").title()


def normalize_policy_id(raw: str, policy_type: str = "") -> tuple[str | None, bool]:
    """-> (canonical id or None, repaired?). Uses the digit pattern, then policy_type to disambiguate."""
    s = raw.strip().upper().replace("_", "-")
    m = re.match(r"^(?:NY-)?([A-Z]+(?:-GEN)?)-?(\d{1,3})$", s)
    prefix = _PREFIX_FIX.get(m.group(1)) if m else None
    if not prefix:
        return None, True
    n = int(m.group(2))
    fam = _norm_type(policy_type)
    candidates = [pid for pid in CATALOG if pid.startswith(prefix)]
    guess = f"{prefix}-{n % 10 if n >= 10 else n:02d}"
    same_family = [pid for pid in candidates if CATALOG[pid].policy_type.startswith(fam[:5])] if fam else []
    if guess in CATALOG and (n < 10 or not same_family or guess in same_family):
        pid = guess                    # a clean id is trusted over a mislabeled policy_type
    elif len(same_family) == 1:
        pid = same_family[0]
    else:
        pid = guess if guess in CATALOG else None
    return pid, pid != s


def parse(text: str) -> list[TraceRecord]:
    text = text.replace("\r\n", "\n")
    heads = list(_HEADER.finditer(text))
    spans = [(m.start(), heads[i + 1].start() if i + 1 < len(heads) else len(text), m) for i, m in enumerate(heads)]
    if heads and heads[0].start() > 0:
        spans.insert(0, (0, heads[0].start(), None))
    records: list[TraceRecord] = []
    anon = 0
    for start, end, m in spans:
        chunk = text[start:end]
        # a chunk may silently contain a second call whose header was lost: split on a journey change
        pieces, last_j, cut = [], None, 0
        for jm in _JOURNEY.finditer(chunk):
            j = JOURNEY_ALIASES.get(jm.group(1).strip().lower())
            if j and last_j and j != last_j and _POLICY.search(chunk, cut, jm.start()):
                pieces.append((chunk[cut:jm.start()], last_j))
                cut = jm.start()
            last_j = j or last_j
        pieces.append((chunk[cut:], last_j))
        for n, (piece, journey) in enumerate(pieces):
            rec = _record(piece, m if n == 0 else None, journey)
            if rec is None:
                continue
            if not rec.cid:
                anon += 1
                rec.cid = f"UNKNOWN-{anon:02d}"
                rec.issues.append("header_missing")
            records.append(rec)
    return records


def _record(piece: str, head, journey) -> TraceRecord | None:
    pols = list(_POLICY.finditer(piece))
    if not pols:
        return None
    issues = []
    cid, created = "", ""
    if head:
        created = head.group(1).strip().rstrip(",")
        digits = re.findall(r"\d+", head.group(2))
        cid = "CID-" + "-".join(digits)
        if not re.match(r"2026-", created):
            issues.append("timestamp_suspect")
    rm, rate = _RESULT.search(piece), _RATE.search(piece)
    seen: dict[str, dict] = {}
    for i, pm in enumerate(pols):
        body = piece[pm.end(): pols[i + 1].start() if i + 1 < len(pols) else len(piece)]
        f = {k: (rx.search(body).group(1).strip() if rx.search(body) else "") for k, rx in _FIELD.items()}
        pid, repaired = normalize_policy_id(pm.group(1), f["policy_type"])
        row = {"policy_id": pid or pm.group(1).strip(), "raw_policy_id": pm.group(1).strip(),
               "id_repaired": repaired, "resolved": pid is not None,
               "verdict": _norm_verdict(f["verdict"]), "raw_verdict": f["verdict"],
               "evidence": f["evidence"][:300], "reason": f["reason"][:500],
               "severity": f["severity"].upper(), "policy_type": f["policy_type"]}
        key = row["policy_id"]
        if key in seen:
            if seen[key]["verdict"] != row["verdict"] and row["verdict"] != "NA" and seen[key]["verdict"] != "NA":
                issues.append(f"conflicting_duplicate:{key}")
            if seen[key]["verdict"] == "NA" and row["verdict"] != "NA":
                seen[key] = row
            continue
        seen[key] = row
    verdicts = list(seen.values())
    if not journey:
        prefixes = Counter(v["policy_id"][:3] for v in verdicts if v["resolved"] and not v["policy_id"].startswith("GEN"))
        journey = {"BEX": "bex", "DIS": "dis", "DDC": "ddc", "PFB": "pfb"}.get(
            prefixes.most_common(1)[0][0] if prefixes else "", "gen")
        issues.append("journey_inferred")
    return TraceRecord(cid, created, journey, (rm.group(1) == "true") if rm else None,
                       float(rate.group(1)) if rate and float(rate.group(1)) <= 1 else None, verdicts, issues)


def load(path: str | Path | None = None) -> list[TraceRecord]:
    return parse(Path(path or DEFAULT_PATH).read_text(encoding="utf-8", errors="replace"))


def classify(row: dict) -> tuple[str, str | None, str]:
    text = f"{row['evidence']} {row['reason']}".lower()
    for bucket, pids, rx, lever, category in BUCKETS:
        if re.match(pids, row["policy_id"]) and re.search(rx, text):
            return bucket, lever, category
    for bucket, pids, _, lever, category in BUCKETS:      # truncated evidence: the policy's default bucket
        if re.match(pids, row["policy_id"]) and not re.match(r"GEN-0[25789]", row["policy_id"]):
            return bucket, lever, category
    p = CATALOG.get(row["policy_id"])
    return "unclassified", (p.lever if p else None), "policy_compliance"


def analyze(records: list[TraceRecord], evidence: bool = True, examples: int = 3) -> dict:
    per_policy = defaultdict(lambda: {"PASS": 0, "FAIL": 0, "NA": 0})
    buckets: dict[str, dict] = {}
    contradictions, drift, repaired, unresolved = [], Counter(), Counter(), Counter()
    for rec in records:
        for r in rec.verdicts:
            if not r["resolved"]:
                unresolved[r["raw_policy_id"]] += 1
                continue
            if r["id_repaired"]:
                repaired[f"{r['raw_policy_id']} -> {r['policy_id']}"] += 1
            p = CATALOG[r["policy_id"]]
            if r["severity"] and r["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW") and r["severity"] != p.severity:
                drift[r["policy_id"]] += 1
            verdict = r["verdict"]
            if verdict == "FAIL" and _NOT_TRIGGERED.search(r["reason"]) and not _VIOLATION.search(r["reason"]):
                contradictions.append({"cid": rec.cid if evidence else _mask(rec.cid), "policy_id": r["policy_id"],
                                       "verdict": verdict, **({"reason": r["reason"][:200]} if evidence else {})})
                verdict = "NA"                                   # the judge's own reasoning says not triggered
            per_policy[r["policy_id"]][verdict] += 1
            if verdict != "FAIL":
                continue
            bucket, lever, category = classify(r)
            b = buckets.setdefault(bucket, {"bucket": bucket, "lever": lever, "category": category,
                                            "fail_verdicts": 0, "calls": set(), "policies": Counter(),
                                            "examples": []})
            b["fail_verdicts"] += 1
            b["calls"].add(rec.cid)
            b["policies"][r["policy_id"]] += 1
            if evidence and len(b["examples"]) < examples:
                b["examples"].append({"cid": rec.cid, "policy_id": r["policy_id"],
                                      "evidence": r["evidence"][:160], "reason": r["reason"][:220]})
    policy_rows = []
    for pid in sorted(per_policy, key=lambda k: list(CATALOG).index(k)):
        c = per_policy[pid]
        judged = c["PASS"] + c["FAIL"]
        p = CATALOG[pid]
        policy_rows.append({"policy_id": pid, "title": p.title, "severity": p.severity, "hard": p.hard,
                            "lever": p.lever, **c, "fail_rate": round(c["FAIL"] / judged, 3) if judged else 0.0})
    bucket_rows = sorted(({**b, "calls": len(b["calls"]), "policies": dict(b["policies"])}
                          for b in buckets.values()), key=lambda b: -b["fail_verdicts"])
    lever_evidence = defaultdict(lambda: {"fail_verdicts": 0, "calls": 0, "buckets": []})
    for b in bucket_rows:
        key = b["lever"] or "outside_lever_set"
        lever_evidence[key]["fail_verdicts"] += b["fail_verdicts"]
        lever_evidence[key]["calls"] += b["calls"]
        lever_evidence[key]["buckets"].append(b["bucket"])
    n_verdicts = sum(len(r.verdicts) for r in records)
    return {
        "source": "writeback compliance verdicts (redacted production traces)",
        "records": len(records),
        "journeys": dict(Counter(r.journey for r in records)),
        "overall_result_false": sum(r.overall_result is False for r in records),
        "verdicts_parsed": n_verdicts,
        "policies": policy_rows,
        "failure_buckets": bucket_rows,
        "lever_evidence": dict(lever_evidence),
        "judge_health": {
            "verdict_reason_contradictions": len(contradictions),
            "contradiction_rate": round(len(contradictions) / n_verdicts, 3) if n_verdicts else 0.0,
            "contradictions": contradictions[:12],
            "severity_drift": dict(drift),
            "policy_ids_repaired": sum(repaired.values()),
            "repairs": dict(repaired.most_common(15)),
            "unresolved_policy_ids": dict(unresolved),
            "record_issues": dict(Counter(i.split(":")[0] for r in records for i in r.issues)),
        },
        "evidence_included": evidence,
    }
