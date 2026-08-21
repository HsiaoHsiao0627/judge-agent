# -*- coding: utf-8 -*-
"""schema.py — 輸出 schema 驗證，確保每筆輸出結構固定、可稽核、可被下游代理人安全消費。"""

DECISIONS = {"execute", "return_for_recalc", "escalate_human", "reject"}
NEXT = {"frontend", "claim_agent", "human_review", "end"}


def validate_output(o: dict) -> list:
    errs = []
    for k in ["schema_version", "agent", "case_id", "line", "timestamp", "retry_count",
              "decision", "next_agent", "fairness_check", "fraud_flags",
              "escalation_triggers", "confidence", "reasons", "audit_trail"]:
        if k not in o:
            errs.append(f"缺欄位: {k}")
    if o.get("decision") not in DECISIONS:
        errs.append(f"decision 非法: {o.get('decision')}")
    if o.get("next_agent") not in NEXT:
        errs.append(f"next_agent 非法: {o.get('next_agent')}")
    if o.get("decision") == "reject":
        errs.append("違反規則：法官不得自動 reject(詐欺應 escalate_human)")
    c = o.get("confidence")
    if not (isinstance(c, (int, float)) and 0 <= c <= 1):
        errs.append(f"confidence 超範圍: {c}")
    for f in o.get("fraud_flags", {}).get("flags", []):
        if f["status"] not in {"hit", "pass", "na"}:
            errs.append(f"flag status 非法: {f}")
    return errs
