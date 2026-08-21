# -*- coding: utf-8 -*-
"""
decision.py — 法官代理人核心決策邏輯

決策優先序：升級觸發(金額超門檻/詐欺blocking/詐欺warning累計/先例不足/信心過低)
  → 退回重算(圍籬外，retry_count<max_retries) → 放行。

詐欺一律 escalate_human，絕不自動 reject（法官不得單獨拍板詐欺結論，這是設計原則，
不是尚未實作——reject 保留在 schema 列舉但邏輯永遠不會產生）。
"""
from datetime import datetime, timezone, timedelta

from .config import JudgeConfig
from .fairness import fairness_check
from .fraud_rules import evaluate_fraud
from .confidence import compute_confidence, fraud_check_coverage
from .tpl_similar import NoOpTplSimilarRetriever

TZ = timezone(timedelta(hours=8))  # Asia/Taipei


def judge_decide(case: dict, cfg: JudgeConfig, amount_sources: dict, fault_retriever=None,
                  reasoning_reviewer=None, tpl_similar_retriever=None) -> dict:
    line = case.get("line")
    amount_source = amount_sources.get(line)
    if amount_source is None:
        raise ValueError(f"未知的 line='{line}'，amount_sources 只有 {list(amount_sources)}")

    fair = fairness_check(case, cfg, amount_source)
    fraud = evaluate_fraud(case, cfg)
    confidence = compute_confidence(fair, fraud, cfg)
    coverage = fraud_check_coverage(fraud)
    retry_count = int(case.get("retry_count", 0))
    fault_consistency = fault_retriever.retrieve(case) if fault_retriever else {"applicable": False}
    reasoning_review = reasoning_reviewer.review(case) if reasoning_reviewer else {
        "applicable": False, "reviewed": False, "flagged": False, "issues": []}

    claim = case.get("claim_amount")
    amt_thr = cfg.amount_threshold.get(line, float("inf"))
    triggers = {
        "amount_over_threshold": claim is not None and claim > amt_thr,
        "fraud_blocking": fraud["blocking_hit"],
        "precedent_insufficient": bool(fair.get("precedent_insufficient")),
        "low_confidence": confidence < cfg.confidence_threshold,
        "reasoning_review_flagged": bool(reasoning_review.get("flagged")),
    }
    fraud_warning_escalate = fraud["warning_score"] >= cfg.warning_threshold
    must_escalate = any(triggers.values()) or fraud_warning_escalate

    within = fair.get("within_tolerance")
    retry_exhausted = False

    if must_escalate:
        decision, next_agent = "escalate_human", "human_review"
    elif within is False:
        if retry_count < cfg.max_retries:
            decision, next_agent = "return_for_recalc", "claim_agent"
        else:
            decision, next_agent = "return_for_recalc", "claim_agent"
            retry_exhausted = True
    elif within is True:
        decision, next_agent = "execute", "frontend"
    else:  # within is None：無法重估
        decision, next_agent = "escalate_human", "human_review"
        triggers["precedent_insufficient"] = True

    # tpl_similar：只在案件觸發escalate_human時才呼叫(省LLM成本)，不是每案必打。
    # 這條線純粹補充參考資訊給human_review用，不影響上面已經拍板的decision，
    # 呼叫時機放在decision確定之後，避免retriever結果反過來影響決策邏輯。
    tpl_similar_queried = bool(tpl_similar_retriever) and not isinstance(tpl_similar_retriever, NoOpTplSimilarRetriever) \
                          and decision == "escalate_human"
    tpl_similar = (
        tpl_similar_retriever.retrieve(case)
        if tpl_similar_queried
        else {"applicable": False, "reason": "未觸發escalate_human，不查詢相似案例(省LLM成本)"}
    )

    reasons = []
    if triggers["amount_over_threshold"]:
        reasons.append(f"金額超過 {line} 轉人工門檻")
    if triggers["fraud_blocking"]:
        reasons.append("觸發詐欺硬紅旗(blocking)")
    if fraud_warning_escalate:
        reasons.append(f"詐欺 warning 累計達 {fraud['warning_score']}")
    if triggers["precedent_insufficient"]:
        reasons.append("先例覆蓋不足")
    if triggers["low_confidence"]:
        reasons.append(f"信心 {confidence} 低於門檻")
    if triggers["reasoning_review_flagged"]:
        reasons.append("審推理發現理賠代理人reasoning有問題: " + "; ".join(reasoning_review.get("issues", [])))
    if fair.get("extreme_dispersion"):
        reasons.append(f"comparable pool 離散度極端(dispersion_ratio={fair.get('dispersion_ratio')})，建議留意")
    if decision == "return_for_recalc":
        claim_str = f"{claim:,.0f}" if claim is not None else "N/A"
        reasons.append(f"金額 {claim_str} 落在 Tukey 圍籬 {fair.get('fair_range')} 之外"
                       + ("(已達退回上限，轉查缺漏)" if retry_exhausted else ""))
    if decision == "execute":
        reasons.append("公平合理且無阻擋，放行")

    return {
        "schema_version": cfg.schema_version, "agent": "judge_agent",
        "case_id": case.get("case_id"), "line": line,
        "timestamp": datetime.now(TZ).isoformat(timespec="seconds"),
        "retry_count": retry_count, "retry_exhausted": retry_exhausted,
        "decision": decision, "next_agent": next_agent,
        "fairness_check": fair,
        "fraud_flags": fraud,
        "escalation_triggers": triggers,
        "fraud_warning_escalate": fraud_warning_escalate,
        "confidence": confidence,
        "llm_assisted": bool(reasoning_review.get("reviewed")) or tpl_similar_queried,
        "reasons": reasons,
        "audit_trail": {
            "precedents_used_n": fair.get("precedent_n"),
            "amount_comparable_tier": fair.get("tier"),
            "rules_evaluated": len(fraud["flags"]),
            "rules_hit": [f["flag_id"] for f in fraud["flags"] if f["status"] == "hit"],
            "rules_na": [f["flag_id"] for f in fraud["flags"] if f["status"] == "na"],
            "fraud_check_coverage": coverage,
            "fault_consistency": fault_consistency,
            "reasoning_review": reasoning_review,
            "tpl_similar": tpl_similar,
        },
    }
