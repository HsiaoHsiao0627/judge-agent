# -*- coding: utf-8 -*-
"""
decision.py — 法官代理人核心決策邏輯（2026-10-07 改版）

法官只做一件決定：是否同意規則代理人的結論。
  同意 → execute（放行）；不同意或無法確認 → escalate_human（轉人工）。

2026-11-03 TPL 退回重算（只限 cfg.recalc_lines）：沒過的檢查「全部」都是理賠代理人改得動的項目時，
  不直接轉人工，而是 return_for_recalc、next_agent="claim_agent"，附 recalc_request 告訴理賠代理人
  要往哪個方向重算；最多退回 cfg.max_retries(3) 次，第 4 次審核仍不通過 → escalate_human、retry_exhausted。
  理賠代理人改得動的項目：
    - 總額落在 Tukey 合理區間外（偏高→調低／偏低→調高）、建議金額高於請款金額（調低）
    - 精神慰撫金落在合理區間外（審的是理賠代理人的建議值 suggested_mental_damage）
    - 推理審查被標記（請理賠代理人帶著問題重寫說明）
  其餘不通過（規則代理人結論／交叉比對、先例不足、超過轉人工門檻、信心過低、逐項上限）重算也不會改變，
  只要有任何一項就直接轉人工。法官只指出方向，不指定金額；金額仍由理賠代理人自己的模型決定。

「同意」需要以下四項檢查都沒有失敗：
  1. rule_agent_conclusion   規則代理人的受理結論是「可受理」（RSA: ELIGIBLE_FOR_PROCESS／TPL: 理賠）
  2. rule_agent_cross_check  規則代理人自己的交叉比對（Agreement Gate）已通過、結論已放行
  3. amount                  理賠金額合理：RSA 有價案件用收費標準規則 ±5%，其餘用 Tukey 圍籬；
                             超過轉人工門檻、先例不足或信心過低都算無法確認
  4. reasoning_review        理賠代理人的推理與引用沒有被 LLM 審查標記（未啟用審查時為「未審查」，不擋放行）
  5. item_amounts            TPL 逐項上限（看護費用；工作損失只依案件薪資證明計算）
  6. mental_damage           TPL 精神慰撫金單獨審查（與總額同一比較池、Tukey 圍籬；沒提供逐項金額時不擋）
  呼叫端沒有送 rule_agent 時，1、2 無法確認，一律轉人工。

詐欺只做標註（suspected_fraud），不影響放行與否；命中的規則仍完整列在 fraud_flags。
法官仍然不修改金額，也不產生 reject。

case["rule_agent"] 格式：{"decision": "ELIGIBLE_FOR_PROCESS" | "理賠" | ..., "released": bool,
                          "agreement_status": "AGREED" | ... (選填), "source": 選填}
"""
from datetime import datetime, timezone, timedelta

from .config import JudgeConfig
from .fairness import fairness_check
from .fraud_rules import evaluate_fraud
from .item_checks import evaluate_items
from .mental_damage import evaluate_mental_damage
from .confidence import compute_confidence, fraud_check_coverage
from .tpl_similar import NoOpTplSimilarRetriever

TZ = timezone(timedelta(hours=8))  # Asia/Taipei

ELIGIBLE_DECISIONS = {"ELIGIBLE_FOR_PROCESS", "理賠"}
DECISION_NAMES = {"ELIGIBLE_FOR_PROCESS": "可受理", "EXCLUDED": "不在保障範圍", "NEED_MORE_INFORMATION": "資料不足"}
CHECK_LABELS = {
    "rule_agent_conclusion": "規則代理人的受理結論",
    "rule_agent_cross_check": "規則代理人的交叉比對",
    "amount": "理賠金額",
    "reasoning_review": "理賠代理人的推理",
    "item_amounts": "逐項金額",
    "mental_damage": "精神慰撫金",
}
DIRECTION_NAMES = {"lower": "調低", "higher": "調高"}


def _check(cid, status, detail):
    return {"id": cid, "label": CHECK_LABELS[cid], "status": status, "detail": detail}


def _rule_agent_checks(case):
    ra = case.get("rule_agent")
    if not isinstance(ra, dict) or not ra.get("decision"):
        msg = "沒有收到規則代理人的結果，無法確認"
        return [_check("rule_agent_conclusion", "na", msg), _check("rule_agent_cross_check", "na", msg)]
    dec = ra.get("decision")
    name = DECISION_NAMES.get(dec, dec)
    concl = (_check("rule_agent_conclusion", "pass", f"規則代理人判斷「{name}」，可進入理賠")
             if dec in ELIGIBLE_DECISIONS
             else _check("rule_agent_conclusion", "fail", f"規則代理人判斷「{name}」，不是可受理的結論"))
    released = ra.get("released")
    agree = ra.get("agreement_status")
    suffix = f"（{agree}）" if agree else ""
    if released is True:
        cross = _check("rule_agent_cross_check", "pass", f"交叉比對通過，結論已放行{suffix}")
    elif released is False:
        cross = _check("rule_agent_cross_check", "fail", f"交叉比對沒有完全通過，結論尚未放行{suffix}")
    else:
        cross = _check("rule_agent_cross_check", "na", "規則代理人沒有提供交叉比對結果，無法確認")
    return [concl, cross]


def _amount_check(case, fair, confidence, cfg):
    line = case.get("line")
    claim = case.get("claim_amount")
    thr = cfg.amount_threshold.get(line, float("inf"))
    if claim is None:
        return _check("amount", "fail", "沒有收到理賠代理人的建議金額")
    # 2026-10-21：核定金額不得高於請款金額（呼叫端須送 requested_amount，沒送就略過這一條）
    requested = case.get("requested_amount")
    if requested is not None and claim > requested:
        return _check("amount", "fail", f"建議金額 {claim:,.0f} 高於請款金額 {requested:,.0f}")
    if claim > thr:
        return _check("amount", "fail", f"金額 {claim:,.0f} 超過 {line} 轉人工門檻 {thr:,.0f}")
    if fair.get("basis") == "fee_schedule_rule":
        if fair.get("within_tolerance"):
            return _check("amount", "pass", f"與收費標準規則金額 {fair['rule_amount']:,} 相差 {fair['deviation_pct']:+.1f}%，"
                                            f"在 ±{cfg.rsa_rule_tolerance * 100:g}% 內")
        detail = "、".join(f"{i['item']} {i['amount']:,}" + ("(資料歸納)" if i["source"] == "data_inferred" else "")
                          for i in fair.get("rule_items", []))
        return _check("amount", "fail", f"金額 {claim:,.0f} 與收費標準規則金額 {fair['rule_amount']:,} 相差 "
                                        f"{fair['deviation_pct']:+.1f}%，超出 ±{cfg.rsa_rule_tolerance * 100:g}%（規則明細：{detail}）")
    if fair.get("precedent_insufficient") or fair.get("within_tolerance") is None:
        return _check("amount", "fail", f"可比對的歷史案件不足（{fair.get('precedent_n') or 0} 筆），無法判斷金額是否合理")
    if confidence < cfg.confidence_threshold:
        return _check("amount", "fail", f"歷史案件金額過於分散，信心 {confidence} 低於門檻 {cfg.confidence_threshold}")
    if fair.get("within_tolerance"):
        return _check("amount", "pass", f"落在 {fair.get('precedent_n')} 筆歷史案件的合理區間 {fair.get('fair_range')} 內")
    return _check("amount", "fail", f"金額 {claim:,.0f} 落在歷史案件合理區間 {fair.get('fair_range')} 之外")


def _reasoning_check(rr):
    if not rr.get("applicable"):
        return _check("reasoning_review", "na", "未審查（理賠代理人沒有提供推理，或審查未啟用）")
    if rr.get("flagged"):
        issues = "; ".join(rr.get("issues", [])) or "審查標記有問題"
        return _check("reasoning_review", "fail", issues)
    return _check("reasoning_review", "pass", "引用的案例與條款跟理賠代理人說的一致")


def _item_check(item_checks):
    """TPL 逐項核對（hfrl 分支）：有任何一項超出上限 → 不同意；全部 na → 不擋。"""
    hits = [it for it in item_checks.get("items", []) if it["status"] == "hit"]
    if hits:
        return _check("item_amounts", "fail", "；".join(it["reason"] for it in hits))
    if any(it["status"] == "pass" for it in item_checks.get("items", [])):
        return _check("item_amounts", "pass", "逐項金額都在上限內")
    return _check("item_amounts", "na", "沒有可核對的逐項金額")


def _suspected_fraud(fraud, cfg):
    hits = [f for f in fraud["flags"] if f["status"] == "hit"]
    level = ("blocking" if fraud["blocking_hit"]
             else "warning" if fraud["warning_score"] >= cfg.warning_threshold
             else "none")
    return {
        "suspected": level != "none",
        "level": level,
        "hit_flags": [{"flag_id": f["flag_id"], "severity": f["severity"], "reason": f.get("reason")} for f in hits],
        "warning_score": fraud["warning_score"], "warning_threshold": cfg.warning_threshold,
        "note": "僅供承辦人員參考的標註，不影響放行與否",
    }


def _direction(value, fair_range):
    if not fair_range or value is None:
        return None
    return "lower" if value > fair_range[1] else "higher" if value < fair_range[0] else None


def _recalc_target(check, case, fair, confidence, mental, reasoning_review, cfg):
    """沒過的檢查 → 理賠代理人該怎麼重算；回傳 None 代表重算改變不了，必須轉人工。
    判斷順序跟 _amount_check 相同，確保「為什麼沒過」與「能不能重算」一致。"""
    cid = check["id"]
    if cid == "amount":
        claim, requested = case.get("claim_amount"), case.get("requested_amount")
        if claim is None:
            return None
        if requested is not None and claim > requested:
            return {"target": "total", "direction": "lower", "bound": requested, "reason": check["detail"]}
        if claim > cfg.amount_threshold.get(case.get("line"), float("inf")):
            return None
        if fair.get("basis") != "tukey_iqr" or fair.get("precedent_insufficient") or confidence < cfg.confidence_threshold:
            return None
        d = _direction(claim, fair.get("fair_range"))
        return d and {"target": "total", "direction": d, "fair_range": fair["fair_range"], "reason": check["detail"]}
    if cid == "mental_damage":
        mf = mental.get("fairness") or {}
        if mental.get("amount_source") != "claim_agent" or mf.get("precedent_insufficient"):
            return None
        d = _direction(mental.get("claim_amount"), mf.get("fair_range"))
        return d and {"target": "mental", "direction": d, "fair_range": mf["fair_range"], "reason": check["detail"]}
    if cid == "reasoning_review":
        return {"target": "reasoning", "issues": list(reasoning_review.get("issues") or []) or [check["detail"]],
                "reason": check["detail"]}
    return None


def judge_decide(case: dict, cfg: JudgeConfig, amount_sources: dict, fault_retriever=None,
                  reasoning_reviewer=None, tpl_similar_retriever=None) -> dict:
    line = case.get("line")
    amount_source = amount_sources.get(line)
    if amount_source is None:
        raise ValueError(f"未知的 line='{line}'，amount_sources 只有 {list(amount_sources)}")

    fair = fairness_check(case, cfg, amount_source)
    fraud = evaluate_fraud(case, cfg)
    item_checks = evaluate_items(case, cfg)
    mental_damage = evaluate_mental_damage(case, cfg, amount_source)
    confidence = compute_confidence(fair, fraud, cfg)
    coverage = fraud_check_coverage(fraud)
    retry_count = int(case.get("retry_count", 0))
    fault_consistency = fault_retriever.retrieve(case) if fault_retriever else {"applicable": False}
    reasoning_review = reasoning_reviewer.review(case) if reasoning_reviewer else {
        "applicable": False, "reviewed": False, "flagged": False, "issues": []}
    claim = case.get("claim_amount")

    checks = _rule_agent_checks(case) + [_amount_check(case, fair, confidence, cfg), _reasoning_check(reasoning_review),
                                       _item_check(item_checks)]
    if line == "TPL":
        checks.append(_check("mental_damage", mental_damage["status"], mental_damage["reason"]))
    by_id = {c["id"]: c for c in checks}
    blocking = [c for c in checks
                if c["status"] == "fail" or (c["status"] == "na" and c["id"].startswith("rule_agent"))]
    agree = not blocking
    targets = [_recalc_target(c, case, fair, confidence, mental_damage, reasoning_review, cfg) for c in blocking]
    recalc_possible = bool(blocking) and line in cfg.recalc_lines and all(targets)
    if agree:
        decision, next_agent = "execute", "frontend"
    elif recalc_possible and retry_count < cfg.max_retries:
        decision, next_agent = "return_for_recalc", "claim_agent"
    else:
        decision, next_agent = "escalate_human", "human_review"
    suspected_fraud = _suspected_fraud(fraud, cfg)

    amt_thr = cfg.amount_threshold.get(line, float("inf"))
    triggers = {
        "rule_agent_missing": by_id["rule_agent_conclusion"]["status"] == "na",
        "rule_agent_not_eligible": by_id["rule_agent_conclusion"]["status"] == "fail",
        "rule_agent_cross_check_failed": by_id["rule_agent_cross_check"]["status"] in ("fail", "na"),
        "amount_over_threshold": claim is not None and claim > amt_thr,
        "amount_out_of_range": fair.get("within_tolerance") is False,
        "precedent_insufficient": bool(fair.get("precedent_insufficient")),
        "low_confidence": confidence < cfg.confidence_threshold,
        "reasoning_review_flagged": bool(reasoning_review.get("flagged")),
        "mental_damage_failed": mental_damage["status"] == "fail",
    }
    retry_exhausted = recalc_possible and retry_count >= cfg.max_retries
    recalc_request = None
    if decision == "return_for_recalc":
        recalc_request = {"target_agent": "claim_agent", "return_no": retry_count + 1,
                          "max_returns": cfg.max_retries, "targets": targets}
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

    if agree:
        reasons = ["規則代理人的結論、交叉比對、理賠金額"
                   + ("、精神慰撫金" if line == "TPL" and mental_damage["status"] == "pass" else "")
                   + "與推理都沒有問題，同意規則代理人的結論，放行"]
    else:
        reasons = [f"{c['label']}：{c['detail']}" for c in blocking]
        if decision == "return_for_recalc":
            how = "、".join(("重寫推理說明" if t["target"] == "reasoning" else
                            ("總額" if t["target"] == "total" else "精神慰撫金") + DIRECTION_NAMES[t["direction"]])
                           for t in targets)
            reasons.insert(0, f"退回理賠代理人重算（第 {retry_count + 1}/{cfg.max_retries} 次）：{how}")
        elif retry_exhausted:
            reasons.insert(0, f"已退回理賠代理人重算 {cfg.max_retries} 次仍未通過，轉人工")
    if suspected_fraud["suspected"]:
        reasons.append("標註疑似詐欺（僅供參考，不影響放行）："
                       + "、".join(h["reason"] or h["flag_id"] for h in suspected_fraud["hit_flags"]))

    return {
        "schema_version": cfg.schema_version, "agent": "judge_agent",
        "case_id": case.get("case_id"), "line": line,
        "timestamp": datetime.now(TZ).isoformat(timespec="seconds"),
        "retry_count": retry_count, "retry_exhausted": retry_exhausted,
        "decision": decision, "next_agent": next_agent,
        "fairness_check": fair,
        "agreement": {"agree": agree, "checks": checks},
        "recalc_request": recalc_request,
        "suspected_fraud": suspected_fraud,
        "fraud_flags": fraud,
        "item_checks": item_checks,
        "mental_damage_check": mental_damage,
        "escalation_triggers": triggers,
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
