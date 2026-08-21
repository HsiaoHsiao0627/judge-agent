# -*- coding: utf-8 -*-
"""
fraud_rules.py — 詐欺旗標規則 A/B/C/D

每條回傳 status：hit / pass / na。缺資料或不適用一律顯式標 na（不靜默略過）。

已停用(業務決定，非資料缺口)：C4(金額vs傷勢)、C5(金額vs肇責)。
已併入 fairness_check(重複判斷故停用)：C2(拖吊費公式)。
RSA 有真實資料算出的信號(非佔位)：B1(同卡同日同地點重複)、B3(使用次數vs方案上限)、
C3(同卡同日多次請求)——由理賠代理人/RSAValCaseLoader 算好餵入 case dict，
TPL 因無對應資料表仍為 na。
A1-A4(保單時序)：TPL/RSA 目前資料庫皆無保單生效/到期日、報案日欄位，且雙方都確認
目前資料無已知詐保案例可供驗證，維持條件判斷邏輯（有資料判斷、無資料標 na），
不強行找額外資料源。
"""
from datetime import date
from .config import JudgeConfig


def _d(x):
    """容錯把字串/日期轉成 date；None → None。
    真實 TPL accident_date 含時間('2024-10-20 17:55:00')，date.fromisoformat 無法直接解析，
    先截前10碼。"""
    if x is None:
        return None
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def _flag(fid, cat, sev, status, lines, evidence=None, reason=""):
    return {"flag_id": fid, "category": cat, "severity": sev, "status": status,
            "applicable_lines": lines, "evidence": evidence or {}, "reason": reason}


def A1(c, cfg: JudgeConfig):
    a, p = _d(c.get("accident_date")), _d(c.get("policy_effective_date"))
    if a is None or p is None:
        return _flag("A1_ACCIDENT_BEFORE_POLICY", "A", "blocking", "na", ["RSA", "TPL"], reason="缺事故日或保單生效日")
    hit = a < p
    return _flag("A1_ACCIDENT_BEFORE_POLICY", "A", "blocking", "hit" if hit else "pass", ["RSA", "TPL"],
                 {"accident_date": str(a), "policy_effective_date": str(p)}, "事故日早於保單生效日" if hit else "")


def A2(c, cfg: JudgeConfig):
    a, e = _d(c.get("accident_date")), _d(c.get("policy_expiry_date"))
    if a is None or e is None:
        return _flag("A2_ACCIDENT_AFTER_EXPIRY", "A", "blocking", "na", ["RSA", "TPL"], reason="缺事故日或保單到期日")
    hit = a > e
    return _flag("A2_ACCIDENT_AFTER_EXPIRY", "A", "blocking", "hit" if hit else "pass", ["RSA", "TPL"],
                 {"accident_date": str(a), "policy_expiry_date": str(e)}, "事故日晚於保單到期日" if hit else "")


def A3(c, cfg: JudgeConfig):
    a, r = _d(c.get("accident_date")), _d(c.get("report_date"))
    if a is None or r is None:
        return _flag("A3_ACCIDENT_AFTER_REPORT", "A", "blocking", "na", ["RSA", "TPL"], reason="缺事故日或報案日")
    hit = a > r
    return _flag("A3_ACCIDENT_AFTER_REPORT", "A", "blocking", "hit" if hit else "pass", ["RSA", "TPL"],
                 {"accident_date": str(a), "report_date": str(r)}, "事故日晚於報案日(時序矛盾)" if hit else "")


def A4(c, cfg: JudgeConfig):
    a, p = _d(c.get("accident_date")), _d(c.get("policy_effective_date"))
    if a is None or p is None:
        return _flag("A4_CLAIM_SOON_AFTER_POLICY", "A", "warning", "na", ["RSA", "TPL"], reason="缺事故日或保單生效日")
    days = (a - p).days
    hit = 0 <= days <= cfg.policy_new_days
    return _flag("A4_CLAIM_SOON_AFTER_POLICY", "A", "warning", "hit" if hit else "pass", ["RSA", "TPL"],
                 {"days_after_effective": days}, f"投保後 {days} 天內出險" if hit else "")


def A5(c, cfg: JudgeConfig):
    v = c.get("policy_endorsement_recent")
    if v is None:
        return _flag("A5_RECENT_ENDORSEMENT", "A", "warning", "na", ["RSA", "TPL"], reason="無保單異動資料")
    return _flag("A5_RECENT_ENDORSEMENT", "A", "warning", "hit" if v else "pass", ["RSA", "TPL"],
                 {"endorsement_recent": v}, "出險前加保/提高保額" if v else "")


def B1(c, cfg: JudgeConfig):
    v = c.get("duplicate_claim")
    if v is None:
        return _flag("B1_DUPLICATE_CLAIM", "B", "blocking", "na", ["RSA", "TPL"], reason="無重複請領比對資料")
    return _flag("B1_DUPLICATE_CLAIM", "B", "blocking", "hit" if v else "pass", ["RSA", "TPL"],
                 {"duplicate": v}, "同事故重複請領" if v else "")


def B2(c, cfg: JudgeConfig):
    v = c.get("receipt_reused")
    if v is None:
        return _flag("B2_RECEIPT_REUSED", "B", "blocking", "na", ["RSA", "TPL"], reason="無收據號/hash 可比對")
    return _flag("B2_RECEIPT_REUSED", "B", "blocking", "hit" if v else "pass", ["RSA", "TPL"],
                 {"receipt_reused": v}, "單據跨案重用" if v else "")


def B3(c, cfg: JudgeConfig):
    n = c.get("claim_freq_window")
    thr = c.get("claim_freq_threshold", 3)
    if n is None:
        return _flag("B3_HIGH_FREQUENCY", "B", "warning", "na", ["RSA", "TPL"], reason="無請領頻率資料")
    if thr is None:  # RSA「不限次」方案，明確設為 None(非缺資料)
        return _flag("B3_HIGH_FREQUENCY", "B", "warning", "na", ["RSA", "TPL"],
                      {"window_count": n}, reason="方案不限使用次數，此規則不適用")
    hit = n > thr
    return _flag("B3_HIGH_FREQUENCY", "B", "warning", "hit" if hit else "pass", ["RSA", "TPL"],
                 {"window_count": n, "threshold": thr}, f"短期內請領 {n} 次" if hit else "")


def B4(c, cfg: JudgeConfig):
    v = c.get("payout_account_shared")
    if v is None:
        return _flag("B4_SHARED_PAYOUT_ACCOUNT", "B", "warning", "na", ["RSA", "TPL"], reason="無收款帳戶資料")
    return _flag("B4_SHARED_PAYOUT_ACCOUNT", "B", "warning", "hit" if v else "pass", ["RSA", "TPL"],
                 {"shared": v}, "同收款帳戶跨請領人" if v else "")


def C1(c, cfg: JudgeConfig):
    claim, limit = c.get("claim_amount"), c.get("coverage_limit")
    if claim is None or limit is None:
        return _flag("C1_OVER_COVERAGE_LIMIT", "C", "blocking", "na", ["RSA", "TPL"], reason="缺請領金額或保額上限")
    hit = claim > limit
    return _flag("C1_OVER_COVERAGE_LIMIT", "C", "blocking", "hit" if hit else "pass", ["RSA", "TPL"],
                 {"claim": claim, "limit": limit}, "請領金額超過保額上限" if hit else "")


def C2(c, cfg: JudgeConfig):
    """已停用(非缺資料，是重複)：RSAAmountComparator 的 fairness_check 已用
    category+km分箱比對真實 comparable 金額分佈(Tukey圍籬)，涵蓋了『拖吊費是否
    與距離合理對應』的判斷意圖，且比固定費率公式更貼近真實市場價格分佈。"""
    return _flag("C2_TOW_COST_MISMATCH", "C", "warning", "na", ["RSA"],
                 reason="已併入 fairness_check(km分箱比對)，此規則停用避免重複判斷")


def C3(c, cfg: JudgeConfig):
    if c.get("line") != "RSA":
        return _flag("C3_ABNORMAL_RESCUE_COUNT", "C", "warning", "na", ["RSA"], reason="非 RSA 案件")
    n = c.get("rescue_count")
    thr = c.get("rescue_count_threshold", 1)
    if n is None:
        return _flag("C3_ABNORMAL_RESCUE_COUNT", "C", "warning", "na", ["RSA"], reason="缺救援次數")
    hit = n > thr
    return _flag("C3_ABNORMAL_RESCUE_COUNT", "C", "warning", "hit" if hit else "pass", ["RSA"],
                 {"rescue_count": n, "threshold": thr}, f"單一事故救援 {n} 次" if hit else "")


def C4(c, cfg: JudgeConfig):
    """業務決定不做(非資料缺口待補)：schema 沒有經驗證的嚴重度分帶依據，已確認過，
    正式停用，不列入信心/決策計算。"""
    return _flag("C4_AMOUNT_VS_INJURY", "C", "warning", "na", ["TPL"],
                 reason="業務決定不做：無經驗證的嚴重度分帶依據")


def C5(c, cfg: JudgeConfig):
    """業務決定不做(非資料缺口待補)：schema 沒有分攤前總損害額欄位，已確認過，
    正式停用，不列入信心/決策計算。"""
    return _flag("C5_AMOUNT_VS_FAULT", "C", "warning", "na", ["TPL"],
                 reason="業務決定不做：無分攤前總損害額欄位")


def D1(c, cfg: JudgeConfig):
    ra, claim = c.get("receipt_amount"), c.get("claim_amount")
    if ra is None or claim is None:
        return _flag("D1_RECEIPT_AMOUNT_MISMATCH", "D", "warning", "na", ["RSA", "TPL"], reason="缺收據金額")
    hit = ra != claim
    return _flag("D1_RECEIPT_AMOUNT_MISMATCH", "D", "warning", "hit" if hit else "pass", ["RSA", "TPL"],
                 {"receipt_amount": ra, "claim": claim}, "收據金額≠請領金額" if hit else "")


def D2(c, cfg: JudgeConfig):
    rd, ad = _d(c.get("receipt_date")), _d(c.get("accident_date"))
    win = c.get("doc_date_window_days", 30)
    if rd is None or ad is None:
        return _flag("D2_DOC_DATE_MISMATCH", "D", "warning", "na", ["RSA", "TPL"], reason="缺文件或事故日期")
    hit = abs((rd - ad).days) > win
    return _flag("D2_DOC_DATE_MISMATCH", "D", "warning", "hit" if hit else "pass", ["RSA", "TPL"],
                 {"receipt_date": str(rd), "accident_date": str(ad)}, "文件日期與事故日期不一致" if hit else "")


def D3(c, cfg: JudgeConfig):
    v = c.get("doc_internal_conflict")
    if v is None:
        return _flag("D3_DOC_INTERNAL_CONFLICT", "D", "warning", "na", ["RSA", "TPL"], reason="無文件一致性比對")
    return _flag("D3_DOC_INTERNAL_CONFLICT", "D", "warning", "hit" if v else "pass", ["RSA", "TPL"],
                 {"conflict": v}, "文件間日期/地點/當事人矛盾" if v else "")


FRAUD_RULES = [A1, A2, A3, A4, A5, B1, B2, B3, B4, C1, C2, C3, C4, C5, D1, D2, D3]


def evaluate_fraud(case: dict, cfg: JudgeConfig) -> dict:
    flags = [rule(case, cfg) for rule in FRAUD_RULES]
    blocking_hit = any(f["severity"] == "blocking" and f["status"] == "hit" for f in flags)
    warning_score = sum(1 for f in flags if f["severity"] == "warning" and f["status"] == "hit")
    return {"blocking_hit": blocking_hit, "warning_score": warning_score,
            "warning_threshold": cfg.warning_threshold, "flags": flags}
