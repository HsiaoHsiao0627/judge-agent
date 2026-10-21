# -*- coding: utf-8 -*-
"""
fairness.py — 金額公平性檢核（Tukey IQR 圍籬 + 離散度信心懲罰）

為什麼改設計：對真實 297 筆 TPL_val 案件實測，comparable pool 的離散度(IQR/中位數)
只有 0.7%(2/297)低到讓固定 ±20% 站得住；其餘 99.3% 案件的先例池天生重尾(少數重傷案件
金額極高、多數不高，是理賠金額的常見特性)。固定 ±20% 對幾乎所有案件都會系統性誤判。

改用 Tukey IQR 圍籬當主要判斷依據（不是「正常/離散過高」二元切換，因為真實資料顯示
幾乎不存在「正常」那一邊）：
  邊界 = [max(0, Q1-k×IQR), Q3+k×IQR]，k 是可調的寬鬆度旋鈕
  (對真實 val 資料實測：TPL k=1.5→攔5.1%, k=1.0→7.7%, k=0.5→11.8%；RSA k=1.5→攔2.0%)
  離散度(dispersion_ratio = IQR/median) 是連續信心懲罰因子，不是決策開關
  落在離散度極端尾端(預設 p90≈1.6)的案件，額外標記 extreme_dispersion
"""
import statistics as _stats
from .config import JudgeConfig
from .amount_sources.rsa_fee_rules import compute_rule_amount


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _rule_fairness(case: dict, cfg: JudgeConfig, rule: dict) -> dict:
    """RSA 費率表規則路徑：claim_amount 與規則金額相差在 ±rsa_rule_tolerance 內即合理。"""
    claim = case.get("claim_amount")
    expected = rule["amount"]
    tol = cfg.rsa_rule_tolerance
    lo, hi = expected * (1 - tol), expected * (1 + tol)
    within = None if claim is None else (lo <= claim <= hi)
    deviation_pct = (claim - expected) / expected * 100.0 if (claim is not None and expected) else None
    return {
        "claim_amount": claim, "median": expected,
        "fair_range": [round(lo), round(hi)], "fair_range_basis": f"fee_schedule_rule(±{tol * 100:g}%)",
        "basis": "fee_schedule_rule",
        "rule_amount": expected, "rule_items": rule["items"],
        "rule_uses_data_inferred": rule.get("uses_data_inferred", False),
        "q1": None, "q3": None, "iqr": None, "dispersion_ratio": None,
        "extreme_dispersion": False, "extreme_dispersion_threshold": cfg.extreme_dispersion_threshold,
        "deviation_pct": round(deviation_pct, 2) if deviation_pct is not None else None,
        "within_tolerance": within,
        "precedent_insufficient": False, "precedent_n": None, "tier": "rule",
        "match_criteria": None,
        "amount_basis": "道路救援服務項目及收費標準逐項計價（部分項目為資料歸納，見 rule_items[].source）",
        "judgment_amounts_available": False,
    }


def fairness_check(case: dict, cfg: JudgeConfig, amount_source) -> dict:
    """amount_source 需實作 expected_amounts(case) -> dict，
    回傳 {'general_amounts', 'n', 'tier', 'match_criteria', 'amount_basis', ...}。
    RSA 且 rsa_rule_check_enabled：先試費率表規則；規則不適用（議價/缺欄位）才走 Tukey，
    並把不適用原因放在 rule_check 欄位供稽核。"""
    rule_check = None
    if case.get("line") == "RSA" and cfg.rsa_rule_check_enabled:
        rule = compute_rule_amount(case)
        if rule["applicable"]:
            return _rule_fairness(case, cfg, rule)
        rule_check = {"applicable": False, "reason": rule["reason"],
                      "negotiable": rule["negotiable"], "missing_fields": rule["missing_fields"]}
    out = _tukey_fairness(case, cfg, amount_source)
    if rule_check is not None:
        out["rule_check"] = rule_check
    return out


def _tukey_fairness(case: dict, cfg: JudgeConfig, amount_source) -> dict:
    claim = case.get("claim_amount")
    exp = amount_source.expected_amounts(case)
    amounts = sorted(exp.get("general_amounts") or [])
    n = exp.get("n", len(amounts))
    tier = exp.get("tier")

    precedent_insufficient = n < cfg.min_precedents
    if not amounts or claim is None:
        return {"claim_amount": claim, "fair_range": None, "median": None,
                "dispersion_ratio": None, "basis": "no_comparables",
                "within_tolerance": None, "deviation_pct": None,
                "precedent_insufficient": True, "precedent_n": n, "tier": tier,
                "extreme_dispersion": False,
                "judgment_amounts_available": exp.get("judgment_upper_bound_available", False)}

    median = _stats.median(amounts)
    q1, q3 = _percentile(amounts, 0.25), _percentile(amounts, 0.75)
    iqr = q3 - q1
    dispersion_ratio = (iqr / median) if median else None

    k = (getattr(cfg, "tukey_k_by_line", None) or {}).get(case.get("line"), cfg.tukey_k)
    lo = max(0.0, q1 - k * iqr)
    hi = q3 + k * iqr
    within = lo <= claim <= hi
    deviation_pct = (claim - median) / median * 100.0 if median else None

    extreme_dispersion = (dispersion_ratio is not None
                          and dispersion_ratio > cfg.extreme_dispersion_threshold)

    return {
        "claim_amount": claim, "median": round(median),
        "fair_range": [round(lo), round(hi)], "fair_range_basis": f"tukey_iqr_fence(k={k})",
        "basis": "tukey_iqr",
        "q1": round(q1), "q3": round(q3), "iqr": round(iqr),
        "dispersion_ratio": round(dispersion_ratio, 3) if dispersion_ratio is not None else None,
        "extreme_dispersion": extreme_dispersion,
        "extreme_dispersion_threshold": cfg.extreme_dispersion_threshold,
        "deviation_pct": round(deviation_pct, 2) if deviation_pct is not None else None,
        "within_tolerance": within,
        "precedent_insufficient": precedent_insufficient, "precedent_n": n, "tier": tier,
        "match_criteria": exp.get("match_criteria"),
        "amount_basis": exp.get("amount_basis", "TPL_train 實付理賠金額(一般案例)"),
        "judgment_amounts_available": exp.get("judgment_upper_bound_available", False),
    }
