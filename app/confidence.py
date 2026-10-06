# -*- coding: utf-8 -*-
"""
confidence.py — 信心分數 + 詐欺檢查覆蓋率（獨立稽核指標）

信心分數只取決於：(1) 先例支持度 (2) comparable pool 離散度。

刻意排除的兩件事，都是實測時真的踩到的 bug：
  - claim 偏離度：偏離大是「該退回重算」的訊號，不代表法官對自身判斷沒把握；
    算進信心會讓大幅偏離案件被 low_confidence 攔去人工，使 return_for_recalc 迴圈失效。
  - 詐欺規則的 na 比例：A1-A5/B1/B2/B4/D1-D3 在真實 TPL 資料上結構性缺欄位，
    na 比例恆常偏高(~94%)，算進信心會讓每筆案件恆低於門檻、全部轉人工，
    等於讓「系統還沒接的檢查維度」偽裝成「這個案子不確定」。
    詐欺檢查覆蓋率改用獨立的 fraud_check_coverage 呈現，僅供稽核參考，不影響決策。
"""
from .config import JudgeConfig


def compute_confidence(fair: dict, fraud: dict, cfg: JudgeConfig) -> float:
    if fair.get("basis") == "fee_schedule_rule":
        # 規則路徑不依賴先例池，先例數/離散度公式不適用
        return cfg.rsa_rule_confidence
    conf = 1.0
    n = fair.get("precedent_n", 0)
    if n < cfg.min_precedents:
        conf *= 0.5
    else:
        conf *= min(1.0, 0.6 + 0.1 * n)

    dispersion_ratio = fair.get("dispersion_ratio")
    if dispersion_ratio is not None:
        conf *= max(0.4, 1.0 - 0.15 * dispersion_ratio)
        if fair.get("extreme_dispersion"):
            conf *= 0.8

    return round(max(0.0, min(1.0, conf)), 3)


def fraud_check_coverage(fraud: dict) -> dict:
    """獨立稽核指標：詐欺規則裡有多少比例『實際查得動』(status != na)。"""
    flags = fraud["flags"]
    checked = [f for f in flags if f["status"] != "na"]
    return {"total_rules": len(flags), "checked_rules": len(checked),
            "coverage_pct": round(len(checked) / len(flags) * 100, 1) if flags else 0.0,
            "na_flag_ids": [f["flag_id"] for f in flags if f["status"] == "na"]}
