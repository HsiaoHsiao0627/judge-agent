# -*- coding: utf-8 -*-
"""
mental_damage.py — TPL 精神慰撫金審查（2026-11-03 新增）

TPL 案件除了審理傷害險總額(fairness.py)，另外單獨審理精神慰撫金(請求項目 8)：
理賠代理人建議的精神慰撫金 case["suggested_mental_damage"]（理賠代理人 breakdown.net_mental；
沒有送時退回審保戶請款的 case["claim_item_amounts"]["8"]，並在 amount_source 註明），與「同一個比較池」
(TPL_train，tiered 或 kNN，跟總額用同一套比對方式)中有請求精神慰撫金的先例金額比較，
用與總額相同的 Tukey IQR 圍籬判斷是否合理。

  - 沒有提供精神慰撫金逐項金額 → na（不擋放行，與其他逐項檢查一致）
  - 有提供，但可比對的先例少於 min_precedents → fail（無法確認，轉人工；與總額一致）
  - 落在圍籬外 → fail；落在圍籬內 → pass

只判斷、不修改金額。精神慰撫金與收入無關，因此比較基準取自訓練資料先例；
收入相關的工作損失審查只依案件提供的薪資證明(見 item_checks.py)。
"""
from .config import JudgeConfig
from .fairness import _tukey_fairness

MENTAL_DAMAGE_CODE = "8"


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


class _ItemPool:
    """把比較池的 expected_amounts 轉成單一請求項目版本，讓 _tukey_fairness 可以直接沿用。"""

    def __init__(self, amount_source, item_code):
        self.src, self.code = amount_source, item_code

    def expected_amounts(self, case):
        return self.src.expected_amounts(case, item_code=self.code)


def evaluate_mental_damage(case: dict, cfg: JudgeConfig, amount_source) -> dict:
    if case.get("line") != "TPL":
        return {"applicable": False, "status": "na", "reason": "僅適用 TPL"}
    amt, source = _num(case.get("suggested_mental_damage")), "claim_agent"
    if amt is None or amt <= 0:
        amt, source = _num((case.get("claim_item_amounts") or {}).get(MENTAL_DAMAGE_CODE)), "requested"
    if amt is None or amt <= 0:
        return {"applicable": False, "status": "na", "claim_amount": None, "amount_source": None,
                "reason": "沒有理賠代理人建議或保戶請款的精神慰撫金金額"}
    fair = _tukey_fairness({**case, "claim_amount": amt}, cfg, _ItemPool(amount_source, MENTAL_DAMAGE_CODE))
    if fair.get("precedent_insufficient") or fair.get("within_tolerance") is None:
        status = "fail"
        reason = f"可比對的精神慰撫金先例不足（{fair.get('precedent_n') or 0} 筆），無法判斷是否合理"
    elif fair["within_tolerance"]:
        status = "pass"
        reason = f"精神慰撫金 {amt:,.0f} 落在 {fair['precedent_n']} 筆先例的合理區間 {fair['fair_range']} 內"
    else:
        status = "fail"
        reason = f"精神慰撫金 {amt:,.0f} 落在 {fair['precedent_n']} 筆先例的合理區間 {fair['fair_range']} 之外"
    if source == "requested":
        reason = "（審的是保戶請款金額）" + reason
    return {"applicable": True, "status": status, "claim_amount": amt, "amount_source": source,
            "reason": reason, "fairness": fair}
