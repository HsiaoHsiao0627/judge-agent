# -*- coding: utf-8 -*-
"""
item_checks.py — TPL 逐項金額核對（2026-10 新增）

背景：只看總額時，TPL 同條件案件的實付金額常相差兩倍以上，灌水 2~3 倍的總額混在正常差異
裡，統計圍籬分不出來(留一驗證：漏放天花板約 73%)。若理賠代理人逐項提供金額
(case["claim_item_amounts"] = {"3": 看護費用, "5": 工作損失, ...})，部分項目可以用
「單價 × 天數」的上限核對：

  I3 看護費用 ≤ care_daily_cap × (住院天數 + 醫囑專人照護天數)
  I5 工作損失 ≤ (案件薪資證明的月薪/30) × (住院天數 + 醫囑休養天數)

收入判斷（2026-11-03）：工作損失只依案件本身提供的薪資證明(case["average_salary"]，
驗證時即驗證集的 employment_certificate)與醫囑天數計算，不再乘上從 TPL_train 核定金額
分布推得的倍數——收入不可依訓練資料的經驗判賠。沒有薪資證明就標 na，不從訓練資料補估。
看護費用的每日上限仍由 TPL_train 推得（與收入無關）。
缺任何一項輸入(沒逐項金額、沒薪資、醫囑讀不到天數)一律標 na，不猜測、不觸發。
命中只會讓法官不同意、轉人工，不修改金額。
"""
from .advice_parse import rest_days, care_days
from .config import JudgeConfig


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _item(item_id, label, status, evidence=None, reason=""):
    return {"item_id": item_id, "label": label, "status": status, "evidence": evidence or {}, "reason": reason}


def _amount(case, code):
    amounts = case.get("claim_item_amounts") or {}
    return _num(amounts.get(code))


def check_care(case: dict, cfg: JudgeConfig) -> dict:
    amt = _amount(case, "3")
    if amt is None or amt <= 0:
        return _item("I3_CARE_FEE", "看護費用", "na", reason="無看護費用逐項金額")
    cd = care_days(case.get("doctor_advice"))
    ind = _num(case.get("inpatient_days")) or 0
    days = ind + (cd or 0)
    if days <= 0:
        return _item("I3_CARE_FEE", "看護費用", "na", reason="醫囑未載明照護期間且無住院天數")
    cap = cfg.item_care_daily_cap * days
    hit = amt > cap
    return _item("I3_CARE_FEE", "看護費用", "hit" if hit else "pass",
                 {"amount": amt, "inpatient_days": ind, "care_days_advised": cd, "cap": round(cap)},
                 f"看護費用 {amt:,.0f} 超過上限 {cap:,.0f}（{cfg.item_care_daily_cap:,.0f}/日 × {days:g} 日）" if hit else "")


def check_work_loss(case: dict, cfg: JudgeConfig) -> dict:
    amt = _amount(case, "5")
    if amt is None or amt <= 0:
        return _item("I5_WORK_LOSS", "工作損失", "na", reason="無工作損失逐項金額")
    sal = _num(case.get("average_salary"))
    if not sal or sal <= 0:
        return _item("I5_WORK_LOSS", "工作損失", "na", reason="無薪資證明")
    rd = rest_days(case.get("doctor_advice"))
    ind = _num(case.get("inpatient_days")) or 0
    if rd is None:
        return _item("I5_WORK_LOSS", "工作損失", "na", reason="醫囑未載明休養期間")
    days = ind + rd
    # 收入只採案件提供的薪資證明，不乘任何由訓練資料推得的倍數
    cap = sal / 30 * days
    hit = amt > cap
    return _item("I5_WORK_LOSS", "工作損失", "hit" if hit else "pass",
                 {"amount": amt, "average_salary": sal, "inpatient_days": ind, "rest_days_advised": rd,
                  "cap": round(cap), "income_basis": "案件提供的薪資證明(average_salary)，未使用訓練資料"},
                 f"工作損失 {amt:,.0f} 超過依薪資證明計算的金額 {cap:,.0f}"
                 f"（月薪 {sal:,.0f} ÷ 30 × {days:g} 日）" if hit else "")


def evaluate_items(case: dict, cfg: JudgeConfig) -> dict:
    if case.get("line") != "TPL":
        return {"applicable": False, "hit": False, "items": []}
    items = [check_care(case, cfg), check_work_loss(case, cfg)]
    return {"applicable": any(i["status"] != "na" for i in items),
            "hit": any(i["status"] == "hit" for i in items), "items": items}
