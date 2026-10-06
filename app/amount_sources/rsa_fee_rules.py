# -*- coding: utf-8 -*-
"""
rsa_fee_rules.py — 依「道路救援服務項目及收費標準」逐項計價的 RSA 規則金額

用途：給法官當 RSA 金額公平性的比對基準（取代費率表有明確價格案件的 Tukey 圍籬）。
法官仍然不修改金額，只用規則金額判斷 claim_amount 是否在容許誤差內。

驗證（2026-10-07，RSA_val.db 339 筆，val 未參與規則調整）：
  費率表有明確價格的 243 筆中，196 筆規則金額落在 signed_amount ±5% 內（81%），多數完全相等。
  議價案件（處理情形含「特殊作業」「載重費」）規則算不出來，回傳 negotiable=True，交回 Tukey。

每個計價項目都標註 source：
  "fee_schedule"  費率表白紙黑字
  "data_inferred" 費率表未載明、從 RSA_train.db 歸納（車價201/501、低底盤），待兆豐確認
"""
import datetime
import re

# 國定假日：資料涵蓋 2026-02~04；其他日期請擴充這份清單（缺漏時只會少算假日加費，不會報錯）
HOLIDAYS = {
    "2026-01-01",
    "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18",
    "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22",
    "2026-02-27", "2026-02-28",
    "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06",
}

HIGHWAY_KW = ("國道", "高速公路", "快速道路", "快速公路", "高架")
EXPRESSWAY_RE = re.compile(r"台\d+線")          # 台61/64/74/76/78… 快速公路
NEGOTIABLE_STEPS = {"特殊作業", "載重費"}
REPAIR_ADDON_STEPS = {"接電排空", "接電", "打氣一輪", "打氣二輪", "打氣三輪", "換備胎", "代送油料"}

# 費率表未載明，從 RSA_train.db 歸納；有「車價」步驟的案件一律不加夜間/假日費
VEHICLE_PRICE_FEE = {"車價0": 0, "車價201": 1500, "車價501": 3500}
LOW_CHASSIS_FEE = 900

# judge-agent 讀 RSA_train.db，category_id 語意以 train 為準（val 的 3/4 對調，呼叫端應傳名稱）
TRAIN_CATEGORY_NAMES = {1: "拖吊", 2: "特殊作業", 3: "道路救援空趟", 4: "急修"}


def is_night_or_holiday(report_date: str, report_time: str) -> bool:
    """夜間 18:00~翌日 08:00、國定假日或週末。"""
    h = int(report_time[:2])
    if h >= 18 or h < 8:
        return True
    if report_date in HOLIDAYS:
        return True
    return datetime.date.fromisoformat(report_date[:10]).weekday() >= 5


def _parking_fee(steps) -> int:
    """停車場加費：地下室/立體停車場 500 元，每增加一樓層加收 300 元（B1/無樓層數視為第一層）。"""
    for s in steps:
        for prefix in ("地下室B", "立體停車場"):
            if s.startswith(prefix):
                digits = "".join(ch for ch in s[len(prefix):] if ch.isdigit())
                floor = int(digits) if digits else 1
                return 500 + 300 * max(0, floor - 1)
    return 0


def resolve_category(case: dict):
    """優先用 category_name；沒有才用 category_id（train 語意）。"""
    name = case.get("category_name")
    if name:
        return name
    cid = case.get("category_id")
    return TRAIN_CATEGORY_NAMES.get(cid) if cid is not None else None


REQUIRED_FIELDS = ("processing_steps", "report_date", "report_time")


def missing_fields(case: dict) -> list:
    miss = [f for f in REQUIRED_FIELDS if not case.get(f)]
    if resolve_category(case) is None:
        miss.append("category_name/category_id")
    if resolve_category(case) in ("拖吊", "特殊作業") and case.get("km") is None:
        miss.append("km")
    return miss


def compute_rule_amount(case: dict) -> dict:
    """回傳 {'applicable', 'amount', 'items', 'negotiable', 'missing_fields', 'reason'}。
    applicable=False 時呼叫端應退回 Tukey 圍籬。"""
    miss = missing_fields(case)
    if miss:
        return {"applicable": False, "amount": None, "items": [], "negotiable": None,
                "missing_fields": miss, "reason": f"缺欄位 {miss}，無法依費率表計價"}

    category = resolve_category(case)
    steps = set(case["processing_steps"])
    km = case.get("km")
    location = case.get("breakdown_location") or ""
    items = []

    negotiable = bool(steps & NEGOTIABLE_STEPS)
    if negotiable:
        return {"applicable": False, "amount": None, "items": [], "negotiable": True, "missing_fields": [],
                "reason": f"處理情形含議價項目 {sorted(steps & NEGOTIABLE_STEPS)}，費率表無固定價格"}

    if category == "急修":
        items.append({"item": "急修(換備胎)" if "換備胎" in steps else "急修", "amount": 800 if "換備胎" in steps else 600,
                      "source": "fee_schedule"})
    elif category in ("拖吊", "特殊作業"):
        highway = any(k in location for k in HIGHWAY_KW) or bool(EXPRESSWAY_RE.search(location))
        items.append({"item": "基本拖吊(10公里內)" + ("-國道/快速道路" if highway else "-一般道路"),
                      "amount": 1500 if highway else 1200, "source": "fee_schedule"})
        if km > 10:
            items.append({"item": f"超程 {km - 10} 公里 × 50", "amount": 50 * (km - 10), "source": "fee_schedule"})
        if "全載拖吊" in steps:
            items.append({"item": "全載", "amount": 500, "source": "fee_schedule"})
        p = _parking_fee(steps)
        if p:
            items.append({"item": "停車場加費", "amount": p, "source": "fee_schedule"})
        if "加人／加車作業" in steps:
            items.append({"item": "加人/加車", "amount": 600, "source": "fee_schedule"})
        if steps & REPAIR_ADDON_STEPS:
            items.append({"item": "拖吊附帶急修", "amount": 600, "source": "fee_schedule"})
        if "國道第二類現場處理" in steps:
            items.append({"item": "國道第二類", "amount": 900, "source": "fee_schedule"})
    elif category == "道路救援空趟":
        items.append({"item": "空趟費", "amount": 300, "source": "fee_schedule"})
    else:
        return {"applicable": False, "amount": None, "items": [], "negotiable": None, "missing_fields": [],
                "reason": f"未知服務類別 {category}"}

    tiers = [s for s in steps if s.startswith("車價")]
    for t in tiers:
        head = t.split("、")[0]
        if VEHICLE_PRICE_FEE.get(head):
            items.append({"item": f"{head} 加價", "amount": VEHICLE_PRICE_FEE[head], "source": "data_inferred"})
        if "低底盤" in t:
            items.append({"item": "低底盤", "amount": LOW_CHASSIS_FEE, "source": "data_inferred"})

    if category != "道路救援空趟" and not tiers and is_night_or_holiday(case["report_date"], case["report_time"]):
        items.append({"item": "夜間/假日加費", "amount": 350, "source": "fee_schedule"})

    return {"applicable": True, "amount": sum(i["amount"] for i in items), "items": items,
            "negotiable": False, "missing_fields": [], "reason": None,
            "uses_data_inferred": any(i["source"] == "data_inferred" for i in items)}
