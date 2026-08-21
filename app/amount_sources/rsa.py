# -*- coding: utf-8 -*-
"""
rsa.py — RSA 金額公平性資料來源（B部分）

比對維度跟 TPL 不同：RSA 無肇責、無傷勢，改用 category_id(拖吊/特殊作業/道路救援
空趟/急修) + km(拖吊公里數) 分箱比對。已用真實 RSA_train.db(700筆)/RSA_val.db(339筆)驗證過。

已知資料落差(誠實揭露，非程式問題)：RSA_train.db 的「道路救援空趟」類別只有 1 筆，
RSA_val.db 卻有 35 筆——這類案件的 fairness_check 幾乎必然 precedent_insufficient。

project_name 欄位(如「自費購道援險(30,000元,60K,3次)」)可解析出保額上限與使用次數
上限，讓 C1(超保額)、B3(超次數)對 RSA 都是真數字。B1(同卡同日同地點重複)、
C3(同卡同日多次請求)也是從真實資料算出，不是假設。
"""
import re
from typing import Optional
from ._db import open_readonly

CATEGORY_NAMES = {1: "拖吊", 2: "特殊作業", 3: "道路救援空趟", 4: "急修"}


def parse_project_limits(project_name: str) -> dict:
    """例：'自費購道援險(30,000元,60K,3次)' → limit=30000, km_cap=60, count_cap=3
    'K' 數字語意未完全確認(單次里程上限或方案總里程)，僅存放不用於判斷。"""
    limit_m = re.search(r"([\d,]+)元", project_name or "")
    km_m = re.search(r"([\d,]+|不限)K", project_name or "")
    cnt_m = re.search(r"([\d,]+|不限)次", project_name or "")
    limit = int(limit_m.group(1).replace(",", "")) if limit_m else None
    km_cap = None if (km_m and km_m.group(1) == "不限") else (int(km_m.group(1).replace(",", "")) if km_m else None)
    count_cap = None if (cnt_m and cnt_m.group(1) == "不限") else (int(cnt_m.group(1).replace(",", "")) if cnt_m else None)
    return {"project_name": project_name, "coverage_limit": limit, "km_cap": km_cap, "count_cap": count_cap}


def get_card_coverage(con, card_no: str) -> dict:
    cur = con.cursor()
    cur.execute("SELECT p.project_name FROM card c JOIN project p ON c.project_id = p.project_id "
                "WHERE c.card_no = ?", (card_no,))
    row = cur.fetchone()
    if row is None:
        return {"card_no": card_no, "found": False}
    return {"card_no": card_no, "found": True, **parse_project_limits(row[0])}


class RSAAmountComparator:
    """比對維度：category_id(必要) + km 分箱。只用 train 當比較池，維持 train/val 隔離。"""

    KM_BINS = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 10**6)]

    def __init__(self, train_db_path: str, min_precedents: int = 3, max_comparables: int = 30):
        self.con = open_readonly(train_db_path)
        self.min_precedents = min_precedents
        self.max_comparables = max_comparables

    def close(self):
        self.con.close()

    def _km_bin(self, km):
        if km is None:
            return None
        for lo, hi in self.KM_BINS:
            if lo <= km < hi:
                return (lo, hi)
        return self.KM_BINS[-1]

    def expected_amounts(self, case: dict) -> dict:
        category_id, km = case.get("category_id"), case.get("km")
        cur = self.con.cursor()
        amounts, tier, match_criteria = [], None, None

        km_bin = self._km_bin(km)
        if category_id is not None and km_bin is not None:
            lo, hi = km_bin
            cur.execute("SELECT signed_amount FROM service_request WHERE category_id = ? "
                        "AND km >= ? AND km < ? LIMIT ?", (category_id, lo, hi, self.max_comparables))
            amounts = [r[0] for r in cur.fetchall() if r[0] is not None]
            if len(amounts) >= self.min_precedents:
                tier = 1
                match_criteria = {"category_id": category_id,
                                   "category_name": CATEGORY_NAMES.get(category_id), "km_bin": [lo, hi]}

        if tier is None and category_id is not None:
            cur.execute("SELECT signed_amount FROM service_request WHERE category_id = ? LIMIT ?",
                        (category_id, self.max_comparables))
            amounts2 = [r[0] for r in cur.fetchall() if r[0] is not None]
            if len(amounts2) >= self.min_precedents:
                amounts, tier = amounts2, 2
                match_criteria = {"category_id": category_id, "category_name": CATEGORY_NAMES.get(category_id)}
            elif len(amounts2) > len(amounts):
                amounts = amounts2

        return {"general_amounts": amounts, "n": len(amounts), "tier": tier, "match_criteria": match_criteria,
                "amount_basis": "RSA_train 實際請款金額(signed_amount)，依服務類別+公里分箱比對",
                "judgment_upper_bound_available": False}


class RSAValCaseLoader:
    """從 RSA 案件庫載入案件並算出真實詐欺信號。正式環境通常不會用到（案件由理賠代理人
    傳入），此為測試/驗證用；但這裡算 B1/B3/C3 的 SQL 邏輯，正式環境的理賠代理人
    產生 case dict 時應該複用同一套查詢。"""

    def __init__(self, db_path: str):
        self.con = open_readonly(db_path)

    def close(self):
        self.con.close()

    def list_request_nos(self, limit: Optional[int] = None) -> list:
        cur = self.con.cursor()
        q = "SELECT request_no FROM service_request ORDER BY request_no" + (f" LIMIT {limit}" if limit else "")
        cur.execute(q)
        return [r[0] for r in cur.fetchall()]

    def load(self, request_no: int, retry_count: int = 0) -> dict:
        cur = self.con.cursor()
        cur.execute("SELECT card_no, driver_id, report_date, report_time, complete_time, "
                    "breakdown_location, vendor_id, category_id, km, signed_amount "
                    "FROM service_request WHERE request_no = ?", (request_no,))
        row = cur.fetchone()
        if row is None:
            return {"case_id": str(request_no), "line": "RSA", "found": False}
        (card_no, driver_id, report_date, report_time, complete_time,
         location, vendor_id, category_id, km, signed_amount) = row
        coverage = get_card_coverage(self.con, card_no) if card_no else {"found": False}

        cur.execute("SELECT COUNT(*) FROM service_request WHERE card_no=? AND report_date=? "
                    "AND breakdown_location=? AND request_no!=?", (card_no, report_date, location, request_no))
        dup_n = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM service_request WHERE card_no=?", (card_no,))
        freq_n = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT request_no) FROM service_request WHERE card_no=? AND report_date=?",
                    (card_no, report_date))
        same_day_n = cur.fetchone()[0]

        return {
            "case_id": str(request_no), "line": "RSA", "found": True, "retry_count": retry_count,
            "claim_amount": signed_amount, "category_id": category_id,
            "category_name": CATEGORY_NAMES.get(category_id), "km": km, "card_no": card_no,
            "report_date": report_date, "report_time": report_time, "complete_time": complete_time,
            "breakdown_location": location, "vendor_id": vendor_id,
            "coverage_limit": coverage.get("coverage_limit"),
            "coverage_km_cap": coverage.get("km_cap"), "coverage_count_cap": coverage.get("count_cap"),
            "coverage_project_name": coverage.get("project_name"),
            "duplicate_claim": dup_n > 0,
            "claim_freq_window": freq_n, "claim_freq_threshold": coverage.get("count_cap"),
            "rescue_count": same_day_n, "rescue_count_threshold": 1,
        }


class MockRSAAmountSource:
    """找不到真實 RSA_train.db 時的備援，讓服務至少能啟動並誠實回報。"""

    def expected_amounts(self, case: dict) -> dict:
        base = case.get("_mock_market") or case.get("claim_amount", 3000) or 3000
        general = [round(base * f) for f in (0.4, 0.6, 0.8, 1.0, 1.1, 1.3, 1.6)]
        return {"general_amounts": general, "n": len(general), "tier": None,
                "match_criteria": {"mock": True}, "amount_basis": "mock(未找到真實 RSA_train.db)",
                "judgment_upper_bound_available": False}
