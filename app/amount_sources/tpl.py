# -*- coding: utf-8 -*-
"""
tpl.py — TPL 金額公平性資料來源（B部分）

金額比對邏輯：只用 TPL_train 當比較池，依 own_fault_pct + injury_types 三級放寬比對，
維持 train/val 隔離。已用真實 TPL_train.db(696筆)/TPL_val.db(297筆)驗證過。

判決語料庫(case_features.db)沒有金額欄位，所以「判決書金額上界」這個概念不存在——
這不是疏漏，是資料本身的限制，也剛好呼應「判決書是非典型爭議案件」的選擇偏誤原則：
只用一般案例金額當基準反而更乾淨。
"""
from typing import Optional
from ._db import open_readonly


def get_case_claim_breakdown(con, case_no: str) -> dict:
    cur = con.cursor()
    cur.execute("SELECT coverage_name, payment_amount FROM payments WHERE case_no = ?", (case_no,))
    rows = cur.fetchall()
    if not rows:
        return {"case_no": case_no, "found": False}
    injury, property_, other, has_null = 0.0, 0.0, 0.0, False
    for name, amt in rows:
        if amt is None:
            has_null = True
            continue
        if "傷害" in name:
            injury += amt
        elif "財損" in name:
            property_ += amt
        else:
            other += amt  # 超額責任附加條款等，歸屬不明確，不算進任一上限
    return {"case_no": case_no, "found": True, "injury_amount": round(injury, 2),
            "property_amount": round(property_, 2), "other_amount": round(other, 2),
            "total_amount": round(injury + property_ + other, 2),
            "has_null_payment_rows": has_null, "n_payment_rows": len(rows)}


def get_case_profile(con, case_no: str) -> dict:
    cur = con.cursor()
    cur.execute("SELECT own_fault_pct, other_fault_pct, third_fault_pct, accident_cause, "
                "accident_date FROM cases WHERE case_no = ?", (case_no,))
    row = cur.fetchone()
    if row is None:
        return {"case_no": case_no, "found": False}
    own_fault, other_fault, third_fault, cause, accident_date = row
    cur.execute("SELECT DISTINCT injury_type, disability_level FROM claim_episodes WHERE case_no = ?", (case_no,))
    ep = cur.fetchall()
    injury_types = {r[0] for r in ep if r[0]}
    disability_levels = {r[1] for r in ep if r[1] and r[1] != "(null)"}
    return {"case_no": case_no, "found": True, "own_fault_pct": own_fault, "other_fault_pct": other_fault,
            "third_fault_pct": third_fault, "accident_cause": cause, "accident_date": accident_date,
            "injury_types": injury_types, "has_disability": bool(disability_levels),
            "disability_levels": disability_levels}


def get_coverage_limits(con, case_no: str) -> dict:
    cur = con.cursor()
    cur.execute("SELECT coverage_type, insured_amount FROM coverages WHERE case_no = ?", (case_no,))
    injury_base, injury_riders, property_base = 0.0, 0.0, 0.0
    for ctype, amt in cur.fetchall():
        if amt is None:
            continue
        if ctype.startswith("31-"):
            injury_base += amt
        elif ctype.startswith(("33-", "86-")):
            injury_riders += amt
        elif ctype.startswith("32-"):
            property_base += amt
    return {"injury_limit_total": round(injury_base + injury_riders, 2) if (injury_base or injury_riders) else None,
            "property_limit": round(property_base, 2) if property_base else None}


class AmountComparator:
    """只用 TPL_train 當比較池，維持 train/val 隔離；三級放寬比對。"""

    def __init__(self, train_db_path: str, min_precedents: int = 3,
                 max_comparables: int = 30, fault_tolerance_tier2: float = 15.0):
        self.con = open_readonly(train_db_path)
        self.min_precedents = min_precedents
        self.max_comparables = max_comparables
        self.fault_tolerance_tier2 = fault_tolerance_tier2

    def close(self):
        self.con.close()

    def _amounts_for(self, case_nos):
        if not case_nos:
            return []
        cur = self.con.cursor()
        qmarks = ",".join("?" * len(case_nos))
        cur.execute(
            f"SELECT case_no, SUM(CASE WHEN coverage_name LIKE '%傷害%' THEN payment_amount "
            f"ELSE 0 END) AS inj FROM payments WHERE case_no IN ({qmarks}) GROUP BY case_no",
            list(case_nos),
        )
        return [(r[0], r[1]) for r in cur.fetchall() if r[1] is not None and r[1] > 0]

    def expected_amounts(self, case: dict) -> dict:
        own_fault = case.get("own_fault_pct")
        injury_types = set(case.get("injury_types") or [])
        cause = case.get("accident_cause")
        cur = self.con.cursor()

        tier1_cases = []
        if own_fault is not None:
            cur.execute("SELECT case_no FROM cases WHERE own_fault_pct = ?", (own_fault,))
            candidates = [r[0] for r in cur.fetchall()]
            if injury_types and candidates:
                qm = ",".join("?" * len(candidates))
                im = ",".join("?" * len(injury_types))
                cur.execute(
                    f"SELECT DISTINCT case_no FROM claim_episodes WHERE case_no IN ({qm}) "
                    f"AND injury_type IN ({im})", candidates + list(injury_types),
                )
                tier1_cases = [r[0] for r in cur.fetchall()]
            else:
                tier1_cases = candidates

        amounts = self._amounts_for(tier1_cases[:self.max_comparables])
        tier, match_criteria = 1, {"own_fault_pct": own_fault, "injury_types": list(injury_types)}

        if len(amounts) < self.min_precedents:
            tier2_cases = []
            if own_fault is not None:
                cur.execute(
                    "SELECT case_no FROM cases WHERE own_fault_pct BETWEEN ? AND ?",
                    (own_fault - self.fault_tolerance_tier2, own_fault + self.fault_tolerance_tier2),
                )
                tier2_cases = [r[0] for r in cur.fetchall()]
            amounts2 = self._amounts_for(tier2_cases[:self.max_comparables])
            if len(amounts2) >= self.min_precedents:
                amounts, tier = amounts2, 2
                match_criteria = {"own_fault_pct_range":
                    [own_fault - self.fault_tolerance_tier2, own_fault + self.fault_tolerance_tier2]}

        if len(amounts) < self.min_precedents and cause is not None:
            cur.execute("SELECT case_no FROM cases WHERE accident_cause = ?", (cause,))
            tier3_cases = [r[0] for r in cur.fetchall()]
            amounts3 = self._amounts_for(tier3_cases[:self.max_comparables])
            if len(amounts3) > len(amounts):
                amounts, tier = amounts3, 3
                match_criteria = {"accident_cause": cause}

        return {"general_amounts": [a for _, a in amounts], "n": len(amounts),
                "tier": tier if amounts else None, "match_criteria": match_criteria,
                "amount_basis": "TPL_train 實付理賠金額(傷害險部分)，判決語料無金額欄位不適用",
                "judgment_upper_bound_available": False}


class TPLValCaseLoader:
    """從 TPL 案件庫載入案件，組成法官代理人可吃的 case dict。
    正式環境通常不會用到這個 class（案件由理賠代理人傳入），此為測試/驗證用。"""

    def __init__(self, db_path: str):
        self.con = open_readonly(db_path)

    def close(self):
        self.con.close()

    def list_case_nos(self, limit: Optional[int] = None) -> list:
        cur = self.con.cursor()
        q = "SELECT case_no FROM cases ORDER BY case_no" + (f" LIMIT {limit}" if limit else "")
        cur.execute(q)
        return [r[0] for r in cur.fetchall()]

    def load(self, case_no: str, retry_count: int = 0) -> dict:
        breakdown = get_case_claim_breakdown(self.con, case_no)
        profile = get_case_profile(self.con, case_no)
        limits = get_coverage_limits(self.con, case_no)
        if not breakdown.get("found") or not profile.get("found"):
            return {"case_id": case_no, "line": "TPL", "found": False}
        return {
            "case_id": case_no, "line": "TPL", "found": True, "retry_count": retry_count,
            "claim_amount": breakdown["injury_amount"], "claim_breakdown": breakdown,
            "coverage_limit": limits.get("injury_limit_total"), "coverage_limits_detail": limits,
            "own_fault_pct": profile["own_fault_pct"], "other_fault_pct": profile["other_fault_pct"],
            "injury_types": list(profile["injury_types"]), "has_disability": profile["has_disability"],
            "disability_levels": list(profile["disability_levels"]),
            "accident_cause": profile["accident_cause"], "accident_date": profile["accident_date"],
            "has_null_payment_rows": breakdown["has_null_payment_rows"],
        }


class MockAmountSource:
    """找不到真實 TPL_train.db 時的備援，讓服務至少能啟動並誠實回報。"""

    def expected_amounts(self, case: dict) -> dict:
        base = case.get("_mock_market") or case.get("claim_amount", 100_000) or 100_000
        general = [round(base * f) for f in (0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.4, 1.9, 2.6)]
        return {"general_amounts": general, "n": len(general), "tier": None,
                "match_criteria": {"mock": True}, "amount_basis": "mock(未找到真實 TPL_train.db)",
                "judgment_upper_bound_available": False}
