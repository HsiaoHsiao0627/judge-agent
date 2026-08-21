# -*- coding: utf-8 -*-
"""
from_cases.py — 從 RSA_train.db 反向生成「金額合不合理」「有沒有詐欺風險」QA

原則：答案不是LLM編的，是重用你已經驗證過的 fairness.py(Tukey IQR圍籬) 跟
fraud_rules.py(17旗標) 算出來的——這支腳本只是把既有模組的輸出轉成QA格式，
不重新發明判斷邏輯，避免微調資料的答案跟正式服務的判斷邏輯兩套標準不一致。

每筆案件產生兩種QA：
1. 金額合理性QA：問「這個案件核定金額合不合理」，答案引用Tukey圍籬的
   fair_range/median，並誠實說明案件是否落在區間內
2. 詐欺風險QA：問「有沒有詐欺風險」，答案列出真的hit的旗標(不是na)，
   誠實區分「查過沒問題」跟「這個維度查不了」

先例不足(precedent_insufficient)的案件不產生金額合理性QA——沒有比較基準的
判斷本身就不該教給模型當作範例，這種案件在正式服務裡也是直接轉人工，
不該學會「憑空判斷」。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import JudgeConfig
from app.fairness import fairness_check
from app.fraud_rules import evaluate_fraud
from app.amount_sources.rsa import RSAAmountComparator, RSAValCaseLoader, CATEGORY_NAMES


def _amount_qa(case: dict, fair: dict) -> dict:
    cat_name = case.get("category_name") or "此類別"
    km = case.get("km")
    claim = case.get("claim_amount")
    lo, hi = fair["fair_range"]
    median = fair["median"]
    within = fair["within_tolerance"]
    question = f"RSA道路救援案件，服務類別為{cat_name}，拖吊/服務里程{km}公里，核定金額{claim:,.0f}元，這個金額合理嗎？"
    if within:
        answer = (f"合理。根據同類別、同里程區間的{fair['precedent_n']}筆歷史案例統計"
                   f"(Tukey IQR圍籬，中位數{median:,.0f}元，合理區間{lo:,.0f}~{hi:,.0f}元)，"
                   f"本案{claim:,.0f}元落在合理區間內。")
    else:
        answer = (f"不在合理區間內，需要留意。根據同類別、同里程區間的{fair['precedent_n']}筆歷史案例統計"
                   f"(中位數{median:,.0f}元，合理區間{lo:,.0f}~{hi:,.0f}元)，"
                   f"本案{claim:,.0f}元超出此區間，建議退回理賠代理人重新核算或補充說明差異原因。")
    return {"question": question, "answer": answer, "task_type": "amount_fairness",
            "case_id": case.get("case_id"), "grounded_by": "fairness.py(Tukey IQR)"}


def _fraud_qa(case: dict, fraud: dict) -> dict:
    hits = [f for f in fraud["flags"] if f["status"] == "hit"]
    checked_n = sum(1 for f in fraud["flags"] if f["status"] != "na")
    question = f"RSA道路救援案件(案號{case.get('case_id')})，有沒有詐欺風險？"
    if fraud["blocking_hit"]:
        reasons = "、".join(f["reason"] for f in hits if f["severity"] == "blocking")
        answer = f"有，觸發硬性詐欺旗標：{reasons}。此類案件應直接轉人工複核，不可自動放行或拒賠。"
    elif hits:
        reasons = "、".join(f["reason"] for f in hits)
        answer = f"有警示訊號但非硬性阻擋：{reasons}。建議列入觀察，但不需直接轉人工。"
    else:
        answer = (f"目前檢查的{checked_n}項詐欺規則均未觸發。"
                   f"但本案有17項規則中的{17-checked_n}項因資料欄位缺失無法檢查(誠實標記為na，"
                   f"不代表沒問題，只是這些維度目前查不了)。")
    return {"question": question, "answer": answer, "task_type": "fraud_risk",
            "case_id": case.get("case_id"), "grounded_by": "fraud_rules.py(17旗標)"}


def generate_case_qa(train_db_path: str, output_path: str, limit: int = None) -> int:
    cfg = JudgeConfig()
    comparator = RSAAmountComparator(train_db_path, min_precedents=cfg.min_precedents)
    loader = RSAValCaseLoader(train_db_path)

    request_nos = loader.list_request_nos(limit=limit)
    qa_pairs = []
    skipped_insufficient = 0

    for request_no in request_nos:
        case = loader.load(request_no)
        if not case.get("found"):
            continue

        fair = fairness_check(case, cfg, comparator)
        if fair.get("precedent_insufficient") or fair.get("within_tolerance") is None:
            skipped_insufficient += 1
            continue
        qa_pairs.append(_amount_qa(case, fair))

        fraud = evaluate_fraud(case, cfg)
        qa_pairs.append(_fraud_qa(case, fraud))

    comparator.close()
    loader.close()

    with open(output_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"跳過 {skipped_insufficient} 筆先例不足的案件(不產生金額QA，避免教模型憑空判斷)")
    return len(qa_pairs)


if __name__ == "__main__":
    n = generate_case_qa("data/RSA_train.db", "qa_generation/output_cases.jsonl")
    print(f"產生 {n} 筆QA，寫入 qa_generation/output_cases.jsonl")
