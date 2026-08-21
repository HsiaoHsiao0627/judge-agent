# -*- coding: utf-8 -*-
"""
from_cases_tpl.py — 從 TPL_train.db 反向生成「金額合不合理」「有沒有詐欺風險」QA

跟 from_cases.py(RSA版) 同一個原則：答案不是LLM編的，是重用你已經驗證過的
fairness.py(Tukey IQR圍籬) 跟 fraud_rules.py(17旗標) 算出來的——這支腳本只是把
既有模組的輸出轉成QA格式，不重新發明判斷邏輯，避免微調資料的答案跟正式服務的
判斷邏輯兩套標準不一致。

TPL跟RSA的差異：
1. amount_source 換成 app.amount_sources.tpl.AmountComparator/TPLValCaseLoader
2. TPL的claim_amount是「傷害險部分實付金額」(injury_amount)，不是RSA的核定金額，
   問題模板措辭跟著調整
3. fraud_rules.py對TPL大部分規則會是na(B1/B3/C3需要RSA理賠代理人算好餵入的欄位，
   TPL資料庫沒有對應表)——這是誠實反映資料現況，不是腳本bug，答案裡會如實呈現
   「幾項查得了/幾項查不了」

先例不足(precedent_insufficient)的案件不產生金額合理性QA——跟RSA版同一個理由：
沒有比較基準的判斷本身就不該教給模型當作範例。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import JudgeConfig
from app.fairness import fairness_check
from app.fraud_rules import evaluate_fraud
from app.amount_sources.tpl import AmountComparator, TPLValCaseLoader


def _amount_qa(case: dict, fair: dict) -> dict:
    case_no = case.get("case_id")
    claim = case.get("claim_amount")
    injury_types = "、".join(case.get("injury_types") or []) or "未標記傷害類型"
    own_fault = case.get("own_fault_pct")
    lo, hi = fair["fair_range"]
    median = fair["median"]
    within = fair["within_tolerance"]

    question = (f"TPL第三人責任險案件(案號{case_no})，傷害類型為{injury_types}，"
                f"己方過失比例{own_fault}%，傷害險部分實付金額{claim:,.0f}元，這個金額合理嗎？")
    if within:
        answer = (f"合理。根據同肇責比例/傷害類型比對池的{fair['precedent_n']}筆歷史案例統計"
                   f"(Tukey IQR圍籬，比對層級tier{fair['tier']}，中位數{median:,.0f}元，"
                   f"合理區間{lo:,.0f}~{hi:,.0f}元)，本案{claim:,.0f}元落在合理區間內。")
    else:
        answer = (f"不在合理區間內，需要留意。根據同肇責比例/傷害類型比對池的{fair['precedent_n']}筆"
                   f"歷史案例統計(比對層級tier{fair['tier']}，中位數{median:,.0f}元，"
                   f"合理區間{lo:,.0f}~{hi:,.0f}元)，本案{claim:,.0f}元超出此區間，"
                   f"建議退回理賠代理人重新核算或補充說明差異原因。")
    if fair.get("extreme_dispersion"):
        answer += "（提醒：本案比對池離散度極高，屬於重尾分佈的極端案例，判斷時信心應相應調降。）"
    return {"question": question, "answer": answer, "task_type": "amount_fairness",
            "case_id": case_no, "line": "TPL", "grounded_by": "fairness.py(Tukey IQR)"}


def _fraud_qa(case: dict, fraud: dict) -> dict:
    """改版：不只講「幾項查得了/查不了」的聚合摘要，而是列出實際查得了哪條規則、
    帶入這個案件的具體數字(claim/limit)——這樣答案文字才會隨案件變化，
    不會696個案件產出幾乎一樣的模板。

    對真實TPL_train.db 696筆案件實測：17條規則裡，只有C1(請領金額 vs 保額上限)
    對TPL有真實資料可查，其餘16條(A1-A5保單時序、B1-B4重複請領信號、C2-C5停用、
    D1-D3單據比對)全部因TPL資料庫沒有對應欄位或業務決定停用而是na——這是
    fraud_rules.py檔頭已經寫明的事實，不是這次才發現。C1在696筆裡從未觸發過
    (claim_amount從未超過coverage_limit)，但claim/limit的實際數字每案不同，
    帶入具體數字後696筆裡有681組是唯一的(不含14組真的剛好金額相同的案件)。
    """
    case_no = case.get("case_id")
    claim = case.get("claim_amount")
    limit = case.get("coverage_limit")

    hits = [f for f in fraud["flags"] if f["status"] == "hit"]
    checked = [f for f in fraud["flags"] if f["status"] != "na"]
    na_flags = [f for f in fraud["flags"] if f["status"] == "na"]

    question = f"TPL第三人責任險案件(案號{case_no})，有沒有詐欺風險？"

    if fraud["blocking_hit"]:
        reasons = "、".join(f["reason"] for f in hits if f["severity"] == "blocking")
        answer = f"有，觸發硬性詐欺旗標：{reasons}。此類案件應直接轉人工複核，不可自動放行或拒賠。"
        return {"question": question, "answer": answer, "task_type": "fraud_risk",
                "case_id": case_no, "line": "TPL", "grounded_by": "fraud_rules.py(17旗標)"}

    if hits:
        reasons = "、".join(f["reason"] for f in hits)
        answer = f"有警示訊號但非硬性阻擋：{reasons}。建議列入觀察，但不需直接轉人工。"
        return {"question": question, "answer": answer, "task_type": "fraud_risk",
                "case_id": case_no, "line": "TPL", "grounded_by": "fraud_rules.py(17旗標)"}

    # 目前TPL資料現況下，checked幾乎必為[C1]，但寫成通用邏輯，若之後TPL補齊
    # 保單時序/重複請領等欄位，其他規則自然會加入checked清單，不用改這支腳本
    if checked:
        checked_desc = "、".join(
            f"{f['flag_id']}(請領金額{claim:,.0f}元 vs 保額上限{limit:,.0f}元，未超過)"
            if f["flag_id"] == "C1_OVER_COVERAGE_LIMIT" and claim is not None and limit is not None
            else f"{f['flag_id']}(通過)"
            for f in checked
        )
    else:
        checked_desc = "無"

    # na按(category, reason)分組，不能只取category裡第一個reason當代表——
    # 同category內不同規則的na原因其實不一樣(例如C類：C2是RSA專屬的
    # 「已併入fairness_check(km分箱比對)」，C3是「非RSA案件」，C4/C5是
    # 「業務決定不做」，三種原因完全不同，混在一起用同一句話描述會失真)
    na_groups = {}
    for f in na_flags:
        key = (f["category"], f.get("reason", ""))
        na_groups.setdefault(key, []).append(f["flag_id"])
    category_names = {"A": "保單時序", "B": "重複請領/帳戶信號", "C": "金額/嚴重度比對", "D": "單據一致性"}
    na_desc_parts = []
    for (cat, reason), flag_ids in sorted(na_groups.items()):
        na_desc_parts.append(f"{category_names.get(cat, cat)}類{'/'.join(flag_ids)}({reason})")
    na_desc = "、".join(na_desc_parts)

    answer = (f"本案實際可查的規則是：{checked_desc}，均未觸發。"
              f"其餘{len(na_flags)}項規則標記na：{na_desc}。"
              f"na不代表沒問題，只是這些維度目前資料查不了。")

    return {"question": question, "answer": answer, "task_type": "fraud_risk",
            "case_id": case_no, "line": "TPL", "grounded_by": "fraud_rules.py(17旗標)"}


def generate_case_qa(train_db_path: str, output_path: str, limit: int = None) -> int:
    cfg = JudgeConfig()
    comparator = AmountComparator(train_db_path, min_precedents=cfg.min_precedents)
    loader = TPLValCaseLoader(train_db_path)

    case_nos = loader.list_case_nos(limit=limit)
    qa_pairs = []
    skipped_insufficient = 0

    for case_no in case_nos:
        case = loader.load(case_no)
        if not case.get("found"):
            continue

        fair = fairness_check(case, cfg, comparator)
        if fair.get("precedent_insufficient") or fair.get("within_tolerance") is None:
            skipped_insufficient += 1
        else:
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
    n = generate_case_qa("data/TPL_train.db", "qa_generation/output_cases_tpl.jsonl")
    print(f"產生 {n} 筆QA，寫入 qa_generation/output_cases_tpl.jsonl")
