# -*- coding: utf-8 -*-
"""
from_graph_tpl.py — 從 GraphRAG API 的 /v1/retrieve/features 端點反向生成
判決先例肇責/傾向一致性QA(對應 app/fault_consistency.py 的 A 部分)

架構原則跟 from_graph.py(RSA版) 一致：不直接連Neo4j，打GraphRAG API的HTTP端點
(POST /v1/retrieve/features)，理由見 app/fault_consistency.py 檔頭說明——這支端點是
純結構化Jaccard比對(injuries/bodyparts交集)，不呼叫LLM、不含向量rerank，只提供
法律佐證，不影響金額判斷。

跟 from_graph.py 的差異：
1. RSA版用 /v1/retrieve/rsa_similar 找「相似案例的實付金額」；TPL版用
   /v1/retrieve/features 找「傷勢/肇責特徵相近的判決先例」，回傳的是肇責比例分佈跟
   判決傾向，不是金額——這是 fault_consistency.py 已經講清楚的架構隔離，
   Precedent schema本身沒有金額欄位，這條線不該、也不會碰 fairness_check
2. judgment_fault_pct 的警語必須原樣帶進QA答案：GraphRAG API明確標註這個數字
   「歸屬對象未確認，不可直接類比查詢案件」，只能當背景參考，QA答案不能把它包裝成定論
3. body_parts資料缺口誠實反映：TPL案件dict目前沒有body_parts欄位(理賠代理人介面
   尚未對齊)，一律傳空陣列，QA答案裡明確標記 bodyparts_available=False，
   不假裝有資料——這跟 from_graph.py 對 quality_flags 不能隱藏是同一個原則
"""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.amount_sources.tpl import TPLValCaseLoader


def _fetch_features(base_url: str, api_key: str, injuries: list, bodyparts: list,
                     top_k: int = 8, min_strength: int = 2, min_jaccard: float = 0.15,
                     timeout: float = 10.0) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = httpx.post(
        f"{base_url.rstrip('/')}/v1/retrieve/features",
        json={"injuries": injuries, "bodyparts": bodyparts, "top_k": top_k,
              "min_strength": min_strength, "min_jaccard": min_jaccard},
        headers=headers, timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _fault_consistency_qa(case: dict, features_resp: dict, bodyparts_available: bool) -> dict:
    case_no = case.get("case_id")
    injury_types = "、".join(case.get("injury_types") or []) or "未標記傷害類型"
    own_fault = case.get("own_fault_pct")
    question = (f"TPL第三人責任險案件(案號{case_no})，傷害類型為{injury_types}，"
                f"己方過失比例{own_fault}%，有沒有類似傷勢/肇責特徵的判決先例可供參考？")

    precedents = features_resp.get("precedents", [])
    bp_note = "" if bodyparts_available else "（本案body_parts欄位目前缺失，此次檢索僅以injury_types比對，非完整比對）"

    if not precedents:
        answer = (f"結構化篩選後無符合先例(Jaccard門檻內)。{bp_note}"
                   f"此類案件應視為先例覆蓋不足，建議轉人工複核，不宜直接類比其他判決的肇責比例或傾向。")
        return {"question": question, "answer": answer, "task_type": "fault_consistency",
                "case_id": case_no, "line": "TPL",
                "grounded_by": "GraphRAG API(/v1/retrieve/features)"}

    fault_pcts = [p.get("judgment_fault_pct") for p in precedents if p.get("judgment_fault_pct") is not None]
    dispositions = [p.get("disposition_type") for p in precedents if p.get("disposition_type")]
    citations = [p.get("statute_no") for p in precedents if p.get("statute_no")]

    lines = [f"結構化篩選後找到{len(precedents)}筆傷勢/肇責特徵相近的判決先例。{bp_note}"]
    if fault_pcts:
        lines.append(f"這些判決中出現的肇責比例有：{fault_pcts}（歸屬對象未確認，不可直接類比本案，僅供背景參考，"
                      f"不能當作定論使用）。")
    if dispositions:
        lines.append(f"判決傾向分佈：{'、'.join(dispositions)}。")
    if citations:
        lines.append(f"相關法條引用：{'、'.join(citations)}。")
    lines.append("此檢索結果只提供法律佐證，不影響金額判斷，金額合理性請另外參考fairness.py的結果。")
    answer = " ".join(lines)

    return {"question": question, "answer": answer, "task_type": "fault_consistency",
            "case_id": case_no, "line": "TPL",
            "grounded_by": "GraphRAG API(/v1/retrieve/features)"}


def generate_graph_qa(train_db_path: str, graphrag_api_url: str, graphrag_api_key: str,
                       output_path: str, limit: int = None) -> int:
    loader = TPLValCaseLoader(train_db_path)
    case_nos = loader.list_case_nos(limit=limit)

    qa_pairs = []
    failed = 0
    skipped_no_features = 0
    for case_no in case_nos:
        case = loader.load(case_no)
        if not case.get("found"):
            continue

        injuries = list(case.get("injury_types") or [])
        bodyparts = list(case.get("body_parts") or [])  # TPL目前一律是空陣列，見檔頭說明
        bodyparts_available = bool(case.get("body_parts"))

        if not injuries and not bodyparts:
            skipped_no_features += 1
            continue

        try:
            features_resp = _fetch_features(graphrag_api_url, graphrag_api_key, injuries, bodyparts)
        except httpx.HTTPError as e:
            # 呼叫失敗就跳過這筆，不要生出假答案；累計失敗數方便你事後檢查API是否穩定
            failed += 1
            continue

        qa_pairs.append(_fault_consistency_qa(case, features_resp, bodyparts_available))

    loader.close()

    with open(output_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"無injury_types/body_parts可查而跳過 {skipped_no_features} 筆")
    print(f"GraphRAG API呼叫失敗 {failed} 筆(已跳過，不產生假QA)")
    return len(qa_pairs)


if __name__ == "__main__":
    import os
    n = generate_graph_qa(
        train_db_path="data/TPL_train.db",
        graphrag_api_url=os.environ["GRAPHRAG_API_URL"],
        graphrag_api_key=os.environ.get("GRAPHRAG_API_KEY"),
        output_path="qa_generation/output_graph_tpl.jsonl",
        limit=None,
    )
    print(f"產生 {n} 筆QA，寫入 qa_generation/output_graph_tpl.jsonl")
