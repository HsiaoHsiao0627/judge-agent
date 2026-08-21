# -*- coding: utf-8 -*-
"""
from_graph.py — 從圖資料庫(GraphRAG API的 /v1/retrieve/rsa_similar)反向生成案例相似性QA

架構原則：不直接連Neo4j，打GraphRAG API的HTTP端點——跟fault_consistency.py同一個
決定(見交接文件設計邏輯第4節)。

/v1/retrieve/rsa_similar 回傳的 similar_cases 本身沒有金額欄位(只有category_name/
km_bucket/region/相似度分數)，要組出「相似案例大概賠多少」這種QA，需要跨接
RSA_train.db 用 request_no 查回實際 signed_amount——這是刻意的跨資料庫設計，
不是圖資料庫漏欄位。

quality_flags(例如 low_sample_category、region_not_comparable、
highway_case_location_imprecise)必須誠實併入答案，不能隱藏——這是GraphRAG API
自己文件裡寫的規則(呼叫端不應隱藏這些flag)，微調資料延續同一個原則，
不然等於教模型學會替資料品質問題打包票。
"""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.amount_sources.rsa import RSAValCaseLoader, CATEGORY_NAMES


def _fetch_similar(base_url: str, api_key: str, request_no: int, top_k: int = 5, timeout: float = 10.0) -> dict:
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = httpx.post(f"{base_url.rstrip('/')}/v1/retrieve/rsa_similar",
                       json={"request_no": request_no, "top_k": top_k},
                       headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _similarity_qa(case: dict, similar_resp: dict, loader: RSAValCaseLoader) -> dict:
    cat_name = case.get("category_name") or "此類別"
    question = (f"RSA道路救援案件(案號{case.get('case_id')}，服務類別{cat_name}，"
                f"里程{case.get('km')}公里)，有哪些相似的歷史案例可供參考？")

    similar_cases = similar_resp.get("similar_cases", [])
    if not similar_cases:
        return {"question": question,
                "answer": "圖資料庫檢索後無足夠相似的歷史案例，此案件應視為先例不足，建議轉人工複核，不宜直接類比其他案件金額。",
                "task_type": "case_similarity", "case_id": case.get("case_id"),
                "grounded_by": "GraphRAG API(/v1/retrieve/rsa_similar)"}

    lines = []
    all_flags = set()
    for sc in similar_cases:
        req_no = sc["request_no"]
        ref_case = loader.load(req_no)
        amt_str = f"{ref_case['claim_amount']:,.0f}元" if ref_case.get("found") and ref_case.get("claim_amount") is not None else "金額未知"
        flags = sc.get("quality_flags", [])
        all_flags.update(flags)
        flag_note = f"(資料品質提醒：{'、'.join(flags)})" if flags else ""
        lines.append(f"案號{req_no}：{sc.get('category_name', '未知類別')}、"
                      f"{sc.get('km_bucket', '未知里程區間')}、相似度分數{sc.get('final_score', 0):.3f}，"
                      f"實付{amt_str}{flag_note}")

    answer = "以下是圖資料庫檢索到的相似案例：\n" + "\n".join(lines)
    if all_flags:
        answer += f"\n\n整體提醒：本次檢索結果中有{len(all_flags)}種資料品質提醒訊號，判斷時應一併考量，不宜完全忽略。"
    return {"question": question, "answer": answer, "task_type": "case_similarity",
            "case_id": case.get("case_id"), "grounded_by": "GraphRAG API(/v1/retrieve/rsa_similar) + RSA_train.db"}


def generate_graph_qa(train_db_path: str, graphrag_api_url: str, graphrag_api_key: str,
                       output_path: str, limit: int = None) -> int:
    loader = RSAValCaseLoader(train_db_path)
    request_nos = loader.list_request_nos(limit=limit)

    qa_pairs = []
    failed = 0
    for request_no in request_nos:
        case = loader.load(request_no)
        if not case.get("found"):
            continue
        try:
            similar_resp = _fetch_similar(graphrag_api_url, graphrag_api_key, request_no)
        except httpx.HTTPError as e:
            # 呼叫失敗就跳過這筆，不要生出假答案；累計失敗數方便你事後檢查API是否穩定
            failed += 1
            continue
        qa_pairs.append(_similarity_qa(case, similar_resp, loader))

    loader.close()

    with open(output_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"GraphRAG API呼叫失敗 {failed} 筆(已跳過，不產生假QA)")
    return len(qa_pairs)


if __name__ == "__main__":
    import os
    n = generate_graph_qa(
        train_db_path="data/RSA_train.db",
        graphrag_api_url=os.environ["GRAPHRAG_API_URL"],
        graphrag_api_key=os.environ.get("GRAPHRAG_API_KEY"),
        output_path="qa_generation/output_graph.jsonl",
        limit=None,
    )
    print(f"產生 {n} 筆QA，寫入 qa_generation/output_graph.jsonl")
