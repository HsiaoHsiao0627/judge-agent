# -*- coding: utf-8 -*-
"""
rsa_hybrid.py — RSA 保單條款 BM25 + Vector 候選聯集 + LLM rerank

比照 fault_consistency.py 的服務邊界：Vector 檢索不直接連 Neo4j，改打
GraphRAG API 既有的 POST /v1/retrieve/clause 端點(不是新增的，那支API
本來就有，涵蓋TPL/RSA/法規三類條款、時效版本判斷，比另外加端點更完整)。
BM25 沿用既有的 RSABM25Retriever(rsa_bm25.py)，不重複寫一份。

為什麼需要候選聯集而不是單用一種：實測發現 BM25 精確詞匹配跟 Vector
語意匹配各有各的失誤模式(例：問句用「理賠限額」，正確答案條文用「補償」
不用「理賠」，BM25因詞頻誤判把不相關條文排第一；Vector靠語意抓對但
BM25排序錯)。兩邊candidate各撈10筆取聯集(依clause_id去重複)，交給
LLM排序後修正回來，比只用其中一種或做分數層級加權(RRF等，因為BM25
分數跟cosine相似度尺度不可比)更穩定。

Rerank用的是微調過的Judge Agent模型(Vertex AI tuned endpoint)，這是
Vertex AI資源，沒辦法像reasoning_review.py那樣用GEMINI_API_KEY打
公開API，需要Vertex OAuth(ADC)，因此這裡是這支服務裡唯一用google-genai
SDK而非httpx直接呼叫REST的地方——原因是endpoint本質不同，不是風格
不一致。

失敗時的保守原則(比照全案既有慣例)：
- GraphRAG API 呼叫失敗 → 只用 BM25 候選繼續(不是整個檢索失敗)，
  在回傳結果裡明確標記 vector_unavailable=True
- Rerank(LLM)呼叫失敗 → 退回BM25+Vector聯集的原始順序(不排序但不是
  沒結果)，標記 rerank_unavailable=True
- 兩者都失敗 → 才真的回空結果
"""
import os
import logging
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

TUNED_MODEL_ENDPOINT = os.environ.get(
    "TUNED_MODEL_ENDPOINT",
    "projects/624489584685/locations/us/endpoints/5493969332926939136",
)


class RSAHybridClauseRetriever:
    def __init__(self, bm25_retriever, graphrag_base_url: str,
                 graphrag_api_key: Optional[str] = None,
                 gcp_project: str = "cki-scu-2026", timeout: float = 15.0):
        """bm25_retriever: 既有的 RSABM25Retriever 實例(main.py 已經建立過，
        這裡複用同一個連線，不重新開一份db連線)。"""
        self.bm25 = bm25_retriever
        self.graphrag_base_url = graphrag_base_url.rstrip("/")
        self.graphrag_api_key = graphrag_api_key
        self.timeout = timeout

        from google import genai
        self._generate_client = genai.Client(vertexai=True, project=gcp_project, location="us")

    def close(self):
        pass  # bm25連線的生命週期由main.py管理，這裡不重複關閉

    def _vector_candidates(self, query: str, top_k: int = 10) -> tuple[list, Optional[str]]:
        """回傳 (候選list, 錯誤原因或None)。候選為 [(clause_id, content, quality_flags), ...]

        呼叫GraphRAG API既有的 /v1/retrieve/clause(不是我方新增的端點；
        這支API已經涵蓋TPL/RSA/法規三類條款、時效版本判斷，比另外加一個
        端點更完整，不要重複造輪子)。"""
        headers = {"X-API-Key": self.graphrag_api_key} if self.graphrag_api_key else {}
        try:
            resp = httpx.post(
                f"{self.graphrag_base_url}/v1/retrieve/clause",
                json={"query": query, "top_k": top_k, "insurance_line": "RSA"},
                headers=headers, timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            return [], f"GraphRAG API 呼叫失敗: {e}"
        return [
            (r["clause_id"], r["content"], r.get("quality_flags", []))
            for r in data.get("results", [])
        ], None

    def _rerank(self, query: str, candidates: list) -> tuple[list, Optional[str]]:
        """candidates: [(clause_id, content), ...]；回傳 (排序後clause_id list, 錯誤原因或None)"""
        from google.genai import types

        numbered = "\n\n".join(
            f"[{i+1}] clause_id={cid}\n{content}"
            for i, (cid, content) in enumerate(candidates)
        )
        prompt = f"""你是保單條款檢索排序助手。以下是使用者問題，以及多個候選條款(可能包含不相關的)。
請依據「條款內容能不能直接回答使用者問題」排序，最相關的排最前面。
只需輸出排序後的 clause_id 清單(用逗號分隔)，不要其他文字。

使用者問題: {query}

候選條款:
{numbered}

輸出格式範例: RSA-CLAUSE-SELF::第2條,RSA-CLAUSE-SELF::第3條,...
"""
        try:
            resp = self._generate_client.models.generate_content(
                model=TUNED_MODEL_ENDPOINT,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = resp.text or ""
        except Exception as e:
            return [], f"Rerank模型呼叫失敗: {type(e).__name__}: {e}"

        order = [cid.strip() for cid in text.split(",") if cid.strip()]
        return order, None

    def search(self, query: str, top_k: int = 5) -> dict:
        bm25_result = self.bm25.search(query, top_k=10)
        # rsa_bm25.py 已補上 clause_id 欄位(見rsa_bm25.py的修改)，前提是
        # main.py 讀的 data/RSA_BM25_clauses.db 是重建過的版本(external_key=
        # RSA-CLAUSE-XXX::第N條格式)，否則這裡拿到的會是舊格式(PDF檔名)，
        # 跟Vector的clause_id對不上，聯集會退化成兩組互不重疊的候選。
        bm25_candidates = [
            (r["clause_id"], r["text"])
            for r in bm25_result.get("results", [])
            if r.get("clause_id")
        ]

        vector_candidates, vector_error = self._vector_candidates(query, top_k=10)

        merged = {}          # clause_id -> content
        quality_flags_map = {}  # clause_id -> quality_flags(僅vector來源有，BM25沒有這個概念)
        for cid, content in bm25_candidates:
            merged[cid] = content
        for cid, content, flags in vector_candidates:
            merged.setdefault(cid, content)
            if flags:
                quality_flags_map[cid] = flags

        if not merged:
            return {
                "query": query, "results": [], "count": 0,
                "vector_unavailable": vector_error is not None,
                "vector_error": vector_error,
                "reason": "BM25與Vector候選皆為空",
            }

        candidates = list(merged.items())
        order, rerank_error = self._rerank(query, candidates)

        if not order:
            # rerank失敗，退回原始聯集順序(bm25候選在前)，不是回空結果
            ordered_ids = [cid for cid, _ in candidates]
        else:
            ordered_ids = [cid for cid in order if cid in merged]
            # 保底：rerank若有幻覺clause_id(不在候選裡)，那筆略過並記log
            hallucinated = [cid for cid in order if cid not in merged]
            if hallucinated:
                logger.warning("rerank輸出了不在候選清單裡的clause_id: %s (問題: %s)",
                                hallucinated, query)

        results = [
            {
                "clause_id": cid,
                "content": merged[cid],
                # quality_flags(例如date_unknown)原樣往下游帶，不隱藏——
                # 比照全案既有的judgment_fault_pct警語處理原則。
                "quality_flags": quality_flags_map.get(cid, []),
            }
            for cid in ordered_ids[:top_k]
        ]

        return {
            "query": query, "results": results, "count": len(results),
            "vector_unavailable": vector_error is not None,
            "vector_error": vector_error,
            "rerank_unavailable": rerank_error is not None,
            "rerank_error": rerank_error,
        }


class MockRSAHybridClauseRetriever:
    """GRAPHRAG_API_URL 未設定，或初始化失敗時的占位，誠實回報未執行。"""
    def search(self, query: str, top_k: int = 5) -> dict:
        return {"query": query, "results": [], "count": 0,
                "reason": "RSAHybridClauseRetriever 未啟用(mock)"}

    def close(self):
        pass


def get_hybrid_clause_retriever(bm25_retriever) -> "RSAHybridClauseRetriever":
    base_url = os.environ.get("GRAPHRAG_API_URL")
    if not base_url:
        print("[警告] 未設定 GRAPHRAG_API_URL，RSA混合條文檢索(BM25+Vector+rerank)"
              "將回退為 Mock，只有 /v1/clause_search 的純BM25端點可用")
        return MockRSAHybridClauseRetriever()

    api_key = os.environ.get("GRAPHRAG_API_KEY")
    gcp_project = os.environ.get("GCP_PROJECT", "cki-scu-2026")
    try:
        return RSAHybridClauseRetriever(bm25_retriever, base_url, api_key, gcp_project)
    except Exception as e:
        print(f"[警告] RSAHybridClauseRetriever 初始化失敗: {e}，回退為 Mock")
        return MockRSAHybridClauseRetriever()
