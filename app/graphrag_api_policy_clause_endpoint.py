# -*- coding: utf-8 -*-
"""
graphrag_api_policy_clause_endpoint.py
=========================================
⚠️ 這不是給 judge-agent 用的，是要加進「GraphRAG API」那個獨立服務
(你另一份程式碼庫，不在這次上傳的 judge_agent.zip 裡)。

新增 POST /v1/retrieve/policy_clause，比照你現有 /v1/retrieve/features
的風格：輸入問句，內部處理embedding+Neo4j向量查詢，回傳結構化結果——
呼叫端(judge-agent)不用知道embedding模型、不用碰Neo4j連線細節。

把這段邏輯貼進你 GraphRAG API 服務的 api.py(或對應的路由檔)，
並在該服務的 requirements.txt 加上 google-genai。

環境變數(在 GraphRAG API 服務本身設定，不是judge-agent)：
  NEO4J_URI / NEO4J_USER(或NEO4J_USERNAME) / NEO4J_PASSWORD  (應該已經有了)
  GCP_PROJECT (預設 cki-scu-2026)
"""
import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 3072
VECTOR_INDEX_NAME = "policy_clause_embedding_index"

_embed_client = None  # lazy init，跟你其他地方的client初始化風格保持一致


def _get_embed_client():
    global _embed_client
    if _embed_client is None:
        from google import genai
        # gemini-embedding-001 透過 Vertex AI 只在 us-central1 開放
        _embed_client = genai.Client(
            vertexai=True,
            project=os.environ.get("GCP_PROJECT", "cki-scu-2026"),
            location="us-central1",
        )
    return _embed_client


class PolicyClauseRequest(BaseModel):
    query: str
    top_k: int = 10
    insurance_line: Optional[str] = None  # "RSA" | "TPL" | None(不限)


class PolicyClauseResult(BaseModel):
    clause_id: str
    content: str
    score: float


class PolicyClauseResponse(BaseModel):
    query: str
    results: List[PolicyClauseResult]
    count: int


@router.post("/v1/retrieve/policy_clause", response_model=PolicyClauseResponse)
def retrieve_policy_clause(req: PolicyClauseRequest):
    from google.genai import types
    from neo4j import GraphDatabase  # 這支服務本來就會連Neo4j，沿用既有driver會更好，
                                      # 這裡示範獨立寫法，你可以改成用服務既有的driver實例

    client = _get_embed_client()
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[req.query],
        config=types.EmbedContentConfig(
            output_dimensionality=EMBED_DIMENSIONS,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    query_embedding = resp.embeddings[0].values

    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, password))

    where_clause = "WHERE node.insurance_line = $insurance_line" if req.insurance_line else ""

    with driver.session() as session:
        result = session.run(f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
            YIELD node, score
            {where_clause}
            RETURN node.clause_id AS clause_id, node.content AS content, score
            ORDER BY score DESC
        """, index_name=VECTOR_INDEX_NAME, top_k=req.top_k,
             query_embedding=query_embedding, insurance_line=req.insurance_line)
        results = [
            PolicyClauseResult(clause_id=r["clause_id"], content=r["content"], score=r["score"])
            for r in result
        ]
    driver.close()

    return PolicyClauseResponse(query=req.query, results=results, count=len(results))
