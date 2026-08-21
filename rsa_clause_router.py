# -*- coding: utf-8 -*-
"""
rsa_clause_router.py
=======================
FastAPI 路由，把 RSAClauseRetriever 包成 HTTP endpoint，比照你既有
GraphRAG API的路由命名慣例(/v1/retrieve/...)。

整合方式：在你既有的 main.py 裡加入：

    from rsa_clause_router import router as rsa_clause_router
    app.include_router(rsa_clause_router)

retriever 用 FastAPI 的 lifespan/startup event 建立一次，不要每個
request都重新建立(會重新開Neo4j連線、重新初始化genai client，
拖慢每次請求的延遲)。
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from rsa_clause_retrieval import RSAClauseRetriever, ClauseResult

logger = logging.getLogger(__name__)
router = APIRouter()

_retriever: RSAClauseRetriever | None = None


def get_retriever() -> RSAClauseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RSAClauseRetriever()
    return _retriever


class ClauseQueryRequest(BaseModel):
    question: str
    top_k: int = 5


class ClauseQueryResponseItem(BaseModel):
    clause_id: str
    content: str
    sources: List[str]


class ClauseQueryResponse(BaseModel):
    question: str
    results: List[ClauseQueryResponseItem]


@router.post("/v1/retrieve/rsa_clause", response_model=ClauseQueryResponse)
def retrieve_rsa_clause(req: ClauseQueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不可為空")

    try:
        retriever = get_retriever()
        results: List[ClauseResult] = retriever.retrieve(req.question, top_k=req.top_k)
    except Exception as e:
        logger.exception("RSA條款檢索失敗: %s", req.question)
        raise HTTPException(status_code=500, detail=f"檢索失敗: {e}")

    return ClauseQueryResponse(
        question=req.question,
        results=[
            ClauseQueryResponseItem(clause_id=r.clause_id, content=r.content, sources=r.sources)
            for r in results
        ],
    )
