# -*- coding: utf-8 -*-
"""
main.py — 法官代理人 API

⚠️ 部署提醒：這是內部服務，不是給瀏覽器直接連的。部署到 Cloud Run 時
【絕對不要】加 --allow-unauthenticated。只有協調層應該能呼叫這支服務
（透過 Cloud Run 的 IAM Invoker 機制，不是這支程式自己做身份驗證）。

啟動時會嘗試連接 data/TPL_train.db、data/RSA_train.db（隨容器映像一起打包的唯讀
比較池資料）。連不到就退回 Mock 並在啟動 log 印出明確警告——服務仍會啟動，
但誠實標示這不是真實資料在跑，不要讓人誤以為在算真實理賠案件。
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .config import JudgeConfig
from .decision import judge_decide
from .schema import validate_output
from .fault_consistency import get_fault_retriever
from .tpl_similar import get_tpl_similar_retriever, NoOpTplSimilarRetriever
from .reasoning_review import get_reasoning_reviewer, NoOpReasoningReviewer
from .amount_sources.tpl import AmountComparator, KNNAmountComparator, MockAmountSource
from .amount_sources.rsa import RSAAmountComparator, MockRSAAmountSource
from .clause_sources.rsa_bm25 import RSABM25Retriever, MockRSABM25Retriever
from .clause_sources.rsa_hybrid import get_hybrid_clause_retriever, MockRSAHybridClauseRetriever

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

app = FastAPI(title="法官代理人 API（內部服務，僅供協調層呼叫）")

CFG = JudgeConfig()
FAULT_RETRIEVER = get_fault_retriever()
TPL_SIMILAR_RETRIEVER = get_tpl_similar_retriever()
REASONING_REVIEWER = get_reasoning_reviewer()
AMOUNT_SOURCES: dict = {}
RSA_CLAUSE_RETRIEVER = None
RSA_HYBRID_RETRIEVER = None
STARTUP_WARNINGS: list = []


@app.on_event("startup")
def _startup():
    global RSA_CLAUSE_RETRIEVER, RSA_HYBRID_RETRIEVER
    tpl_db = DATA_DIR / "TPL_train.db"
    rsa_db = DATA_DIR / "RSA_train.db"
    rsa_bm25_db = DATA_DIR / "RSA_BM25_clauses.db"

    if tpl_db.exists():
        if CFG.tpl_pool == "knn":
            AMOUNT_SOURCES["TPL"] = KNNAmountComparator(str(tpl_db), k_neighbors=CFG.knn_neighbors,
                                                        min_precedents=CFG.min_precedents)
            print(f"[啟動] TPL KNNAmountComparator(k={CFG.knn_neighbors}) 已連接: {tpl_db}")
        else:
            AMOUNT_SOURCES["TPL"] = AmountComparator(str(tpl_db), min_precedents=CFG.min_precedents)
            print(f"[啟動] TPL AmountComparator 已連接: {tpl_db}")
    else:
        AMOUNT_SOURCES["TPL"] = MockAmountSource()
        msg = f"[警告] 找不到 {tpl_db}，TPL 案件將使用 Mock 金額來源，不是真實判斷！"
        print(msg)
        STARTUP_WARNINGS.append(msg)

    if rsa_db.exists():
        AMOUNT_SOURCES["RSA"] = RSAAmountComparator(str(rsa_db), min_precedents=CFG.min_precedents)
        print(f"[啟動] RSA AmountComparator 已連接: {rsa_db}")
    else:
        AMOUNT_SOURCES["RSA"] = MockRSAAmountSource()
        msg = f"[警告] 找不到 {rsa_db}，RSA 案件將使用 Mock 金額來源，不是真實判斷！"
        print(msg)
        STARTUP_WARNINGS.append(msg)

    if rsa_bm25_db.exists():
        RSA_CLAUSE_RETRIEVER = RSABM25Retriever(str(rsa_bm25_db))
        print(f"[啟動] RSA BM25條文檢索 已連接: {rsa_bm25_db}")
    else:
        RSA_CLAUSE_RETRIEVER = MockRSABM25Retriever()
        msg = f"[警告] 找不到 {rsa_bm25_db}，RSA BM25條文檢索將使用 Mock，查詢會回空結果！"
        print(msg)
        STARTUP_WARNINGS.append(msg)

    # 混合檢索(BM25+Vector+rerank)複用上面剛建立的BM25 retriever，
    # 不重新開一份db連線。若GRAPHRAG_API_URL未設定會自動回退為Mock，
    # 這時 /v1/clause_search(純BM25)仍然可用，只有新增的
    # /v1/clause_search_hybrid 端點會回空結果。
    RSA_HYBRID_RETRIEVER = get_hybrid_clause_retriever(RSA_CLAUSE_RETRIEVER)
    if isinstance(RSA_HYBRID_RETRIEVER, MockRSAHybridClauseRetriever):
        STARTUP_WARNINGS.append(
            "[警告] RSA混合條文檢索(BM25+Vector+rerank)未啟用，"
            "/v1/clause_search_hybrid 將回空結果"
        )
    else:
        print("[啟動] RSA混合條文檢索(BM25+Vector+rerank) 已連接")


@app.on_event("shutdown")
def _shutdown():
    for src in AMOUNT_SOURCES.values():
        close = getattr(src, "close", None)
        if callable(close):
            close()
    if RSA_CLAUSE_RETRIEVER is not None:
        RSA_CLAUSE_RETRIEVER.close()
    if RSA_HYBRID_RETRIEVER is not None:
        RSA_HYBRID_RETRIEVER.close()


class JudgeRequest(BaseModel):
    """案件輸入格式。欄位對照 amount_sources/tpl.py、rsa.py 的 case dict 結構——
    理賠代理人應該產生跟 TPLValCaseLoader.load()/RSAValCaseLoader.load() 相同形狀的物件。
    這裡刻意用寬鬆的 dict-passthrough 而非嚴格 pydantic 欄位表，因為 TPL/RSA 兩線的
    欄位集合本來就不同，嚴格定義反而會擋掉合法輸入；正確性由 fraud_rules 的逐條 na
    判斷把關，不是靠這一層擋。"""
    case_id: str
    line: str  # "TPL" | "RSA"

    model_config = ConfigDict(extra="allow")


@app.post("/v1/judge")
def judge(req: JudgeRequest):
    case = req.model_dump()
    if case.get("line") not in AMOUNT_SOURCES:
        raise HTTPException(400, f"未知的 line='{case.get('line')}'，只支援 {list(AMOUNT_SOURCES)}")

    result = judge_decide(case, CFG, AMOUNT_SOURCES, FAULT_RETRIEVER, REASONING_REVIEWER, TPL_SIMILAR_RETRIEVER)
    errs = validate_output(result)
    if errs:
        # 不應該發生；若發生了要看得到，不要靜默吞掉
        result["_schema_validation_errors"] = errs
        print(f"[錯誤] 案件 {case.get('case_id')} 輸出未通過 schema 驗證: {errs}")
    return result


class ClauseSearchRequest(BaseModel):
    """RSA保單條款BM25關鍵字檢索。只涵蓋rsa_clause(條款本文)跟rsa_fee_schedule
    (收費標準，僅供參考，承保判斷仍以rsa_clause為準)，不含rsa_case(過往案例)。"""
    query: str
    top_k: int = 5
    include_fee_schedule: bool = True


@app.post("/v1/clause_search")
def clause_search(req: ClauseSearchRequest):
    source_types = ("rsa_clause", "rsa_fee_schedule") if req.include_fee_schedule else ("rsa_clause",)
    return RSA_CLAUSE_RETRIEVER.search(req.query, top_k=req.top_k, source_types=source_types)


class ClauseSearchHybridRequest(BaseModel):
    """BM25+Vector候選聯集+LLM rerank版本的RSA條文檢索，比純BM25(/v1/clause_search)
    更能處理口語化問法(例如問句用「理賠限額」，正確答案條文用「補償」)，代價是
    多打一次GraphRAG API跟一次rerank模型，延遲比純BM25高。"""
    query: str
    top_k: int = 5


@app.post("/v1/clause_search_hybrid")
def clause_search_hybrid(req: ClauseSearchHybridRequest):
    return RSA_HYBRID_RETRIEVER.search(req.query, top_k=req.top_k)


@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "amount_sources": {
            line: ("real" if not isinstance(src, (MockAmountSource, MockRSAAmountSource)) else "mock")
            for line, src in AMOUNT_SOURCES.items()
        },
        "rsa_clause_bm25": "real" if not isinstance(RSA_CLAUSE_RETRIEVER, MockRSABM25Retriever) else "mock",
        "rsa_clause_hybrid": "real" if not isinstance(RSA_HYBRID_RETRIEVER, MockRSAHybridClauseRetriever) else "mock",
        "reasoning_review": "noop" if isinstance(REASONING_REVIEWER, NoOpReasoningReviewer) else type(REASONING_REVIEWER).__name__,
        "tpl_similar": "noop" if isinstance(TPL_SIMILAR_RETRIEVER, NoOpTplSimilarRetriever) else type(TPL_SIMILAR_RETRIEVER).__name__,
        "startup_warnings": STARTUP_WARNINGS,
    }
