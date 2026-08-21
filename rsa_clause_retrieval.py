# -*- coding: utf-8 -*-
"""
rsa_clause_retrieval.py
==========================
Judge Agent 用的 RSA 保單條款檢索模組。整合 BM25 + Vector 候選聯集 +
LLM rerank，取代單一方法各自的排序盲點(BM25精確詞匹配 vs Vector語意
匹配，兩者各有各的失誤模式，聯集後交給LLM排序比單一方法或分數加權
融合更穩定)。

設計原則(比照專案既有慣例)：
  - 連線(Neo4j driver, genai Client)在模組載入時建立一次並重複使用，
    不是每次呼叫都重新建立連線(Cloud Run單一instance可能處理多個
    請求，重複建立連線是效能浪費)
  - 所有敏感設定走環境變數，不寫死在程式碼裡
  - 回傳結構化結果(dataclass)，不是裸字串，方便上層(Judge Agent)
    決定怎麼組裝最終回答

使用方式：
    from rsa_clause_retrieval import RSAClauseRetriever

    retriever = RSAClauseRetriever()  # 模組層級初始化一次即可
    results = retriever.retrieve("自用道路救援保單的保障型方案，理賠限額跟次數限制是什麼？")
    for r in results:
        print(r.clause_id, r.content)

環境變數：
  NEO4J_URI / NEO4J_USER(或NEO4J_USERNAME) / NEO4J_PASSWORD
  GCP_PROJECT (預設 cki-scu-2026)
  BM25_DB_LOCAL_PATH   (預設 /tmp/rsa_bm25_knowledge_base_V1.db，
                         Cloud Run容器內的本機路徑，由 sync_bm25_db.py
                         在服務啟動時從GCS下載到這裡)
"""

import os
import json
import math
import sqlite3
import logging
import collections
import dataclasses
from typing import List

logger = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 3072
TUNED_MODEL_ENDPOINT = os.environ.get(
    "TUNED_MODEL_ENDPOINT",
    "projects/624489584685/locations/us/endpoints/5493969332926939136",
)
VECTOR_INDEX_NAME = "policy_clause_embedding_index"
CANDIDATE_K = 10


@dataclasses.dataclass
class ClauseResult:
    clause_id: str
    content: str
    sources: List[str]  # 這筆是從哪些檢索方法撈到的，例如 ["bm25", "vector"]


class RSAClauseRetriever:
    def __init__(self):
        gcp_project = os.environ.get("GCP_PROJECT", "cki-scu-2026")

        neo4j_uri = os.environ.get("NEO4J_URI")
        neo4j_user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
        neo4j_password = os.environ.get("NEO4J_PASSWORD")
        missing = [n for n, v in [("NEO4J_URI", neo4j_uri),
                                   ("NEO4J_USER/NEO4J_USERNAME", neo4j_user),
                                   ("NEO4J_PASSWORD", neo4j_password)] if not v]
        if missing:
            raise RuntimeError(f"缺少環境變數: {', '.join(missing)}")

        bm25_db_path = os.environ.get("BM25_DB_LOCAL_PATH", "/tmp/rsa_bm25_knowledge_base_V1.db")
        if not os.path.exists(bm25_db_path):
            raise RuntimeError(
                f"找不到 BM25 db: {bm25_db_path}。"
                f"請確認服務啟動時有先執行 sync_bm25_db.py 從 GCS 下載，"
                f"或 BM25_DB_LOCAL_PATH 設定是否正確。"
            )

        from neo4j import GraphDatabase
        from google import genai

        self._neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._bm25_db_path = bm25_db_path

        # 兩個地區不同的client：embedding模型只在us-central1開放，
        # 微調模型endpoint則部署在us多地區，不能共用同一個client。
        self._embed_client = genai.Client(vertexai=True, project=gcp_project, location="us-central1")
        self._generate_client = genai.Client(vertexai=True, project=gcp_project, location="us")

        import jieba
        import zhconv
        self._jieba = jieba
        self._zhconv = zhconv

        # 啟動時先觸發一次jieba建字典快取，避免第一個request才觸發造成明顯延遲
        list(self._jieba.cut("初始化"))

        logger.info("RSAClauseRetriever 初始化完成 (bm25_db=%s)", bm25_db_path)

    def close(self):
        self._neo4j_driver.close()

    # ---------- BM25 ----------

    def _tokenize(self, text):
        simplified = self._zhconv.convert(text, "zh-cn")
        tokens = [t.strip().lower() for t in self._jieba.cut(simplified) if t.strip()]
        return [t for t in tokens if any(c.isalnum() for c in t)]

    def _bm25_candidates(self, question):
        con = sqlite3.connect(self._bm25_db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        try:
            stats = dict(cur.execute("SELECT stat_key, stat_value FROM bm25_stats").fetchall())
            N, avgdl = int(stats["N"]), float(stats["avgdl"])
            k1, b = float(stats["k1"]), float(stats["b"])

            valid_types = ("rsa_clause", "rsa_fee_schedule")
            placeholders = ",".join("?" * len(valid_types))
            valid_chunk_ids = {r[0] for r in cur.execute(
                f"SELECT chunk_id FROM chunks WHERE source_type IN ({placeholders})", valid_types
            ).fetchall()}
            doc_len = dict(cur.execute(
                f"SELECT chunk_id, token_count FROM chunks WHERE source_type IN ({placeholders})", valid_types
            ).fetchall())

            query_tokens = self._tokenize(question)
            scores = collections.defaultdict(float)

            for term in set(query_tokens):
                row = cur.execute("SELECT idf FROM bm25_vocab WHERE term=?", (term,)).fetchone()
                if row is None:
                    continue
                idf = row[0]
                for chunk_id, tf in cur.execute(
                    "SELECT chunk_id, term_frequency FROM bm25_postings WHERE term=?", (term,)
                ).fetchall():
                    if chunk_id not in valid_chunk_ids:
                        continue
                    dl = doc_len.get(chunk_id, avgdl)
                    denom = tf + k1 * (1 - b + b * dl / avgdl)
                    scores[chunk_id] += idf * (tf * (k1 + 1)) / denom if denom else 0

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:CANDIDATE_K]

            results = []
            for chunk_id, _score in ranked:
                row = cur.execute("""
                    SELECT c.text_content, d.external_key
                    FROM chunks c JOIN documents d ON c.document_id = d.document_id
                    WHERE c.chunk_id = ?
                """, (chunk_id,)).fetchone()
                results.append((row["external_key"], row["text_content"]))
            return results
        finally:
            con.close()

    # ---------- Vector ----------

    def _vector_candidates(self, question):
        from google.genai import types

        resp = self._embed_client.models.embed_content(
            model=EMBED_MODEL,
            contents=[question],
            config=types.EmbedContentConfig(
                output_dimensionality=EMBED_DIMENSIONS,
                task_type="RETRIEVAL_QUERY",
            ),
        )
        query_embedding = resp.embeddings[0].values

        with self._neo4j_driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                YIELD node, score
                WHERE node.insurance_line = 'RSA'
                RETURN node.clause_id AS clause_id, node.content AS content, score
                ORDER BY score DESC
            """, index_name=VECTOR_INDEX_NAME, top_k=CANDIDATE_K, query_embedding=query_embedding)
            return [(r["clause_id"], r["content"]) for r in result]

    # ---------- Rerank ----------

    def _rerank(self, question, candidates):
        """candidates: List[ClauseResult]，回傳排序後的 clause_id list"""
        from google.genai import types

        numbered = "\n\n".join(
            f"[{i+1}] clause_id={c.clause_id}\n{c.content}"
            for i, c in enumerate(candidates)
        )
        prompt = f"""你是保單條款檢索排序助手。以下是使用者問題，以及多個候選條款(可能包含不相關的)。
請依據「條款內容能不能直接回答使用者問題」排序，最相關的排最前面。
只需輸出排序後的 clause_id 清單(用逗號分隔)，不要其他文字。

使用者問題: {question}

候選條款:
{numbered}

輸出格式範例: RSA-CLAUSE-SELF::第2條,RSA-CLAUSE-SELF::第3條,...
"""
        resp = self._generate_client.models.generate_content(
            model=TUNED_MODEL_ENDPOINT,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = resp.text or ""
        return [cid.strip() for cid in text.split(",") if cid.strip()]

    # ---------- Public API ----------

    def retrieve(self, question: str, top_k: int = 5) -> List[ClauseResult]:
        bm25_raw = self._bm25_candidates(question)
        vec_raw = self._vector_candidates(question)

        merged: dict[str, ClauseResult] = {}
        for clause_id, content in bm25_raw:
            merged[clause_id] = ClauseResult(clause_id, content, ["bm25"])
        for clause_id, content in vec_raw:
            if clause_id in merged:
                merged[clause_id].sources.append("vector")
            else:
                merged[clause_id] = ClauseResult(clause_id, content, ["vector"])

        candidates = list(merged.values())
        if not candidates:
            logger.warning("RSA條款檢索沒有任何候選結果，問題: %s", question)
            return []

        order = self._rerank(question, candidates)

        ranked_results = []
        for clause_id in order:
            if clause_id in merged:
                ranked_results.append(merged[clause_id])
            else:
                logger.warning(
                    "rerank輸出的clause_id '%s' 不在候選清單裡，可能是LLM編造的，已略過。問題: %s",
                    clause_id, question,
                )

        # 保底：如果rerank整批失敗或格式不對，退回原始聯集順序而不是回傳空結果
        if not ranked_results:
            logger.warning("rerank結果為空，退回未排序的候選聯集。問題: %s", question)
            ranked_results = candidates

        return ranked_results[:top_k]
