# -*- coding: utf-8 -*-
"""
test_hybrid_retrieval.py
===========================
把 BM25 Top-10 跟 Vector Top-10 的候選(依clause_id去重複後取聯集)，
丟給LLM做最終排序，取代單一方法各自的排序結果。

依據：BM25 精確詞匹配 vs Vector 語意匹配各有各的盲點(參考本次
"理賠限額"問句案例:BM25因為"理賠"詞頻誤把第3條排第一，正確答案
第2條用"補償"而非"理賠"，Vector靠語意抓對但BM25排序錯誤)，
候選層級聯集比分數層級加權(RRF等)更適合這種尺度不可比的情況。

環境變數：
  GEMINI_API_KEY
  NEO4J_URI / NEO4J_USER(或NEO4J_USERNAME) / NEO4J_PASSWORD

執行：
  python test_hybrid_retrieval.py "你的問題"
"""

import sys
import json
import math
import sqlite3
import collections

DB_PATH = "rsa_bm25_knowledge_base_V1.db"
VECTOR_INDEX_NAME = "policy_clause_embedding_index"
EMBED_MODEL = "gemini-embedding-001"
DIMENSIONS = 3072
CANDIDATE_K = 10  # 各自撈多少候選再聯集


def tokenize_query(text, jieba, zhconv):
    simplified = zhconv.convert(text, "zh-cn")
    tokens = [t.strip().lower() for t in jieba.cut(simplified) if t.strip()]
    return [t for t in tokens if any(c.isalnum() for c in t)]


def bm25_candidates(question, source_type="rsa_clause_all"):
    """回傳 [(clause_id, content, doc_title, bm25_score), ...]"""
    import jieba
    import zhconv

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

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

    query_tokens = tokenize_query(question, jieba, zhconv)
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
    for chunk_id, score in ranked:
        row = cur.execute("""
            SELECT c.text_content, d.external_key, d.title
            FROM chunks c JOIN documents d ON c.document_id = d.document_id
            WHERE c.chunk_id = ?
        """, (chunk_id,)).fetchone()
        results.append({
            "clause_id": row["external_key"],
            "content": row["text_content"],
            "source": "bm25",
            "raw_score": score,
        })
    con.close()
    return results


def vector_candidates(embed_client, types, question):
    from neo4j import GraphDatabase
    import os

    resp = embed_client.models.embed_content(
        model=EMBED_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS, task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = resp.embeddings[0].values

    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
            YIELD node, score
            WHERE node.insurance_line = 'RSA'
            RETURN node.clause_id AS clause_id, node.content AS content, score
            ORDER BY score DESC
        """, index_name=VECTOR_INDEX_NAME, top_k=CANDIDATE_K, query_embedding=query_embedding)
        results = [{
            "clause_id": r["clause_id"], "content": r["content"],
            "source": "vector", "raw_score": r["score"],
        } for r in result]

    driver.close()
    return results


def rerank_with_llm(client, types, question, candidates):
    """把聯集後的候選丟給LLM排序，回傳排序後的clause_id清單"""
    numbered = "\n\n".join(
        f"[{i+1}] clause_id={c['clause_id']}\n{c['content']}"
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
    resp = client.models.generate_content(
        model="projects/624489584685/locations/us/endpoints/5493969332926939136",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    order = [cid.strip() for cid in (resp.text or "").split(",") if cid.strip()]
    return order


def main():
    if len(sys.argv) < 2:
        print('用法: python test_hybrid_retrieval.py "你的問題"')
        sys.exit(1)
    question = sys.argv[1]

    from google import genai
    from google.genai import types
    import os

    # gemini-embedding-001 透過 Vertex AI 只在 us-central1 開放(不是 us 多地區)，
    # 微調模型endpoint則是部署在 us 多地區，兩個模型地區不同，要用兩個client。
    embed_client = genai.Client(vertexai=True, project="cki-scu-2026", location="us-central1")
    generate_client = genai.Client(vertexai=True, project="cki-scu-2026", location="us")

    print(f"問題: {question}\n")

    bm25_results = bm25_candidates(question)
    print(f"BM25 候選: {[r['clause_id'] for r in bm25_results]}")

    vec_results = vector_candidates(embed_client, types, question)
    print(f"Vector 候選: {[r['clause_id'] for r in vec_results]}")

    # 依clause_id去重複聯集(同一clause_id若兩邊都撈到，保留內容較完整的一筆即可)
    merged = {}
    for r in bm25_results + vec_results:
        merged[r["clause_id"]] = r
    candidates = list(merged.values())
    print(f"\n聯集後候選數: {len(candidates)}")

    order = rerank_with_llm(generate_client, types, question, candidates)
    print(f"\n=== LLM rerank 後排序 ===")
    for i, cid in enumerate(order[:5], 1):
        c = merged.get(cid)
        if c:
            print(f"[{i}] {cid}")
            print(f"    {c['content'][:120]}...")
        else:
            print(f"[{i}] {cid} (⚠️ 不在候選清單裡，LLM可能自己編了一個，需要留意)")


if __name__ == "__main__":
    main()
