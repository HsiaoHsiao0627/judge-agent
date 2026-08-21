# -*- coding: utf-8 -*-
"""
test_bm25_retrieval.py
=========================
測試重建後的 BM25 索引查詢效果(只查 rsa_clause，排除案例)，
跟 test_retrieval.py(vector版本)拿同樣的問題互相比對。

執行：
  pip install jieba zhconv
  python test_bm25_retrieval.py "你的問題"
"""

import sys
import json
import math
import sqlite3
import collections

DB_PATH = "rsa_bm25_knowledge_base_V1.db"
TOP_K = 5


def tokenize_query(text, jieba, zhconv):
    simplified = zhconv.convert(text, "zh-cn")
    tokens = [t.strip().lower() for t in jieba.cut(simplified) if t.strip()]
    tokens = [t for t in tokens if any(c.isalnum() for c in t)]
    return tokens


def bm25_score(con, query_tokens, source_type="rsa_clause"):
    cur = con.cursor()

    stats = dict(cur.execute("SELECT stat_key, stat_value FROM bm25_stats").fetchall())
    N = int(stats["N"])
    avgdl = float(stats["avgdl"])
    k1 = float(stats["k1"])
    b = float(stats["b"])

    # 限定在 source_type 範圍內的 chunk_id 集合(排除案例)
    valid_chunk_ids = {r[0] for r in cur.execute(
        "SELECT chunk_id FROM chunks WHERE source_type=?", (source_type,)
    ).fetchall()}

    doc_len = dict(cur.execute(
        "SELECT chunk_id, token_count FROM chunks WHERE source_type=?", (source_type,)
    ).fetchall())

    scores = collections.defaultdict(float)

    for term in set(query_tokens):
        row = cur.execute(
            "SELECT idf FROM bm25_vocab WHERE term=?", (term,)
        ).fetchone()
        if row is None:
            continue  # 語料裡完全沒出現過的詞，略過
        idf = row[0]

        postings = cur.execute(
            "SELECT chunk_id, term_frequency FROM bm25_postings WHERE term=?", (term,)
        ).fetchall()

        for chunk_id, tf in postings:
            if chunk_id not in valid_chunk_ids:
                continue
            dl = doc_len.get(chunk_id, avgdl)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score = idf * (tf * (k1 + 1)) / denom if denom else 0
            scores[chunk_id] += score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main():
    if len(sys.argv) < 2:
        print('用法: python test_bm25_retrieval.py "你的問題"')
        sys.exit(1)
    question = sys.argv[1]

    import jieba
    import zhconv

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    query_tokens = tokenize_query(question, jieba, zhconv)
    print(f"問題: {question}")
    print(f"查詢分詞(簡體): {query_tokens}\n")

    ranked = bm25_score(con, query_tokens, source_type="rsa_clause")

    print(f"=== 前 {TOP_K} 筆 BM25 結果 ===")
    for i, (chunk_id, score) in enumerate(ranked[:TOP_K], 1):
        row = con.execute("""
            SELECT c.text_content, d.external_key, d.title
            FROM chunks c JOIN documents d ON c.document_id = d.document_id
            WHERE c.chunk_id = ?
        """, (chunk_id,)).fetchone()
        print(f"\n[{i}] chunk_id={chunk_id}  score={score:.4f}")
        print(f"    clause_id={row['external_key']}  來源={row['title']}")
        content = row["text_content"]
        print(f"    內容: {content[:150]}{'...' if len(content) > 150 else ''}")

    con.close()


if __name__ == "__main__":
    main()
