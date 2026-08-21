# -*- coding: utf-8 -*-
"""
test_retrieval.py
===================
測試條款向量檢索是否正常運作：把問題轉成向量(RETRIEVAL_QUERY)，
查詢Neo4j的向量索引，印出最相似的條款內容跟分數。

環境變數：
  GEMINI_API_KEY
  NEO4J_URI / NEO4J_USER(或NEO4J_USERNAME) / NEO4J_PASSWORD

執行：
  python test_retrieval.py "你的問題"
"""

import os
import sys

MODEL = "gemini-embedding-001"
DIMENSIONS = 3072
VECTOR_INDEX_NAME = "policy_clause_embedding_index"  # 資料庫裡本來就存在的正式索引名稱，不是我們自建的
TOP_K = 5


def embed_query(client, types, question):
    resp = client.models.embed_content(
        model=MODEL,
        contents=[question],
        config=types.EmbedContentConfig(
            output_dimensionality=DIMENSIONS,
            task_type="RETRIEVAL_QUERY",  # 注意:查詢端要用RETRIEVAL_QUERY，
                                           # 跟建庫時的RETRIEVAL_DOCUMENT配對
                                           # 才能拿到校準過的相似度分數
        ),
    )
    return resp.embeddings[0].values


def main():
    if len(sys.argv) < 2:
        print('用法: python test_retrieval.py "你的問題"')
        sys.exit(1)
    question = sys.argv[1]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ 缺少環境變數 GEMINI_API_KEY")
        sys.exit(1)

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    missing = [n for n, v in [("NEO4J_URI", uri), ("NEO4J_USER/NEO4J_USERNAME", user),
                              ("NEO4J_PASSWORD", password)] if not v]
    if missing:
        print(f"❌ 缺少環境變數: {', '.join(missing)}")
        sys.exit(1)

    from google import genai
    from google.genai import types
    from neo4j import GraphDatabase

    client = genai.Client(api_key=api_key)
    print(f"問題: {question}")
    query_embedding = embed_query(client, types, question)
    print(f"已算出查詢向量(維度: {len(query_embedding)})")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
            YIELD node, score
            RETURN node.clause_id AS clause_id, node.doc_title AS doc_title,
                   node.article_title AS article_title, node.content AS content,
                   score
            ORDER BY score DESC
        """, index_name=VECTOR_INDEX_NAME, top_k=TOP_K, query_embedding=query_embedding)

        print(f"\n=== 前 {TOP_K} 筆最相似的條款 ===")
        for i, record in enumerate(result, 1):
            print(f"\n[{i}] clause_id={record['clause_id']}  score={record['score']:.4f}")
            print(f"    來源: {record['doc_title']} / {record.get('article_title') or '(無條名)'}")
            content = record["content"] or ""
            print(f"    內容: {content[:150]}{'...' if len(content) > 150 else ''}")

    driver.close()


if __name__ == "__main__":
    main()
