# -*- coding: utf-8 -*-
"""
load_embeddings_to_neo4j.py
=============================
把 embed_policy_clauses.py 產生的 clause_embeddings.jsonl 寫進 Neo4j AuraDB，
並建立向量索引，讓後續RAG檢索可以用 db.index.vector.queryNodes 查詢。

前置條件：
  1. Neo4j裡已經有 (:PolicyClause {clause_id: ...}) 節點存在
     (由原本匯入 policy_clauses_full.json 的流程建立，本script只負責補上embedding屬性)
  2. clause_embeddings.jsonl 已經跑完 embed_policy_clauses.py 產生

環境變數(比照專案既有慣例，NEO4J_USER / NEO4J_USERNAME 都支援避免silent fallback)：
  NEO4J_URI       例如 neo4j+s://<instance_id>.databases.neo4j.io
  NEO4J_USER 或 NEO4J_USERNAME   AuraDB Free的username就是instance_id，不是"neo4j"
  NEO4J_PASSWORD

執行：
  pip install neo4j
  python load_embeddings_to_neo4j.py

中斷後重跑：SET屬性本身是幂等操作(idempotent)，重跑不會造成重複或錯誤，
但仍會印出進度方便追蹤，不會跳過已寫入的節點重新確認(因為SET很便宜，
沒有API呼叫成本，不需要像embedding那樣避免重複呼叫)。
"""

import os
import json
import sys

EMBEDDINGS_FILE = "clause_embeddings.jsonl"
VECTOR_INDEX_NAME = "policy_clause_embedding_index"  # 資料庫裡既有的正式索引，不是自建
DIMENSIONS = 3072
BATCH_SIZE = 50  # 用 UNWIND 批次寫入，減少往返次數


def get_neo4j_credentials():
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    missing = [name for name, val in [
        ("NEO4J_URI", uri), ("NEO4J_USER/NEO4J_USERNAME", user), ("NEO4J_PASSWORD", password)
    ] if not val]
    if missing:
        print(f"❌ 缺少環境變數: {', '.join(missing)}")
        sys.exit(1)
    return uri, user, password


def load_embeddings(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                print(f"  ⚠️ 略過一行無法解析的資料: {line[:80]}...")
    return records


def main():
    uri, user, password = get_neo4j_credentials()

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))

    records = load_embeddings(EMBEDDINGS_FILE)
    print(f"讀取到 {len(records)} 筆向量，準備寫入 Neo4j")

    if not records:
        print("沒有資料可寫入，結束。")
        driver.close()
        return

    # 明確檢查向量維度是否一致，維度不對會讓向量索引整個建立失敗，
    # 提早在這裡發現比讓Neo4j丟出模糊錯誤好。
    dims_seen = {len(r["embedding"]) for r in records}
    if dims_seen != {DIMENSIONS}:
        print(f"❌ 偵測到不一致的向量維度: {dims_seen}，預期全部應為 {DIMENSIONS}。")
        print("   請確認 embed_policy_clauses.py 的 DIMENSIONS 設定是否有變動過，"
              "或 clause_embeddings.jsonl 是否混雜了不同批次/不同維度設定的結果。")
        driver.close()
        sys.exit(1)

    with driver.session() as session:
        # 這個資料庫本來就已經有 policy_clause_embedding_index 這個向量索引
        # (由其他既有流程建立，讀取次數3141次，顯示是正在被使用的正式索引)。
        # 本script「不」建立新索引，只確認既有索引存在且ONLINE，避免重複建立
        # 造成混淆，也避免誤判成需要自己建一個。
        status = session.run(
            "SHOW INDEXES YIELD name, state, type, labelsOrTypes, properties "
            "WHERE name = $index_name "
            "RETURN name, state, type, labelsOrTypes, properties",
            index_name=VECTOR_INDEX_NAME,
        ).single()
        if status is None:
            print(f"❌ 找不到名為 {VECTOR_INDEX_NAME} 的索引，"
                  f"請確認索引名稱是否正確(先執行 SHOW INDEXES 確認實際名稱)，"
                  f"不要在共用資料庫上自行建立新的向量索引。")
            driver.close()
            sys.exit(1)
        print(f"確認既有索引: name={status['name']}, state={status['state']}, "
              f"type={status['type']}, labels={status['labelsOrTypes']}, "
              f"properties={status['properties']}")
        if status["state"] != "ONLINE":
            print(f"⚠️ 索引狀態不是 ONLINE（目前是 {status['state']}），先不要繼續寫入。")
            driver.close()
            sys.exit(1)

        # 2. 批次寫入embedding屬性到既有的 PolicyClause 節點
        matched_total = 0
        not_found = []

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            result = session.run("""
                UNWIND $batch AS row
                MATCH (c:PolicyClause {clause_id: row.clause_id})
                SET c.embedding = row.embedding,
                    c.char_count = row.char_count
                RETURN row.clause_id AS clause_id
            """, batch=batch)
            matched_ids = {r["clause_id"] for r in result}
            matched_total += len(matched_ids)

            batch_ids = {r["clause_id"] for r in batch}
            missing_in_batch = batch_ids - matched_ids
            if missing_in_batch:
                not_found.extend(missing_in_batch)

            print(f"  批次 {i}~{i+len(batch)}: 成功寫入 {len(matched_ids)}/{len(batch)} 筆")

        print(f"\n完成。共寫入 {matched_total}/{len(records)} 筆節點的 embedding 屬性。")

        if not_found:
            print(f"⚠️ 有 {len(not_found)} 個 clause_id 在Neo4j裡找不到對應的 (:PolicyClause) 節點，"
                  f"embedding未寫入(不是靜默跳過，明確列出以下清單供人工確認)：")
            for cid in not_found:
                print(f"    - {cid}")
            print("   請確認這些 clause_id 是否在原本匯入 policy_clauses_full.json 時"
                  "就已經漏建節點，或是clause_id本身有拼字/型別不一致(例如字串vs數字)的問題。")

    driver.close()

    print("\n下一步：可以用以下Cypher語法測試向量檢索是否正常運作，")
    print("query_embedding 需要用同一個 gemini-embedding-001 模型、")
    print("task_type=\"RETRIEVAL_QUERY\"(對應embedding時用的RETRIEVAL_DOCUMENT)算出：")
    print("""
    CALL db.index.vector.queryNodes(
        'policyClauseEmbeddings', 5, $query_embedding
    ) YIELD node, score
    RETURN node.clause_id, node.content, score
    ORDER BY score DESC
    """)


if __name__ == "__main__":
    main()
