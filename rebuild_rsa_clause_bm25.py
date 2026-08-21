# -*- coding: utf-8 -*-
"""
rebuild_rsa_clause_bm25.py
=============================
把 rsa_bm25_knowledge_base_V1.db 裡的 rsa_clause(20筆) + rsa_fee_schedule
(12筆)，用 Neo4j 匯出的 32 筆 RSA PolicyClause(export_rsa_clauses_from_
neo4j.py 產生的 rsa_clauses_export.json)重新寫入，確保兩邊內容一致、
clause_id 對得上，之後才能做 BM25/Vector 候選聯集。

依 doc_title 分流回正確的 source_type(收費標準文件 -> rsa_fee_schedule，
其餘兩份附加條款文件 -> rsa_clause)，避免內容被重複插入成兩份。

不動 rsa_case(700筆)。

因為BM25的idf/avgdl是全語料庫層級的統計量，只要chunks數量有變動，
理論上全部的idf都會微幅變化，所以這裡是「整個bm25_vocab/bm25_postings/
bm25_stats 全部重算」，不是只補差異，避免統計量不一致。

安全機制：
  1. 執行前自動備份原始db(.bak後綴)
  2. 執行後印出前後對照統計，明確列出改動範圍

環境變數：
  無(純本機檔案操作)

執行：
  pip install jieba zhconv
  python rebuild_rsa_clause_bm25.py
"""

import os
import sys
import json
import math
import shutil
import sqlite3
import hashlib
import datetime
import collections

DB_PATH = "rsa_bm25_knowledge_base_V1.db"
EXPORT_PATH = "rsa_clauses_export.json"

FEE_SCHEDULE_DOC_TITLE = "汽車道路救援服務項目及收費標準"
REBUILD_SOURCE_TYPES = ("rsa_clause", "rsa_fee_schedule")


def source_type_for(doc_title):
    """依doc_title分流，保留原本db區分rsa_clause / rsa_fee_schedule的慣例，
    避免收費標準內容被重複插入成兩份。"""
    if doc_title == FEE_SCHEDULE_DOC_TITLE:
        return "rsa_fee_schedule"
    return "rsa_clause"


def backup_db(path):
    if not os.path.exists(path):
        print(f"❌ 找不到 {path}")
        sys.exit(1)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.{ts}.bak"
    shutil.copy2(path, backup_path)
    print(f"已備份原始db: {backup_path}")
    return backup_path


def tokenize(text, jieba, zhconv):
    """轉簡體 + jieba分詞，比照既有chunks的tokenized_text慣例"""
    simplified = zhconv.convert(text, "zh-cn")
    tokens = [t.strip().lower() for t in jieba.cut(simplified) if t.strip()]
    # 過濾掉純標點符號的token(既有資料裡似乎也有做這層過濾)
    tokens = [t for t in tokens if any(c.isalnum() for c in t)]
    return tokens


def main():
    if not os.path.exists(EXPORT_PATH):
        print(f"❌ 找不到 {EXPORT_PATH}，請先執行 export_rsa_clauses_from_neo4j.py")
        sys.exit(1)

    import jieba
    import zhconv

    with open(EXPORT_PATH, encoding="utf-8") as f:
        clauses = json.load(f)
    clauses = [c for c in clauses if c.get("content")]  # 跳過content為空的
    print(f"讀取到 {len(clauses)} 筆有效RSA條款(content非空)")

    backup_db(DB_PATH)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # ===== 1. 記錄改動前的狀態 =====
    before_doc_count = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    before_chunk_count = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    placeholders = ",".join("?" * len(REBUILD_SOURCE_TYPES))
    old_clause_doc_ids = [r[0] for r in cur.execute(
        f"SELECT document_id FROM documents WHERE source_type IN ({placeholders})",
        REBUILD_SOURCE_TYPES
    ).fetchall()]
    old_clause_chunk_count = cur.execute(
        f"SELECT COUNT(*) FROM chunks WHERE source_type IN ({placeholders})",
        REBUILD_SOURCE_TYPES
    ).fetchone()[0]
    print(f"\n改動前: documents={before_doc_count}, chunks={before_chunk_count}, "
          f"舊 rsa_clause+rsa_fee_schedule chunks={old_clause_chunk_count}")

    # ===== 2. 刪除舊的 rsa_clause + rsa_fee_schedule documents/chunks =====
    cur.execute(f"DELETE FROM chunks WHERE source_type IN ({placeholders})", REBUILD_SOURCE_TYPES)
    if old_clause_doc_ids:
        cur.executemany("DELETE FROM documents WHERE document_id=?",
                         [(d,) for d in old_clause_doc_ids])
    print(f"已刪除舊的 rsa_clause+rsa_fee_schedule: {old_clause_chunk_count} chunks, "
          f"{len(old_clause_doc_ids)} documents")

    # ===== 3. 插入新的32筆(一條文=一document=一chunk) =====
    now = datetime.datetime.now().isoformat()
    inserted = 0
    skipped_duplicate_key = []

    for clause in clauses:
        clause_id = clause["clause_id"]
        content = clause["content"]
        doc_title = clause.get("doc_title") or ""
        article_title = clause.get("article_title") or ""
        source_type = source_type_for(doc_title)

        # external_key 必須唯一，用 clause_id 本身(格式類似 RSA-CLAUSE-SELF::第2條)
        existing = cur.execute(
            "SELECT document_id FROM documents WHERE external_key=?", (clause_id,)
        ).fetchone()
        if existing:
            skipped_duplicate_key.append(clause_id)
            continue

        cur.execute("""
            INSERT INTO documents (external_key, source_type, title, source_file,
                                    case_no, page_count, extraction_method, source_db, created_at)
            VALUES (?, ?, ?, NULL, NULL, NULL, 'neo4j_export', 'Neo4j PolicyClause', ?)
        """, (clause_id, source_type, f"{doc_title} {article_title}".strip(), now))
        document_id = cur.lastrowid

        tokens = tokenize(content, jieba, zhconv)
        chunk_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        cur.execute("""
            INSERT INTO chunks (document_id, source_type, source_table, source_record_id,
                                 chunk_no, page_start, page_end, char_start, char_end,
                                 text_content, tokenized_text, token_count, chunk_hash)
            VALUES (?, ?, 'Neo4j.PolicyClause', NULL, 1, NULL, NULL, NULL, NULL,
                    ?, ?, ?, ?)
        """, (document_id, source_type, content, json.dumps(tokens, ensure_ascii=False),
              len(tokens), chunk_hash))
        inserted += 1

    print(f"已插入新的 rsa_clause+rsa_fee_schedule: {inserted} chunks")
    if skipped_duplicate_key:
        print(f"⚠️ 有 {len(skipped_duplicate_key)} 筆因 external_key 重複被跳過(不應該發生，"
              f"代表匯出資料裡有重複的clause_id): {skipped_duplicate_key}")

    # ===== 4. 全庫重算 bm25_vocab / bm25_postings / bm25_stats =====
    # idf/avgdl 是全語料庫統計量，chunks異動後全部要重算，不是只算異動部分。
    print("\n開始全庫重算 BM25 統計量(vocab/postings/stats)...")

    all_chunks = cur.execute("SELECT chunk_id, tokenized_text FROM chunks").fetchall()
    N = len(all_chunks)

    doc_freq = collections.Counter()
    term_freq_per_chunk = {}  # chunk_id -> Counter(term)
    total_len = 0

    for row in all_chunks:
        chunk_id = row["chunk_id"]
        tokens = json.loads(row["tokenized_text"])
        total_len += len(tokens)
        tf = collections.Counter(tokens)
        term_freq_per_chunk[chunk_id] = tf
        for term in tf:
            doc_freq[term] += 1

    avgdl = total_len / N if N else 0

    cur.execute("DELETE FROM bm25_postings")
    cur.execute("DELETE FROM bm25_vocab")

    for term, df in doc_freq.items():
        # 標準 Robertson/Okapi IDF 公式，若你原本build流程用的是不同公式，
        # 這裡的數值會跟原本略有差異，請留意下游有沒有寫死假設某種公式。
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        cur.execute("INSERT INTO bm25_vocab (term, document_frequency, idf) VALUES (?, ?, ?)",
                    (term, df, idf))

    postings_rows = []
    for chunk_id, tf in term_freq_per_chunk.items():
        for term, freq in tf.items():
            postings_rows.append((term, chunk_id, freq))
    cur.executemany(
        "INSERT INTO bm25_postings (term, chunk_id, term_frequency) VALUES (?, ?, ?)",
        postings_rows
    )

    cur.execute("UPDATE bm25_stats SET stat_value=? WHERE stat_key='N'", (str(N),))
    cur.execute("UPDATE bm25_stats SET stat_value=? WHERE stat_key='avgdl'", (str(avgdl),))
    cur.execute("UPDATE bm25_stats SET stat_value=? WHERE stat_key='created_at'",
                (datetime.datetime.now().isoformat(),))
    # k1, b 維持原本db裡的既有值(1.5 / 0.75)，本次不重新調校

    con.commit()

    # ===== 5. 印出改動後對照 =====
    after_doc_count = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    after_chunk_count = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    after_clause_count = cur.execute(
        "SELECT COUNT(*) FROM chunks WHERE source_type='rsa_clause'"
    ).fetchone()[0]
    after_fee_count = cur.execute(
        "SELECT COUNT(*) FROM chunks WHERE source_type='rsa_fee_schedule'"
    ).fetchone()[0]
    vocab_count = cur.execute("SELECT COUNT(*) FROM bm25_vocab").fetchone()[0]
    postings_count = cur.execute("SELECT COUNT(*) FROM bm25_postings").fetchone()[0]

    print(f"\n=== 改動後對照 ===")
    print(f"documents: {before_doc_count} -> {after_doc_count}")
    print(f"chunks: {before_chunk_count} -> {after_chunk_count}")
    print(f"rsa_clause chunks: -> {after_clause_count}")
    print(f"rsa_fee_schedule chunks: -> {after_fee_count}")
    print(f"(合計應為32: rsa_clause 20 + rsa_fee_schedule 12)")
    print(f"bm25_vocab: {vocab_count} 個詞")
    print(f"bm25_postings: {postings_count} 筆")
    print(f"N (total docs): {N}, avgdl: {avgdl:.2f}")

    con.close()
    print("\n完成。原始db已備份，若結果有問題可以用備份檔還原。")


if __name__ == "__main__":
    main()
