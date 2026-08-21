# -*- coding: utf-8 -*-
"""
export_rsa_clauses_from_neo4j.py
===================================
從 Neo4j 匯出 insurance_line='RSA' 的 PolicyClause 節點，存成JSON。
純讀取，不寫入任何東西，安全性最高，方便你先確認資料內容再進行下一步
(rebuild_rsa_clause_bm25.py 才會真正改寫 BM25 db)。

環境變數：
  NEO4J_URI / NEO4J_USER(或NEO4J_USERNAME) / NEO4J_PASSWORD

執行：
  python export_rsa_clauses_from_neo4j.py
"""

import os
import sys
import json


def main():
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    missing = [n for n, v in [("NEO4J_URI", uri), ("NEO4J_USER/NEO4J_USERNAME", user),
                              ("NEO4J_PASSWORD", password)] if not v]
    if missing:
        print(f"❌ 缺少環境變數: {', '.join(missing)}")
        sys.exit(1)

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # 先確認 insurance_line 欄位實際有哪些值，避免篩選條件跟實際資料寫法對不上
        # (例如是 'RSA' 還是 'rsa' 或其他寫法)，明確印出來讓你確認，不是用猜的。
        distinct = session.run(
            "MATCH (c:PolicyClause) RETURN DISTINCT c.insurance_line AS line, count(c) AS cnt"
        )
        print("=== PolicyClause.insurance_line 實際分佈 ===")
        lines_seen = []
        for r in distinct:
            print(f"  {r['line']!r}: {r['cnt']} 筆")
            lines_seen.append(r["line"])

        if "RSA" not in lines_seen:
            print("\n⚠️ 沒有找到值剛好是 'RSA' 的節點，上面列出的實際值可能跟預期不同。")
            print("   請確認要用哪個值篩選，本次先不自動猜測繼續執行。")
            driver.close()
            sys.exit(1)

        result = session.run("""
            MATCH (c:PolicyClause)
            WHERE c.insurance_line = 'RSA'
            RETURN c.clause_id AS clause_id,
                   c.doc_title AS doc_title,
                   c.article_title AS article_title,
                   c.content AS content,
                   c.status AS status,
                   c.insurance_line AS insurance_line
            ORDER BY c.clause_id
        """)
        records = [dict(r) for r in result]

    driver.close()

    print(f"\n共撈出 {len(records)} 筆 RSA 條款節點")

    missing_content = [r["clause_id"] for r in records if not r.get("content")]
    if missing_content:
        print(f"⚠️ 有 {len(missing_content)} 筆 content 是空的，這些之後BM25索引不到任何內容: {missing_content}")

    status_counter = {}
    for r in records:
        status_counter[r.get("status")] = status_counter.get(r.get("status"), 0) + 1
    print(f"status 分佈: {status_counter}")
    print("(如果有非'active'或None以外的狀態，例如'deprecated'，請確認要不要排除，"
          "本次匯出「沒有」依status過濾，全部撈出來，過濾與否留給你決定)")

    out_path = "rsa_clauses_export.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n已輸出: {out_path}")
    print("請打開檢查內容是否符合預期，確認沒問題後再執行 rebuild_rsa_clause_bm25.py")


if __name__ == "__main__":
    main()
