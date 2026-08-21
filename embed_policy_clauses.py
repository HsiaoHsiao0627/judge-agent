# -*- coding: utf-8 -*-
"""
embed_policy_clauses.py
=========================
把157條PolicyClause的內容送進 gemini-embedding-001 算3072維向量，
逐筆checkpoint存檔，中斷可從斷點續跑(比照專案既有的embedding pipeline慣例)。

Embedding文字組成：doc_title + article_title + content
(把文件名稱跟條名一起餵進去，讓向量帶有一點來源脈絡，不是只有純條文內容)

輸出：
  clause_embeddings.jsonl   每行一筆 {clause_id, embedding, char_count}
  (jsonl而非單一大json，是為了checkpoint時只需append，不用整檔重寫)

環境變數：
  GEMINI_API_KEY        Gemini API金鑰(Colab用 userdata.get 或 getpass 設定)

執行：
  pip install google-genai
  python embed_policy_clauses.py

中斷後重跑：自動跳過已經在 clause_embeddings.jsonl 裡的 clause_id，接續未完成的部分。

=== 這版修正的地方(相較原始草稿) ===
1. char_count 改用 .get() 安全讀取，缺欄位時即時計算並明確印出警告，
   不假設輸入檔一定有這個欄位（避免 KeyError 整批中斷）。
2. 新增「回傳向量數量 == 送出筆數」的檢查，數量對不上時整批視為失敗、
   不寫入 checkpoint（下次重跑會自動補做），避免向量跟clause錯位配對
   卻沒有任何錯誤訊息的靜默風險。
3. EmbedContentConfig 加上 task_type="RETRIEVAL_DOCUMENT"，這是官方文件
   建議的檢索場景設定，之後查詢端要記得對應改用 RETRIEVAL_QUERY，
   兩邊task_type要配對才能拿到校準過的相似度分數。
4. 下一步接 load_embeddings_to_neo4j.py 把這裡產出的向量寫進 Neo4j，
   本腳本本身「不」寫資料庫，職責維持單一。
"""

import os
import json
import time
import sys

INPUT_FILE = "policy_clauses_full.json"
OUTPUT_FILE = "clause_embeddings.jsonl"
MODEL = "gemini-embedding-001"
DIMENSIONS = 3072
BATCH_SIZE = 10          # 每次API呼叫送幾條(官方文件顯示上限是250筆/2萬token，10很保守)
MAX_CONTENT_CHARS = 2000  # content過長時截斷，避免超過embedding模型輸入上限
                          # 注意：這是「字元數」而非「token數」的近似截斷，
                          # 中文字通常 1字元對應 >1 token，若條款內容逼近此長度
                          # 建議之後改用實際 tokenizer 計算，目前資料長度風險不高。
RETRY_LIMIT = 3
RETRY_WAIT_SEC = 5


def build_embed_text(clause):
    """組合要送進embedding的文字：文件名+條名+內容"""
    parts = [clause["doc_title"]]
    if clause.get("article_title"):
        parts.append(clause["article_title"])
    content = clause["content"]
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
    parts.append(content)
    return "\n".join(parts)


def get_char_count(clause):
    """安全讀取char_count，缺欄位時即時計算並明確警告(不靜默補)"""
    if "char_count" in clause:
        return clause["char_count"]
    computed = len(clause.get("content", ""))
    print(f"  ⚠️ clause_id={clause.get('clause_id')} 缺少 char_count 欄位，"
          f"已即時計算為 {computed}（請確認 policy_clauses_full.json 的產生流程）")
    return computed


def load_done_ids(path):
    """讀取已完成的clause_id集合，供resume使用"""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["clause_id"])
            except json.JSONDecodeError:
                continue  # 忽略斷行寫入不完整的最後一行，重跑時會補回
    return done


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ 缺少環境變數 GEMINI_API_KEY")
        sys.exit(1)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    with open(INPUT_FILE, encoding="utf-8") as f:
        clauses = json.load(f)

    done_ids = load_done_ids(OUTPUT_FILE)
    todo = [c for c in clauses if c["clause_id"] not in done_ids]

    print(f"總條文數: {len(clauses)}")
    print(f"已完成(resume跳過): {len(done_ids)}")
    print(f"本次待處理: {len(todo)}")

    if not todo:
        print("全部已完成，無需處理。")
        return

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            texts = [build_embed_text(c) for c in batch]

            resp = None
            for attempt in range(1, RETRY_LIMIT + 1):
                try:
                    resp = client.models.embed_content(
                        model=MODEL,
                        contents=texts,
                        config=types.EmbedContentConfig(
                            output_dimensionality=DIMENSIONS,
                            task_type="RETRIEVAL_DOCUMENT",
                        ),
                    )
                    # 關鍵防呆：回傳向量數量必須跟送出的筆數完全一致，
                    # 否則下面的 zip 會把向量錯位配對到錯的 clause 卻不報錯。
                    if len(resp.embeddings) != len(batch):
                        print(f"  ❌ 批次{i}~{i+len(batch)} 回傳向量數量({len(resp.embeddings)}) "
                              f"與送出筆數({len(batch)})不符，整批視為失敗，不寫入checkpoint")
                        resp = None
                    break
                except Exception as e:
                    print(f"  ⚠️ 批次{i}~{i+len(batch)} 第{attempt}次失敗: {e}")
                    if attempt == RETRY_LIMIT:
                        print(f"  ❌ 批次{i}~{i+len(batch)} 重試已達上限，跳過此批次"
                              f"(這批的clause_id不會寫入checkpoint，下次重跑會自動補做)")
                        resp = None
                    else:
                        time.sleep(RETRY_WAIT_SEC)

            if resp is None:
                continue

            for clause, emb in zip(batch, resp.embeddings):
                out_f.write(json.dumps({
                    "clause_id": clause["clause_id"],
                    "embedding": emb.values,
                    "char_count": get_char_count(clause),
                }, ensure_ascii=False) + "\n")
                out_f.flush()  # 每筆立即落盤，中斷不遺失已算好的向量

            done_count = min(i + BATCH_SIZE, len(todo))
            print(f"  已完成 {done_count}/{len(todo)}")

    final_done = load_done_ids(OUTPUT_FILE)
    print(f"\n完成。checkpoint檔內共有 {len(final_done)}/{len(clauses)} 條向量。")
    if len(final_done) < len(clauses):
        missing = len(clauses) - len(final_done)
        print(f"⚠️ 仍有 {missing} 條未成功(通常是重試上限用盡)，重跑本script即可自動補做。")
        print("下一步：確認全部完成後，執行 load_embeddings_to_neo4j.py 把向量寫進 Neo4j。")
    else:
        print("下一步：執行 load_embeddings_to_neo4j.py 把向量寫進 Neo4j，並建立向量索引。")


if __name__ == "__main__":
    main()
