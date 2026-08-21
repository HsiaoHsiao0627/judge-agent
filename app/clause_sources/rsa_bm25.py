# -*- coding: utf-8 -*-
"""
rsa_bm25.py — RSA 保單條款 BM25 檢索（唯讀，隨容器映像打包）

資料來源：data/RSA_BM25_clauses.db，內含已算好的BM25索引(bm25_vocab的idf、
bm25_postings的term_frequency、bm25_stats的k1/b/avgdl/N)，查詢時只需要算
query端的分數加總，不需要重建索引。

⚠️ 關鍵修正(實測才發現，寫程式碼時看不出來)：資料庫裡 chunks.tokenized_text
用的詞是**簡體字**(例如「资料」不是「資料」)，但 chunks.text_content 原文是
繁體。如果直接拿使用者輸入的繁體查詢字串去比對 bm25_vocab，兩邊文字系統不一致，
幾乎所有詞都比對不到、分數全是0——查詢「拖吊費」得到空結果，但資料庫裡明明有
「拖吊救援」這種內容，不是資料不存在，是繁簡不一致害查詢端配不到。
修法：查詢字串先用 zhconv 轉簡體再分詞，跟建索引時的處理方式一致。

只涵蓋 source_type='rsa_clause'(20筆，兩份RSA附加條款逐條切塊)跟
'rsa_fee_schedule'(12筆，拖吊/急修收費標準，明確標註「僅供收費參考，
承保與不保判斷應以rsa_clause為準」)。'rsa_case'(700筆，RSA案例本身)
不在條文檢索範圍內，避免把過往案例敘述誤當成保單條款回傳。
"""
import sqlite3
from pathlib import Path

import jieba
import zhconv

from ..amount_sources._db import open_readonly

DEFAULT_SOURCE_TYPES = ("rsa_clause", "rsa_fee_schedule")


class RSABM25Retriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.con = open_readonly(db_path)
        cur = self.con.cursor()
        cur.execute("SELECT stat_key, stat_value FROM bm25_stats")
        stats = dict(cur.fetchall())
        self.k1 = float(stats["k1"])
        self.b = float(stats["b"])
        self.avgdl = float(stats["avgdl"])
        self.N = int(stats["N"])

    def close(self):
        self.con.close()

    @staticmethod
    def _tokenize(query: str) -> list:
        """轉簡體後分詞，跟建索引時的前處理一致(否則查不到，見檔頭說明)。"""
        simplified = zhconv.convert(query, "zh-cn")
        return [t.strip() for t in jieba.cut(simplified) if t.strip()]

    def search(self, query: str, top_k: int = 5,
               source_types: tuple = DEFAULT_SOURCE_TYPES) -> dict:
        terms = self._tokenize(query)
        if not terms:
            return {"query": query, "matched_terms": [], "results": [],
                    "count": 0, "reason": "查詢字串分詞後無有效詞"}

        cur = self.con.cursor()
        placeholders = ",".join("?" * len(terms))
        cur.execute(
            f"SELECT term, idf FROM bm25_vocab WHERE term IN ({placeholders})",
            terms,
        )
        term_idf = dict(cur.fetchall())

        matched_terms = [t for t in terms if t in term_idf]
        if not matched_terms:
            return {"query": query, "matched_terms": [], "results": [],
                    "count": 0, "reason": "分詞後的詞彙都不在BM25索引裡(vocab miss)"}

        stype_placeholders = ",".join("?" * len(source_types))
        term_placeholders = ",".join("?" * len(matched_terms))
        # 只在允許的 source_type 範圍內查 postings，並帶出 chunk 的 token_count 當文件長度(dl)
        cur.execute(
            f"""
            SELECT p.term, p.chunk_id, p.term_frequency, c.token_count,
                   c.text_content, c.chunk_no, d.title, c.source_type, d.external_key
            FROM bm25_postings p
            JOIN chunks c ON p.chunk_id = c.chunk_id
            JOIN documents d ON c.document_id = d.document_id
            WHERE p.term IN ({term_placeholders})
              AND c.source_type IN ({stype_placeholders})
            """,
            matched_terms + list(source_types),
        )
        rows = cur.fetchall()

        scores = {}
        chunk_meta = {}
        for term, chunk_id, tf, dl, text, chunk_no, title, stype, external_key in rows:
            idf = term_idf[term]
            dl = dl or self.avgdl
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score = idf * (tf * (self.k1 + 1)) / denom if denom else 0.0
            scores[chunk_id] = scores.get(chunk_id, 0.0) + score
            chunk_meta[chunk_id] = {
                "chunk_id": chunk_id, "chunk_no": chunk_no, "title": title,
                "source_type": stype, "text": text,
                # clause_id：只有重建過的db(external_key=RSA-CLAUSE-XXX::第N條格式)才有意義，
                # 舊格式db這個值會是PDF檔名，呼叫端(rsa_hybrid.py)靠這個跟Vector結果的
                # clause_id對齊做候選聯集，所以db沒重建的話聯集會失效(兩邊id體系對不上)。
                "clause_id": external_key,
            }

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results = [
            {**chunk_meta[cid], "score": round(score, 4)}
            for cid, score in ranked
        ]
        return {
            "query": query, "matched_terms": matched_terms,
            "unmatched_terms": [t for t in terms if t not in term_idf],
            "results": results, "count": len(results),
        }


class MockRSABM25Retriever:
    """找不到 data/RSA_BM25_clauses.db 時的占位，誠實回報未執行，不假裝有結果。"""
    def search(self, query: str, top_k: int = 5, source_types: tuple = DEFAULT_SOURCE_TYPES) -> dict:
        return {"query": query, "matched_terms": [], "results": [], "count": 0,
                "reason": "RSA_BM25_clauses.db 未找到，BM25條文檢索不可用(mock)"}

    def close(self):
        pass
