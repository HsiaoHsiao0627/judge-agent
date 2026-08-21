# -*- coding: utf-8 -*-
"""
from_clauses.py — 從 RSA_BM25_clauses.db 的 rsa_clause/rsa_fee_schedule 反向生成QA

原則：條款文字本身就是答案，不需要LLM編造，純規則抽取最安全。每筆QA都保留
source_chunk_id/doc_title，微調資料稽核時可以回頭核對是不是真的來自原文，
不是模型自己編的——這跟你法官代理人「每個決策都要可稽核」是同一個原則。

兩種QA：
1. 全條款QA：一條(或一個附件段落)對一個QA，答案是該段原文，問題用模板生成
2. 細顆粒度QA：針對「經濟型/保障型/豪華型」這種結構化條列，額外拆成逐項QA，
   訓練效果通常比整條硬塞給模型好(模型更容易學到「問限額答限額」而不是整段背誦)

不做的事：不用LLM去「潤飾」或「改寫」答案——這裡的答案必須是條款原文或從條款
精確抽取的數字，LLM介入只會增加幻覺風險，微調資料的正確性比多樣性重要。
"""
import json
import re
import sqlite3
from pathlib import Path

QUESTION_TEMPLATES_CLAUSE = [
    "{vehicle_use}道路救援保單的{article}規定什麼？",
    "根據{doc_title}，{article}的內容是？",
]
QUESTION_TEMPLATES_FEE = [
    "道路救援服務項目及收費標準裡，{section}的收費規則是什麼？",
]


def _extract_vehicle_use(text: str) -> str:
    m = re.search(r"【車輛用途】(\S+?)。", text)
    return m.group(1) if m else "RSA"


def _extract_article(text: str) -> str:
    m = re.search(r"【(第[一二三四五六七八九十]+條[^】]*)】", text)
    if m:
        return m.group(1)
    m2 = re.search(r"【(附件[一二三四五六七八九十]+[^】]*)】", text)
    return m2.group(1) if m2 else "本條款"


def _extract_tiers(text: str) -> list:
    """專門解析第二條(保險金額)那種「一、經濟型:...;二、保障型:...;三、豪華型:...」結構，
    拆成逐項QA。抓不到就回傳空列表，不強行湊數。"""
    tiers = []
    for name in ("經濟型", "保障型", "豪華型"):
        m = re.search(rf"{name}[:：]([^。]+(?:;[^。]+)*)。", text)
        if m:
            tiers.append({"tier": name, "detail": m.group(1)})
    return tiers


def _clean_answer_text(raw: str) -> str:
    """去掉【資料類型】【車輛用途】這類結構標籤，留下條款正文本身當答案，
    避免QA的答案裡出現內部資料標記，讀起來不像正式回覆。"""
    lines = [l for l in raw.split("\n") if l.strip() and not l.startswith("【資料類型】")
             and not l.startswith("【車輛用途】")]
    return " ".join(lines).strip()


def generate_clause_qa(db_path: str, output_path: str) -> int:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("""
        SELECT c.chunk_id, c.source_type, c.text_content, d.title
        FROM chunks c JOIN documents d ON c.document_id = d.document_id
        WHERE c.source_type IN ('rsa_clause', 'rsa_fee_schedule')
        ORDER BY c.chunk_id
    """)
    rows = cur.fetchall()
    con.close()

    qa_pairs = []
    for chunk_id, source_type, text, doc_title in rows:
        answer = _clean_answer_text(text)
        if source_type == "rsa_clause":
            vehicle_use = _extract_vehicle_use(text)
            article = _extract_article(text)
            question = f"{vehicle_use}道路救援保單的{article}規定什麼？"
            qa_pairs.append({
                "question": question, "answer": answer,
                "source_chunk_id": chunk_id, "source_type": source_type,
                "doc_title": doc_title,
            })
            for tier in _extract_tiers(text):
                q2 = f"{vehicle_use}道路救援保單的{tier['tier']}方案，理賠限額跟次數限制是什麼？"
                a2 = f"{tier['tier']}：{tier['detail']}。"
                qa_pairs.append({
                    "question": q2, "answer": a2,
                    "source_chunk_id": chunk_id, "source_type": "rsa_clause_tier",
                    "doc_title": doc_title,
                })
        else:  # rsa_fee_schedule
            m = re.search(r"【(附件[一二三四五六七八九十]+[^】]*)】", text)
            section = m.group(1) if m else "此收費項目"
            question = f"道路救援服務項目及收費標準裡，{section}的收費規則是什麼？"
            qa_pairs.append({
                "question": question, "answer": answer,
                "source_chunk_id": chunk_id, "source_type": source_type,
                "doc_title": doc_title,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    return len(qa_pairs)


if __name__ == "__main__":
    n = generate_clause_qa("data/RSA_BM25_clauses.db", "qa_generation/output_clauses.jsonl")
    print(f"產生 {n} 筆QA，寫入 qa_generation/output_clauses.jsonl")
