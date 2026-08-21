# -*- coding: utf-8 -*-
"""
from_clauses_tpl.py — 從 policy_clauses.db 反向生成TPL條款QA

實際schema只有一張表：policy_clauses(id, text, article, source)，124筆，
來源10份PDF（超額責任附加條款/傷害責任多倍保障附加條款/主保險條款各自用+營業用，
加上強制汽車責任保險給付標準兩個修正版本）。跟RSA的from_clauses.py同一個原則：
條款文字本身就是答案，不需要LLM編造，純規則抽取最安全。

已知資料品質問題（誠實跳過，不硬修補）：
    強制汽車責任保險給付標準1150225.pdf（民國115年2月25日=2026-02-25修正）
    的PDF擷取破碎，9筆裡5筆內容只剩條號標題、正文遺失（條文邊界被切壞）。
    同一份標準另有1150529.pdf（2026-05-29修正，更新且擷取完整），因此
    1150225.pdf整份跳過不產生QA——原因是擷取品質不足+已被更新版取代，
    不是用regex硬拼湊出內容，那樣等於自己編條文。

standard_version標記：兩份「強制汽車責任保險給付標準」文件都在2026-07-01法規
修正之前，這裡標記實際修正日期(從檔名解析，例如"2026-05-29")而非籠統字串，
方便之後精確篩選。保單本身的附加條款(超額責任/傷害責任多倍保障)不受此次
強制險給付標準修正影響，不加此標記。
"""
import argparse
import json
import re
import sqlite3

CONFIG = {
    "db_path": "policy_clauses.db",
    "output_path": "qa_generation/output_clauses_tpl.jsonl",
    "min_text_length": 30,  # 低於此長度視為擷取破碎(只剩條號標題)，跳過不產生QA
}

# 從檔名判斷文件類別與用途，用來組問題模板
DOC_CATEGORY_PATTERNS = [
    (re.compile(r"超額責任附加條款"), "超額責任附加條款"),
    (re.compile(r"傷害責任多倍保障附加條款"), "傷害責任多倍保障附加條款"),
    (re.compile(r"強制汽車責任保險給付標準"), "強制汽車責任保險給付標準"),
    (re.compile(r"第三人責任保險"), "第三人責任保險主保險條款"),
]

# 已知擷取破碎、已被更新版取代，整份跳過
SKIP_SOURCES = {"強制汽車責任保險給付標準1150225.pdf"}

# 強制汽車責任保險給付標準的檔名日期(民國年轉西元年)，用來標記standard_version
ROC_DATE_PATTERN = re.compile(r"(\d{3})(\d{2})(\d{2})")


def introspect(db_path: str):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    print(f"=== {db_path} ===")
    for (tname,) in tables:
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({tname});").fetchall()]
        count = cur.execute(f"SELECT COUNT(*) FROM {tname};").fetchone()[0]
        print(f"  table={tname}  rows={count}  columns={cols}")
    print("\n  各來源文件筆數/平均長度/短文字(<30字)筆數：")
    for (src,) in cur.execute("SELECT DISTINCT source FROM policy_clauses").fetchall():
        rows = cur.execute("SELECT LENGTH(text) FROM policy_clauses WHERE source=?", (src,)).fetchall()
        n = len(rows)
        avg_len = sum(l for (l,) in rows) / n if n else 0
        short = sum(1 for (l,) in rows if l < CONFIG["min_text_length"])
        print(f"    {src[:55]:55s} rows={n:3d} avg_len={avg_len:6.0f} short={short}")
    conn.close()


def _doc_category(source: str) -> str:
    for pattern, label in DOC_CATEGORY_PATTERNS:
        if pattern.search(source):
            return label
    return "保單條款"


def _vehicle_use(source: str) -> str:
    if "營業用" in source:
        return "營業用"
    if "自用" in source:
        return "自用"
    return ""  # 強制險給付標準不分自用/營業用，留空


def _clean_text(raw: str) -> str:
    """去除PDF擷取的空白行/純數字頁碼行雜訊，保留實質內容行。"""
    lines = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.isdigit():  # 頁碼/註腳編號雜訊行(例如獨立一行"1"、"2")
            continue
        lines.append(s)
    return " ".join(lines).strip()


def _standard_version_from_filename(source: str):
    """從「強制汽車責任保險給付標準」檔名解析民國年月日，轉西元日期字串。
    非此類文件回傳None。"""
    if "給付標準" not in source:
        return None
    m = ROC_DATE_PATTERN.search(source)
    if not m:
        return None
    roc_year, month, day = m.groups()
    western_year = int(roc_year) + 1911
    return f"{western_year}-{month}-{day}"


def generate_clause_qa(db_path: str, output_path: str) -> int:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT id, text, article, source FROM policy_clauses ORDER BY id")
    rows = cur.fetchall()
    con.close()

    qa_pairs = []
    skipped_broken_source = 0
    skipped_short = 0

    for row_id, text, article, source in rows:
        if source in SKIP_SOURCES:
            skipped_broken_source += 1
            continue

        answer = _clean_text(text)
        if len(answer) < CONFIG["min_text_length"]:
            skipped_short += 1
            continue

        category = _doc_category(source)
        vehicle_use = _vehicle_use(source)
        article_clean = article.replace(" ", "")

        if vehicle_use:
            question = f"{vehicle_use}汽車{category}的{article_clean}規定什麼？"
        else:
            question = f"{category}的{article_clean}規定什麼？"

        qa = {
            "question": question, "answer": answer,
            "task_type": "clause_interpretation",
            "source_row_id": row_id, "source_document": source,
        }

        std_version = _standard_version_from_filename(source)
        if std_version:
            qa["standard_version"] = std_version

        qa_pairs.append(qa)

    with open(output_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"跳過 {skipped_broken_source} 筆(來自已知擷取破碎、被取代的文件: {SKIP_SOURCES})")
    print(f"跳過 {skipped_short} 筆(文字長度<{CONFIG['min_text_length']}字，判斷為擷取不完整)")
    return len(qa_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["introspect", "generate"], default="generate")
    parser.add_argument("--db_path", default=CONFIG["db_path"])
    args = parser.parse_args()

    if args.step == "introspect":
        introspect(args.db_path)
    else:
        n = generate_clause_qa(args.db_path, CONFIG["output_path"])
        print(f"產生 {n} 筆QA，寫入 {CONFIG['output_path']}")
