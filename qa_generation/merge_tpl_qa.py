# -*- coding: utf-8 -*-
"""
merge_tpl_qa.py — 合併 from_cases_tpl.py / from_graph_tpl.py /（若有）from_clauses_tpl.py
的輸出，依 task_type 平衡取樣到目標題數，寫出最終微調用的 all_qa_tpl.jsonl。

跑法（先各自跑完生成腳本，再跑這支合併——這次只用cases_tpl+clauses_tpl兩個來源，
graph_tpl(GraphRAG API)先不用，見下方SOURCES設定）：
    python qa_generation/from_cases_tpl.py
    python qa_generation/from_clauses_tpl.py
    python qa_generation/merge_tpl_qa.py

平衡取樣原則：cases_tpl 光是696筆案件就能生出1392題(amount_fairness+fraud_risk各半)，
如果直接取前1000題會讓 fault_consistency(判決先例) 這個推理能力在訓練資料裡幾乎缺席。
這裡改成先看每個task_type各有多少題，再按比例平均分配到目標題數，缺哪個來源就如實
在log印出來，不會用其他來源硬湊補齊。

standard_version標記：只有 fault_consistency (task_type) 的QA會標
standard_version="pre-2026-07-01"，因為它的答案依據是判決語料庫(GraphRAG API的
Precedent資料)，這批語料全部早於2026-07-01法規修正——這次先不過濾，只標記，
之後要依法規版本篩選訓練資料時直接用這個欄位即可，不用重新生成整批。
amount_fairness/fraud_risk的QA答案依據是TPL_train.db實付理賠金額/資料完整性，
跟該法規修正無直接關聯，不加此標記。

去重：同一task_type內，答案文字逐字相同的QA只保留1筆（用第一筆遇到的，問題本身
不同不影響判斷——這是刻意的，逐字相同的答案代表訓練資訊量只有1筆的價值，保留
全部只會讓模型學到「輸出這串固定文字」而不是真正的推理）。去重後空出來的配額
一樣用「剩餘可用題數」補到其他task_type，不會讓總數縮水。
"""
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

TARGET_TOTAL = 1000
STANDARD_VERSION_TAG = "pre-2026-07-01"
TASK_TYPES_NEEDING_TAG = {"fault_consistency"}

SOURCES = [
    ("qa_generation/output_cases_tpl.jsonl", "cases"),
    ("qa_generation/output_clauses_tpl.jsonl", "clauses"),
    # graph(GraphRAG API)這次不用，先只依循TPL_train.db+policy_clauses.db兩個來源；
    # 之後要加回來的話把下面這行取消註解即可，merge邏輯不用改
    # ("qa_generation/output_graph_tpl.jsonl", "graph"),
]


def load_jsonl(path: str) -> list:
    p = Path(path)
    if not p.exists():
        print(f"[!] 找不到 {path}，此來源這次先跳過")
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    all_qa = []
    for path, label in SOURCES:
        items = load_jsonl(path)
        print(f"讀到 {label}: {len(items)} 題")
        all_qa.extend(items)

    if not all_qa:
        print("[!] 三個來源都沒有資料，先跑生成腳本")
        return

    for qa in all_qa:
        if qa.get("task_type") in TASK_TYPES_NEEDING_TAG:
            qa["standard_version"] = STANDARD_VERSION_TAG

    by_type = defaultdict(list)
    for qa in all_qa:
        by_type[qa.get("task_type", "unknown")].append(qa)

    print("\n=== 各task_type現有題數(去重前) ===")
    for t, items in by_type.items():
        print(f"  {t}: {len(items)}")

    print("\n=== 去重(同task_type內答案文字逐字相同只留1筆) ===")
    for t in list(by_type.keys()):
        seen_answers = set()
        deduped = []
        for qa in by_type[t]:
            ans = qa.get("answer", "")
            if ans in seen_answers:
                continue
            seen_answers.add(ans)
            deduped.append(qa)
        removed = len(by_type[t]) - len(deduped)
        if removed:
            print(f"  {t}: 去除 {removed} 筆重複答案，剩 {len(deduped)} 題")
        by_type[t] = deduped

    # 平均分配基礎配額，優先讓每個task_type都有代表性；配額不足的來源缺口，
    # 用其他來源的「剩餘可用題數」補滿到TARGET_TOTAL（不是憑空生成，是動用真實
    # 存在但超過基礎配額的資料），這樣才不會因為某個來源題數少就拉低總量。
    n_types = len(by_type)
    per_type_quota = TARGET_TOTAL // n_types
    remainder = TARGET_TOTAL - per_type_quota * n_types

    for t in by_type:
        random.shuffle(by_type[t])

    allocated = {}
    shortfall_total = 0
    for i, (t, items) in enumerate(sorted(by_type.items())):
        quota = per_type_quota + (1 if i < remainder else 0)
        allocated[t] = min(quota, len(items))
        if len(items) < quota:
            shortfall = quota - len(items)
            shortfall_total += shortfall
            print(f"  [!] {t} 只有 {len(items)} 題，基礎配額缺 {shortfall} 題，將從其他來源剩餘題數補")

    remaining_budget = TARGET_TOTAL - sum(allocated.values())
    if remaining_budget > 0:
        surplus_types = [t for t in by_type if len(by_type[t]) > allocated[t]]
        while remaining_budget > 0 and surplus_types:
            for t in list(surplus_types):
                if remaining_budget <= 0:
                    break
                if len(by_type[t]) > allocated[t]:
                    allocated[t] += 1
                    remaining_budget -= 1
                else:
                    surplus_types.remove(t)

    sampled = []
    for t, n in allocated.items():
        sampled.extend(by_type[t][:n])
    random.shuffle(sampled)

    final_shortfall = TARGET_TOTAL - len(sampled)
    print(f"\n最終配額：{ {t: allocated[t] for t in sorted(allocated)} }")
    print(f"平衡取樣後：{len(sampled)} 題（目標 {TARGET_TOTAL}，"
          f"真實總缺口 {final_shortfall} 題——只有全部來源加總都不夠1000才會缺）")

    with open("qa_generation/all_qa_tpl.jsonl", "w", encoding="utf-8") as f:
        for qa in sampled:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print("輸出：qa_generation/all_qa_tpl.jsonl")


if __name__ == "__main__":
    main()
