# -*- coding: utf-8 -*-
"""
eval_tpl_lora_v1.py — 用 vertex_sft_val.jsonl(101題) 打 tpl-claims-qa-lora-v1 endpoint，
比對輸出跟原始正確答案(fairness.py/fraud_rules.py算出來的)是否一致。

架構重點：
1. tuned endpoint在 us multi-region，跟embedding/一般呼叫慣用的 us-central1 是分開的
   genai.Client——這是你們V2就踩過的坑，這裡直接開兩個Client示範正確用法
   (這支腳本其實只需要us這個Client，us-central1那個是保留給你之後串embedding用的示範)
2. thinking_budget=0：沿用你們解過的 gemini-3.5-flash MALFORMED_FUNCTION_CALL 問題
3. 「準確率」怎麼算——不是逐字比對答案文字(tuned模型措辭不會跟訓練資料一模一樣)，
   是比對「結論」是否一致：
   - amount_fairness: 從答案文字判斷模型說「合理」還是「不合理」，跟正確答案的
     『合理。』/『不在合理區間內』開頭比對
   - fraud_risk: 判斷模型有沒有指出「有詐欺風險/觸發旗標」vs「均未觸發」，
     跟正確答案比對
   - clause_interpretation: 條款類不容易用二元判斷評分，這裡先只列出來人工看，
     不計入準確率分母(不然會掩蓋真正的verdict準確率數字)
"""
import json
import re
import time

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

PROJECT = "cki-scu-2026"
TUNED_ENDPOINT = "projects/624489584685/locations/us/endpoints/5547731053478674432"
VAL_PATH = "vertex_sft_val.jsonl"


def load_val_examples(path: str) -> list:
    """從Vertex SFT格式(systemInstruction+contents)還原出question/answer/task_type。
    注意：轉換時task_type被丟在原始all_qa_tpl.jsonl裡，SFT格式本身沒有這個欄位，
    所以這支腳本假設你手邊還留著all_qa_tpl.jsonl可以對照——如果沒有，用question
    文字本身判斷task_type(見_infer_task_type)。"""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            question = d["contents"][0]["parts"][0]["text"]
            gold_answer = d["contents"][1]["parts"][0]["text"]
            examples.append({"question": question, "gold_answer": gold_answer})
    return examples


def _infer_task_type(question: str) -> str:
    if "有沒有詐欺風險" in question:
        return "fraud_risk"
    if "這個金額合理嗎" in question:
        return "amount_fairness"
    return "clause_interpretation"


def _extract_verdict_amount(text: str) -> str:
    """從答案文字判斷amount_fairness的結論：合理/不合理/無法判斷"""
    if re.search(r"^不在合理區間|不合理|超出.*區間|建議退回", text):
        return "不合理"
    if re.search(r"^合理", text):
        return "合理"
    return "無法判斷"


def _extract_verdict_fraud(text: str) -> str:
    """從答案文字判斷fraud_risk的結論：有風險/無風險/無法判斷"""
    if re.search(r"^有，觸發|^有警示訊號", text):
        return "有風險"
    if "均未觸發" in text:
        return "無風險"
    return "無法判斷"


def call_tuned_model(client: genai.Client, question: str, system_instruction: str) -> str:
    response = client.models.generate_content(
        model=TUNED_ENDPOINT,
        contents=question,
        config=GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,  # 評測要看模型真實判斷，不要隨機性
            thinking_config=ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


SYSTEM_INSTRUCTION = (
    "你是兆豐產物保險TPL(第三人責任險)理賠案件的法官代理人(Judge Agent)。"
    "你的判斷必須基於實際案件資料與既有規則模組(金額公平性Tukey IQR圍籬、"
    "17項詐欺旗標規則、保單條款)，不可憑空推測。若資料不足以支撐判斷，"
    "必須誠實說明資料缺口(標記為na或建議轉人工複核)，不可用臆測的數字或結論"
    "取代缺失的資訊。"
)


def main():
    client = genai.Client(vertexai=True, project=PROJECT, location="us")

    examples = load_val_examples(VAL_PATH)
    print(f"共{len(examples)}題validation")

    results = []
    for i, ex in enumerate(examples):
        task_type = _infer_task_type(ex["question"])
        try:
            model_answer = call_tuned_model(client, ex["question"], SYSTEM_INSTRUCTION)
        except Exception as e:
            print(f"  [{i+1}/{len(examples)}] 呼叫失敗: {e}")
            results.append({**ex, "task_type": task_type, "model_answer": None, "match": None})
            time.sleep(1)
            continue

        if task_type == "amount_fairness":
            gold_verdict = _extract_verdict_amount(ex["gold_answer"])
            model_verdict = _extract_verdict_amount(model_answer)
            match = (gold_verdict == model_verdict) and gold_verdict != "無法判斷"
        elif task_type == "fraud_risk":
            gold_verdict = _extract_verdict_fraud(ex["gold_answer"])
            model_verdict = _extract_verdict_fraud(model_answer)
            match = (gold_verdict == model_verdict) and gold_verdict != "無法判斷"
        else:
            gold_verdict = model_verdict = None
            match = None  # clause類不計入verdict準確率

        results.append({**ex, "task_type": task_type, "model_answer": model_answer,
                         "gold_verdict": gold_verdict, "model_verdict": model_verdict, "match": match})
        print(f"  [{i+1}/{len(examples)}] {task_type}: match={match}")

    with open("eval_results_tpl_v1.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 統計：只對amount_fairness/fraud_risk算verdict準確率，clause類另外列
    scoreable = [r for r in results if r["task_type"] in ("amount_fairness", "fraud_risk")]
    failed_calls = [r for r in scoreable if r["match"] is None and r["model_answer"] is None]
    unparseable = [r for r in scoreable if r["match"] is None and r["model_answer"] is not None]
    scored = [r for r in scoreable if r["match"] is not None]
    correct = sum(1 for r in scored if r["match"])

    print(f"\n=== 結果 ===")
    print(f"amount_fairness + fraud_risk 共{len(scoreable)}題")
    print(f"  API呼叫失敗: {len(failed_calls)}")
    print(f"  結論判讀不出來(無法用regex判斷合理/不合理): {len(unparseable)}")
    print(f"  可判分: {len(scored)}題，正確{correct}題")
    if scored:
        print(f"  Verdict準確率: {correct/len(scored)*100:.1f}%")

    clause_n = sum(1 for r in results if r["task_type"] == "clause_interpretation")
    print(f"\nclause_interpretation {clause_n}題已寫入eval_results_tpl_v1.jsonl，建議人工抽查(不計入準確率)")
    print("完整結果：eval_results_tpl_v1.jsonl")


if __name__ == "__main__":
    main()
