# -*- coding: utf-8 -*-
"""
reasoning_review.py — 審推理(公平性判斷的第二支腿)

背景：公平性判斷 = 獨立重估(fairness.py的Tukey IQR，看純結構化數字) + 審推理
(這支模組，看理賠代理人的reasoning文字有沒有站得住腳)。兩者是獨立訊號，缺一不可——
獨立重估抓不到「引用幻覺」、「漏查不保事項」這類推理過程的問題，這正是為什麼需要LLM。

鐵律：這支模組只做「一致性稽核」，絕不重新計算金額、絕不覆蓋fairness_check的結論。
跟詐欺旗標同一個精神——LLM輔助為輔，發現問題只能觸發escalate_human或
return_for_recalc並附上具體理由，不能自己「修正」理賠代理人的輸出。

失敗時的保守原則：
- 理賠代理人沒有給reasoning欄位(例如理賠代理人尚未真正接上、過渡期) → applicable=False，
  不觸發升級，這是「沒東西可審」不是「審查失敗」
- 有reasoning但LLM呼叫失敗(逾時/API錯誤/JSON解析失敗) → 必須誠實回報失敗原因，
  且不能默認放行——金額判斷這種事，審查失敗不能當作沒問題，見get_fallback_result()

backend可切換(env var REASONING_REVIEW_BACKEND，預設gemini)，目前只實作Gemini，
Claude/Ollama先留SEAM。
"""
import json
import os
from abc import ABC, abstractmethod

import httpx

REVIEW_SYSTEM_PROMPT = """你是保險理賠審查的稽核助手，任務是稽核「理賠代理人」的推理輸出\
是否忠實於它自己檢索到的證據，不是重新判斷理賠金額是否合理(那是另一套獨立的統計檢核在做)。

請只做以下四件事的稽核，不要給出你自己認為合理的金額：

1. citation_fidelity：reasoning裡描述的案例/條款內容，是否忠實反映cited_case_ids/
   cited_policy_ids實際指向的文字？有沒有幻覺、誤讀、無中生有的細節？
2. unaddressed_policy_flags：檢索到的policy_results裡，有沒有出現「不保事項」、
   「除外」、「限制」等條款，但reasoning/cited_policy_ids完全沒提到、沒有明確說明
   為何不適用？列出這些條款的doc_id。
3. amount_reasoning_consistent：reasoning裡描述的推論過程，跟suggested_amount數字
   本身是否對得上(不是各說各話)？
4. claim_agent_confidence_assessment：理賠代理人自己標示的confidence，對照證據的
   質與量(相似案例數量、命中來源數)是否合理？

只回傳以下格式的JSON，不要有其他文字：
{
  "citation_fidelity": "consistent" | "inconsistent" | "unable_to_verify",
  "unaddressed_policy_flags": ["doc_id", ...],
  "amount_reasoning_consistent": true | false,
  "claim_agent_confidence_assessment": "reasonable" | "overconfident" | "underconfident",
  "issues": ["具體問題描述，繁體中文"]
}"""


def _build_user_content(case: dict) -> str:
    parts = [
        f"【理賠代理人的suggested_amount(=claim_amount)】{case.get('claim_amount')}",
        f"【理賠代理人的reasoning】{case.get('reasoning', '')}",
        f"【cited_case_ids】{case.get('cited_case_ids', [])}",
        f"【cited_policy_ids】{case.get('cited_policy_ids', [])}",
        f"【claim_agent自評confidence】{case.get('claim_agent_confidence', '')}",
    ]
    case_results = case.get("retrieved_case_results") or []
    policy_results = case.get("retrieved_policy_results") or []
    parts.append("【檢索到的相似案例(retrieved_case_results)】")
    for r in case_results:
        parts.append(f"  doc_id={r.get('doc_id')}: {r.get('text', '')}")
    parts.append("【檢索到的保單條款(retrieved_policy_results)】")
    for r in policy_results:
        parts.append(f"  doc_id={r.get('doc_id')}: {r.get('text', '')}")
    return "\n".join(parts)


class ReasoningReviewer(ABC):
    @abstractmethod
    def review(self, case: dict) -> dict:
        """回傳 {'applicable', 'reviewed', 'flagged', 'issues', 'raw': {...}}"""
        ...


class NoOpReasoningReviewer(ReasoningReviewer):
    def review(self, case: dict) -> dict:
        return {"applicable": False, "reviewed": False, "flagged": False, "issues": [],
                "reason": "LLM backend 未設定，reasoning_review 未啟用"}


def _not_applicable_no_reasoning() -> dict:
    return {"applicable": False, "reviewed": False, "flagged": False, "issues": [],
            "reason": "案件無 reasoning 欄位可審(理賠代理人尚未提供，非審查失敗)"}


def _review_failed(reason: str) -> dict:
    """審查失敗一律 flagged=True，保守轉人工——不能因為審查失敗就當作沒問題。"""
    return {"applicable": True, "reviewed": False, "flagged": True,
            "issues": [f"reasoning_review 呼叫失敗，保守起見轉人工複核: {reason}"],
            "reason": reason}


class GeminiReasoningReviewer(ReasoningReviewer):
    def __init__(self, api_key: str, model: str = "gemini-flash-latest", timeout: float = 20.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={api_key}")

    def review(self, case: dict) -> dict:
        if not case.get("reasoning"):
            return _not_applicable_no_reasoning()

        user_content = _build_user_content(case)
        payload = {
            "system_instruction": {"parts": [{"text": REVIEW_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0},
        }
        try:
            resp = httpx.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            verdict = json.loads(text)
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            return _review_failed(f"{type(e).__name__}: {e}")

        flagged = (
            verdict.get("citation_fidelity") == "inconsistent"
            or bool(verdict.get("unaddressed_policy_flags"))
            or verdict.get("amount_reasoning_consistent") is False
        )
        return {
            "applicable": True, "reviewed": True, "flagged": flagged,
            "issues": verdict.get("issues", []),
            "citation_fidelity": verdict.get("citation_fidelity"),
            "unaddressed_policy_flags": verdict.get("unaddressed_policy_flags", []),
            "amount_reasoning_consistent": verdict.get("amount_reasoning_consistent"),
            "claim_agent_confidence_assessment": verdict.get("claim_agent_confidence_assessment"),
            "backend": "gemini", "model": self.model,
        }


def get_reasoning_reviewer() -> ReasoningReviewer:
    backend = os.environ.get("REASONING_REVIEW_BACKEND", "gemini").lower()
    if backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            return GeminiReasoningReviewer(api_key)
        print("[警告] REASONING_REVIEW_BACKEND=gemini 但未設定 GEMINI_API_KEY，reasoning_review 回退為 NoOp")
        return NoOpReasoningReviewer()
    # SEAM：claude / ollama 之後要接的話在這裡加對應的Reviewer實作
    print(f"[警告] REASONING_REVIEW_BACKEND='{backend}' 尚未實作，reasoning_review 回退為 NoOp")
    return NoOpReasoningReviewer()
