# -*- coding: utf-8 -*-
"""
tpl_similar.py — TPL 相似案例檢索(B部分的姊妹模組，方向不同)

GraphRAG API 新增的 POST /v1/answer/tpl_similar 是獨立上線的端點，跟
fault_consistency.py 打的 /v1/retrieve/features 是不同方向：那支是「TPL案件
vs 法院判決」找肇責/傾向一致性；這支是「TPL案件 vs TPL案件」找相似的理賠
案例本身(傷勢/部位/肇責分桶/事故原因)，純粹補充參考資訊，不影響金額判斷——
跟 fault_consistency 同一個架構原則：這條線只寫進 audit_trail，不參與
escalation_triggers、不覆蓋 fairness_check 的結論。

⚠️ 跟 fault_consistency.py 最大的差異：/v1/retrieve/features 是純結構化查詢，
不呼叫LLM；/v1/answer/tpl_similar 內部一定會呼叫LLM生成answer(見GraphRAG API
的 answer_tpl_similar())，代表**每叫一次這支client，就多一次LLM延遲+費用**。
目前 decision.py 已經有 reasoning_review 每案必呼叫一次LLM，如果把這支也
無條件接進 judge_decide()，等於每個TPL案件的判斷會固定觸發兩次LLM呼叫。

這裡先只提供client本身(SEAM: interface + stub + real)，**是否要在
judge_decide()裡無條件呼叫、還是只在特定情況(例如precedent_insufficient/
escalate_human時)才呼叫**，留給你決定後再接進 decision.py——這是成本/延遲
的取捨，不是技術限制，不應該由這支client自己隱性決定。

question 目前用固定模板(不是使用者自由輸入)，因為這裡是自動化判斷流程，
不是人工查詢介面；若之後要讓理賠專員在human_review階段自己問問題，可以另外
開一個交談式的呼叫路徑，不用共用這支自動化用的固定模板。
"""
import os
from abc import ABC, abstractmethod

import httpx

DEFAULT_QUESTION = "這件案件有哪些相似的TPL案例？請摘要這些案例的肇責比例分佈與傷勢特徵。"


class TplSimilarRetriever(ABC):
    @abstractmethod
    def retrieve(self, case: dict) -> dict:
        """回傳 {'applicable', 'similar_cases', 'answer', 'citations', ...}"""
        ...


class NoOpTplSimilarRetriever(TplSimilarRetriever):
    def retrieve(self, case: dict) -> dict:
        return {"applicable": False, "reason": "GraphRAG API 未設定(缺 GRAPHRAG_API_URL)，SEAM 未啟用",
                "similar_cases": [], "answer": None, "citations": []}


class HTTPTplSimilarRetriever(TplSimilarRetriever):
    def __init__(self, base_url: str, api_key: str = None, timeout: float = 20.0,
                 top_k: int = 8, min_jaccard: float = 0.15):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout  # 比fault_consistency的10秒長，因為這支端點內部會呼叫LLM生成answer
        self.top_k = top_k
        self.min_jaccard = min_jaccard

    def retrieve(self, case: dict) -> dict:
        if case.get("line") != "TPL":
            return {"applicable": False, "reason": "RSA無TPL案件比對意義，不適用此端點",
                     "similar_cases": [], "answer": None, "citations": []}

        case_no = case.get("case_id")
        if not case_no:
            return {"applicable": False, "reason": "案件缺少 case_id，無法查詢",
                     "similar_cases": [], "answer": None, "citations": []}

        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/answer/tpl_similar",
                json={"case_no": case_no, "question": DEFAULT_QUESTION,
                      "top_k": self.top_k, "min_jaccard": self.min_jaccard},
                headers=headers, timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # 案件在圖資料庫裡查無(可能尚未匯入/case_no格式不符)，
                # 誠實回報，不當成「applicable=False代表沒有相似案件」處理
                return {"applicable": False, "reason": f"GraphRAG API 查無此案件: {case_no}",
                         "similar_cases": [], "answer": None, "citations": []}
            return {"applicable": False, "reason": f"GraphRAG API 呼叫失敗: {e}",
                     "similar_cases": [], "answer": None, "citations": []}
        except httpx.HTTPError as e:
            # 呼叫失敗(逾時/連線錯誤等)要讓人看得到，不能靜默吞掉退回 applicable=False 卻不留原因
            return {"applicable": False, "reason": f"GraphRAG API 呼叫失敗: {e}",
                     "similar_cases": [], "answer": None, "citations": []}

        similar_cases = data.get("similar_cases", [])
        return {
            "applicable": bool(similar_cases),
            "reason": "" if similar_cases else "結構化篩選後無符合相似案件(min_jaccard門檻內)",
            "similar_case_count": len(similar_cases),
            "similar_cases": similar_cases,
            "similarity_caveat": "相似度為傷勢/部位/肇責分桶/事故原因等結構化特徵比對結果，"
                                  "不代表金額或肇責比例可直接類比套用到本案",
            "answer": data.get("answer"),
            "citations": data.get("citations", []),
        }


def get_tpl_similar_retriever() -> TplSimilarRetriever:
    base_url = os.environ.get("GRAPHRAG_API_URL")
    api_key = os.environ.get("GRAPHRAG_API_KEY")
    if base_url:
        return HTTPTplSimilarRetriever(base_url, api_key)
    print("[警告] 未設定 GRAPHRAG_API_URL，tpl_similar 將回退為 NoOp(不影響決策，但不提供相似案例參考)")
    return NoOpTplSimilarRetriever()
