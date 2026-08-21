# -*- coding: utf-8 -*-
"""
fault_consistency.py — 判決先例的肇責/傾向一致性檢索（A部分）

GraphRAG API 是獨立上線的 Cloud Run 服務，這裡打 POST /v1/retrieve/features（真實
schema 見 GraphRAG API 的 api.py：RetrieveByFeaturesRequest / RetrieveResponse /
Precedent）。不直接連 Neo4j。

這支端點是純結構化 Jaccard 比對(injuries/bodyparts 交集)，不呼叫LLM、不含向量
rerank——這對「同傷勢部位/同案件類型」這種離散屬性本來就是合適的排序方式，
不需要另外加語意層。只提供法律佐證(肇責比例分佈、判決傾向、裁判字號引用)，
不影響金額判斷——Precedent schema 本身就沒有金額欄位，這條線不會、也不該碰
fairness_check，不需要額外的『金額不可信』警語欄位，架構上金額風險已經隔離。

judgment_fault_pct 的警語必須原樣往下游帶：GraphRAG API 的 Precedent schema
明確標註這個數字「歸屬對象未確認，不可直接類比查詢案件」，法官代理人不能把它當
定論使用，只能當背景參考。

bodyparts 資料缺口(誠實已知、非疏漏)：TPL 案件 dict 目前有 injury_types
(來自 TPL_train.db 的 claim_episodes 表)，但沒有 body_parts 欄位——這是理賠代理人
介面尚未對齊的部分。目前 body_parts 一律傳空陣列，並在回傳結果裡顯式標記
bodyparts_available=False，不要假裝有資料，等理賠代理人補上這個欄位後再拿掉這個標記。

RSA 案件沒有傷勢/肇責特徵，此端點不適用，直接回 applicable=False。
"""
import os
from abc import ABC, abstractmethod

import httpx


class FaultConsistencyRetriever(ABC):
    @abstractmethod
    def retrieve(self, case: dict) -> dict:
        """回傳 {'applicable', 'fault_pcts', 'dispositions', 'citations', ...}"""
        ...


class NoOpFaultConsistencyRetriever(FaultConsistencyRetriever):
    def retrieve(self, case: dict) -> dict:
        return {"applicable": False, "reason": "GraphRAG API 未設定(缺 GRAPHRAG_API_URL)，SEAM 未啟用",
                "fault_pcts": [], "dispositions": [], "citations": []}


class HTTPFaultConsistencyRetriever(FaultConsistencyRetriever):
    def __init__(self, base_url: str, api_key: str = None, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def retrieve(self, case: dict) -> dict:
        if case.get("line") != "TPL":
            return {"applicable": False, "reason": "RSA 無傷勢/肇責特徵，不適用此端點",
                     "fault_pcts": [], "dispositions": [], "citations": []}

        injuries = list(case.get("injury_types") or [])
        bodyparts = list(case.get("body_parts") or [])
        bodyparts_available = bool(case.get("body_parts"))

        if not injuries and not bodyparts:
            return {"applicable": False, "reason": "案件無 injury_types/body_parts 可查，無法檢索先例",
                     "fault_pcts": [], "dispositions": [], "citations": [],
                     "bodyparts_available": bodyparts_available}

        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            resp = httpx.post(
                f"{self.base_url}/v1/retrieve/features",
                json={"injuries": injuries, "bodyparts": bodyparts, "top_k": 8,
                      "min_strength": 2, "min_jaccard": 0.15},
                headers=headers, timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            # 呼叫失敗要讓人看得到，不能靜默吞掉退回 applicable=False 卻不留原因
            return {"applicable": False, "reason": f"GraphRAG API 呼叫失敗: {e}",
                     "fault_pcts": [], "dispositions": [], "citations": [],
                     "bodyparts_available": bodyparts_available}

        precedents = data.get("precedents", [])
        return {
            "applicable": bool(precedents),
            "reason": "" if precedents else "結構化篩選後無符合先例(min_jaccard門檻內)",
            "precedent_count": data.get("count", len(precedents)),
            "fault_pcts": [p.get("judgment_fault_pct") for p in precedents
                            if p.get("judgment_fault_pct") is not None],
            "fault_pcts_caveat": "歸屬對象未確認，不可直接類比本案，僅供背景參考",
            "dispositions": [p.get("disposition_type") for p in precedents if p.get("disposition_type")],
            "citations": [p.get("statute_no") for p in precedents if p.get("statute_no")],
            "bodyparts_available": bodyparts_available,
        }


def get_fault_retriever() -> FaultConsistencyRetriever:
    base_url = os.environ.get("GRAPHRAG_API_URL")
    api_key = os.environ.get("GRAPHRAG_API_KEY")
    if base_url:
        return HTTPFaultConsistencyRetriever(base_url, api_key)
    print("[警告] 未設定 GRAPHRAG_API_URL，fault_consistency 將回退為 NoOp(不影響決策，但不提供法律佐證)")
    return NoOpFaultConsistencyRetriever()
