# -*- coding: utf-8 -*-
"""
seams.py — 三個對外整合點：OCR / 協調層 / 通知

每個都是「介面 + 誠實的樁(stub)」。樁不假裝有真結果，缺什麼就明確回報缺什麼，
不虛構金額、不虛構判決——這樣測試時看到的任何「假資料」都清楚標示 simulated=True，
不會被誤認成真實案件結果。

# TODO SEAM 標記的地方，是接上真實系統時要換的程式碼。
"""
import os
import json as _json
import httpx
from abc import ABC, abstractmethod


# ============================================================
# 1. OCR 擷取 —— 保單文件（保單號/被保險人/保險期間）跟理賠佐證文件
#    （收據/估價單/診斷金額日期）是兩種不同性質的文件，分開擷取、
#    分開回傳，不要混在同一組欄位裡靠猜的方式合併。
# ============================================================
class OCRExtractor(ABC):
    @abstractmethod
    async def extract(self, policy_files: list, evidence_files: list) -> dict:
        """回傳 {'policy': {...}, 'evidence': {...}}，兩者結構皆為
        {'simulated': bool, 'fields': {...}, 'per_file': [...], 'errors': [...]}"""
        ...


def _empty_ocr_block(file_paths: list, note: str) -> dict:
    return {
        "simulated": True,
        "fields": {},  # 刻意留空，不虛構任何擷取欄位
        "per_file": [{"file": f, "status": "not_processed"} for f in file_paths],
        "errors": [],
        "note": note,
    }


class StubOCRExtractor(OCRExtractor):
    """尚未設定 GEMINI_API_KEY 前的樁。誠實回報「未執行」，
    不虛構任何擷取欄位，避免測試者誤以為 OCR 已經真的在跑。"""

    async def extract(self, policy_files: list, evidence_files: list) -> dict:
        note = "OCR 尚未設定 GEMINI_API_KEY，此為樁。設定後會自動改用 RealOCRExtractor。"
        return {
            "policy": _empty_ocr_block(policy_files, note),
            "evidence": _empty_ocr_block(evidence_files, note),
        }


class RealOCRExtractor(OCRExtractor):
    """呼叫 Gemini API 做多模態文件擷取。金鑰一律從環境變數讀，不寫死在程式碼裡。

    多檔案合併規則：同一欄位在不同檔案讀到不同值時，不靜默覆蓋、不挑一個當正確
    答案，改標記進 fields['_conflicts']，交給後續流程（後台人員複核／法官代理人
    的詐欺旗標）判斷，不在這層就替使用者決定哪個值才是對的。
    """

    POLICY_PROMPT = (
        "你是保險保單文件擷取助手。請從這張保單影像中擷取以下欄位，"
        "只回傳 JSON，不要有其他文字、不要用 markdown code block：\n"
        '{"policy_no": "文字或null", "insured_name": "文字或null", '
        '"policy_period_start": "YYYY-MM-DD或null", "policy_period_end": "YYYY-MM-DD或null", '
        '"vehicle_plate_no": "文字或null（車牌號碼）", '
        '"vehicle_brand_model": "文字或null（車輛廠牌型號）", '
        '"vehicle_use": "自用或營業用或null", '
        '"rsa_addon_purchased": "true或false或null（保單上是否明確載明投保道路救援／道路救援服務附加條款，'
        '文件沒提到就填null，不要用常識或險種名稱去猜測）"}\n'
        "看不清楚或文件中沒有某欄位，該欄位填 null，不要用猜測值填充。"
    )
    EVIDENCE_PROMPT = (
        "你是保險理賠文件擷取助手。請從這份文件中擷取以下欄位，"
        "只回傳 JSON，不要有其他文字、不要用 markdown code block：\n"
        '{"amount": 數字或null, "date": "YYYY-MM-DD或null", '
        '"diagnosis": "文字或null（僅醫療相關文件適用，非醫療文件此欄位填null）", '
        '"service_type": "TOWING或BATTERY_JUMP或FUEL_DELIVERY或LOCKOUT或TIRE_CHANGE或OTHER或null'
        '（僅道路救援服務單據適用，文件沒提到就填null）", '
        '"service_location": "文字或null（服務地點，僅道路救援服務單據適用）", '
        '"towing_distance_km": "數字或null（拖吊公里數，單據沒有明確載明就填null，不要用猜測值填充）"}\n'
        "看不清楚或文件中沒有某欄位，該欄位填 null，不要用猜測值填充。"
    )

    # 2026-08 新增：Gemini 在沒有強制結構化輸出（response_schema）時，即使
    # prompt 裡明確要求了完整欄位清單，實測發現它仍可能整個省略某些欄位
    # （不是回傳 null，是連 key 都不見），導致 vehicle_use／rsa_addon_purchased／
    # service_type／service_location／towing_distance_km 這類欄位平白漏抓。
    # 加上 response_schema 強制規定「這八／六個欄位都必須存在」，讓 Gemini
    # 不能再省略——不確定的值它還是可以填 null，只是不能整個不回傳這個欄位。
    POLICY_RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "policy_no": {"type": "string", "nullable": True},
            "insured_name": {"type": "string", "nullable": True},
            "policy_period_start": {"type": "string", "nullable": True},
            "policy_period_end": {"type": "string", "nullable": True},
            "vehicle_plate_no": {"type": "string", "nullable": True},
            "vehicle_brand_model": {"type": "string", "nullable": True},
            "vehicle_use": {"type": "string", "nullable": True},
            "rsa_addon_purchased": {"type": "boolean", "nullable": True},
        },
        "required": [
            "policy_no", "insured_name", "policy_period_start", "policy_period_end",
            "vehicle_plate_no", "vehicle_brand_model", "vehicle_use", "rsa_addon_purchased",
        ],
    }
    EVIDENCE_RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "nullable": True},
            "date": {"type": "string", "nullable": True},
            "diagnosis": {"type": "string", "nullable": True},
            "service_type": {"type": "string", "nullable": True},
            "service_location": {"type": "string", "nullable": True},
            "towing_distance_km": {"type": "number", "nullable": True},
        },
        "required": [
            "amount", "date", "diagnosis",
            "service_type", "service_location", "towing_distance_km",
        ],
    }

    def __init__(self, api_key: str, model: str = "gemini-flash-latest"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY 未設定")
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def extract(self, policy_files: list, evidence_files: list) -> dict:
        return {
            "policy": await self._extract_block(
                policy_files, self.POLICY_PROMPT, self.POLICY_RESPONSE_SCHEMA
            ),
            "evidence": await self._extract_block(
                evidence_files, self.EVIDENCE_PROMPT, self.EVIDENCE_RESPONSE_SCHEMA
            ),
        }

    async def _extract_block(self, file_paths: list, prompt: str, response_schema: dict) -> dict:
        fields: dict = {}
        conflicts: dict = {}
        per_file = []
        errors = []

        for path in file_paths:
            try:
                extracted = await self._extract_one(path, prompt, response_schema)
                per_file.append({"file": path, "status": "ok", "raw": extracted})
                for key, value in extracted.items():
                    if value is None:
                        continue
                    if key in fields and fields[key] != value:
                        conflicts.setdefault(key, [fields[key]]).append(value)
                    else:
                        fields[key] = value
            except Exception as e:
                per_file.append({"file": path, "status": "error", "error": str(e)})
                errors.append(f"{path}: {type(e).__name__}: {e}")

        if conflicts:
            fields["_conflicts"] = conflicts

        return {
            "simulated": False,
            "fields": fields,
            "per_file": per_file,
            "errors": errors,
        }

    async def _extract_one(self, file_path: str, prompt: str, response_schema: dict) -> dict:
        from google.genai import types

        uploaded = await self._client.aio.files.upload(file=file_path)
        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        return _json.loads(text)


def get_ocr_extractor() -> OCRExtractor:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return RealOCRExtractor(api_key)
    return StubOCRExtractor()


# ============================================================
# 1.5 描述文字解析 —— 把保戶自由填寫的 description 解析成 Rule Agent
#     需要的結構化事故欄位（車輛用途/服務類型/地點/道路類型/是否已聯絡
#     指定救援中心等）。這批欄位是「事故情境」，跟 OCR 抓的「文件metadata」
#     (保單號/金額/日期)本質不同、欄位名也不重疊，理論上不會互相衝突。
#
#     只解析明確寫在文字裡的資訊，沒提到的欄位一律填 null，不用常識或
#     刻板印象猜測——這是延續整個專案「不確定資料寧可標記不要靜默修正」
#     的原則，猜錯比留白危險，因為猜錯的欄位會被Rule Agent當成保戶親口
#     確認的事實去用。
#
#     刻意不解析 risk_review.*（是否涉嫌詐欺/資料是否有矛盾）：這些是
#     下游風控該做的判斷，不該讓「解析保戶自己講的話」這個步驟越權
#     替保戶的誠信背書或定罪。也不解析 documents.*／policy.*（需要
#     實際文件或保單資料庫查詢，不是文字裡找得到的東西）。
# ============================================================
class DescriptionParser(ABC):
    @abstractmethod
    async def parse(self, description: str) -> dict:
        """回傳 {'simulated': bool, 'fields': {...}, 'evidence_quotes': {...},
        'errors': [...]}。fields 裡每個值若為 null，代表文字中沒有明確提及，
        不是解析失敗；evidence_quotes 是每個非null欄位對應到原文的哪一段話，
        方便人工複核時快速核對，不用整段description重新讀一次。"""
        ...


# Rule Agent 目前判定缺少的欄位裡，真正能從自由文字合理推斷的子集
# （對照案例 CLM-3C23880E1F 的 rsa_missing_fields 清單篩出來的）
_DESCRIPTION_PARSE_FIELDS = (
    "vehicle_use", "requested_service", "location", "road_type",
    "contacted_designated_center", "special_operation_required",
    "claims_bridge_or_toll_fees", "vehicle_loaded_and_unwilling_to_unload",
    "claims_passenger_or_cargo_transport_cost",
)


def _empty_description_parse(note: str) -> dict:
    return {
        "simulated": True,
        "fields": {k: None for k in _DESCRIPTION_PARSE_FIELDS},
        "evidence_quotes": {},
        "errors": [],
        "note": note,
    }


class StubDescriptionParser(DescriptionParser):
    """尚未設定 GEMINI_API_KEY 前的樁。誠實回報「未執行」，全部欄位填 null，
    不假裝已經解析過。"""

    async def parse(self, description: str) -> dict:
        return _empty_description_parse(
            "描述解析尚未設定 GEMINI_API_KEY，此為樁。設定後會自動改用 RealDescriptionParser。"
        )


class RealDescriptionParser(DescriptionParser):
    """呼叫 Gemini 從保戶填寫的事故描述文字裡解析結構化欄位。"""

    PROMPT_TEMPLATE = (
        "你是道路救援理賠案件的事故描述解析助手。以下是保戶填寫的事故描述文字，"
        "請只根據這段文字裡「明確寫到」的內容擷取欄位，只回傳 JSON，"
        "不要有其他文字、不要用 markdown code block：\n\n"
        '{{"vehicle_use": "自用或營業用或null", '
        '"requested_service": "TOWING或BATTERY_JUMP或FUEL_DELIVERY或LOCKOUT或TIRE_CHANGE或OTHER或null", '
        '"location": "文字或null", '
        '"road_type": "國道或快速道路或一般道路或其他或null", '
        '"contacted_designated_center": "true或false或null", '
        '"special_operation_required": "true或false或null", '
        '"claims_bridge_or_toll_fees": "true或false或null", '
        '"vehicle_loaded_and_unwilling_to_unload": "true或false或null", '
        '"claims_passenger_or_cargo_transport_cost": "true或false或null", '
        '"evidence_quotes": {{"欄位名": "原文中支持這個判斷的那一段話"}}}}\n\n'
        "規則：\n"
        "1. 文字沒有明確提到的欄位，一律填 null，不要用常識、刻板印象、或「通常都是這樣」去猜測。\n"
        "2. evidence_quotes 只需要放非null欄位對應的原文片段，null欄位不用放進去。\n"
        "3. 不要推斷是否涉及詐欺或資料是否矛盾，這不是這個任務的範圍。\n\n"
        "事故描述文字：\n{description}"
    )

    def __init__(self, api_key: str, model: str = "gemini-flash-latest"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY 未設定")
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def parse(self, description: str) -> dict:
        if not description or not description.strip():
            return _empty_description_parse("description 為空，無內容可解析。")

        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=self.PROMPT_TEMPLATE.format(description=description),
            )
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            parsed = _json.loads(text)
        except Exception as e:
            result = _empty_description_parse(f"解析失敗，全部欄位維持null: {type(e).__name__}: {e}")
            result["simulated"] = False
            result["errors"] = [f"{type(e).__name__}: {e}"]
            return result

        fields = {}
        for key in _DESCRIPTION_PARSE_FIELDS:
            value = parsed.get(key)
            if value in (None, "null", ""):
                fields[key] = None
            elif key in ("contacted_designated_center", "special_operation_required",
                         "claims_bridge_or_toll_fees", "vehicle_loaded_and_unwilling_to_unload",
                         "claims_passenger_or_cargo_transport_cost"):
                # LLM有時會把布林值回傳成字串"true"/"false"，統一轉成真布林值，
                # 而不是留著字串讓下游Rule Agent比對時因型別不符而誤判
                fields[key] = str(value).strip().lower() == "true" if value is not None else None
            else:
                fields[key] = value

        return {
            "simulated": False,
            "fields": fields,
            "evidence_quotes": parsed.get("evidence_quotes", {}),
            "errors": [],
        }


def get_description_parser() -> DescriptionParser:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return RealDescriptionParser(api_key)
    return StubDescriptionParser()


# ============================================================
# 2. 協調層（規則→理賠→法官代理人）
# ============================================================
class OrchestratorClient(ABC):
    @abstractmethod
    async def submit(self, case_id: str, claim_data: dict) -> dict:
        """回傳協調層/法官代理人的最終結果 dict，至少含 decision 欄位。"""
        ...


class HTTPOrchestratorClient(OrchestratorClient):
    """真的去打協調層 API。若打不通(帳單暫停/尚未部署)，明確回報連線失敗，
    不偽裝成功——呼叫端會依此決定要不要落到模擬結果。"""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def _get_id_token(self) -> str | None:
        """Cloud Run 服務間呼叫需要用目標服務網址當 audience 換一張 Google 簽發的
        ID token，否則會被 Cloud Run IAM 層直接擋在門口(連程式碼都進不去)。
        本機或非 GCP 環境跑不出 token 是正常的，回傳 None、讓呼叫端照舊只帶 API Key。"""
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token
            request = google.auth.transport.requests.Request()
            return google.oauth2.id_token.fetch_id_token(request, self.base_url)
        except Exception:
            return None

    async def submit(self, case_id: str, claim_data: dict) -> dict:
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        token = await self._get_id_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/submit-claim",
                json={"case_id": case_id, **claim_data},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()


class SimulatedOrchestratorClient(OrchestratorClient):
    """協調層還沒部署/打不通時的替代品。回傳的每一筆結果都標記 simulated=True，
    決策邏輯只是粗略示範(金額>50萬轉人工，其餘放行)，不是真的規則/理賠/法官代理人在跑。
    測試介面時可以看到完整流程走完，但絕不能把這個結果當成真實理賠判斷。"""

    async def submit(self, case_id: str, claim_data: dict) -> dict:
        amount = claim_data.get("claim_amount") or 0
        decision = "escalate_human" if amount > 500_000 else "execute"
        return {
            "simulated": True,
            "case_id": case_id,
            "decision": decision,
            "next_agent": "human_review" if decision == "escalate_human" else "frontend",
            "reasons": ["模擬結果：協調層尚未部署或無法連線，此為佔位判斷，非真實審核"],
            "note": "這不是真實的規則/理賠/法官代理人輸出，僅供介面流程測試用。",
        }


def get_orchestrator_client() -> OrchestratorClient:
    base_url = os.environ.get("ORCHESTRATOR_URL")
    api_key = os.environ.get("ORCHESTRATOR_API_KEY")
    if base_url:
        return HTTPOrchestratorClient(base_url, api_key)
    return SimulatedOrchestratorClient()


async def submit_to_orchestrator_with_fallback(case_id: str, claim_data: dict) -> dict:
    """先試真實協調層；打不通(帳單暫停/尚未部署常見情況)就明確標記後退回模擬結果，
    不讓整條背景任務因為協調層還沒上線就整個失敗掛掉。"""
    client = get_orchestrator_client()
    if isinstance(client, SimulatedOrchestratorClient):
        return await client.submit(case_id, claim_data)
    try:
        return await client.submit(case_id, claim_data)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        fallback = await SimulatedOrchestratorClient().submit(case_id, claim_data)
        fallback["orchestrator_error"] = f"{type(e).__name__}: {e}"
        return fallback


# ============================================================
# 2.5 TPL 理賠代理人（Lin423423/tpl-claim-agent-api）—— 只服務「第三人責任險」
#     案件，回傳的是金額建議 + confidence，不是 execute/escalate_human 這種
#     決策，所以不能直接套進上面 OrchestratorClient 的合約，另外開一個 SEAM。
#     這支服務部署時沒開 --allow-unauthenticated，Cloud Run IAM 層跟服務內部
#     的 X-API-Key 驗證是兩道分開的關卡，呼叫時兩個都要帶，缺一個都會被擋。
# ============================================================
class TplClaimAgentClient(ABC):
    @abstractmethod
    async def get_suggestion(self, accident_area: str, own_fault_pct: float, injury_desc: str) -> dict:
        """回傳 TPL Claim Agent 的原始結果 dict（suggested_items/total_suggested_amount/
        confidence），或樁版本標記 simulated=True 的佔位結果。這一層只負責「問到了什麼」，
        不負責把結果轉成 execute/escalate_human/return_for_recalc——那是 judge-agent
        的工作（見 JudgeAgentClient），職責分開避免這裡混雜業務判斷。"""
        ...


class SimulatedTplClaimAgentClient(TplClaimAgentClient):
    """TPL_CLAIM_AGENT_URL 還沒設定/服務還沒部署時的樁。誠實回報「尚未執行」，
    不虛構任何金額建議或信心度，避免測試者誤以為已經有真實的 RAG 檢索結果。"""

    async def get_suggestion(self, accident_area: str, own_fault_pct: float, injury_desc: str) -> dict:
        return {
            "simulated": True,
            "suggested_items": [],
            "total_suggested_amount": None,
            "confidence": None,
            "note": "TPL_CLAIM_AGENT_URL 尚未設定，此為樁，非真實金額建議。",
        }


class HTTPTplClaimAgentClient(TplClaimAgentClient):
    """真的去打 tpl-claim-agent-api 的 /v1/tpl/claim。timeout 刻意拉長到 8 分鐘：
    該服務冷啟動實測約 7 分鐘（LoRA 4-bit 量化模型載入），demo 前建議另外對該服務
    設定 --min-instances=1 降低冷啟動機率，但 client 端仍保留寬鬆 timeout 當保險。
    打不通時明確拋出例外，不偽裝成功，由呼叫端（main.py）決定要不要轉人工。"""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 480.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def _get_id_token(self) -> str | None:
        """跟 HTTPOrchestratorClient 一樣：Cloud Run 服務間呼叫需要用目標服務網址
        當 audience 換一張 Google 簽發的 ID token，否則會被 Cloud Run IAM 層擋在
        門口（連 X-API-Key 驗證都進不去）。本機或非 GCP 環境跑不出 token 是正常的，
        回傳 None、讓呼叫端照舊只帶 X-API-Key（適用本機測試 tpl-claim-agent-api
        時該服務若暫時開放未驗證存取的情境）。"""
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token
            request = google.auth.transport.requests.Request()
            return google.oauth2.id_token.fetch_id_token(request, self.base_url)
        except Exception:
            return None

    async def get_suggestion(self, accident_area: str, own_fault_pct: float, injury_desc: str) -> dict:
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        token = await self._get_id_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/tpl/claim",
                json={
                    "accident_area": accident_area,
                    "own_fault_pct": own_fault_pct,
                    "injury_desc": injury_desc,
                },
                headers=headers,
            )
            resp.raise_for_status()
            result = resp.json()
            result["simulated"] = False
            return result


def get_tpl_claim_agent_client() -> TplClaimAgentClient:
    base_url = os.environ.get("TPL_CLAIM_AGENT_URL")
    api_key = os.environ.get("TPL_CLAIM_AGENT_API_KEY")
    if base_url:
        return HTTPTplClaimAgentClient(base_url, api_key)
    return SimulatedTplClaimAgentClient()


# ============================================================
# 2.6 法官代理人（HsiaoHsiao0627/judge-agent）—— 內部服務，只做金額公平性
#     (Tukey IQR)、詐欺旗標(A/B/C/D)、審推理，回傳 execute/return_for_recalc/
#     escalate_human。這支服務部署時沒開 --allow-unauthenticated、也沒有
#     X-API-Key 這層驗證，只靠 Cloud Run IAM Invoker 擋，所以呼叫時只需要
#     帶 ID token，不用像 TPL Claim Agent 那樣額外帶 API Key。
# ============================================================
class JudgeAgentClient(ABC):
    @abstractmethod
    async def judge(self, case: dict) -> dict:
        """case 必須是跟 judge-agent 的 TPLValCaseLoader.load()/RSAValCaseLoader.load()
        相同形狀的 dict（見 main.py 的 _build_judge_case_from_tpl()）。回傳 judge-agent
        的完整輸出（decision/next_agent/reasons/fairness_check/fraud_flags/...），
        或樁版本標記 simulated=True 的保守佔位結果（一律 escalate_human，不猜放行）。"""
        ...


class SimulatedJudgeAgentClient(JudgeAgentClient):
    """JUDGE_AGENT_URL 還沒設定/服務還沒部署時的樁。金額判斷這種事沒有真實
    法官代理人把關時，保守起見一律轉人工，不能因為服務沒接上就悄悄放行。"""

    async def judge(self, case: dict) -> dict:
        return {
            "simulated": True,
            "case_id": case.get("case_id"), "line": case.get("line"),
            "decision": "escalate_human", "next_agent": "human_review",
            "reasons": ["JUDGE_AGENT_URL 尚未設定，此為樁，非真實法官代理人判斷，保守轉人工"],
            "confidence": 0.0,
        }


class HTTPJudgeAgentClient(JudgeAgentClient):
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _get_id_token(self) -> str | None:
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token
            request = google.auth.transport.requests.Request()
            return google.oauth2.id_token.fetch_id_token(request, self.base_url)
        except Exception:
            return None

    async def judge(self, case: dict) -> dict:
        headers = {}
        token = await self._get_id_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/v1/judge", json=case, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            result["simulated"] = False
            return result


def get_judge_agent_client() -> JudgeAgentClient:
    base_url = os.environ.get("JUDGE_AGENT_URL")
    if base_url:
        return HTTPJudgeAgentClient(base_url)
    return SimulatedJudgeAgentClient()


# ============================================================
# 3. 通知（email/簡訊現在；LINE 推播之後）
# ============================================================
class Notifier(ABC):
    @abstractmethod
    async def notify(self, case_id: str, contact_email: str | None,
                      contact_phone: str | None, result: dict) -> None:
        ...


class StubNotifier(Notifier):
    """尚未接真實 SMTP/簡訊供應商前的樁：只記錄「本來會送出什麼」，不真的寄送。
    介面在同一個 case_id 查詢頁一樣看得到最終結果，方便沒有信箱/簡訊帳號時測試。"""

    async def notify(self, case_id: str, contact_email, contact_phone, result: dict) -> None:
        print(f"[通知樁] 案件 {case_id} 完成，決定={result.get('decision')}。"
              f"本來會寄送到 email={contact_email} phone={contact_phone}（尚未接真實服務）。")


# TODO SEAM ── 之後依管道擴充，例如：
# class LineNotifier(Notifier):
#     async def notify(self, case_id, contact_email, contact_phone, result):
#         # 呼叫 LINE Messaging API push message，用 LINE user_id 取代 email/phone
#         ...


def get_notifier() -> Notifier:
    return StubNotifier()
