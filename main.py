# -*- coding: utf-8 -*-
"""
main.py — 理賠案件受理 API（測試系統，無身份驗證）

設計原則（呼應這次的五個決定）：
  1. 送出後立刻回「已受理」，實際處理在背景執行，最終用通知管道告知結果。
  2. 完全無登入/驗證，任何人皆可呼叫（測試階段限定，正式上線前必須補上，
     包含下面新增的 /v1/admin/* 後台端點——這批端點同樣刻意先不加驗證，
     正式上線前要補 Admin API Key 或帳密登入）。
  3. 上傳文件會嘗試 OCR 擷取欄位（見 seams.py，GEMINI_API_KEY 設定後改用真實
     Gemini 擷取，否則為樁）。保單照片跟理賠佐證文件是兩種不同性質的上傳，
     分開處理：保單照片擷取 policy_no/insured_name/保險期間，佐證文件擷取
     amount/date/diagnosis。保戶表單欄位留空時，OCR 讀到的值才會拿來自動
     帶入——這個「自動帶入」會誠實記在 ocr_filled_fields，後台看得出來
     哪些欄位是 OCR 補的、不是保戶手填的，不會混為一談。
  3.5 OCR 之後另外跑一次「描述解析」：把保戶自由填寫的 description 解析成
     Rule Agent 需要的事故欄位（車輛用途/服務類型/地點/道路類型/是否已
     聯絡指定救援中心等，同樣見 seams.py 的 DescriptionParser）。只解析
     文字裡明確寫到的內容，沒提到的一律留 null，不用常識猜測。這批欄位
     跟 OCR 抓的欄位定義不重疊，但仍防呆處理：萬一解析結果撞到已有值的
     欄位，一律以既有值（保戶手填或OCR）為準，解析結果捨棄並記錄，不覆蓋。
  4. 沒有「申請人工複核」端點（保戶端）。後台人員標記複核狀態的端點有。
  5. 沒有任何法遵揭露文字或個資同意書邏輯。

管道無關設計：這支 API 不知道、也不需要知道呼叫者是網頁表單還是未來的 LINE
webhook——兩者都只是呼叫同一個 create_claim() + 背景流程，channel 參數只是
記錄用，不影響任何業務邏輯。這是為了呼應「以後可能要延伸到 LINE」的需求，
現在就不要把管道邏輯埋進核心流程裡。
"""
import os
import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

import store
import seams

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="兆豐理賠案件受理 API（測試系統）")

# 測試階段對任何來源開放；上線前務必收窄成實際前端網域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    store.init_db()


# ============================================================
# 案件受理（管道無關：網頁前端與未來 LINE webhook 都呼叫這支）
# ============================================================
@app.post("/v1/claims")
async def submit_claim(
    background_tasks: BackgroundTasks,
    insurance_type: str = Form(...),
    policy_no: Optional[str] = Form(None),
    applicant_name: Optional[str] = Form(None),
    claim_amount: Optional[float] = Form(None),
    incident_date: Optional[str] = Form(None),
    description: str = Form(""),
    contact_email: Optional[str] = Form(None),
    contact_phone: Optional[str] = Form(None),
    channel: str = Form("web"),
    # 2026-08 新增：RSA Rule Agent 的 decide_rsa_v3() 需要知道保單事故當下
    # 是否有效、有沒有投保道路救援附加條款，這兩個欄位在此之前完全沒有
    # 管道收集，導致 RSA 案件不管多完整都一定卡在 NEED_MORE_INFORMATION。
    # 只在申請險種是車險／道路救援相關時前端才會顯示，其他險種留空即可，
    # 沒填就是 None，交給下游誠實判斷「不知道」，不強迫二選一。
    policy_active: Optional[str] = Form(None),
    rsa_addon_purchased: Optional[str] = Form(None),
    # 2026-08 新增：TPL Claim Agent（tpl-claim-agent-api）需要這三個欄位才能
    # 呼叫 /v1/tpl/claim，且 own_fault_pct 是理賠金額計算的關鍵數字，刻意不
    # 靠描述解析用 LLM 從自由文字猜，而是讓保戶/客服明確填寫。只在申請險種
    # 是第三人責任險時前端才會顯示，其他險種留空即可。
    accident_area: Optional[str] = Form(None),
    own_fault_pct: Optional[float] = Form(None),
    injury_desc: Optional[str] = Form(None),
    policy_documents: list[UploadFile] = File(default=[]),
    evidence_documents: list[UploadFile] = File(default=[]),
):
    if not contact_email and not contact_phone:
        raise HTTPException(400, "email 與電話至少要留一個，否則無法通知結果")

    # policy_no/申請人/金額/日期 可以用手打，也可以靠上傳的保單照片／佐證文件
    # 讓 OCR 補上——但兩邊都沒有的話，這筆案件註定資料不全，直接擋在受理前，
    # 不要讓一個完全沒有保單資訊、也沒有金額的案件進到後續流程。
    if not policy_no and not policy_documents:
        raise HTTPException(400, "請填寫保單號，或上傳保單照片供系統辨識")
    if applicant_name is None and not policy_documents:
        raise HTTPException(400, "請填寫申請人姓名，或上傳保單照片供系統辨識")
    if claim_amount is None and not evidence_documents:
        raise HTTPException(400, "請填寫申請理賠金額，或上傳收據／估價單等佐證文件供系統辨識")
    if not incident_date and not evidence_documents:
        raise HTTPException(400, "請填寫事故／就醫日期，或上傳佐證文件供系統辨識")

    submitted_fields = {
        "policy_no": policy_no, "applicant_name": applicant_name,
        "insurance_type": insurance_type, "claim_amount": claim_amount,
        "incident_date": incident_date, "description": description,
        "contact_email": contact_email, "contact_phone": contact_phone,
        "policy_active": policy_active, "rsa_addon_purchased": rsa_addon_purchased,
        "accident_area": accident_area, "own_fault_pct": own_fault_pct,
        "injury_desc": injury_desc,
    }

    case_id = store.create_claim(submitted_fields, file_paths={}, channel=channel)

    # 存檔（SEAM：測試階段存本地磁碟，上線後改存 Cloud Storage，
    # 用簽章網址讓前端直接上傳，避免大檔案吃掉 API 的請求大小限制）
    case_dir = UPLOAD_DIR / case_id
    saved = {"policy": [], "evidence": []}
    for doc_type, docs in (("policy", policy_documents), ("evidence", evidence_documents)):
        if not docs:
            continue
        sub_dir = case_dir / doc_type
        sub_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            dest = sub_dir / doc.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(doc.file, f)
            saved[doc_type].append(str(dest))
    store.update_status(case_id, "received", file_paths=json.dumps(saved))

    background_tasks.add_task(_process_claim, case_id, submitted_fields, saved)

    return {
        "case_id": case_id,
        "status": "received",
        "message": "您的理賠申請已受理，審核完成後將以您留下的聯絡方式通知結果。",
    }


# ============================================================
# 背景處理流程：OCR → 自動帶入空欄位 → 協調層 → 通知
# ============================================================
# OCR 擷取欄位名 -> 表單欄位名的對應。只有表單欄位「留空」時才會用這裡的值
# 自動帶入，且一律記錄進 ocr_filled_fields，不會悄悄覆蓋保戶自己填的資料。
_POLICY_FIELD_MAP = {"policy_no": "policy_no", "insured_name": "applicant_name"}
_EVIDENCE_FIELD_MAP = {"amount": "claim_amount", "date": "incident_date"}


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


# 2026-08 更新：法官代理人（judge-agent）已經接上，取代原本 claim-intake 自己
# 猜的「confidence==low 就轉人工」門檻。真正的金額公平性(Tukey IQR)、詐欺旗標、
# 審推理都在 judge-agent 那邊做，claim-intake 這裡只負責把 TPL Claim Agent 的
# 輸出轉成 judge-agent 看得懂的 case dict 形狀（對照 judge-agent 的
# TPLValCaseLoader.load()），不重新判斷任何金額合理性——這條界線很重要：
# claim-intake 不能把猜測的業務規則混進來冒充法官代理人的判斷。
def _build_judge_case_from_tpl(case_id: str, submitted_fields: dict, ocr_result: dict,
                                suggestion: dict, retry_count: int) -> dict:
    policy_fields = (ocr_result.get("policy") or {}).get("fields", {})
    context = suggestion.get("_retrieved_context") or {}
    items = suggestion.get("suggested_items", [])
    if not isinstance(items, list):
        items = []

    # reasoning_review 要看的是「理賠代理人有沒有忠實反映自己檢索到的證據」，
    # TPL Claim Agent 是逐項(item)給 reasoning，這裡合併成一段整案 reasoning
    # 文字，並把每項引用的案號/條款號去重彙總，不做任何金額或合理性的判斷。
    reasoning_parts, cited_case_ids, cited_policy_ids = [], set(), set()
    for item in items:
        if not isinstance(item, dict):
            continue
        category = item.get("item_category", "未分類")
        summary = item.get("reasoning_summary")
        if summary:
            reasoning_parts.append(f"【{category}】{summary}")
        for case_no in ((item.get("similar_case_reference") or {}).get("case_nos") or []):
            if case_no:
                cited_case_ids.add(case_no)
        clause_id = (item.get("amount_basis") or {}).get("policy_clause_id")
        if clause_id:
            cited_policy_ids.add(clause_id)

    own_fault_pct = submitted_fields.get("own_fault_pct")
    try:
        own_fault_pct = float(own_fault_pct) if own_fault_pct not in (None, "") else None
    except (TypeError, ValueError):
        own_fault_pct = None

    return {
        "case_id": case_id, "line": "TPL", "retry_count": retry_count,
        # ---- fairness_check（Tukey IQR）/ fraud_rules 用的結構化欄位 ----
        # claim_amount 用 TPL Claim Agent 套用肇責比例「之後」的建議總額，
        # 因為 judge-agent 是在核對「這個要核准的金額」是否公平合理，
        # 不是核對套用肇責前的基礎金額。
        "claim_amount": suggestion.get("total_suggested_amount"),
        "own_fault_pct": own_fault_pct,
        "other_fault_pct": (100 - own_fault_pct) if own_fault_pct is not None else None,
        # 以下欄位 claim-intake 目前沒有對應的結構化資料來源，誠實留 null／[]，
        # 讓 judge-agent 的 fraud_rules／fairness 顯式回報 na，不要用猜測值填充：
        "injury_types": [],
        "accident_cause": None,
        "has_disability": None,
        "disability_levels": [],
        "coverage_limit": None,
        "report_date": None,
        "accident_date": submitted_fields.get("incident_date"),
        "policy_effective_date": policy_fields.get("policy_period_start"),
        "policy_expiry_date": policy_fields.get("policy_period_end"),
        # ---- reasoning_review 用的欄位 ----
        "reasoning": "\n".join(reasoning_parts) if reasoning_parts else None,
        "cited_case_ids": sorted(cited_case_ids),
        "cited_policy_ids": sorted(cited_policy_ids),
        "claim_agent_confidence": suggestion.get("confidence"),
        "retrieved_case_results": [
            {"doc_id": c.get("case_no"), "text": c.get("summary")}
            for c in (context.get("similar_cases") or []) if isinstance(c, dict)
        ],
        "retrieved_policy_results": [
            {"doc_id": p.get("id"), "text": p.get("text")}
            for p in (context.get("policy_clauses") or []) if isinstance(p, dict)
        ],
    }


async def _process_claim(case_id: str, submitted_fields: dict, file_paths: dict):
    try:
        store.update_status(case_id, "ocr_processing")
        ocr = seams.get_ocr_extractor()
        ocr_result = await ocr.extract(file_paths.get("policy", []), file_paths.get("evidence", []))
        store.update_status(case_id, "ocr_done", ocr_result=json.dumps(ocr_result, ensure_ascii=False))

        # 只補保戶留空的欄位，絕不覆蓋保戶已填的值
        policy_fields = (ocr_result.get("policy") or {}).get("fields", {})
        evidence_fields = (ocr_result.get("evidence") or {}).get("fields", {})
        autofill, filled_names = {}, []
        for ocr_key, form_key in {**_POLICY_FIELD_MAP, **_EVIDENCE_FIELD_MAP}.items():
            source = policy_fields if ocr_key in _POLICY_FIELD_MAP else evidence_fields
            ocr_value = source.get(ocr_key)
            if _blank(submitted_fields.get(form_key)) and not _blank(ocr_value):
                autofill[form_key] = ocr_value
                filled_names.append(form_key)

        if autofill:
            store.apply_ocr_autofill(case_id, autofill, filled_names)
            submitted_fields = {**submitted_fields, **autofill}

        # 描述解析：把保戶自由填寫的description解析成Rule Agent需要的事故欄位
        # (車輛用途/服務類型/地點/道路類型/是否已聯絡指定救援中心等)。
        # 這批欄位跟OCR抓的欄位(policy_no/claim_amount等)定義上不重疊，
        # 但仍防呆處理：萬一解析結果撞到已經有值的欄位名(保戶手填或OCR
        # 帶入的)，一律以既有值為準，解析結果那筆捨棄並記錄下來，不覆蓋
        # ——這是你要求的「OCR判斷結果優先於description」原則的落實。
        store.update_status(case_id, "description_parsing")
        parser = seams.get_description_parser()
        parse_result = await parser.parse(submitted_fields.get("description", ""))

        dropped_due_to_conflict = []
        parsed_fields = parse_result.get("fields", {})
        for key, value in list(parsed_fields.items()):
            if value is not None and not _blank(submitted_fields.get(key)):
                dropped_due_to_conflict.append(key)
                parsed_fields[key] = None  # 不覆蓋，改填null，等同沒解析到

        if dropped_due_to_conflict:
            print(f"[警告] 案件 {case_id} 描述解析結果與既有欄位衝突，"
                  f"已捨棄解析值、保留原值: {dropped_due_to_conflict}")

        store.apply_description_parse(case_id, parse_result, dropped_due_to_conflict)
        non_null_parsed = {k: v for k, v in parsed_fields.items() if v is not None}
        submitted_fields = {**submitted_fields, **non_null_parsed}

        # 保戶手填 + OCR 自動帶入之後，關鍵欄位若仍缺，誠實停在這裡轉人工，
        # 不要讓協調層拿到 None/0 這種會被誤判成「金額為零」的資料去做決策。
        missing = [f for f in ("policy_no", "applicant_name", "claim_amount", "incident_date")
                   if _blank(submitted_fields.get(f))]
        if missing:
            store.update_status(
                case_id, "escalated_human",
                error_message=f"OCR 辨識後仍缺少必要欄位：{', '.join(missing)}，需人工確認",
            )
            return

        store.update_status(case_id, "pipeline_processing")

        # 2026-08 三次更新：把「描述解析結果」跟「OCR 從保單／佐證文件讀到的
        # RSA 專屬欄位」合併成 rsa_fields，送給 RSA Rule Agent。合併順序：
        # OCR 優先於描述解析——保戶上傳的文件比自由文字敘述更可靠，呼應
        # 「OCR 判斷結果優先於 description」的既有原則，這裡延伸到 RSA 專屬
        # 欄位上。towing_distance_km 只有 OCR 讀得到（描述解析沒有這個欄位），
        # 沒讀到就誠實留 None，不用其他資訊推算。
        policy_ocr_fields = (ocr_result.get("policy") or {}).get("fields", {})
        evidence_ocr_fields = (ocr_result.get("evidence") or {}).get("fields", {})

        rsa_fields = dict(parsed_fields)
        if not _blank(evidence_ocr_fields.get("service_type")):
            rsa_fields["requested_service"] = evidence_ocr_fields["service_type"]
        if not _blank(evidence_ocr_fields.get("service_location")):
            rsa_fields["location"] = evidence_ocr_fields["service_location"]
        if not _blank(evidence_ocr_fields.get("towing_distance_km")):
            rsa_fields["towing_distance_km"] = evidence_ocr_fields["towing_distance_km"]
        if not _blank(policy_ocr_fields.get("vehicle_use")):
            rsa_fields["vehicle_use"] = policy_ocr_fields["vehicle_use"]

        # rsa_addon_purchased 是表單上的手動選單（是／否／不確定），代表保戶
        # 或客服人員的明確輸入，永遠優先。只有保戶留空（未確認／不確定）時，
        # 才用 OCR 從保單文件上讀到的值頂上——這裡是唯一會回頭修改
        # submitted_fields 的地方，因為這個欄位本來就該是「手動優先、OCR 補位」
        # 的邏輯，其餘 RSA 欄位（vehicle_use/requested_service/...）沒有對應的
        # 手動輸入管道，不需要這層判斷。
        ocr_addon = policy_ocr_fields.get("rsa_addon_purchased")
        if _blank(submitted_fields.get("rsa_addon_purchased")) and ocr_addon is not None:
            submitted_fields["rsa_addon_purchased"] = "是" if ocr_addon else "否"

        # 2026-08 新增：把「有沒有上傳保單照片／佐證文件」轉換成 RSA Rule
        # Agent 需要的 documents.available_types。這裡只能誠實地做到「有
        # 上傳檔案就視為對應文件存在」，沒辦法驗證檔案內容是否真的符合
        # policy_record／service_request_record 的定義——那屬於 OCR 或
        # 人工複核的範圍，不是這一層該做的判斷。目前只有 RSA 案件會讀
        # 這個欄位，其他險種送過去也不影響。
        available_document_types = []
        if file_paths.get("policy"):
            available_document_types.append("policy_record")
        if file_paths.get("evidence"):
            available_document_types.append("service_request_record")

        claim_data = {**submitted_fields, "ocr_result": ocr_result,
                      "description_parsed": parse_result,
                      "rsa_fields": rsa_fields,
                      "available_document_types": available_document_types}

        # 2026-08 新增：第三人責任險（TPL）案件不走通用協調層，改走
        # tpl-claim-agent-api（金額建議）→ judge-agent（金額公平性/詐欺旗標/
        # 審推理，拍板 execute/return_for_recalc/escalate_human）這條專屬管線。
        # claim-intake 只負責把 TPL Claim Agent 的輸出轉成 judge-agent 看得懂
        # 的 case dict（_build_judge_case_from_tpl），不自己判斷金額合不合理。
        if insurance_type == "第三人責任險":
            missing_tpl = [f for f in ("accident_area", "own_fault_pct", "injury_desc")
                           if _blank(submitted_fields.get(f))]
            if missing_tpl:
                store.update_status(
                    case_id, "escalated_human",
                    error_message=f"第三人責任險案件缺少必要欄位：{', '.join(missing_tpl)}，需人工確認",
                )
                return

            tpl_client = seams.get_tpl_claim_agent_client()
            judge_client = seams.get_judge_agent_client()

            # judge-agent 的 decision.py 在 decision=="return_for_recalc" 且
            # retry_count < max_retries(judge-agent 端設定，預設2) 時會要求「退回
            # 重算」，next_agent="claim_agent"——這裡最多重打 3 輪(retry_count
            # 0/1/2)跟 judge-agent 預設的 max_retries=2 對齊。claim-intake 目前
            # 沒有能力依 judge 的理由調整 TPL Claim Agent 的輸入去真正「重算」，
            # 只能誠實地重新問一次；若三輪後仍是 return_for_recalc，就不再繼續
            # 重試，讓下面的 final_status 邏輯把它視同需要人工介入。
            judge_result, last_suggestion = None, None
            for retry_count in range(3):
                try:
                    last_suggestion = await tpl_client.get_suggestion(
                        submitted_fields["accident_area"],
                        float(submitted_fields["own_fault_pct"]),
                        submitted_fields["injury_desc"],
                    )
                except Exception as e:
                    # 打不通(冷啟動/帳單暫停/尚未部署)就明確標記，轉人工，
                    # 不讓整條背景任務因為 TPL agent 還沒就緒就整個失敗掛掉
                    last_suggestion = {
                        "simulated": True, "suggested_items": [],
                        "total_suggested_amount": None, "confidence": None,
                        "tpl_agent_error": f"{type(e).__name__}: {e}",
                    }

                judge_case = _build_judge_case_from_tpl(
                    case_id, submitted_fields, ocr_result, last_suggestion, retry_count
                )
                try:
                    judge_result = await judge_client.judge(judge_case)
                except Exception as e:
                    judge_result = {
                        "simulated": True, "decision": "escalate_human", "next_agent": "human_review",
                        "reasons": [f"法官代理人呼叫失敗，保守轉人工: {type(e).__name__}: {e}"],
                        "confidence": 0.0,
                    }
                    break
                if judge_result.get("decision") != "return_for_recalc":
                    break

            result = {**judge_result, "tpl_claim_agent_suggestion": last_suggestion}
        else:
            result = await seams.submit_to_orchestrator_with_fallback(case_id, claim_data)

        # return_for_recalc 走到這裡代表上面的重試迴圈已經試過、仍然沒有拍板
        # execute，claim-intake 沒有能力再自動重算，視同需要人工介入。
        final_status = ("escalated_human"
                         if result.get("decision") in ("escalate_human", "return_for_recalc")
                         else "completed")
        store.update_status(case_id, final_status, pipeline_result=json.dumps(result, ensure_ascii=False))

        notifier = seams.get_notifier()
        await notifier.notify(case_id, submitted_fields.get("contact_email"),
                               submitted_fields.get("contact_phone"), result)
    except Exception as e:
        # 誠實記錄錯誤，不要吞掉、不要假裝成功
        store.update_status(case_id, "error", error_message=f"{type(e).__name__}: {e}")


# ============================================================
# 狀態查詢（測試/除錯用；正式產品的主要通知管道是 email/簡訊，不是這支）
# ============================================================
@app.get("/v1/claims/{case_id}")
def get_claim_status(case_id: str):
    claim = store.get_claim(case_id)
    if claim is None:
        raise HTTPException(404, "查無此案件編號")
    return claim


@app.get("/v1/health")
def health():
    return {"status": "ok"}


# ============================================================
# 後台總覽（/admin 頁面用；⚠️ 測試階段刻意不加驗證，正式上線前必須補
# Admin API Key 或帳密登入，比照未來 policy 管理端點的做法）
# ============================================================
def _parse_admin_filters(
    status: Optional[str], channel: Optional[str], insurance_type: Optional[str],
    review_status: Optional[str], q: Optional[str],
    date_from: Optional[str], date_to: Optional[str],
) -> dict:
    return {
        "status": status, "channel": channel, "insurance_type": insurance_type,
        "review_status": review_status, "q": q,
        "date_from": date_from, "date_to": date_to,
    }


@app.get("/v1/admin/claims")
def admin_list_claims(
    status: Optional[str] = None,
    channel: Optional[str] = None,
    insurance_type: Optional[str] = None,
    review_status: Optional[str] = None,
    q: Optional[str] = Query(None, description="模糊搜尋 case_id / policy_no / applicant_name"),
    date_from: Optional[str] = Query(None, description="ISO 日期，篩 created_at 起"),
    date_to: Optional[str] = Query(None, description="ISO 日期，篩 created_at 迄"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    filters = _parse_admin_filters(status, channel, insurance_type, review_status, q, date_from, date_to)
    rows, total = store.list_claims(filters, limit=page_size, offset=(page - 1) * page_size)
    return {
        "items": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if page_size else 0,
    }


@app.get("/v1/admin/claims/export.csv")
def admin_export_claims_csv(
    status: Optional[str] = None,
    channel: Optional[str] = None,
    insurance_type: Optional[str] = None,
    review_status: Optional[str] = None,
    q: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    filters = _parse_admin_filters(status, channel, insurance_type, review_status, q, date_from, date_to)
    csv_text = store.export_claims_csv(filters)
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=claims_export.csv"},
    )


@app.get("/v1/admin/claims/{case_id}")
def admin_get_claim_detail(case_id: str):
    claim = store.get_claim(case_id)
    if claim is None:
        raise HTTPException(404, "查無此案件編號")
    return claim


@app.post("/v1/admin/claims/{case_id}/review")
def admin_set_review(
    case_id: str,
    review_status: str = Form(...),
    reviewed_by: Optional[str] = Form(None),
    review_note: Optional[str] = Form(None),
):
    try:
        ok = store.set_review(case_id, review_status, reviewed_by, review_note)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, "查無此案件編號")
    return store.get_claim(case_id)


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    print(f"[警告] 找不到前端資料夾：{FRONTEND_DIR}，'/' 無法提供網頁，但 /v1/... 的 API 端點不受影響。")
