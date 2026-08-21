# 法官代理人 API（內部服務）

金額公平性(Tukey IQR 圍籬)、詐欺旗標(A/B/C/D)、升級觸發、退回重算迴圈——邏輯已在
Colab 筆記本用真實 `TPL_train.db`/`TPL_val.db`(696/297筆)、`RSA_train.db`/`RSA_val.db`
(700/339筆)驗證過，這裡是把同一套邏輯包裝成可部署的 FastAPI 服務。

## ⚠️ 這是內部服務，不是給瀏覽器連的

部署到 Cloud Run 時**絕對不要加 `--allow-unauthenticated`**。只有協調層應該能呼叫
這支服務，透過 Cloud Run 的 IAM Invoker 機制（協調層的服務帳號需要這個服務的
`roles/run.invoker`），不是這支程式自己做身份驗證。

## 本機測試

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`data/TPL_train.db`、`data/RSA_train.db` 必須存在（唯讀比較池，隨專案一起打包）。
啟動 log 會明確顯示連上真實資料還是退回 Mock：

```
[啟動] TPL AmountComparator 已連接: data/TPL_train.db
[啟動] RSA AmountComparator 已連接: data/RSA_train.db
```

測試呼叫：

```bash
curl -X POST http://localhost:8000/v1/judge \
  -H "Content-Type: application/json" \
  -d '{"case_id":"TEST001","line":"RSA","claim_amount":3000,"coverage_limit":30000,"category_id":1,"km":20,"retry_count":0}'
```

## 已經真的測試過的部分（不是紙上設計）

- 用真實 `TPL_val.db`/`RSA_val.db` 抽案件、送進 `/v1/judge`，schema 全部驗證通過
- **併發測試**：同時送 10 個請求，全部正確處理（見下方「已修正的真實 bug」）
- 批次跑 50 筆真實 TPL + 50 筆真實 RSA 案件，決策分佈合理（`execute`/`escalate_human`/
  `return_for_recalc` 都有出現，符合筆記本裡驗證過的比例）

## 已修正的真實 bug（實測才發現，寫程式碼時看不出來）

**FastAPI 用執行緒池處理請求，SQLite 連線預設只能在建立它的執行緒使用。** 啟動時
建立的資料庫連線被不同執行緒的請求呼叫，會直接 500 錯誤——如果沒有實際送請求測試，
這會是「部署到 Cloud Run 後每一筆案件都失敗」，而且錯誤只在正式環境的 log 才看得到。
修法：`amount_sources/_db.py` 的連線加上 `check_same_thread=False`（這裡只有讀取、
沒有寫入，關掉執行緒歸屬檢查是安全的）。已用 10 個併發請求重新驗證過。

## 架構重點

- **金額公平性(B)**：`AmountComparator`(TPL)/`RSAAmountComparator`(RSA) 各自比對真實
  結構化資料，Tukey IQR 圍籬取代固定 ±20%（理由見 `app/fairness.py` 檔頭說明）
- **判決先例的肇責/傾向一致性(A)**：`app/fault_consistency.py` 是 SEAM，目前用
  `NoOpFaultConsistencyRetriever` 佔位。正確做法是打既有的 GraphRAG API
  (`https://graphrag-api-557608333258.asia-east1.run.app`)，**不是**直接連 Neo4j。
  需要 `API_SPEC.md` 才能確定請求/回應格式，尚未取得
- **信心分數**：只反映金額判斷本身的把握度(先例數+離散度)，刻意不受 claim 偏離度
  或詐欺規則 na 比例影響——這兩個都是實測時踩到的真 bug，理由見 `app/confidence.py`
- **輸出 schema**：與規則/理賠代理人對齊，`decision` 只有四種
  (`execute`/`return_for_recalc`/`escalate_human`/`reject`)，`reject` 保留在列舉但
  邏輯永遠不會產生——詐欺一律轉人工，法官不得自動拍板拒賠

## 部署到 Cloud Run

```powershell
gcloud run deploy judge-agent `
  --source . `
  --region asia-east1 `
  --project cki-scu-2026
```

**注意沒有 `--allow-unauthenticated`。** 部署完，只有協調層(或你自己用
`gcloud run services proxy`)能連。之後把 `judge-agent` 服務的呼叫端(協調層)服務帳號
加上這支服務的 `roles/run.invoker`：

```powershell
gcloud run services add-iam-policy-binding judge-agent `
  --region=asia-east1 --project=cki-scu-2026 `
  --member="serviceAccount:<協調層的服務帳號>" --role=roles/run.invoker
```

部署驗證（因為沒開放公開存取，用 proxy 測）：

```powershell
gcloud run services proxy judge-agent --region=asia-east1 --project=cki-scu-2026
```

再對本機 proxy 網址跑一次上面「本機測試」的 curl 指令，確認 `/v1/health` 顯示
`"amount_sources": {"TPL": "real", "RSA": "real"}`。

## 尚待補的東西（誠實列出，非隱藏）

- `app/fault_consistency.py` 的 HTTP client：需要 `API_SPEC.md`
- A1-A4(保單時序旗標)：TPL/RSA 資料庫都沒有保單生效/到期日、報案日欄位，維持誠實
  的 `na`，非疏漏（已與你確認過不需要另外找資料源）
- C4/C5：業務決定不做，非資料缺口
- 理賠代理人產生的 case dict 需要跟 `TPLValCaseLoader.load()`/`RSAValCaseLoader.load()`
  回傳的形狀一致，才能正確餵進這支 API——這是下一個要對齊的介面
