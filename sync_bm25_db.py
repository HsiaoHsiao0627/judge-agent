# -*- coding: utf-8 -*-
"""
sync_bm25_db.py
==================
Cloud Run 容器啟動時執行，把 GCS 上最新的 BM25 db 下載到本機路徑，
讓 rsa_clause_retrieval.py 可以直接用 sqlite3 開啟。

這樣資料更新(重跑 rebuild_rsa_clause_bm25.py 後上傳新版db到GCS)
跟程式碼部署可以分開，不用每次資料變動都重新build image。

環境變數：
  BM25_DB_GCS_URI       例如 gs://cki-scu-2026-tuning/rsa_bm25_knowledge_base_V1.db
  BM25_DB_LOCAL_PATH    預設 /tmp/rsa_bm25_knowledge_base_V1.db

執行時機：在 FastAPI app 啟動時呼叫(例如 main.py 的 startup event，
或是 Dockerfile 的 CMD 前先跑這支script)。
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def sync():
    gcs_uri = os.environ.get("BM25_DB_GCS_URI")
    local_path = os.environ.get("BM25_DB_LOCAL_PATH", "/tmp/rsa_bm25_knowledge_base_V1.db")

    if not gcs_uri:
        print("❌ 缺少環境變數 BM25_DB_GCS_URI")
        sys.exit(1)

    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        print(f"❌ BM25_DB_GCS_URI 格式錯誤，應為 gs://bucket/path，收到: {gcs_uri}")
        sys.exit(1)

    bucket_name, blob_path = gcs_uri[5:].split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        print(f"❌ GCS 上找不到檔案: {gcs_uri}")
        sys.exit(1)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"已下載 {gcs_uri} -> {local_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sync()
