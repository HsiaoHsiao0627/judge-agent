# -*- coding: utf-8 -*-
"""config.py — 法官代理人設定檔，全部門檻皆可調，非寫死。"""
from dataclasses import dataclass, field


@dataclass
class JudgeConfig:
    tukey_k: float = 1.5                 # IQR 圍籬寬鬆度；對真實 val 實測 k=1.5→攔5.1%(TPL)/2.0%(RSA)
    extreme_dispersion_threshold: float = 1.6  # dispersion_ratio 高於此(≈真實資料 p90) → 標記極端離散
    warning_threshold: int = 3           # warning 旗標累計達此數 → 轉人工
    amount_threshold: dict = field(default_factory=lambda: {"RSA": 200_000, "TPL": 500_000})
    min_precedents: int = 3              # 先例數低於此 → 先例覆蓋不足 → 轉人工
    confidence_threshold: float = 0.60   # 信心低於此 → 轉人工
    max_retries: int = 2                 # 退回重算上限
    policy_new_days: int = 14            # 投保後 N 天內出險視為短期 (A4)
    schema_version: str = "1.0"
