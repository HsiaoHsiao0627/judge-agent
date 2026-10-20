# -*- coding: utf-8 -*-
"""醫囑文字解析：休養天數、專人照護天數。只取文字中明確寫出的期間，讀不到回傳 None。"""
import re

_CN = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
       "十": 10, "壹": 1, "貳": 2, "參": 3, "肆": 4, "伍": 5, "陸": 6, "柒": 7, "捌": 8, "玖": 9, "拾": 10, "半": 0.5}
_NUM = r"([0-9０-９]+(?:\.[0-9]+)?|[零一二兩三四五六七八九十壹貳參肆伍陸柒捌玖拾半]+)"
_UNIT = r"(個半月|個月|星期|週|周|天|日)"
_DUR = re.compile(_NUM + r"\s*" + _UNIT)
_UNIT_DAYS = {"個月": 30, "月": 30, "個半月": 30, "星期": 7, "週": 7, "周": 7, "天": 1, "日": 1}


def _to_num(s: str):
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    try:
        return float(s)
    except ValueError:
        pass
    if s == "半":
        return 0.5
    total, cur = 0, 0
    for ch in s:
        v = _CN.get(ch)
        if v is None:
            return None
        if v == 10:
            total += (cur or 1) * 10
            cur = 0
        else:
            cur = v
    return total + cur


def _durations(sentence: str) -> list:
    out = []
    for num, unit in _DUR.findall(sentence):
        n = _to_num(num)
        if n is None:
            continue
        days = n * _UNIT_DAYS[unit] + (15 if unit == "個半月" else 0)
        out.append(days)
    return out


def _parse(text: str, keywords) -> float:
    if not text:
        return None
    found = False
    total = 0.0
    # 多張診斷書以 " | " 串接；同一張取最大值，不同張相加（後續回診常寫「宜再休養」）
    seen = set()
    for cert in str(text).split("|"):
        key = re.sub(r"\s+", "", cert)
        if not key or key in seen:
            continue
        seen.add(key)
        best = None
        for sent in re.split(r"[。；;\n，,]", cert):
            if not any(k in sent for k in keywords):
                continue
            ds = [d for d in _durations(sent) if d <= 730]
            # 「住院期間…共住院 N 日」的住院天數不是休養/照護期間：排除緊接「住院」的數字
            ds_clean = []
            for m in _DUR.finditer(sent):
                pre = sent[max(0, m.start() - 3):m.start()]
                post = sent[m.end():m.end() + 1]
                if "住院" in pre or "診" in pre:
                    continue
                # 日期（113年10月20日、10/20）不是期間
                if m.group(2) in ("日", "天") and (pre.endswith("月") or pre[-1:].isdigit() or "年" in pre):
                    continue
                if m.group(2) == "日" and (post.isdigit() or post in ("起", "至", "~", "－", "-", "、")):
                    continue
                n = _to_num(m.group(1))
                if n is not None:
                    d = n * _UNIT_DAYS[m.group(2)] + (15 if m.group(2) == "個半月" else 0)
                    if d <= 730:
                        ds_clean.append(d)
            if ds_clean:
                best = max(best or 0, max(ds_clean))
        if best is not None:
            found = True
            total += best
    return total if found else None


def rest_days(text: str):
    return _parse(text, ("休養", "休息", "不宜工作", "無法工作"))


def care_days(text: str):
    return _parse(text, ("專人", "照護", "照顧", "看護"))
