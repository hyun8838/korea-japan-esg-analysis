# -*- coding: utf-8 -*-
"""URL 목록의 분류용 스니펫과 최종 분석용 본문을 단계별로 추출한다."""
import os
from pathlib import Path
import argparse
import json
import re
import time

import openpyxl
import requests
import trafilatura
from bs4 import BeautifulSoup
from openpyxl.styles import Font, PatternFill

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

SNIPPET_SRC = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/url_inventory_full.xlsx"
SNIPPET_OUT = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/url_inventory_full_trafilatura.xlsx"
SNIPPET_CHECKPOINT = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/refetch_full_checkpoint.jsonl"
FULL_SRC = SNIPPET_OUT
FULL_OUT = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/url_inventory_full_text.xlsx"
FULL_CHECKPOINT = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/full_text_checkpoint.jsonl"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA}
SNIPPET_LEN = 1500
MAX_TEXT_LEN = 30000  # 엑셀 셀 한도(32,767자) 안전마진
ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
META_CHARSET_RE = re.compile(rb'charset=["\']?([\w-]+)', re.I)



def sanitize(v):
    return ILLEGAL_XML_RE.sub("", v) if isinstance(v, str) else v


def basic_extract(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def fetch_clean_text(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code >= 400:
            return f"[fetch failed: {resp.status_code}]"
        content_type = resp.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            return f"[non-html content-type: {content_type}]"
        if "charset" not in content_type:
            resp.encoding = resp.apparent_encoding or resp.encoding
        text = trafilatura.extract(resp.text, favor_precision=False)
        if not text or len(text.strip()) < 100:
            text = basic_extract(resp.text)
        return sanitize(text.strip()[:SNIPPET_LEN])
    except Exception as e:
        return f"[error: {type(e).__name__}]"


def fetch_full_text(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code >= 400:
            return f"[fetch failed: {resp.status_code}]", 0
        content_type = resp.headers.get("Content-Type", "").lower()
        if "html" not in content_type and content_type:
            return f"[non-html content-type: {content_type}]", 0
        if "charset" not in content_type:
            m = META_CHARSET_RE.search(resp.content[:3000])
            if m:
                resp.encoding = m.group(1).decode("ascii", "ignore")
            else:
                resp.encoding = resp.apparent_encoding or resp.encoding
        text = trafilatura.extract(resp.text, favor_precision=False)
        if not text or len(text.strip()) < 100:
            text = basic_extract(resp.text)
        text = sanitize((text or "").strip())
        return text[:MAX_TEXT_LEN], len(text)
    except Exception as e:
        return f"[error: {type(e).__name__}]", 0


def collect_snippets():
    wb = openpyxl.load_workbook(SNIPPET_SRC, data_only=True)
    ws = wb["url_inventory"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    idx = {h: i for i, h in enumerate(headers)}
    rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]
    target_rows = [r for r in rows if r["status"] in (200, "200")]
    print("target rows (status=200):", len(target_rows), "/ total:", len(rows), flush=True)

    open(SNIPPET_CHECKPOINT, "w", encoding="utf-8").close()
    for i, r in enumerate(target_rows, 1):
        r["snippet_trafilatura"] = fetch_clean_text(r["url"])
        with open(SNIPPET_CHECKPOINT, "a", encoding="utf-8") as f:
            f.write(json.dumps({"url": r["url"], "snippet_trafilatura": r["snippet_trafilatura"]},
                                ensure_ascii=False) + "\n")
        if i % 50 == 0 or i == len(target_rows):
            print(f"[{i}/{len(target_rows)}] done", flush=True)
        time.sleep(0.15)

    non_target = [r for r in rows if r["status"] not in (200, "200")]
    for r in non_target:
        r["snippet_trafilatura"] = ""
    all_rows = target_rows + non_target

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "url_inventory"
    fields = headers + ["snippet_trafilatura"]
    ws2.append(fields)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    for c in range(1, len(fields) + 1):
        cell = ws2.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for r in all_rows:
        ws2.append([sanitize(r.get(f, "")) for f in fields])
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = ws2.dimensions

    wb2.save(SNIPPET_OUT)
    print("saved:", SNIPPET_OUT, "total:", len(all_rows), flush=True)


def collect_full_text():
    wb = openpyxl.load_workbook(FULL_SRC, data_only=True)
    ws = wb["url_inventory"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]
    target_rows = [r for r in rows if r["status"] in (200, "200") and not r.get("is_duplicate_url")]
    other_rows = [r for r in rows if not (r["status"] in (200, "200") and not r.get("is_duplicate_url"))]
    print(f"본문 전체 수집 대상: {len(target_rows)} / 전체: {len(rows)}", flush=True)

    open(FULL_CHECKPOINT, "w", encoding="utf-8").close()
    t0 = time.time()
    for i, r in enumerate(target_rows, 1):
        text, full_len = fetch_full_text(r["url"])
        r["full_text"] = text
        r["full_text_len"] = full_len
        with open(FULL_CHECKPOINT, "a", encoding="utf-8") as f:
            f.write(json.dumps({"url": r["url"], "full_text_len": full_len}, ensure_ascii=False) + "\n")
        if i % 100 == 0 or i == len(target_rows):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(target_rows) - i)
            print(f"[{i}/{len(target_rows)}] done | 경과 {elapsed/60:.1f}분 | 예상잔여 {eta/60:.1f}분", flush=True)
        time.sleep(0.15)

    for r in other_rows:
        r["full_text"] = ""
        r["full_text_len"] = 0
    all_rows = target_rows + other_rows

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "url_inventory"
    fields = headers + ["full_text", "full_text_len"]
    ws2.append(fields)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    for c in range(1, len(fields) + 1):
        cell = ws2.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for r in all_rows:
        ws2.append([sanitize(r.get(f, "")) for f in fields])
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = ws2.dimensions
    wb2.save(FULL_OUT)
    print("saved:", FULL_OUT, flush=True)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("snippets", "full-text"),
                        help="snippets: 분류용 스니펫, full-text: 분석용 본문")
    args = parser.parse_args()
    if args.stage == "snippets":
        collect_snippets()
    else:
        collect_full_text()


if __name__ == "__main__":
    main()
