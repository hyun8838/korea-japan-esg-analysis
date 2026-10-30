# -*- coding: utf-8 -*-
"""
run_full_crawl.py 결과(url_inventory_full.xlsx)에 요약/중복URL 시트를 추가하고
CSV로도 내보낸다. (수집 결과의 URL 중복을 표시한다)
"""
import os
from pathlib import Path
import csv as csvmod
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from collections import Counter, defaultdict

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

SRC = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/url_inventory_full.xlsx"
OUT_CSV = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/url_inventory_full.csv"
OUT_DUP_CSV = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/url_inventory_full_duplicates.csv"

wb = openpyxl.load_workbook(SRC, data_only=True)
ws = wb["url_inventory"]
headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
idx = {h: i for i, h in enumerate(headers)}
rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]

url_counter = Counter(r["url"] for r in rows)
url_to_entities = defaultdict(set)
for r in rows:
    url_to_entities[r["url"]].add(r["entity_id"])
    r["is_duplicate_url"] = url_counter[r["url"]] > 1

header_fill = PatternFill("solid", fgColor="D9E1F2")

# 요약 시트를 맨 앞에 삽입
summary = wb.create_sheet("요약", 0)
summary["A1"] = "URL 수집 결과 요약"
summary["A1"].font = Font(bold=True, size=13)
status_counts = Counter(r["status"] for r in rows)
dup_url_count = sum(1 for u, n in url_counter.items() if n > 1)
img_flag_count = sum(1 for r in rows if r["possible_image_content"])
entity_ids = set(r["entity_id"] for r in rows)
ws_pdf = wb["pdf_extraction"]
pdf_headers = [c.value for c in next(ws_pdf.iter_rows(min_row=1, max_row=1))]
pdf_idx = {h: i for i, h in enumerate(pdf_headers)}
pdf_rows = list(ws_pdf.iter_rows(min_row=2, max_row=ws_pdf.max_row, values_only=True))
ocr_count = sum(1 for r in pdf_rows if r[pdf_idx["ocr_applied"]])

rows_info = [
    ("총 레코드 수", len(rows)),
    ("커버된 entity 수", f"{len(entity_ids)} / 165"),
    ("상태코드 분포", dict(status_counts)),
    ("PDF 발견 수", len(pdf_rows)),
    ("PDF 중 OCR 적용", ocr_count),
    ("HTML 페이지 중 이미지기반 의심(본문<200자+이미지있음)", img_flag_count),
    ("중복 URL 종류 수(2개 이상 entity에 걸침)", dup_url_count),
]
for i, (a, b) in enumerate(rows_info, start=3):
    summary.cell(row=i, column=1, value=a)
    summary.cell(row=i, column=2, value=str(b))
summary.column_dimensions["A"].width = 45
summary.column_dimensions["B"].width = 70

# 중복URL 시트
ws3 = wb.create_sheet("중복URL")
ws3.append(["url", "걸친 entity 수", "entity_id 목록", "대학"])
for c in range(1, 5):
    cell = ws3.cell(row=1, column=c)
    cell.font = Font(bold=True)
    cell.fill = header_fill
univ_by_entity = {r["entity_id"]: r["university"] for r in rows}
dup_items = sorted(((u, e) for u, e in url_to_entities.items() if len(e) > 1), key=lambda x: -len(x[1]))
for u, eids in dup_items:
    eids_sorted = sorted(eids)
    univs = sorted(set(univ_by_entity.get(e, "") for e in eids_sorted))
    ws3.append([u, len(eids_sorted), ", ".join(eids_sorted), ", ".join(univs)])
ws3.freeze_panes = "A2"
ws3.auto_filter.ref = ws3.dimensions
for i, w in enumerate([70, 12, 30, 30], start=1):
    ws3.column_dimensions[get_column_letter(i)].width = w

# is_duplicate_url을 url_inventory 시트에도 반영(마지막 컬럼 "비고" 앞에 삽입)
bigo_col = headers.index("비고") + 1  # 1-indexed
ws.insert_cols(bigo_col)
ws.cell(row=1, column=bigo_col, value="is_duplicate_url").font = Font(bold=True)
ws.cell(row=1, column=bigo_col).fill = header_fill
for r_i, r in enumerate(rows, start=2):
    ws.cell(row=r_i, column=bigo_col, value=r["is_duplicate_url"])

wb.save(SRC)

# CSV 내보내기
with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    fieldnames = headers[:bigo_col-1] + ["is_duplicate_url"] + headers[bigo_col-1:]
    w = csvmod.writer(f)
    w.writerow(fieldnames)
    for r in rows:
        w.writerow([r.get(h, "") if h != "is_duplicate_url" else r["is_duplicate_url"] for h in fieldnames])

with open(OUT_DUP_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csvmod.writer(f)
    w.writerow(["url", "걸친 entity 수", "entity_id 목록", "대학"])
    for u, eids in dup_items:
        eids_sorted = sorted(eids)
        univs = sorted(set(univ_by_entity.get(e, "") for e in eids_sorted))
        w.writerow([u, len(eids_sorted), ", ".join(eids_sorted), ", ".join(univs)])

print("done. total:", len(rows), "duplicates:", dup_url_count)
