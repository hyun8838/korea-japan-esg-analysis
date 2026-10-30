# -*- coding: utf-8 -*-
"""
Target Field 상속·검증, 본문 병합, PDF 포함 필터(A/B/F/H, 중복 제거) 및
ESG 분석용 최종 문서 데이터셋 생성. 수동 검토 결과 반영 후 실행.
"""
import os
from pathlib import Path
import csv
import openpyxl
from openpyxl.styles import Font, PatternFill
from collections import Counter

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

ROOT = f"{COMP_ROOT}/0_crawling"
CLASSIFIED = f"{ROOT}/phase5_page_type/page_type_classified_v3.xlsx"
ENTITY_KR = f"{ROOT}/phase1_entity_registry/entity_registry_kr.csv"
ENTITY_JP = f"{ROOT}/phase1_entity_registry/entity_registry_jp.csv"
PDF_SRC = f"{ROOT}/phase4_url_inventory/url_inventory_full.xlsx"
FULL_TEXT_SRC = f"{ROOT}/phase4_url_inventory/url_inventory_full_text.xlsx"

OUT_XLSX = f"{ROOT}/phase5_page_type/esg_final_dataset_v3.xlsx"
OUT_CSV = f"{ROOT}/phase5_page_type/esg_final_dataset_v3.csv"

PDF_INCLUDE_TYPES = {"A", "B", "F", "H"}


def load_entity_map():
    m = {}
    for path in (ENTITY_KR, ENTITY_JP):
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            m[row["entity_id"]] = {
                "target_field_primary": row["target_field_primary"],
                "college": row["college"],
            }
    return m


def load_full_text_map():
    wb = openpyxl.load_workbook(FULL_TEXT_SRC, data_only=True)
    ws = wb["url_inventory"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]
    return {r["url"]: {"full_text": r.get("full_text") or "", "full_text_len": r.get("full_text_len") or 0}
            for r in rows if r.get("url")}


def main():
    entity_map = load_entity_map()
    full_text_map = load_full_text_map()
    print(f"본문 전체 맵: {len(full_text_map)}건")

    wb = openpyxl.load_workbook(CLASSIFIED, data_only=True)
    ws = wb["page_type_classified"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]

    missing_entity = set()
    missing_full_text = 0
    for r in rows:
        info = entity_map.get(r["entity_id"])
        if info is None:
            missing_entity.add(r["entity_id"])
            r["target_field_primary"] = ""
            r["college"] = ""
        else:
            r["target_field_primary"] = info["target_field_primary"]
            r["college"] = info["college"]
        ft = full_text_map.get(r["url"])
        if ft is None:
            missing_full_text += 1
            r["full_text"] = ""
            r["full_text_len"] = 0
        else:
            r["full_text"] = ft["full_text"]
            r["full_text_len"] = ft["full_text_len"]
    print("entity_id 매칭 안 된 건:", len(missing_entity), missing_entity if missing_entity else "")
    print("본문 전체 매칭 안 된 건(중복 URL 등으로 처음부터 수집 대상 아니었을 수 있음):", missing_full_text)

    excluded_rows = [r for r in rows if r.get("excluded")]
    analysis_rows = [r for r in rows if r["status"] in (200, "200") and r["page_type"] and not r.get("excluded")]
    print("수동검토 제외 처리된 행:", len(excluded_rows))
    print("분석 대상 페이지 수:", len(analysis_rows), "/ 전체:", len(rows))
    no_text = [r for r in analysis_rows if not r["full_text"] or r["full_text"].startswith("[")]
    print("본문 전체 없음/수집실패:", len(no_text))

    page_type_by_url = {r["url"]: r["page_type"] for r in rows if not r.get("excluded")}
    wb_pdf = openpyxl.load_workbook(PDF_SRC, data_only=True)
    ws_pdf = wb_pdf["pdf_extraction"]
    pdf_headers = [c.value for c in next(ws_pdf.iter_rows(min_row=1, max_row=1))]
    pdf_rows = [dict(zip(pdf_headers, r)) for r in ws_pdf.iter_rows(min_row=2, max_row=ws_pdf.max_row, values_only=True)]

    seen_pdf_urls = set()
    kept_pdfs = []
    for pr in pdf_rows:
        source_type = page_type_by_url.get(pr["source_page_url"])
        if source_type not in PDF_INCLUDE_TYPES:
            continue
        if pr["pdf_url"] in seen_pdf_urls:
            continue
        seen_pdf_urls.add(pr["pdf_url"])
        pr["source_page_type"] = source_type
        kept_pdfs.append(pr)
    print(f"PDF 전체 {len(pdf_rows)}건 중 A/B/F/H 페이지 소속+중복제거 후 {len(kept_pdfs)}건 최종 포함")

    final_fields = ["entity_id", "university", "college", "entity_name", "target_field_primary",
                     "page_type", "classify_source", "needs_review_priority", "title", "url",
                     "full_text_len", "full_text"]
    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "esg_final_dataset"
    ws2.append(final_fields)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    for c in range(1, len(final_fields) + 1):
        cell = ws2.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for r in analysis_rows:
        ws2.append([r.get(f, "") for f in final_fields])
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = ws2.dimensions

    ws3 = wb2.create_sheet("pdf_included")
    pdf_fields = ["entity_id", "university", "entity_name", "source_page_url", "source_page_type",
                  "pdf_url", "text_len", "ocr_applied", "text_sample"]
    ws3.append(pdf_fields)
    for c in range(1, len(pdf_fields) + 1):
        cell = ws3.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for pr in kept_pdfs:
        ws3.append([pr.get(f, "") for f in pdf_fields])
    ws3.freeze_panes = "A2"
    ws3.auto_filter.ref = ws3.dimensions

    summary = wb2.create_sheet("요약", 0)
    summary["A1"] = "ESG 분석용 최종 데이터셋 요약"
    summary["A1"].font = Font(bold=True, size=13)
    tf_dist = Counter(r["target_field_primary"] for r in analysis_rows)
    pt_dist = Counter(r["page_type"] for r in analysis_rows)
    src_dist = Counter(r["classify_source"] for r in analysis_rows)
    country_dist = Counter(("KR" if r["entity_id"].startswith("KR") else "JP") for r in analysis_rows)
    info_rows = [
        ("최종 분석 대상 페이지 수", len(analysis_rows)),
        ("KR / JP", dict(country_dist)),
        ("page_type 분포", dict(pt_dist)),
        ("분류 출처 분포", dict(src_dist)),
        ("Target Field 분포", dict(tf_dist)),
        ("본문 전체 없음/실패", len(no_text)),
        ("최종 포함 PDF 수", len(kept_pdfs)),
        ("entity_id 매칭 실패", len(missing_entity)),
        ("needs_review_priority(저신뢰) 건수", sum(1 for r in analysis_rows if r["needs_review_priority"])),
    ]
    for i, (a, b) in enumerate(info_rows, start=3):
        summary.cell(row=i, column=1, value=a)
        summary.cell(row=i, column=2, value=str(b))
    summary.column_dimensions["A"].width = 35
    summary.column_dimensions["B"].width = 70

    wb2.save(OUT_XLSX)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(final_fields)
        for r in analysis_rows:
            w.writerow([r.get(f, "") for f in final_fields])

    print("saved:", OUT_XLSX)
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
