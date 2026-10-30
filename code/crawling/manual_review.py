# -*- coding: utf-8 -*-
"""분류 결과에서 검토표를 생성하고, 사람이 확정한 유형을 반영한다."""
import os
from pathlib import Path
import argparse
import re
from collections import Counter

import openpyxl
from openpyxl.styles import Font, PatternFill

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

ROOT = f"{COMP_ROOT}/0_crawling"
CLASSIFIED = f"{ROOT}/phase5_page_type/page_type_classified_v3.xlsx"
REVIEW = f"{ROOT}/phase5_page_type/저신뢰플래그_수동검토_v3.xlsx"

WEAK_TYPES = {"A", "E", "G", "I", "J"}
VALID_CODES = set("ABCDEFGHIJ")
ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

ACCURACY_NOTE = {
    "A": "학과소개 (실측 62%, B/C와 경계 혼동)",
    "E": "학생/졸업생 활동 (실측 44%, C/D/F/J 전방위 혼동)",
    "G": "관련 협회/기관 (실측 50%, n=4 표본 희소)",
    "I": "커뮤니티 (실측 78%, D/F와 경계 혼동)",
    "J": "기타 (실측 53%, A/I/D/F와 경계 혼동)",
}



def sanitize(v):
    return ILLEGAL_XML_RE.sub("", v) if isinstance(v, str) else v


def build_review_package():
    wb = openpyxl.load_workbook(CLASSIFIED, data_only=True)
    ws = wb["page_type_classified"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]

    review_rows = [r for r in rows if r.get("classify_source") == "dynamic_fewshot" and r.get("page_type") in WEAK_TYPES]
    review_rows.sort(key=lambda r: (r["page_type"], r["university"], r["entity_name"]))
    print(f"검토 대상: {len(review_rows)}건")
    print(Counter(r["page_type"] for r in review_rows))

    fields = ["entity_id", "university", "entity_name", "page_type", "저신뢰_사유",
              "title", "url", "snippet_trafilatura", "확정_page_type", "비고"]

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "저신뢰_수동검토"
    ws2.append(fields)
    header_fill = PatternFill("solid", fgColor="FFF2CC")
    for c in range(1, len(fields) + 1):
        cell = ws2.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill

    for r in review_rows:
        row = [r["entity_id"], r["university"], r["entity_name"], r["page_type"],
               ACCURACY_NOTE.get(r["page_type"], ""), r["title"], r["url"],
               r.get("snippet_trafilatura", ""), "", ""]
        ws2.append([sanitize(v) for v in row])
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = ws2.dimensions
    widths = {"A": 10, "B": 18, "C": 22, "D": 8, "E": 32, "F": 40, "G": 60, "H": 40, "I": 14, "J": 30}
    for col, w in widths.items():
        ws2.column_dimensions[col].width = w

    summary = wb2.create_sheet("요약", 0)
    summary["A1"] = "저신뢰 플래그(A/E/G/I/J) 수동검토 대상 요약"
    summary["A1"].font = Font(bold=True, size=13)
    pt_dist = Counter(r["page_type"] for r in review_rows)
    info = [
        ("전체 검토 대상 건수", len(review_rows)),
        ("선정 기준", "GT 364건 실측 정확도 <80% 카테고리(A 62%/E 44%/G 50%,n=4/I 78%/J 53%) 전체"),
        ("page_type 분포", dict(pt_dist)),
        ("제외 대상", "classify_source='human_gt'인 수동 라벨은 이미 검증된 값이라 제외"),
        ("작성 방법", "'확정_page_type' 열에 최종 코드(A~J)를 입력하세요. 원래 분류가 맞으면 같은 코드를 입력하면 됩니다."),
    ]
    for i, (a, b) in enumerate(info, start=3):
        summary.cell(row=i, column=1, value=a)
        summary.cell(row=i, column=2, value=str(b))
    summary.column_dimensions["A"].width = 28
    summary.column_dimensions["B"].width = 80

    wb2.save(REVIEW)
    print("saved:", REVIEW)


def apply_manual_review():
    wbr = openpyxl.load_workbook(REVIEW, data_only=True)
    wsr = wbr["저신뢰_수동검토"]
    rh = [c.value for c in next(wsr.iter_rows(min_row=1, max_row=1))]
    review_rows = [dict(zip(rh, r)) for r in wsr.iter_rows(min_row=2, max_row=wsr.max_row, values_only=True)]

    from collections import defaultdict, Counter as Ctr
    codes_by_url = defaultdict(list)
    for r in review_rows:
        url = r.get("url")
        if not url:
            continue
        code = (r.get("확정_page_type") or "").strip().upper()
        codes_by_url[url].append(code)

    confirm_by_url = {}
    exclude_urls = set()
    for url, codes in codes_by_url.items():
        nonblank = [c for c in codes if c]
        if not nonblank:
            exclude_urls.add(url)  # 전부 공백 -> 경희대 menuNo 중복/에러페이지류
            continue
        majority_code, _ = Ctr(nonblank).most_common(1)[0]
        if majority_code in VALID_CODES:
            confirm_by_url[url] = majority_code
        else:  # 제외
            exclude_urls.add(url)

    print(f"확정 코드 적용 대상: {len(confirm_by_url)}건")
    print(f"제외 대상(명시적 '제외' + 빈칸/경희대 중복페이지): {len(exclude_urls)}건")

    wb = openpyxl.load_workbook(CLASSIFIED, data_only=True)
    ws = wb["page_type_classified"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    if "excluded" not in headers:
        headers = headers + ["excluded"]
    rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]

    changed, excluded_applied = 0, 0
    for r in rows:
        url = r.get("url")
        r["excluded"] = r.get("excluded") or False
        if url in confirm_by_url:
            new_code = confirm_by_url[url]
            if r.get("page_type") != new_code:
                changed += 1
            r["page_type"] = new_code
            r["classify_source"] = "human_reviewed"
            r["needs_review_priority"] = False
        elif url in exclude_urls:
            r["excluded"] = True
            r["classify_source"] = "human_excluded"
            r["needs_review_priority"] = False
            excluded_applied += 1

    print(f"실제 라벨 변경(모델 예측과 달랐던 건): {changed}건")
    print(f"실제 제외 적용된 행: {excluded_applied}건 (중복 URL로 여러 entity에 걸쳐있으면 review 대상 건수보다 많을 수 있음)")

    dist = Counter(r["page_type"] for r in rows if not r["excluded"] and r.get("page_type"))
    print("최종 page_type 분포(제외 반영):", dist)

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "page_type_classified"
    ws2.append(headers)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    excl_fill = PatternFill("solid", fgColor="F4CCCC")
    reviewed_fill = PatternFill("solid", fgColor="C6E0B4")
    for c in range(1, len(headers) + 1):
        cell = ws2.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for r in rows:
        row = [sanitize(r.get(h, "")) for h in headers]
        ws2.append(row)
        if r.get("excluded"):
            for c in range(1, len(headers) + 1):
                ws2.cell(row=ws2.max_row, column=c).fill = excl_fill
        elif r.get("classify_source") == "human_reviewed":
            for c in range(1, len(headers) + 1):
                ws2.cell(row=ws2.max_row, column=c).fill = reviewed_fill
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = ws2.dimensions
    wb2.save(CLASSIFIED)
    print("갱신 저장:", CLASSIFIED)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "apply"),
                        help="prepare: 수동검토표 생성, apply: 확정 라벨 반영")
    args = parser.parse_args()
    if args.stage == "prepare":
        build_review_package()
    else:
        apply_manual_review()


if __name__ == "__main__":
    main()
