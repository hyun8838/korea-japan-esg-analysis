# -*- coding: utf-8 -*-
"""
GT 풀 통합: 현재 A~J 체계의 라벨 280건과 이전 A~I 체계의 라벨 150건 중
중복 없이 현재 코퍼스에 남아 있는 93건을 A~J 체계로 변환해 병합.
이전 H 라벨은 현재 E/I 등으로 갈라져 자동 매핑할 수 없어
14건을 수동으로 확인해 판정했다.
"""
import os
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

ROOT = f"{COMP_ROOT}/0_crawling"
GT_V2 = f"{ROOT}/phase5_page_type/gt_labeling_sample_v2_trafilatura.xlsx"
GT_V1 = f"{ROOT}/old/phase5_page_type/gt_labeling_sample.xlsx"
CLASSIFIED_CORPUS = f"{ROOT}/old_v2/phase5_page_type/page_type_classified_full.xlsx"
OUT = f"{ROOT}/phase5_page_type/gt_pool_v3.xlsx"

# A~I 라벨을 A~J 라벨로 변환 (H 제외, H는 수동 판정)
V1_TO_V2 = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "F", "F": "G", "G": "H", "I": "J"}

def main():
    wb2 = openpyxl.load_workbook(GT_V2, data_only=True)
    ws2 = wb2.active
    h2 = [c.value for c in next(ws2.iter_rows(min_row=1, max_row=1))]
    r2 = [dict(zip(h2, r)) for r in ws2.iter_rows(min_row=2, max_row=ws2.max_row, values_only=True)]
    v2_pool = [
        {"url": r["url"], "title": r.get("title") or "", "snippet": r.get("snippet_trafilatura") or r.get("snippet") or "",
         "gt_label": (r.get("GT_page_type") or "").strip().upper(), "source": "gt_v2"}
        for r in r2 if (r.get("GT_page_type") or "").strip() and (r.get("GT_page_type") or "").strip() != "제외"
    ]
    urls_v2 = set(r["url"] for r in v2_pool)
    print(f"2차 GT: {len(v2_pool)}건")

    wb1 = openpyxl.load_workbook(GT_V1, data_only=True)
    ws1 = wb1.active
    h1 = [c.value for c in next(ws1.iter_rows(min_row=1, max_row=1))]
    r1 = [dict(zip(h1, r)) for r in ws1.iter_rows(min_row=2, max_row=ws1.max_row, values_only=True)]

    wbc = openpyxl.load_workbook(CLASSIFIED_CORPUS, data_only=True)
    wsc = wbc["page_type_classified"]
    hc = [c.value for c in next(wsc.iter_rows(min_row=1, max_row=1))]
    rc = [dict(zip(hc, r)) for r in wsc.iter_rows(min_row=2, max_row=wsc.max_row, values_only=True)]
    by_url_c = {r["url"]: r for r in rc if r.get("url")}

    # 이전 H 라벨은 원문을 확인해 수동 판정
    manual_h_resolution = {
        "교육대학원 공지 - 식품영양학과": "I",
        "대학동정 게시판목록": "I",
        "질문게시판 - 식품공학과": "I",
        "공지사항 | 커뮤니티 | 식품영양학전공": "I",
        "총합Ｑ＆Ａ": "I",
        "総合Ｑ＆Ａ": "I",
        "2026학년도 2학기 학·석사 연계과정생 모집 안내": "I",
        "공지사항 게시판읽기": "I",
        "공지사항 | 커뮤니티 | 의상디자인전공": "I",
        "학부 공지 - 식품영양학과": "I",
        "環境デザイン学科ニュース": "D",
    }

    added, skipped_ambiguous, skipped_unknown = [], 0, 0
    for r in r1:
        url = r.get("url")
        v1_label = (r.get("GT_page_type") or "").strip().upper()
        if not url or url in urls_v2 or url not in by_url_c:
            continue
        if not v1_label or v1_label == "?":
            continue
        cur = by_url_c[url]
        title = cur.get("title") or ""
        snippet = cur.get("snippet_trafilatura") or ""

        if v1_label == "H":
            v2_label = None
            for key, code in manual_h_resolution.items():
                if key in title:
                    v2_label = code
                    break
            if v2_label is None:
                skipped_ambiguous += 1
                continue
        elif v1_label in V1_TO_V2:
            v2_label = V1_TO_V2[v1_label]
        else:
            skipped_unknown += 1
            continue

        added.append({"url": url, "title": title, "snippet": snippet, "gt_label": v2_label, "source": "gt_v1_remap"})

    print(f"1차 GT에서 추가: {len(added)}건 (애매하여 제외 {skipped_ambiguous}건, 기타 제외 {skipped_unknown}건)")

    final_pool = v2_pool + added
    print(f"최종 GT 풀: {len(final_pool)}건")
    from collections import Counter
    print("코드 분포:", Counter(r["gt_label"] for r in final_pool))

    wb_out = openpyxl.Workbook()
    ws_out = wb_out.active
    ws_out.title = "gt_pool_v3"
    fields = ["url", "title", "snippet", "gt_label", "source"]
    ws_out.append(fields)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    for c in range(1, len(fields) + 1):
        cell = ws_out.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for r in final_pool:
        ws_out.append([r[f] for f in fields])
    ws_out.freeze_panes = "A2"
    ws_out.auto_filter.ref = ws_out.dimensions
    wb_out.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
