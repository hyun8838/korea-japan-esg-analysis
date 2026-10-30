# -*- coding: utf-8 -*-
import os
from pathlib import Path
import csv
from urllib.parse import urlsplit
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

REG_DIR = f"{COMP_ROOT}/0_crawling/phase1_entity_registry"
OUT_XLSX = f"{COMP_ROOT}/0_crawling/phase2_seed_registry/seed_registry.xlsx"

PDF_PAGE_TYPES = "A,B,E,G"


def path_prefix(url, is_hub):
    parts = urlsplit(url)
    segments = [s for s in parts.path.split("/") if s]
    looks_like_file = bool(segments) and "." in segments[-1]
    dir_segments = segments[:-1] if looks_like_file else segments
    if is_hub and dir_segments:
        dir_segments = dir_segments[:-1]
    prefix = "/" + "/".join(dir_segments)
    if not prefix.endswith("/"):
        prefix += "/"
    return parts.netloc, prefix


def load_entities(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def build_seeds(entities, id_prefix):
    seeds = []
    seq = 0
    for e in entities:
        seq += 1
        is_hub = e["org_level"] in ("faculty", "site_hub")
        instr = e["crawl_instruction"]
        if instr == "collect_all_then_dedupe":
            # menuNo 등 쿼리스트링으로만 구분되는 경우 path 기준 좁히기가 무의미 -
            # 도메인 전체를 범위로 잡아 넓게 수집 후 수동분류
            domain = urlsplit(e["url"]).netloc
            prefix = "/"
        else:
            # 허브 경로를 확장하면 다른 단과대가 섞일 수 있어 seed_url 경로로 제한
            domain, prefix = path_prefix(e["url"], is_hub=False)

        deny_extra = []
        notes = []

        if instr == "manual_collect":
            seed_role, crawl_mode, max_depth = "manual", "manual_collect", 0
            notes.append("수동 수집 대상 - 자동 크롤링 제외")
        elif instr == "single_page":
            seed_role, crawl_mode, max_depth = "single_page", "single_page", 0
            notes.append("이 페이지 1건만 수집 (하위 확장 없음)")
        elif instr == "manual_url_list":
            seed_role, crawl_mode, max_depth = "primary", "manual_url_list", 0
            notes.append("seed_url + manual_urls 목록에 있는 URL만 수집")
        elif instr == "hub_with_deny":
            seed_role, crawl_mode, max_depth = "hub", "hub_children", 2
            deny_extra = [u.strip() for u in e["manual_urls"].split(";") if u.strip()]
            notes.append("하위 수집하되 manual_urls에 명시된 경로는 제외(형제 학과 중복 방지)")
        elif instr == "collect_all_then_dedupe":
            seed_role, crawl_mode, max_depth = "hub", "site", 3
            notes.append("경로/쿼리스트링으로 학과 구분 불가 - 전체 수집 후 canonical URL/본문해시로 사후 중복제거 필요. "
                          "같은 도메인의 형제 학과 행은 중복 여부를 별도 확인")
        elif is_hub:
            seed_role, crawl_mode, max_depth = "hub", "hub_children", 2
        elif e["parent_entity_id"]:
            seed_role, crawl_mode, max_depth = "child", "section", 3
        else:
            seed_role, crawl_mode, max_depth = "primary", "section", 3

        manual_review_required = instr in ("collect_all_then_dedupe",)

        # 개별 사이트 구조로 인해 경로를 별도 확인해야 하는 사례
        SITE_PATH_ISSUES = {
            "KR-0032": "포털형 CMS의 실제 콘텐츠가 /-1, /-10 같은 숫자 slug 형태로 루트 바로 아래 존재하며 "
                       "/home/ 하위 경로에 포함되지 않음.",
            "JP-0023": "실제 커리큘럼 페이지가 .../food_and_nutritional_sciences/currriculum/nutritional_science/... "
                       "경로에 있어 seed 자체 디렉토리(.../nutritional_science/) 밖에 위치.",
        }
        if e["entity_id"] in SITE_PATH_ISSUES:
            manual_review_required = True
            notes.append(SITE_PATH_ISSUES[e["entity_id"]])

        base_deny = ["login", "search", "calendar", "print", "/en/", "/english/", "/eng/"]
        deny_patterns = "|".join(base_deny + deny_extra) if deny_extra else "|".join(base_deny)

        seeds.append({
            "seed_id": f"{id_prefix}-S{seq:04d}",
            "entity_id": e["entity_id"],
            "university": e["university"],
            "college": e["college"],
            "entity_name_original": e["entity_name_original"],
            "org_level": e["org_level"],
            "seed_url": e["url"],
            "manual_urls": e["manual_urls"],
            "crawl_instruction": instr,
            "seed_role": seed_role,
            "crawl_mode": crawl_mode,
            "allowed_domains": domain,
            "allowed_path_prefix": prefix,
            "deny_patterns": deny_patterns,
            "max_depth": max_depth,
            "include_pdf": True,
            "pdf_page_types_only": PDF_PAGE_TYPES,
            "page_type_hint": "A",
            "manual_review_required": manual_review_required,
            "notes": " / ".join(notes),
        })
    return seeds


FIELDS = ["seed_id", "entity_id", "university", "college", "entity_name_original", "org_level",
          "seed_url", "manual_urls", "crawl_instruction", "seed_role", "crawl_mode",
          "allowed_domains", "allowed_path_prefix", "deny_patterns", "max_depth",
          "include_pdf", "pdf_page_types_only", "page_type_hint",
          "manual_review_required", "notes"]

HEADER_KO = {
    "seed_id": "seed_id", "entity_id": "entity_id (Phase1 연결)", "university": "대학",
    "college": "학부/College", "entity_name_original": "학과/전공/과정명", "org_level": "org_level",
    "seed_url": "seed_url (root URL)", "manual_urls": "manual_urls (추가/예외 URL)",
    "crawl_instruction": "crawl_instruction (담당자 지정)",
    "seed_role": "seed_role", "crawl_mode": "crawl_mode",
    "allowed_domains": "allowed_domains", "allowed_path_prefix": "allowed_path_prefix (제안값)",
    "deny_patterns": "deny_patterns", "max_depth": "max_depth (제안값)",
    "include_pdf": "include_pdf (수집은 함)",
    "pdf_page_types_only": "PDF 최종포함 page_type 한정 (후처리필터)",
    "page_type_hint": "page_type_hint (제안값)",
    "manual_review_required": "manual_review_required", "notes": "비고",
}

kr_entities = load_entities(f"{REG_DIR}/entity_registry_kr.csv")
jp_entities = load_entities(f"{REG_DIR}/entity_registry_jp.csv")
kr_seeds = build_seeds(kr_entities, "KR")
jp_seeds = build_seeds(jp_entities, "JP")

wb = openpyxl.Workbook()
wb.remove(wb.active)

header_fill = PatternFill("solid", fgColor="D9E1F2")
review_fill = PatternFill("solid", fgColor="FFF2CC")
dup_fill = PatternFill("solid", fgColor="FCE4D6")
manual_fill = PatternFill("solid", fgColor="D9D2E9")

for sheet_name, seeds in [("seed_registry_kr", kr_seeds), ("seed_registry_jp", jp_seeds)]:
    ws = wb.create_sheet(sheet_name)
    ws.append([HEADER_KO[f] for f in FIELDS])
    for c in range(1, len(FIELDS) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for s in seeds:
        row = [s[f] for f in FIELDS]
        ws.append(row)
        r = ws.max_row
        if s["crawl_mode"] == "manual_collect":
            fill = manual_fill
        elif s["crawl_mode"] == "single_page":
            fill = dup_fill
        elif s["manual_review_required"]:
            fill = review_fill
        else:
            fill = None
        if fill:
            for c in range(1, len(FIELDS) + 1):
                ws.cell(row=r, column=c).fill = fill
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    widths = [12, 14, 16, 16, 26, 12, 46, 30, 20, 11, 11, 22, 30, 24, 10, 12, 16, 12, 18, 60]
    for i, w in enumerate(widths[:len(FIELDS)], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

summary = wb.create_sheet("요약", 0)
summary["A1"] = "Seed Registry 요약"
summary["A1"].font = Font(bold=True, size=13)
from collections import Counter
kr_c = Counter(s["crawl_mode"] for s in kr_seeds)
jp_c = Counter(s["crawl_mode"] for s in jp_seeds)
rows_info = [
    ("KR seed 수", len(kr_seeds)), ("JP seed 수", len(jp_seeds)), ("", ""),
    ("KR crawl_mode 분포", dict(kr_c)), ("JP crawl_mode 분포", dict(jp_c)), ("", ""),
    ("범례", ""),
    ("주황 배경", "single_page - 이 페이지 1건만 수집"),
    ("보라 배경", "manual_collect - 자동크롤링 제외, 사용자 직접 수집"),
    ("노랑 배경", "manual_review_required - 경로로 학과 구분 불가, 전체수집 후 사후 중복제거 필요"),
    ("", ""),
    ("경로 주의", "같은 도메인의 형제 학과는 수집 후 중복 여부를 확인"),
]
for i, item in enumerate(rows_info, start=2):
    a, b = item
    summary.cell(row=i, column=1, value=a)
    summary.cell(row=i, column=2, value=str(b))
summary.column_dimensions["A"].width = 20
summary.column_dimensions["B"].width = 90

wb.save(OUT_XLSX)
print("saved:", OUT_XLSX)
print("KR seeds:", len(kr_seeds), "JP seeds:", len(jp_seeds))
