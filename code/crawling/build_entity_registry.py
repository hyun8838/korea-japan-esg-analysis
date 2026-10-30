# -*- coding: utf-8 -*-
"""
대학 목록을 기준으로 학과 Entity Registry를 생성.
remarks에 직접 적어둔 크롤링 지시문을 crawl_instruction으로 구조화해서 함께 저장한다.
"""
import os
from pathlib import Path
import re
import csv
import openpyxl

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

SRC = f"{COMP_ROOT}/file/lists/univ_list.xlsx"
OUT_DIR = f"{COMP_ROOT}/0_crawling/phase1_entity_registry"

wb = openpyxl.load_workbook(SRC, data_only=True)


def ffill(rows, cols):
    last = {c: None for c in cols}
    out = []
    for row in rows:
        row = list(row)
        for c in cols:
            if row[c] is not None and str(row[c]).strip() != "":
                last[c] = row[c]
            else:
                row[c] = last[c]
        out.append(row)
    return out


URL_RE = re.compile(r'https?://[^\s",]+')


def parse_remark(remark):
    """remark 문자열 -> (crawl_instruction, extra_urls, exclude, raw_remark)"""
    r = (remark or "").strip()
    if r in ("", "-"):
        return "normal", [], False, ""
    if "하위 페이지" in r and "제외" in r:
        return "exclude", [], True, r
    if r == "현재 url만 수집":
        return "single_page", [], False, r
    if r.startswith("현재 url와"):
        urls = URL_RE.findall(r)
        return "manual_url_list", urls, False, r
    if r.startswith("특정 링크 하위 페이지는 수집 X") or r.startswith("다음 링크 하위 페이지는 수집 X") or r.startswith("다음 경로로 시작하는 url 하위 페이지 수집 X"):
        urls = URL_RE.findall(r)
        return "hub_with_deny", urls, False, r
    if r.startswith("기타"):
        return "collect_all_then_dedupe", [], False, r
    if r.startswith("수동수집"):
        return "manual_collect", [], False, r
    # 기존 Phase1의 복수 Target Field 통일배정류 메모 등 - 크롤링 지시 아님
    return "normal", [], False, r


def classify_org_level(university, college, department, country):
    dept = (department or "").strip()
    coll = (college or "").strip()
    univ = (university or "").strip()

    if dept == univ and coll == univ:
        return "site_hub"

    grad_markers_kr = ["대학원", "전문대"]
    grad_markers_kr_dept = ["대학원", "박사", "석사"]
    grad_markers_jp = ["大学院", "研究科", "教職大学院"]
    if (country == "KR" and (any(m in coll for m in grad_markers_kr) or any(m in dept for m in grad_markers_kr_dept))) or \
       (country == "JP" and (any(m in coll for m in grad_markers_jp) or any(m in dept for m in grad_markers_jp))):
        return "graduate_program"

    if "연구소" in dept or "研究所" in dept or "研究センター" in dept:
        return "research_unit"

    if "마이크로" in coll or "마이크로" in dept:
        return "micro_major"

    if coll and dept == coll:
        return "faculty"

    if "教員養成課程" in dept and ("家庭科webサイト" in dept or "家庭科Webサイト" in dept):
        return "program"

    if " - " in dept:
        tail = dept.split(" - ")[-1]
        if "コース" in tail:
            return "course"
        if "専攻" in tail:
            return "major"
        if "講座" in tail:
            return "lecture"
        if "類" in tail:
            return "course"
        if "전공" in tail:
            return "major"
        return "course"

    if "전공" in dept or "専攻" in dept:
        return "major"

    return "department"


def build_kr():
    ws = wb["univ_lits_kr"]
    rows = [r for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True) if any(v is not None for v in r)]
    rows = ffill(rows, cols=[0, 1, 2])
    entities = []
    seq = 0
    for r in rows:
        num, university, college, department, target_field, url, remark = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        if department is None:
            continue
        instr, extra_urls, exclude, raw_remark = parse_remark(remark)
        if exclude:
            continue
        seq += 1
        entity_id = f"KR-{seq:04d}"
        org_level = classify_org_level(university, college, department, "KR")
        entities.append({
            "entity_id": entity_id, "country": "KR", "seed_group_num": num,
            "university": university, "college": college,
            "entity_name_original": department, "entity_name_ko": department,
            "org_level": org_level, "parent_entity_id": "",
            "analysis_group_id": f"KR-G{int(num):02d}" if num is not None else None,
            "target_field_primary": target_field, "target_field_secondary": "",
            "url": url, "crawl_instruction": instr,
            "manual_urls": "; ".join(extra_urls), "remarks": raw_remark,
        })
    return entities


def build_jp():
    ws = wb["univ_list_jp_ko"]
    rows = [r for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True) if any(v is not None for v in r)]
    # cols: 0=#,1=Univ원문,2=Univ한글,3=College원문,4=College한글,5=Dept원문,6=Dept한글,7=TF,8=TF한글,9=URL,10=Remarks
    rows = ffill(rows, cols=[0, 1, 2, 3, 4])
    entities = []
    seq = 0
    for r in rows:
        num, univ, univ_ko, coll, coll_ko, dept, dept_ko, tf, tf_ko, url, remark = r
        if dept is None:
            continue
        instr, extra_urls, exclude, raw_remark = parse_remark(remark)
        if exclude:
            continue
        seq += 1
        entity_id = f"JP-{seq:04d}"
        org_level = classify_org_level(univ, coll, dept, "JP")
        entities.append({
            "entity_id": entity_id, "country": "JP", "seed_group_num": num,
            "university": univ, "college": coll,
            "entity_name_original": dept, "entity_name_ko": dept_ko,
            "org_level": org_level, "parent_entity_id": "",
            "analysis_group_id": f"JP-G{int(num):02d}" if num is not None else None,
            "target_field_primary": tf, "target_field_secondary": "",
            "url": url, "crawl_instruction": instr,
            "manual_urls": "; ".join(extra_urls), "remarks": raw_remark,
        })
    return entities


def link_parents(entities):
    faculty_by_uc = {}
    department_exact = {}
    for e in entities:
        key_uc = (e["university"], e["college"])
        if e["org_level"] == "faculty":
            faculty_by_uc[key_uc] = e["entity_id"]
        department_exact[(e["university"], e["college"], e["entity_name_original"])] = e["entity_id"]
    for e in entities:
        dept = e["entity_name_original"]
        key_uc = (e["university"], e["college"])
        if e["org_level"] in ("faculty", "site_hub"):
            continue
        if " - " in dept:
            prefix = dept.split(" - ")[0].strip()
            parent_id = department_exact.get((e["university"], e["college"], prefix))
            if parent_id and parent_id != e["entity_id"]:
                e["parent_entity_id"] = parent_id
                continue
        parent_id = faculty_by_uc.get(key_uc)
        if parent_id and parent_id != e["entity_id"]:
            e["parent_entity_id"] = parent_id
    return entities


def propagate_menuno_group(entities):
    """collect_all_then_dedupe(경로가 menuNo 쿼리스트링으로만 구분되어 자동분리 불가)로 표시된 hub와
    같은 university+college에 속한 형제 행에도 동일 처리를 전파. 자동 중복제거 대신
    "전체 수집 후 수동분류"로 해결."""
    groups_with_menuno = set()
    for e in entities:
        if e["crawl_instruction"] == "collect_all_then_dedupe":
            groups_with_menuno.add((e["university"], e["college"]))
    note = ("menuNo 쿼리스트링으로만 학과가 구분되어 경로 기반 자동 분리 불가. "
            "발견된 하위 URL은 자동 중복제거 대신 수동분류로 어느 학과 소속인지 확정함.")
    for e in entities:
        key = (e["university"], e["college"])
        if key in groups_with_menuno and e["crawl_instruction"] == "normal":
            e["crawl_instruction"] = "collect_all_then_dedupe"
            e["remarks"] = note
    return entities


FIELDS = ["entity_id", "country", "seed_group_num", "university", "college",
          "entity_name_original", "entity_name_ko", "org_level", "parent_entity_id",
          "analysis_group_id", "target_field_primary", "target_field_secondary",
          "url", "crawl_instruction", "manual_urls", "remarks"]


def write_csv(entities, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for e in entities:
            w.writerow(e)


kr = propagate_menuno_group(link_parents(build_kr()))
jp = propagate_menuno_group(link_parents(build_jp()))
write_csv(kr, f"{OUT_DIR}/entity_registry_kr.csv")
write_csv(jp, f"{OUT_DIR}/entity_registry_jp.csv")

from collections import Counter
print("KR entities:", len(kr), "JP entities:", len(jp))
print("KR crawl_instruction:", Counter(e["crawl_instruction"] for e in kr))
print("JP crawl_instruction:", Counter(e["crawl_instruction"] for e in jp))
print("KR org_level:", Counter(e["org_level"] for e in kr))
print("JP org_level:", Counter(e["org_level"] for e in jp))
