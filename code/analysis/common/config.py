# -*- coding: utf-8 -*-
"""한일 ESG 비교분석 공통 설정: 경로, page_type/target_field 라벨, scope 정의."""
import os
from pathlib import Path

if not os.environ.get("ESG_RESEARCH_ROOT"):
    raise RuntimeError("ESG_RESEARCH_ROOT 환경변수에 비공개 연구자료 폴더를 지정하세요.")
ESG_ROOT = Path(os.environ["ESG_RESEARCH_ROOT"]).expanduser()
COMP_ROOT = ESG_ROOT / "comp"
PHASE5_DIR = COMP_ROOT / "0_crawling" / "phase5_page_type"

FINAL_DATASET_XLSX = PHASE5_DIR / "esg_final_dataset_v3.xlsx"
REVIEW_XLSX = PHASE5_DIR / "저신뢰플래그_수동검토_v3.xlsx"
UNIV_LIST_XLSX = COMP_ROOT / "file" / "lists" / "univ_list.xlsx"  # 최초 조사대상 대학·학과 설계 리스트

ANALYSIS_ROOT = COMP_ROOT / "analysis"
OUTPUT_DIR = ANALYSIS_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 언어 코드는 kr(한국)/jp(일본)로 통일 (entity_id 접두어 KR-/JP-와 일치)
LANGS = ["kr", "jp"]

# page_type A~J
PAGE_TYPE_LABELS = {
    "A": {"kr": "학과소개", "jp": "学科紹介", "en": "Department Overview"},
    "B": {"kr": "커리큘럼", "jp": "教育課程", "en": "Curriculum"},
    "C": {"kr": "교수/연구", "jp": "教授/研究", "en": "Faculty & Research"},
    "D": {"kr": "학과활동", "jp": "学科活動", "en": "Department Activities"},
    "E": {"kr": "학생/졸업생활동", "jp": "学生・卒業生活動", "en": "Student & Alumni Activities"},
    "F": {"kr": "진로/취업", "jp": "進路/就職", "en": "Career & Employment"},
    "G": {"kr": "관련협회/기관", "jp": "関連協会・機関", "en": "Related Associations & Institutions"},
    "H": {"kr": "국제교류", "jp": "国際交流", "en": "International Exchange"},
    "I": {"kr": "커뮤니티", "jp": "コミュニティー", "en": "Community"},
    "J": {"kr": "기타", "jp": "その他", "en": "Other"},
}
PAGE_TYPE_ORDER = list(PAGE_TYPE_LABELS)  # A..J

# scope 정의: I(커뮤니티)·J(기타)를 제외한 것이 main.
EXCLUDED_FROM_MAIN_PAGE_TYPES = {"I", "J"}


def scope_of(page_type: str) -> str:
    """단일 page_type 값에 대해 'main' 또는 'all_only'(= main 제외 대상)를 반환한다."""
    return "excluded_from_main" if page_type in EXCLUDED_FROM_MAIN_PAGE_TYPES else "main"


# target_field 1~7 (한일 공통 코드)
TARGET_FIELD_LABELS = {
    1: {"kr": "가정교육", "jp": "家政教育", "en": "Home Economics Education"},
    2: {"kr": "생활과학", "jp": "生活科学", "en": "Human Life Science"},
    3: {"kr": "소비자", "jp": "消費者", "en": "Consumer Studies"},
    4: {"kr": "식품영양", "jp": "食品栄養", "en": "Food & Nutrition"},
    5: {"kr": "아동가족", "jp": "児童家族", "en": "Child & Family Studies"},
    6: {"kr": "의류", "jp": "被服", "en": "Clothing & Textiles"},
    7: {"kr": "주거", "jp": "住居", "en": "Housing Studies"},
}
TARGET_FIELD_ORDER = list(TARGET_FIELD_LABELS)  # 1..7

# ESG 카테고리 표시 순서
ESG_CATEGORY_ORDER = ["S", "E", "DUP_ES", "DUP_SG", "DUP_EG", "G", "ESG_META"]
ESG_CATEGORY_LABEL_KR = {
    "S": "S(사회)", "E": "E(환경)", "G": "G(거버넌스)",
    "DUP_ES": "DUP_ES", "DUP_SG": "DUP_SG", "DUP_EG": "DUP_EG",
    "ESG_META": "ESG_META",
}
ESG_CATEGORY_LABEL_EN = {
    "S": "S (Social)", "E": "E (Environmental)", "G": "G (Governance)",
    "DUP_ES": "DUP_ES", "DUP_SG": "DUP_SG", "DUP_EG": "DUP_EG",
    "ESG_META": "ESG_META",
}


def entity_lang(entity_id: str) -> str:
    s = str(entity_id)
    if s.startswith("KR-"):
        return "kr"
    if s.startswith("JP-"):
        return "jp"
    raise ValueError(f"알 수 없는 entity_id 접두어: {entity_id!r}")


