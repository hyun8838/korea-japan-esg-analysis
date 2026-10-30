# -*- coding: utf-8 -*-
"""최종 분류 데이터 로드와 수동검토 확정 라벨 반영 공통 유틸."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

# 크롤링 실패/비-HTML 콘텐츠 표시 마커 — 실제 페이지 본문이 아니므로 분석 대상에서 제외
_ERROR_MARKER_RE = re.compile(r"^\[(fetch failed|non-html|error)", re.IGNORECASE)
MIN_CLEANED_CHARS = 200  # japan/preprocessing/step1_preprocess.ipynb의 MIN_CHARS=200과 동일 기준


def load_final_dataset(dataset_path: Path = cfg.FINAL_DATASET_XLSX) -> pd.DataFrame:
    df = pd.read_excel(dataset_path, sheet_name="esg_final_dataset")
    df["lang"] = df["entity_id"].apply(cfg.entity_lang)
    return df


def drop_invalid_documents(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """본문 없음(null)/크롤링 실패 마커 행 제거. 길이 기준은 텍스트 정제 이후 각 언어 전처리 스크립트에서 별도 적용."""
    n_before = len(df)
    is_null = df["full_text"].isna()
    is_error = df["full_text"].astype(str).str.match(_ERROR_MARKER_RE)
    valid = df[~(is_null | is_error)].copy()
    if verbose:
        print(
            f"[빈 문서/에러마커 제거] {n_before}건 → {len(valid)}건 "
            f"(null {int(is_null.sum())}건 + 에러마커 {int(is_error.sum())}건 제외, "
            f"중복 제외분 있으면 합계보다 적을 수 있음)"
        )
    return valid


def load_review_overrides(review_path: Path = cfg.REVIEW_XLSX) -> pd.DataFrame:
    """확정_page_type이 채워진 행만 반환. (entity_id, url) 조합이 병합 키.
    수동검토 입력이 대소문자 혼용될 수 있어 대문자로 정규화 ('제외'는 한글이라 영향 없음)."""
    rev = pd.read_excel(review_path, sheet_name="저신뢰_수동검토")
    filled = rev[rev["확정_page_type"].notna()].copy()
    filled["확정_page_type"] = filled["확정_page_type"].astype(str).str.strip().str.upper()
    filled = filled[["entity_id", "url", "확정_page_type"]].drop_duplicates(
        subset=["entity_id", "url"]
    )
    return filled


def apply_review_overrides(
    df: pd.DataFrame,
    dataset_path: Path = cfg.FINAL_DATASET_XLSX,
    review_path: Path = cfg.REVIEW_XLSX,
) -> pd.DataFrame:
    """page_type_reviewed 열 추가. 확정_page_type 있으면 그 값, 없으면 원래 page_type 사용.
    이후 분석은 page_type_reviewed 기준."""
    df = df.copy()
    overrides = load_review_overrides(review_path)
    n_total_flagged = None
    if review_path.exists():
        rev_all = pd.read_excel(review_path, sheet_name="저신뢰_수동검토")
        n_total_flagged = len(rev_all)

    df = df.merge(
        overrides.rename(columns={"확정_page_type": "_confirmed_page_type"}),
        on=["entity_id", "url"],
        how="left",
    )
    df["page_type_reviewed"] = df["_confirmed_page_type"].fillna(df["page_type"])
    df["review_applied"] = df["_confirmed_page_type"].notna()

    excluded_mask = df["_confirmed_page_type"] == "제외"
    n_excluded = int(excluded_mask.sum())
    df = df[~excluded_mask].copy()
    df = df.drop(columns=["_confirmed_page_type"])

    n_applied = int(df["review_applied"].sum())
    if n_total_flagged is not None:
        print(
            f"[저신뢰 플래그 수동검토 반영] 대상 {n_total_flagged}건 중 "
            f"현재 확정 완료·반영 {n_applied}건 "
            f"({n_applied / n_total_flagged * 100:.1f}%), 그중 '제외' 확정 {n_excluded}건은 "
            f"데이터셋에서 제거 — 나머지는 원래 page_type 유지"
        )
    return df


def add_scope_columns(df: pd.DataFrame, page_type_col: str = "page_type_reviewed") -> pd.DataFrame:
    """all(전체) / main(all 중 I,J 제외) 구분을 위한 is_main 불리언 열 추가."""
    df = df.copy()
    df["is_main"] = ~df[page_type_col].isin(cfg.EXCLUDED_FROM_MAIN_PAGE_TYPES)
    return df


def load_scoped_dataset(
    dataset_path: Path = cfg.FINAL_DATASET_XLSX,
    review_path: Path = cfg.REVIEW_XLSX,
) -> pd.DataFrame:
    """분석 파이프라인 진입점: 데이터 로드 → 빈문서 제거 → 리뷰 반영 → scope 열 추가까지 한 번에."""
    df = load_final_dataset(dataset_path)
    df = drop_invalid_documents(df)
    df = apply_review_overrides(df, dataset_path, review_path)
    df = add_scope_columns(df)
    return df


def filter_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    assert scope in ("main", "all"), f"scope는 'main' 또는 'all'만 허용: {scope!r}"
    if scope == "all":
        return df
    return df[df["is_main"]].copy()


def load_enriched(lang: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """문서와 선택된 키워드 언급을 병합한다."""
    docs = pd.read_csv(cfg.OUTPUT_DIR / f"docs_{lang}.csv", dtype={"doc_id": str})
    mentions = pd.read_csv(cfg.OUTPUT_DIR / f"mentions_{lang}_raw.csv", dtype={"doc_id": str})
    mentions = mentions[mentions["is_selected_longest"]].copy()
    meta_cols = [
        "doc_id", "entity_id", "university", "target_field_primary",
        "page_type_reviewed", "is_main", "full_text_len",
    ]
    merged = mentions.merge(docs[meta_cols], on="doc_id", how="left")
    return merged, docs


def doc_densities(docs: pd.DataFrame, matched_mentions: pd.DataFrame) -> np.ndarray:
    """언급이 없는 문서를 포함해 문서 순서대로 키워드 밀도를 반환한다."""
    counts = matched_mentions.groupby("doc_id").size()
    counts = docs["doc_id"].map(counts).fillna(0).to_numpy()
    lens = docs["full_text_len"].to_numpy()
    return counts / lens
