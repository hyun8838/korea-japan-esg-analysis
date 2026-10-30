# -*- coding: utf-8 -*-
"""빈도(TF-IDF 상위 키워드)·연관분석(edge)·LDA(topic) 공용 원문 샘플 문장 추출 유틸.
표에 실제로 등장하는 항목에 한해 원문 문장 예시를 붙인다. 개수 기준: LDA 토픽당 최대 20개,
연관분석 edge 쌍당 최대 4개, 빈도 키워드당 최대 5개. 대학이 겹치지 않도록 다양성 있게 샘플링.

주의할 점 2가지:
1. 목록형 페이지(교원 CV·논문 목록 등, 마침표 없는 나열형 텍스트)가 문장분리기에서 통째로
   하나의 "문장"으로 묶여 수천~수만 자짜리 비정상 문장이 섞일 수 있음.
2. concept이 여러 별칭(영어·가타카나 등)으로 매칭돼 canonical 이름 자체가 문장에 리터럴로
   없는 채 샘플로 뽑힐 수 있음.
sample_rows의 문장 길이 상한과 keyword_hint 인자로 최대한 피하지만 완전히 막을 순 없으므로,
보고서 작성 시 인용 문장에 실제로 해당 키워드가 보이는지 육안 확인할 것.
"""
import pandas as pd

DOC_META_COLS = [
    "doc_id", "entity_id", "university", "college", "entity_name", "title", "url",
    "target_field_primary", "page_type_reviewed",
]

MAX_SENTENCE_LEN = 300  # 이보다 길면 실제 문장이 아니라 목록형 페이지 통짜 텍스트일 가능성이 큼


def attach_doc_meta(mention_rows: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """mention 행에 대학·URL 등 문서 메타를 붙인다.
    mention_rows가 이미 병합된 결과(university 등 보유)일 수 있어, 또 병합하면 pandas가
    university_x/university_y로 접미사를 붙이는 버그가 생김 — 이미 있는 열은 건드리지 않고
    없는 열만 추가로 병합한다."""
    already = set(mention_rows.columns)
    cols = ["doc_id"] + [c for c in DOC_META_COLS if c in docs.columns and c not in already]
    return mention_rows.merge(docs[cols], on="doc_id", how="left")


def sample_rows(rows: pd.DataFrame, n: int, keyword_hint=None) -> pd.DataFrame:
    """대학이 겹치지 않도록 우선순위를 두고 최대 n건 샘플링.
    keyword_hint를 주면 sentence_text에 그 문자열(들)이 리터럴로 포함된 행을 우선(canonical과
    실제 매칭 별칭이 다른 경우 보완). 리스트면 "전부 포함 > 하나라도 포함 > 무관계" 순으로 완화.
    짧은 문장(MAX_SENTENCE_LEN 이하)도 우선하되, 조건 만족 행이 없으면 원래 후보군 전체로 복귀."""
    if rows.empty or n <= 0:
        return rows.head(0)
    if "sentence_text" in rows.columns:
        short = rows[rows["sentence_text"].str.len() <= MAX_SENTENCE_LEN]
        if len(short) > 0:
            rows = short
        if keyword_hint:
            hints = [keyword_hint] if isinstance(keyword_hint, str) else list(keyword_hint)
            all_present = rows[rows["sentence_text"].apply(lambda s: all(h in s for h in hints))]
            if len(all_present) > 0:
                rows = all_present
            else:
                any_present = rows[rows["sentence_text"].apply(lambda s: any(h in s for h in hints))]
                if len(any_present) > 0:
                    rows = any_present
    if "university" in rows.columns and rows["university"].nunique() > 1:
        picked = rows.groupby("university", group_keys=False).head(1)
        if len(picked) >= n:
            return picked.head(n)
        remaining = rows.drop(picked.index)
        extra = remaining.head(n - len(picked))
        return pd.concat([picked, extra], ignore_index=True)
    return rows.head(n)
