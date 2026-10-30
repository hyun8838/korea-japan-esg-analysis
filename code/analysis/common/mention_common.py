# -*- coding: utf-8 -*-
"""언어 공용 문장 단위 ESG mention 추출 엔진.
언어별 concept 사전, 문장 분리 함수, track 판정 규칙을 입력받는다."""
import re
from dataclasses import dataclass, field
from typing import Callable, List


def build_alias_pool(concept_dict: dict, fp_exclude: set, track_of) -> list:
    """(alias, canonical, category, track) 리스트. 길이 내림차순 정렬(긴 표현 우선 매칭)."""
    pool = []
    seen = set()
    for concept in concept_dict["concepts"]:
        canonical = concept["canonical"]
        category = concept["category"]
        for alias in concept["aliases"]:
            if alias in fp_exclude:
                continue
            key = (alias.lower(), canonical)
            if key in seen:
                continue
            seen.add(key)
            pool.append((alias, canonical, category, track_of(alias)))
    return sorted(pool, key=lambda x: len(x[0]), reverse=True)


def _alias_pattern(alias: str):
    if re.fullmatch(r"[A-Za-z]+(?:[-\s][A-Za-z0-9]+)*", alias):
        return re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
    return re.compile(re.escape(alias), re.IGNORECASE)


def collect_mentions_in_sentence(doc_id, sent_id: int, sent_text: str, alias_pool: list) -> list:
    mentions = []
    for alias, canonical, category, track in alias_pool:
        for m in _alias_pattern(alias).finditer(sent_text):
            mentions.append({
                "doc_id": doc_id, "sent_id": sent_id, "sentence_text": sent_text,
                "surface_text": m.group(0), "alias": alias, "canonical": canonical,
                "category": category, "track": track,
                "start_char": m.start(), "end_char": m.end(),
                "is_selected_longest": False, "suppressed_by": "",
            })
    return mentions


def apply_longest_match(mentions: list) -> list:
    """같은 문장 내 겹치는 span 중 가장 긴 것만 is_selected_longest=True로 표시."""
    if not mentions:
        return mentions
    ordered = sorted(mentions, key=lambda x: (-(x["end_char"] - x["start_char"]), x["start_char"]))
    selected_spans = []  # (start, end, alias)
    for m in ordered:
        s, e = m["start_char"], m["end_char"]
        overlap_with = None
        for ss, se, sa in selected_spans:
            if not (e <= ss or s >= se):
                overlap_with = sa
                break
        if overlap_with is None:
            m["is_selected_longest"] = True
            selected_spans.append((s, e, m["alias"]))
        else:
            m["is_selected_longest"] = False
            m["suppressed_by"] = overlap_with
    return ordered


def extract_doc_mentions(doc_id, sentences: List[str], alias_pool: list) -> list:
    """문서 한 건의 문장 리스트 → longest-match 적용된 mention row 리스트."""
    rows = []
    for sent_id, sent in enumerate(sentences):
        raw = collect_mentions_in_sentence(doc_id, sent_id, sent, alias_pool)
        rows.extend(apply_longest_match(raw))
    return rows
