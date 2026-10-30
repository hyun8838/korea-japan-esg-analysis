# -*- coding: utf-8 -*-
"""B. page_type별 ESG concept 특징어 TF-IDF.

ESG concept 단위(전체어휘 아님)로 문서별 TF-IDF를 계산한 뒤,
page_type별로 묶어 평균(mean_tfidf)낸다. 표 구조는 keyword/category/raw_count/doc_freq/mean_tfidf.

주의: 한국어·일본어 어휘집합이 달라 직접 통계비교가 아니라 정성적 병렬대조.
상위 키워드마다 원문 문장 예시(최대 5개, common/sample_text.py 기준)를 함께 산출.
"""
import sys

import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common import sample_text
from common.io_common import load_enriched

SAMPLES_PER_KEYWORD = 5


def build_pagetype_tfidf(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path, top_n: int = 15):
    d_main = docs[docs["is_main"]].copy()
    m_main = merged[merged["is_main"]]

    doc_tokens = m_main.groupby("doc_id")["canonical"].apply(list)
    doc_tokens = doc_tokens.reindex(d_main["doc_id"]).apply(lambda x: x if isinstance(x, list) else [])
    doc_ids = doc_tokens.index.tolist()

    non_empty_idx = [i for i, toks in enumerate(doc_tokens) if len(toks) > 0]
    if not non_empty_idx:
        empty = pd.DataFrame(columns=["page_type", "label", "keyword", "category", "raw_count", "doc_freq", "mean_tfidf"])
        return empty, empty.head(0)

    vec = TfidfVectorizer(analyzer=lambda x: x, lowercase=False)
    tfidf_matrix = vec.fit_transform([doc_tokens.iloc[i] for i in non_empty_idx])
    # sklearn 0.24대(get_feature_names_out 미지원)/1.0+ 양쪽 호환(환경 버전 고정 문제, wordcloud/Pillow와 동일 성격).
    terms = vec.get_feature_names_out() if hasattr(vec, "get_feature_names_out") else vec.get_feature_names()

    # sparse DataFrame에서 (sub > 0).sum(axis=0)이 카운트가 아니라 불리언을 반환하는
    # pandas sparse dtype 버그성 동작이 있어 dense로 변환해서 사용 (ESG concept 어휘만 대상이라 메모리 문제 없음)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), index=[doc_ids[i] for i in non_empty_idx], columns=terms
    )

    category_of = m_main.drop_duplicates("canonical").set_index("canonical")["category"].to_dict()
    d_main_indexed = d_main.set_index("doc_id")

    rows = []
    sample_rows_all = []
    page_types_in_scope = [pt for pt in cfg.PAGE_TYPE_ORDER if pt not in cfg.EXCLUDED_FROM_MAIN_PAGE_TYPES]
    for pt in page_types_in_scope:
        pt_doc_ids = d_main_indexed.index[d_main_indexed["page_type_reviewed"] == pt]
        pt_doc_ids = [d for d in pt_doc_ids if d in tfidf_df.index]
        if not pt_doc_ids:
            continue
        sub = tfidf_df.loc[pt_doc_ids]
        mean_tfidf = sub.mean(axis=0)
        doc_freq = (sub > 0).sum(axis=0)
        pt_mentions = m_main[m_main["doc_id"].isin(pt_doc_ids)]
        raw_count = pt_mentions["canonical"].value_counts()

        top_terms = mean_tfidf.sort_values(ascending=False).head(top_n)
        for term, score in top_terms.items():
            if score <= 0:
                continue
            rows.append({
                "page_type": pt, "label": cfg.PAGE_TYPE_LABELS[pt][lang],
                "label_en": cfg.PAGE_TYPE_LABELS[pt]["en"],
                "keyword": term, "category": category_of.get(term, ""),
                "raw_count": int(raw_count.get(term, 0)), "doc_freq": int(doc_freq.get(term, 0)),
                "mean_tfidf": round(float(score), 5),
            })
            # 원문 샘플: 이 page_type 안에서 이 키워드(canonical)가 등장한 문장 중 최대 N개
            kw_mentions = pt_mentions[pt_mentions["canonical"] == term]
            kw_mentions = sample_text.attach_doc_meta(kw_mentions, docs)
            picked = sample_text.sample_rows(kw_mentions, SAMPLES_PER_KEYWORD, keyword_hint=term)
            for _, r in picked.iterrows():
                sample_rows_all.append({
                    "page_type": pt, "label_en": cfg.PAGE_TYPE_LABELS[pt]["en"],
                    "keyword": term, "category": category_of.get(term, ""),
                    "university": r.get("university", ""), "title": r.get("title", ""),
                    "url": r.get("url", ""), "sentence": r.get("sentence_text", ""),
                })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"B_page_type별_TFIDF_{lang}.csv", index=False, encoding="utf-8-sig")
    samples = pd.DataFrame(sample_rows_all)
    samples.to_csv(out_dir / f"B_page_type별_TFIDF_샘플_{lang}.csv", index=False, encoding="utf-8-sig")
    return table, samples


def build_field_tfidf(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path,
                       top_n: int = 15, min_docs: int = 30):
    """D. target_field(계열)별 ESG concept 특징어 TF-IDF. build_pagetype_tfidf와 동일 방법론,
    page_type 대신 target_field_primary로 묶는다. association.py의 계열별 분석과 동일하게
    문서 30건 미만 계열은 스킵."""
    d_main = docs[docs["is_main"]].copy()
    m_main = merged[merged["is_main"]]

    doc_tokens = m_main.groupby("doc_id")["canonical"].apply(list)
    doc_tokens = doc_tokens.reindex(d_main["doc_id"]).apply(lambda x: x if isinstance(x, list) else [])
    doc_ids = doc_tokens.index.tolist()

    non_empty_idx = [i for i, toks in enumerate(doc_tokens) if len(toks) > 0]
    if not non_empty_idx:
        empty = pd.DataFrame(columns=["target_field", "label", "keyword", "category", "raw_count", "doc_freq", "mean_tfidf"])
        return empty, empty.head(0)

    vec = TfidfVectorizer(analyzer=lambda x: x, lowercase=False)
    tfidf_matrix = vec.fit_transform([doc_tokens.iloc[i] for i in non_empty_idx])
    terms = vec.get_feature_names_out() if hasattr(vec, "get_feature_names_out") else vec.get_feature_names()
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), index=[doc_ids[i] for i in non_empty_idx], columns=terms
    )

    category_of = m_main.drop_duplicates("canonical").set_index("canonical")["category"].to_dict()
    d_main_indexed = d_main.set_index("doc_id")

    rows = []
    sample_rows_all = []
    for tf in cfg.TARGET_FIELD_ORDER:
        tf_doc_ids_all = d_main_indexed.index[d_main_indexed["target_field_primary"] == tf]
        n_docs = len(tf_doc_ids_all)
        if n_docs < min_docs:
            print(f"[{lang}] D. target_field={tf}({cfg.TARGET_FIELD_LABELS[tf][lang]}): 문서 {n_docs}건 < {min_docs}건 → 스킵")
            continue
        tf_doc_ids = [d for d in tf_doc_ids_all if d in tfidf_df.index]
        if not tf_doc_ids:
            continue
        sub = tfidf_df.loc[tf_doc_ids]
        mean_tfidf = sub.mean(axis=0)
        doc_freq = (sub > 0).sum(axis=0)
        tf_mentions = m_main[m_main["doc_id"].isin(tf_doc_ids)]
        raw_count = tf_mentions["canonical"].value_counts()

        top_terms = mean_tfidf.sort_values(ascending=False).head(top_n)
        for term, score in top_terms.items():
            if score <= 0:
                continue
            rows.append({
                "target_field": tf, "label": cfg.TARGET_FIELD_LABELS[tf][lang],
                "label_en": cfg.TARGET_FIELD_LABELS[tf]["en"],
                "keyword": term, "category": category_of.get(term, ""),
                "raw_count": int(raw_count.get(term, 0)), "doc_freq": int(doc_freq.get(term, 0)),
                "mean_tfidf": round(float(score), 5),
            })
            kw_mentions = tf_mentions[tf_mentions["canonical"] == term]
            kw_mentions = sample_text.attach_doc_meta(kw_mentions, docs)
            picked = sample_text.sample_rows(kw_mentions, SAMPLES_PER_KEYWORD, keyword_hint=term)
            for _, r in picked.iterrows():
                sample_rows_all.append({
                    "target_field": tf, "label_en": cfg.TARGET_FIELD_LABELS[tf]["en"],
                    "keyword": term, "category": category_of.get(term, ""),
                    "university": r.get("university", ""), "title": r.get("title", ""),
                    "url": r.get("url", ""), "sentence": r.get("sentence_text", ""),
                })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"D_계열별_TFIDF_{lang}.csv", index=False, encoding="utf-8-sig")
    samples = pd.DataFrame(sample_rows_all)
    samples.to_csv(out_dir / f"D_계열별_TFIDF_샘플_{lang}.csv", index=False, encoding="utf-8-sig")
    return table, samples


ESG_CATEGORY_MEMBERSHIP = {
    "S": {"S", "DUP_ES", "DUP_SG"},
    "E": {"E", "DUP_ES", "DUP_EG"},
    "G": {"G", "DUP_SG", "DUP_EG"},
}  # frequency.py ESG_SUBCATEGORY_MEMBERSHIP과 동일 정의, DUP는 관련 범주 모두에 중복 포함


def build_esg_category_tfidf(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path, top_n: int = 15):
    """F. ESG 범주(E/S/G)별 특징어 TF-IDF.
    page_type/target_field는 문서 속성이라 문서를 그룹으로 나눠 계산했지만, ESG 범주(S/E/G)는
    키워드(concept) 자체의 속성이라 전체 코퍼스로 TF-IDF를 한 번 계산한 뒤 키워드 자기 범주
    기준으로 재정렬해 범주별 top_n을 뽑는다(B/D와 동일 TfidfVectorizer, 그룹 기준만 다름).
    G(+DUP_SG+DUP_EG) 범주는 키워드 자체가 적어(한국 11개, 일본 7개) top_n을 못 채우거나
    표본이 얇을 수 있음 — 해석 시 주의."""
    d_main = docs[docs["is_main"]].copy()
    m_main = merged[merged["is_main"]]

    doc_tokens = m_main.groupby("doc_id")["canonical"].apply(list)
    doc_tokens = doc_tokens.reindex(d_main["doc_id"]).apply(lambda x: x if isinstance(x, list) else [])
    doc_ids = doc_tokens.index.tolist()

    non_empty_idx = [i for i, toks in enumerate(doc_tokens) if len(toks) > 0]
    if not non_empty_idx:
        empty = pd.DataFrame(columns=["esg_category", "keyword", "category", "raw_count", "doc_freq", "mean_tfidf"])
        return empty, empty.head(0)

    vec = TfidfVectorizer(analyzer=lambda x: x, lowercase=False)
    tfidf_matrix = vec.fit_transform([doc_tokens.iloc[i] for i in non_empty_idx])
    terms = vec.get_feature_names_out() if hasattr(vec, "get_feature_names_out") else vec.get_feature_names()
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), index=[doc_ids[i] for i in non_empty_idx], columns=terms
    )

    category_of = m_main.drop_duplicates("canonical").set_index("canonical")["category"].to_dict()

    # 전체 코퍼스(단일 그룹) 기준 mean TF-IDF·doc_freq·raw_count — B/D와 동일 계산이나 그룹 분할 없음
    mean_tfidf_all = tfidf_df.mean(axis=0)
    doc_freq_all = (tfidf_df > 0).sum(axis=0)
    raw_count_all = m_main["canonical"].value_counts()

    rows = []
    sample_rows_all = []
    for esg_cat, member_cats in ESG_CATEGORY_MEMBERSHIP.items():
        cat_terms = [t for t in mean_tfidf_all.index if category_of.get(t, "") in member_cats]
        cat_scores = mean_tfidf_all.loc[cat_terms].sort_values(ascending=False)
        top_terms = cat_scores.head(top_n)
        for term, score in top_terms.items():
            if score <= 0:
                continue
            rows.append({
                "esg_category": esg_cat,
                "keyword": term, "category": category_of.get(term, ""),
                "raw_count": int(raw_count_all.get(term, 0)), "doc_freq": int(doc_freq_all.get(term, 0)),
                "mean_tfidf": round(float(score), 5),
            })
            kw_mentions = m_main[m_main["canonical"] == term]
            kw_mentions = sample_text.attach_doc_meta(kw_mentions, docs)
            picked = sample_text.sample_rows(kw_mentions, SAMPLES_PER_KEYWORD, keyword_hint=term)
            for _, r in picked.iterrows():
                sample_rows_all.append({
                    "esg_category": esg_cat,
                    "keyword": term, "category": category_of.get(term, ""),
                    "university": r.get("university", ""), "title": r.get("title", ""),
                    "url": r.get("url", ""), "sentence": r.get("sentence_text", ""),
                })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"F_ESG범주별_TFIDF_{lang}.csv", index=False, encoding="utf-8-sig")
    samples = pd.DataFrame(sample_rows_all)
    samples.to_csv(out_dir / f"F_ESG범주별_TFIDF_샘플_{lang}.csv", index=False, encoding="utf-8-sig")
    return table, samples


def build_stage2_tfidf_report(lang: str):
    out_dir = cfg.OUTPUT_DIR / lang / "단계2_빈도분석"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged, docs = load_enriched(lang)
    table, samples = build_pagetype_tfidf(merged, docs, lang, out_dir)
    print(f"[{lang}] B. page_type별 TF-IDF 저장 완료: {out_dir} ({len(table)}행, 샘플 {len(samples)}건)")
    field_table, field_samples = build_field_tfidf(merged, docs, lang, out_dir)
    print(f"[{lang}] D. 계열별 TF-IDF 저장 완료: {out_dir} ({len(field_table)}행, 샘플 {len(field_samples)}건)")
    esg_table, esg_samples = build_esg_category_tfidf(merged, docs, lang, out_dir)
    print(f"[{lang}] F. ESG 범주별 TF-IDF 저장 완료: {out_dir} ({len(esg_table)}행, 샘플 {len(esg_samples)}건)")
    return table


TOP_N = 15  # 보고서 표에 실린 상위 개수(tfidf.py의 top_n과 동일해야 함)
OUT = cfg.OUTPUT_DIR / "significance" / "TFIDF_rank_vs_rawcount_spearman.csv"


def _doc_term_tfidf(merged: pd.DataFrame, docs: pd.DataFrame):
    """tfidf.py의 build_*_tfidf 앞부분과 동일한 문서×concept TF-IDF 행렬."""
    d_main = docs[docs["is_main"]].copy()
    m_main = merged[merged["is_main"]]

    doc_tokens = m_main.groupby("doc_id")["canonical"].apply(list)
    doc_tokens = doc_tokens.reindex(d_main["doc_id"]).apply(lambda x: x if isinstance(x, list) else [])
    doc_ids = doc_tokens.index.tolist()
    non_empty_idx = [i for i, toks in enumerate(doc_tokens) if len(toks) > 0]

    vec = TfidfVectorizer(analyzer=lambda x: x, lowercase=False)
    mat = vec.fit_transform([doc_tokens.iloc[i] for i in non_empty_idx])
    terms = vec.get_feature_names_out() if hasattr(vec, "get_feature_names_out") else vec.get_feature_names()
    tfidf_df = pd.DataFrame(mat.toarray(), index=[doc_ids[i] for i in non_empty_idx], columns=terms)
    return tfidf_df, d_main, m_main


def _row(lang, table, group, group_label, scope, tfidf_s: pd.Series, raw_s: pd.Series) -> dict:
    """tfidf_s / raw_s: 같은 index(concept)를 가진 두 값 시리즈."""
    terms = list(tfidf_s.index)
    x = tfidf_s.reindex(terms).astype(float).to_numpy()
    y = raw_s.reindex(terms).fillna(0).astype(float).to_numpy()
    if len(terms) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        rho, p = np.nan, np.nan
    else:
        rho, p = spearmanr(x, y)
    top_t = set(pd.Series(x, index=terms).sort_values(ascending=False).head(TOP_N).index)
    top_r = set(pd.Series(y, index=terms).sort_values(ascending=False).head(TOP_N).index)
    return {
        "lang": lang, "table": table, "group": group, "group_label": group_label,
        "scope": scope, "n_terms": len(terms),
        "spearman_rho": round(float(rho), 4) if rho == rho else None,
        "p_value": (f"{p:.3e}" if p == p else None),
        f"top{TOP_N}_overlap_n": len(top_t & top_r),
        f"top{TOP_N}_overlap_pct": round(len(top_t & top_r) / min(TOP_N, len(terms)) * 100, 1),
    }


def _both_scopes(rows, lang, table, group, group_label, tfidf_all: pd.Series, raw_all: pd.Series):
    """present_in_group(절단 없음)과 published_top15(보고서 표) 두 scope를 모두 추가."""
    present = [t for t in tfidf_all.index if raw_all.get(t, 0) > 0]
    t_present = tfidf_all.loc[present]
    rows.append(_row(lang, table, group, group_label, "present_in_group", t_present, raw_all))
    top = t_present.sort_values(ascending=False).head(TOP_N)
    top = top[top > 0]
    rows.append(_row(lang, table, group, group_label, f"published_top{TOP_N}", top, raw_all))


def build_rank_correlation_report():
    rows = []
    vocab_note = {}
    for lang in cfg.LANGS:
        merged, docs = load_enriched(lang)
        tfidf_df, d_main, m_main = _doc_term_tfidf(merged, docs)
        category_of = m_main.drop_duplicates("canonical").set_index("canonical")["category"].to_dict()
        d_idx = d_main.set_index("doc_id")
        vocab_note[lang] = tfidf_df.shape

        # --- F. ESG 범주별(전체 코퍼스 단일 그룹 TF-IDF를 키워드 범주로 재정렬) ---
        mean_all = tfidf_df.mean(axis=0)
        raw_all = m_main["canonical"].value_counts()
        _both_scopes(rows, lang, "F_ESG범주별", "ALL", "corpus (all concepts)", mean_all, raw_all)
        for esg_cat, member_cats in ESG_CATEGORY_MEMBERSHIP.items():
            cat_terms = [t for t in mean_all.index if category_of.get(t, "") in member_cats]
            _both_scopes(rows, lang, "F_ESG범주별", esg_cat, f"ESG category {esg_cat}",
                         mean_all.loc[cat_terms], raw_all)

        # --- B. page_type별 ---
        for pt in [p for p in cfg.PAGE_TYPE_ORDER if p not in cfg.EXCLUDED_FROM_MAIN_PAGE_TYPES]:
            ids = [d for d in d_idx.index[d_idx["page_type_reviewed"] == pt] if d in tfidf_df.index]
            if not ids:
                continue
            sub = tfidf_df.loc[ids]
            raw = m_main[m_main["doc_id"].isin(ids)]["canonical"].value_counts()
            _both_scopes(rows, lang, "B_page_type별", pt, cfg.PAGE_TYPE_LABELS[pt]["en"],
                         sub.mean(axis=0), raw)

        # --- D. 계열별(tfidf.py와 동일하게 문서 30건 미만 계열 스킵) ---
        for tf in cfg.TARGET_FIELD_ORDER:
            ids_all = d_idx.index[d_idx["target_field_primary"] == tf]
            if len(ids_all) < 30:
                continue
            ids = [d for d in ids_all if d in tfidf_df.index]
            if not ids:
                continue
            sub = tfidf_df.loc[ids]
            raw = m_main[m_main["doc_id"].isin(ids)]["canonical"].value_counts()
            _both_scopes(rows, lang, "D_계열별", tf, cfg.TARGET_FIELD_LABELS[tf]["en"],
                         sub.mean(axis=0), raw)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(out.to_string(index=False))
    print("\n[요약] scope=present_in_group 기준 표별 ρ 분포")
    g = out[out["scope"] == "present_in_group"].groupby(["lang", "table"])["spearman_rho"]
    print(g.agg(["count", "min", "median", "max"]).round(4).to_string())
    print("\n[문서×concept 행렬 크기]", vocab_note)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    if "--rank-correlation" in sys.argv[1:]:
        build_rank_correlation_report()
    else:
        for _lang in cfg.LANGS:
            build_stage2_tfidf_report(_lang)
