# -*- coding: utf-8 -*-
"""한일 비교 문서단위 통계검정. 문서별 밀도를 Mann-Whitney U + Fisher-Pitman 순열검정으로 비교한다.
(common/stats_testing.py mannwhitney_doc_test/fisher_pitman_doc_test)을 사용한다.

검정 대상(문서수 기준 최소표본 n>=5 양쪽 통과분만 검정, 통과 못 하면 기술통계만):
  1. 카테고리별 국가간 밀도 차이(문서단위)
  2. E/S/G 하위범주별 밀도 차이(DUP 중복포함, 문서단위)
  3. target_field별 밀도 차이(문서단위)
  4. page_type별 밀도 차이(문서단위)
  5. 3중교차표(target_field×page_type) 셀별 밀도 차이(문서단위)

ESG 카테고리 구성비(전체 형태) 비교는 검정하지 않고 기술통계로만 제시 — "국가 A 문서집합 vs
국가 B 문서집합"이 아니라 "프로필 전체 vs 전체" 비교라 문서단위 검정으로 안 바뀌기 때문.
category_composition_descriptive.csv로 별도 산출.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common import stats_testing as st
from common.io_common import load_enriched, doc_densities
from stage2_frequency.frequency import ESG_SUBCATEGORY_MEMBERSHIP


def _load_both():
    data = {}
    for lang in ("kr", "jp"):
        merged, docs = load_enriched(lang)
        data[lang] = {"merged": merged[merged["is_main"]], "docs": docs[docs["is_main"]]}
    return data


def _category_composition_descriptive(data: dict) -> pd.DataFrame:
    """카테고리 구성비 — 검정 없이 기술통계만."""
    cats = sorted(set(data["kr"]["merged"]["category"].unique()) | set(data["jp"]["merged"]["category"].unique()))
    total_kr, total_jp = len(data["kr"]["merged"]), len(data["jp"]["merged"])
    rows = []
    for c in cats:
        n_kr = len(data["kr"]["merged"][data["kr"]["merged"]["category"] == c])
        n_jp = len(data["jp"]["merged"][data["jp"]["merged"]["category"] == c])
        rate_kr, rate_jp = n_kr / total_kr if total_kr else 0.0, n_jp / total_jp if total_jp else 0.0
        rows.append({
            "category": c, "n_kr": n_kr, "n_jp": n_jp,
            "구성비_kr": round(rate_kr, 4), "구성비_jp": round(rate_jp, 4),
            "density_ratio": (rate_kr / rate_jp) if rate_jp > 0 else None,
        })
    return pd.DataFrame(rows)


def _per_category_tests(data: dict) -> list:
    cats = sorted(set(data["kr"]["merged"]["category"].unique()) | set(data["jp"]["merged"]["category"].unique()))
    rows = []
    for c in cats:
        dens_ko = doc_densities(data["kr"]["docs"], data["kr"]["merged"][data["kr"]["merged"]["category"] == c])
        dens_ja = doc_densities(data["jp"]["docs"], data["jp"]["merged"][data["jp"]["merged"]["category"] == c])
        rows.append({"family": "카테고리별_밀도", "comparison": f"category={c}", **st.pairwise_doc_test(dens_ko, dens_ja)})
    return rows


def _esg_subcategory_tests(data: dict) -> list:
    rows = []
    for sub, cats in ESG_SUBCATEGORY_MEMBERSHIP.items():
        mask_ko = data["kr"]["merged"]["category"].isin(cats)
        mask_ja = data["jp"]["merged"]["category"].isin(cats)
        dens_ko = doc_densities(data["kr"]["docs"], data["kr"]["merged"][mask_ko])
        dens_ja = doc_densities(data["jp"]["docs"], data["jp"]["merged"][mask_ja])
        rows.append({"family": "ESG하위범주_밀도", "comparison": f"subcategory={sub}", **st.pairwise_doc_test(dens_ko, dens_ja)})
    return rows


def _target_field_density_tests(data: dict) -> list:
    rows = []
    for tf in cfg.TARGET_FIELD_ORDER:
        d_ko = data["kr"]["docs"][data["kr"]["docs"]["target_field_primary"] == tf]
        d_ja = data["jp"]["docs"][data["jp"]["docs"]["target_field_primary"] == tf]
        m_ko = data["kr"]["merged"][data["kr"]["merged"]["target_field_primary"] == tf]
        m_ja = data["jp"]["merged"][data["jp"]["merged"]["target_field_primary"] == tf]
        dens_ko, dens_ja = doc_densities(d_ko, m_ko), doc_densities(d_ja, m_ja)
        rows.append({"family": "target_field별_밀도", "comparison": f"target_field={tf}", **st.pairwise_doc_test(dens_ko, dens_ja)})
    return rows


def _page_type_density_tests(data: dict) -> list:
    page_types = [pt for pt in cfg.PAGE_TYPE_ORDER if pt not in cfg.EXCLUDED_FROM_MAIN_PAGE_TYPES]
    rows = []
    for pt in page_types:
        d_ko = data["kr"]["docs"][data["kr"]["docs"]["page_type_reviewed"] == pt]
        d_ja = data["jp"]["docs"][data["jp"]["docs"]["page_type_reviewed"] == pt]
        m_ko = data["kr"]["merged"][data["kr"]["merged"]["page_type_reviewed"] == pt]
        m_ja = data["jp"]["merged"][data["jp"]["merged"]["page_type_reviewed"] == pt]
        dens_ko, dens_ja = doc_densities(d_ko, m_ko), doc_densities(d_ja, m_ja)
        rows.append({"family": "page_type별_밀도", "comparison": f"page_type={pt}", **st.pairwise_doc_test(dens_ko, dens_ja)})
    return rows


def _crosstab_cell_tests(data: dict) -> list:
    page_types = [pt for pt in cfg.PAGE_TYPE_ORDER if pt not in cfg.EXCLUDED_FROM_MAIN_PAGE_TYPES]
    rows = []
    for tf in cfg.TARGET_FIELD_ORDER:
        for pt in page_types:
            d_ko = data["kr"]["docs"][
                (data["kr"]["docs"]["target_field_primary"] == tf) & (data["kr"]["docs"]["page_type_reviewed"] == pt)
            ]
            d_ja = data["jp"]["docs"][
                (data["jp"]["docs"]["target_field_primary"] == tf) & (data["jp"]["docs"]["page_type_reviewed"] == pt)
            ]
            if len(d_ko) < 5 or len(d_ja) < 5:
                continue  # 문서단위 최소표본 미달 — 검정 불가능한 건 결과에도 안 넣음
            m_ko = data["kr"]["merged"][
                (data["kr"]["merged"]["target_field_primary"] == tf)
                & (data["kr"]["merged"]["page_type_reviewed"] == pt)
            ]
            m_ja = data["jp"]["merged"][
                (data["jp"]["merged"]["target_field_primary"] == tf)
                & (data["jp"]["merged"]["page_type_reviewed"] == pt)
            ]
            dens_ko, dens_ja = doc_densities(d_ko, m_ko), doc_densities(d_ja, m_ja)
            rows.append({
                "family": "3중교차표_셀", "comparison": f"target_field={tf},page_type={pt}",
                **st.pairwise_doc_test(dens_ko, dens_ja),
            })
    return rows


def build_significance_report(out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_both()

    composition_df = _category_composition_descriptive(data)
    composition_df.to_csv(out_dir / "category_composition_descriptive.csv", index=False, encoding="utf-8-sig")

    followup_rows = (
        _per_category_tests(data)
        + _esg_subcategory_tests(data)
        + _target_field_density_tests(data)
        + _page_type_density_tests(data)
        + _crosstab_cell_tests(data)
    )

    followup_df = pd.DataFrame(followup_rows)
    followup_df = st.apply_fdr_to_family(followup_df, p_col="mwu_p", q_col="fdr_q_mwu")
    followup_df = st.apply_fdr_to_family(followup_df, p_col="fp_p", q_col="fdr_q_fp")
    followup_df.to_csv(out_dir / "significance_tests_all.csv", index=False, encoding="utf-8-sig")

    n_tested = int(followup_df["mwu_p"].notna().sum())
    n_sig_mwu = int((followup_df["fdr_q_mwu"] < 0.05).sum())
    n_sig_fp = int((followup_df["fdr_q_fp"] < 0.05).sum())
    print(
        f"[검정 요약] 카테고리 구성비는 기술통계로만 별도 산출 + 후속검정 {len(followup_df)}개 중 "
        f"문서수 기준 검정가능 {n_tested}개 → FDR q<0.05: Mann-Whitney {n_sig_mwu}개 / Fisher-Pitman {n_sig_fp}개"
    )
    return followup_df


def _present_target_fields(merged: pd.DataFrame) -> list:
    return [tf for tf in cfg.TARGET_FIELD_ORDER if (merged["target_field_primary"] == tf).any()]


def _present_page_types(merged: pd.DataFrame) -> list:
    scoped = [pt for pt in cfg.PAGE_TYPE_ORDER if pt not in cfg.EXCLUDED_FROM_MAIN_PAGE_TYPES]
    return [pt for pt in scoped if (merged["page_type_reviewed"] == pt).any()]


def _group_densities(merged: pd.DataFrame, docs: pd.DataFrame, group_col: str, groups: list) -> dict:
    """group -> 문서단위 밀도 배열 딕셔너리(전체 mention 기준, 카테고리 무관)."""
    out = {}
    for g in groups:
        d = docs[docs[group_col] == g]
        m = merged[merged[group_col] == g]
        out[g] = doc_densities(d, m)
    return out


def _omnibus_tests_for_lang(lang: str, merged: pd.DataFrame, docs: pd.DataFrame):
    tfs = _present_target_fields(merged)
    pts = _present_page_types(merged)

    tf_densities = _group_densities(merged, docs, "target_field_primary", tfs)
    pt_densities = _group_densities(merged, docs, "page_type_reviewed", pts)

    rows = [
        {"family": f"{lang}_내부", "comparison": "target_field_밀도_이질성(옴니버스)",
         **st.kruskal_doc_test(tf_densities)},
        {"family": f"{lang}_내부", "comparison": "page_type_밀도_이질성(옴니버스)",
         **st.kruskal_doc_test(pt_densities)},
    ]
    return rows, tf_densities, pt_densities


def _pairwise_posthoc(lang: str, group_label: str, densities: dict) -> list:
    rows = []
    for a, b in itertools.combinations(sorted(densities), 2):
        result = st.pairwise_doc_test(densities[a], densities[b])
        rows.append({"family": f"{lang}_내부", "comparison": f"{group_label}_posthoc:{a}_vs_{b}", **result})
    return rows


def build_within_country_significance(out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for lang in cfg.LANGS:
        merged, docs = load_enriched(lang)
        merged = merged[merged["is_main"]]
        docs = docs[docs["is_main"]]

        omnibus_rows, tf_densities, pt_densities = _omnibus_tests_for_lang(lang, merged, docs)
        posthoc_tf = _pairwise_posthoc(lang, "target_field", tf_densities)
        posthoc_pt = _pairwise_posthoc(lang, "page_type", pt_densities)

        lang_rows = omnibus_rows + posthoc_tf + posthoc_pt
        lang_df = pd.DataFrame(lang_rows)
        # 국가별 FDR 묶음 — 한국 검정끼리, 일본 검정끼리 각각 보정.
        # 옴니버스(kw_p)와 post-hoc(mwu_p/fp_p)은 서로 다른 검정이라 FDR도 따로 적용.
        lang_df = st.apply_fdr_to_family(lang_df, p_col="kw_p", q_col="fdr_q_kw")
        lang_df = st.apply_fdr_to_family(lang_df, p_col="mwu_p", q_col="fdr_q_mwu")
        lang_df = st.apply_fdr_to_family(lang_df, p_col="fp_p", q_col="fdr_q_fp")

        n_tested = int(lang_df["mwu_p"].notna().sum()) + int(lang_df["kw_p"].notna().sum())
        n_sig = int((lang_df["fdr_q_mwu"] < 0.05).sum()) + int((lang_df["fdr_q_kw"] < 0.05).sum())
        print(f"[국가내부 검정:{lang}] {len(lang_df)}개 중 검정가능 {n_tested}개 → FDR q<0.05: {n_sig}개")
        all_rows.append(lang_df)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(out_dir / "significance_within_country.csv", index=False, encoding="utf-8-sig")
    return full


if __name__ == "__main__":
    if "--within-country" in sys.argv[1:]:
        build_within_country_significance(cfg.OUTPUT_DIR / "significance")
    else:
        build_significance_report(cfg.OUTPUT_DIR / "significance")
