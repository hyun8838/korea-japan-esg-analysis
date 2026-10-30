# -*- coding: utf-8 -*-
"""문서별 언급 밀도의 국가 간·국가 내 검정과 FDR 보정."""
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, permutation_test


def benjamini_hochberg(pvalues: list) -> list:
    """Benjamini & Hochberg(1995) FDR 보정. None/NaN 항목은 None 유지.
    dict→DataFrame 변환시 None이 NaN으로 바뀌므로 pd.notna로 걸러야 함(안 그러면 누적계산이 전부 NaN 오염됨).
    """
    valid_idx = [i for i, p in enumerate(pvalues) if pd.notna(p)]
    if not valid_idx:
        return [None] * len(pvalues)
    valid_p = np.array([pvalues[i] for i in valid_idx])
    n = len(valid_p)
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    result_valid = np.empty(n)
    result_valid[order] = q

    out = [None] * len(pvalues)
    for pos, i in enumerate(valid_idx):
        out[i] = round(float(result_valid[pos]), 4)
    return out


def apply_fdr_to_family(df: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    """df[p_col]의 유효한(None 아닌) p-value 전체를 하나의 family로 FDR 보정해 q_col에 저장."""
    df = df.copy()
    df[q_col] = benjamini_hochberg(df[p_col].tolist())
    return df


# 문서별 상대빈도(mention수÷문서길이)를 표집단위로 Mann-Whitney U(순위)와 Fisher-Pitman(순열) 두 검정을 계산 (Bestgen 2017)

def doc_count_check(n_a: int, n_b: int, min_n: int = 5) -> dict:
    """문서단위 검정에 필요한 집단별 최소 문서수를 확인한다."""
    valid = n_a >= min_n and n_b >= min_n
    return {"valid": valid, "n_a": int(n_a), "n_b": int(n_b)}


def rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    """Mann-Whitney U의 순위이연상관(rank-biserial) 효과크기. r = 2*(U/(n_a*n_b)) - 1 (Kerby 2014).
    U는 a 기준이라 r>0이면 그룹 a가 b보다 대체로 큰 경향."""
    return 2 * (u_stat / (n_a * n_b)) - 1


def mannwhitney_doc_test(densities_a, densities_b, min_n: int = 5) -> dict:
    """문서단위 상대빈도 배열 두 집단을 Mann-Whitney U로 비교.
    density_ratio(문서밀도 평균비)를 기술통계로 함께 제시."""
    a = np.asarray(densities_a, dtype=float)
    b = np.asarray(densities_b, dtype=float)
    n_a, n_b = len(a), len(b)
    check = doc_count_check(n_a, n_b, min_n)
    mean_a = float(a.mean()) if n_a else 0.0
    mean_b = float(b.mean()) if n_b else 0.0
    density_ratio = (mean_a / mean_b) if mean_b > 0 else (float("inf") if mean_a > 0 else None)
    base = {**check, "mean_density_a": mean_a, "mean_density_b": mean_b, "density_ratio": density_ratio}
    if not check["valid"]:
        return {**base, "mwu_stat": None, "mwu_p": None, "rank_biserial": None}
    try:
        result = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        # 분산 0 등 검정이 성립하지 않는 퇴화 케이스
        return {**base, "mwu_stat": None, "mwu_p": None, "rank_biserial": None}
    r = rank_biserial_from_u(float(result.statistic), n_a, n_b)
    return {
        **base,
        "mwu_stat": round(float(result.statistic), 4),
        "mwu_p": float(result.pvalue),
        "rank_biserial": round(r, 4),
    }


def fisher_pitman_doc_test(
    densities_a, densities_b, min_n: int = 5, n_resamples: int = 9999, random_state: int = 42
) -> dict:
    """Fisher-Pitman 순열검정(평균차 기반) — Mann-Whitney U 결과의 로버스트니스 체크 (Bestgen 2017)."""
    a = np.asarray(densities_a, dtype=float)
    b = np.asarray(densities_b, dtype=float)
    n_a, n_b = len(a), len(b)
    check = doc_count_check(n_a, n_b, min_n)
    if not check["valid"]:
        return {**check, "fp_stat": None, "fp_p": None}

    def _mean_diff(x, y):
        return np.mean(x) - np.mean(y)

    result = permutation_test(
        (a, b), _mean_diff, permutation_type="independent",
        n_resamples=n_resamples, alternative="two-sided", random_state=random_state,
    )
    return {**check, "fp_stat": round(float(result.statistic), 6), "fp_p": float(result.pvalue)}


def pairwise_doc_test(densities_a, densities_b, min_n: int = 5) -> dict:
    """Mann-Whitney U + Fisher-Pitman을 한 번에 계산해 병합. 국가 간·국가 내부 검정 공용 헬퍼."""
    mwu = mannwhitney_doc_test(densities_a, densities_b, min_n=min_n)
    fp = fisher_pitman_doc_test(densities_a, densities_b, min_n=min_n)
    return {**mwu, **fp}


def kruskal_doc_test(density_groups: dict, min_n: int = 5) -> dict:
    """Kruskal-Wallis H검정 — 문서단위 밀도의 k그룹(학문분야, page_type 등) 이질성 검정. Mann-Whitney U의 k그룹 일반화."""
    groups = {k: np.asarray(v, dtype=float) for k, v in density_groups.items()}
    ns = {k: len(v) for k, v in groups.items()}
    valid = len(groups) >= 2 and all(n >= min_n for n in ns.values())
    base = {"valid": valid, "group_sizes": ns}
    if not valid:
        return {**base, "kw_stat": None, "kw_p": None, "epsilon_squared": None}
    stat, p = kruskal(*groups.values())
    n_total = sum(ns.values())
    k = len(groups)
    # H기반 효과크기 엡실론제곱 (Tomczak & Tomczak 2014), 참고용 수치
    epsilon_sq = (float(stat) - k + 1) / (n_total - k) if n_total > k else None
    return {
        **base,
        "kw_stat": round(float(stat), 4),
        "kw_p": float(p),
        "epsilon_squared": round(epsilon_sq, 4) if epsilon_sq is not None else None,
    }
