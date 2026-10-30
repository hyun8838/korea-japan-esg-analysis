# -*- coding: utf-8 -*-
"""E 하위범주 문서단위 밀도 — 중앙값 확인 + 오염 키워드 제외 민감도 분석.

배경: 보고서 본문의 "E 하위범주 문서단위 밀도비 0.60"은 '문서당 언급 0.70배 ÷ 평균 문서길이
1.16배' 계산으로 나온 값인데, 이는 총합÷총합인 코퍼스단위 비(0.598)를 재현할 뿐 문서단위
평균비(0.604)의 분해는 아니다. 중앙값비(0.37)도 저장 산출물에 없어 검증이 불가능했다.
이 스크립트는 (1) 중앙값을 실제로 계산해 저장하고, (2) 오염 키워드 제외 후 밀도비가
얼마까지 움직이는지 민감도 분석을 한다.

민감도 시나리오:
  - JP_管理栄養  : JP E하위범주에서 '管理' 매칭 중 직후 2글자가 '栄養'인 것(管理栄養士/
                  管理栄養学科/管理栄養学専攻 — 국가자격·학과명 상용구) 제외
  - JP_水        : JP E하위범주에서 canonical='水' 매칭 전건 제외
                  (到達水準/御茶ノ水/水野/（水）=수요일 등 대부분 오탐 확인,
                   한국 사전에는 설계상 대응 키워드가 없음)
  - KR_복지행동  : KR E하위범주(DUP_ES 경유)에서 canonical이 '복지'/'행동'인 매칭 제외
                  (범용어 의존, 대칭 확인용)

밀도 단위는 per 1,000자(문서단위 밀도 = 문서별 mention 수 ÷ full_text_len × 1000).
산출: analysis/output/significance/E_subcategory_density_sensitivity.csv

검증: 기준 시나리오의 U·p·rank_biserial이 저장 산출물 significance_tests_all.csv
(subcategory=E: p=7.845979e-29, rank_biserial=-0.2874) 및 scipy 1.13.1 mannwhitneyu와
완전히 일치함을 확인 (아래 _mannwhitney는 scipy 없이도 돌아가도록 numpy로 재구현한 것).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common.io_common import load_enriched, doc_densities
try:  # frequency.py는 matplotlib 의존 — 없는 환경에서도 이 스크립트는 돌아야 함
    from stage2_frequency.frequency import ESG_SUBCATEGORY_MEMBERSHIP
except ModuleNotFoundError:  # 정의는 stage2_frequency/frequency.py와 동일해야 함
    ESG_SUBCATEGORY_MEMBERSHIP = {
        "E": {"E", "DUP_ES", "DUP_EG"},
        "S": {"S", "DUP_ES", "DUP_SG"},
        "G": {"G", "DUP_SG", "DUP_EG"},
    }

E_CATS = ESG_SUBCATEGORY_MEMBERSHIP["E"]
OUT = cfg.OUTPUT_DIR / "significance" / "E_subcategory_density_sensitivity.csv"


def _mannwhitney(a, b) -> dict:
    """scipy 미설치 환경 대비 numpy 전용 Mann-Whitney U(동점보정 정규근사, 연속성보정).
    scipy.stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic')와 동일한 값을
    내며, 기준 시나리오에서 저장 산출물(significance_tests_all.csv)의 값과 일치하는지 검증한다.
    rank_biserial은 common/stats_testing.rank_biserial_from_u와 같은 Kerby(2014) 공식."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    sv = allv[order]
    i = 0
    tie_term = 0.0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    r_a = ranks[:n_a].sum()
    u_a = r_a - n_a * (n_a + 1) / 2.0
    n = n_a + n_b
    mu = n_a * n_b / 2.0
    sigma = np.sqrt(n_a * n_b / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u_a - mu - np.sign(u_a - mu) * 0.5) / sigma if sigma > 0 else 0.0
    p = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return {"mwu_stat": float(u_a), "mwu_p": min(1.0, float(p)),
            "rank_biserial": round(2 * (u_a / (n_a * n_b)) - 1, 4)}


def _load():
    data = {}
    for lang in ("kr", "jp"):
        merged, docs = load_enriched(lang)
        m = merged[merged["is_main"]].copy()
        d = docs[docs["is_main"]].copy()
        data[lang] = {"merged": m[m["category"].isin(E_CATS)].copy(), "docs": d}
    return data


def _following(df: pd.DataFrame, n: int = 2) -> pd.Series:
    """각 매칭 직후 n글자(문장 내 기준)."""
    def f(r):
        s = str(r["sentence_text"])
        return s[int(r["end_char"]): int(r["end_char"]) + n]
    return df.apply(f, axis=1)


def _stats(docs: pd.DataFrame, mentions: pd.DataFrame) -> dict:
    dens = doc_densities(docs, mentions) * 1000.0  # per 1,000자
    return {
        "n_docs": len(docs),
        "n_mentions": len(mentions),
        "mean": float(np.mean(dens)),
        "median": float(np.median(dens)),
        "q1": float(np.percentile(dens, 25)),
        "q3": float(np.percentile(dens, 75)),
        "pct_zero_docs": float((dens == 0).mean() * 100),
        "_dens": dens,
    }


def _row(label, note, s_kr, s_jp) -> dict:
    mwu = _mannwhitney(s_kr["_dens"], s_jp["_dens"])
    return {
        "시나리오": label,
        "제외내용": note,
        "n_docs_kr": s_kr["n_docs"], "n_docs_jp": s_jp["n_docs"],
        "E언급수_kr": s_kr["n_mentions"], "E언급수_jp": s_jp["n_mentions"],
        "평균밀도_kr": round(s_kr["mean"], 4), "평균밀도_jp": round(s_jp["mean"], 4),
        "평균밀도비_kr÷jp": round(s_kr["mean"] / s_jp["mean"], 4) if s_jp["mean"] else None,
        "중앙값_kr": round(s_kr["median"], 4), "중앙값_jp": round(s_jp["median"], 4),
        "중앙값비_kr÷jp": round(s_kr["median"] / s_jp["median"], 4) if s_jp["median"] else None,
        "Q1_kr": round(s_kr["q1"], 4), "Q3_kr": round(s_kr["q3"], 4),
        "Q1_jp": round(s_jp["q1"], 4), "Q3_jp": round(s_jp["q3"], 4),
        "밀도0문서비율_kr(%)": round(s_kr["pct_zero_docs"], 2),
        "밀도0문서비율_jp(%)": round(s_jp["pct_zero_docs"], 2),
        "mwu_p": mwu["mwu_p"], "rank_biserial": mwu["rank_biserial"],
        "코퍼스단위밀도_kr": round(s_kr["n_mentions"] / docs_chars["kr"] * 1000, 4),
        "코퍼스단위밀도_jp": round(s_jp["n_mentions"] / docs_chars["jp"] * 1000, 4),
        "코퍼스단위밀도비": round(
            (s_kr["n_mentions"] / docs_chars["kr"]) / (s_jp["n_mentions"] / docs_chars["jp"]), 4),
    }


def main():
    data = _load()
    global docs_chars
    docs_chars = {l: float(data[l]["docs"]["full_text_len"].sum()) for l in ("kr", "jp")}

    m_kr, m_jp = data["kr"]["merged"], data["jp"]["merged"]
    d_kr, d_jp = data["kr"]["docs"], data["jp"]["docs"]

    # 오염 마스크
    jp_kanri = (m_jp["canonical"] == "管理") & (_following(m_jp).str.startswith("栄養"))
    jp_mizu = m_jp["canonical"] == "水"
    kr_generic = m_kr["canonical"].isin(["복지", "행동"])

    base_kr, base_jp = _stats(d_kr, m_kr), _stats(d_jp, m_jp)

    rows = [_row("0_기준(현행)", "없음 — 현행 보고서 수치", base_kr, base_jp)]
    rows.append(_row(
        "1_JP_管理栄養제외", f"JP 管理栄養士/学科/学専攻류 {int(jp_kanri.sum())}건 제외",
        base_kr, _stats(d_jp, m_jp[~jp_kanri])))
    rows.append(_row(
        "2_JP_水제외", f"JP 水 {int(jp_mizu.sum())}건 전건 제외",
        base_kr, _stats(d_jp, m_jp[~jp_mizu])))
    rows.append(_row(
        "3_JP_둘다제외", f"JP 管理栄養 {int(jp_kanri.sum())}건 + 水 {int(jp_mizu.sum())}건 제외",
        base_kr, _stats(d_jp, m_jp[~(jp_kanri | jp_mizu)])))
    rows.append(_row(
        "4_KR_복지행동제외", f"KR 복지·행동 {int(kr_generic.sum())}건 제외(대칭 확인)",
        _stats(d_kr, m_kr[~kr_generic]), base_jp))
    rows.append(_row(
        "5_양쪽모두제외", "JP 管理栄養+水, KR 복지+행동 동시 제외",
        _stats(d_kr, m_kr[~kr_generic]), _stats(d_jp, m_jp[~(jp_kanri | jp_mizu)])))

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(out[["시나리오", "E언급수_kr", "E언급수_jp", "평균밀도_kr", "평균밀도_jp",
               "평균밀도비_kr÷jp", "중앙값_kr", "중앙값_jp", "중앙값비_kr÷jp",
               "rank_biserial", "코퍼스단위밀도비"]].to_string(index=False))
    print(f"\n저장: {OUT}")
    print(f"\n[참고] JP 管理 전체 {int((m_jp['canonical']=='管理').sum())}건 중 "
          f"直後가 '栄養'인 것 {int(jp_kanri.sum())}건 "
          f"({jp_kanri.sum()/max(1,(m_jp['canonical']=='管理').sum())*100:.1f}%)")
    print(f"[참고] KR 문서수 {len(d_kr)} 총글자 {docs_chars['kr']:.0f} / "
          f"JP 문서수 {len(d_jp)} 총글자 {docs_chars['jp']:.0f}")


if __name__ == "__main__":
    main()
