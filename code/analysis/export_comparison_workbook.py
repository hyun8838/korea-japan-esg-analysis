# -*- coding: utf-8 -*-
"""한일 분석방법과 분석결과를 비교하는 워크북을 생성한다.

주의: 이 스크립트는 run_analysis_pipeline.py 실행 후(단계 I~IV 산출물이 output/kr, output/jp에 있는 상태) 실행해야 한다.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import config as cfg
from common import io_common as io

OUT = cfg.OUTPUT_DIR
REPORT_PATH = OUT / "한일_비교결과.xlsx"

# I. 분석방법 비교
PREPROCESSING_COMPARISON = pd.DataFrame([
    {"항목": "키워드 추출", "한국": "사전 기반 span 매칭", "일본": "사전 기반 span 매칭", "비고": "겹치는 표현은 longest-match 채택"},
    {"항목": "문장 분리", "한국": "kss(미설치·오류 시 정규식)", "일본": "[。！？\\n]+ 기준 정규식", "비고": "-"},
    {"항목": "전체어휘 LDA 토큰화", "한국": "KoNLPy Okt 명사", "일본": "Janome 漢字·カタカナ 명사", "비고": "ESG 키워드 매칭과 별도"},
    {"항목": "메타블록 토큰", "한국": "KCI, DCOLL", "일본": "CiNii, J-STAGE, NDL", "비고": "학술DB 표기 차이"},
    {"항목": "FP: 水(물)", "한국": "FP 우려로 E범주에서 제외", "일본": "E범주 포함", "비고": "양국 사전의 포함 기준 차이"},
    {"항목": "FP: 法(법)", "한국": "FP 우려로 제외", "일본": "G범주 포함", "비고": "方法/技法 등의 부분 문자열 오탐 위험"},
    {"항목": "집계기준(scope)", "한국": "all=전체, main=all 중 page_type I·J 제외", "일본": "동일", "비고": "-"},
])

PIPELINE_FLOW = pd.DataFrame([
    {"단계": 1, "처리내용": "원문 로드", "목적": "esg_final_dataset_v3.xlsx에서 언어별(KR-/JP-) 행 분리"},
    {"단계": 2, "처리내용": "빈 문서/에러마커 제거", "목적": "null 본문·크롤링 실패 마커([non-html], [fetch failed] 등) 제외"},
    {"단계": 3, "처리내용": "텍스트 정제", "목적": "전화번호·이메일·URL·반복기호·UI 잔재(admin_stopwords) 제거"},
    {"단계": 4, "처리내용": "정제 후 길이 필터", "목적": "200자 미만 문서 제외(japan/preprocessing MIN_CHARS=200 기준)"},
    {"단계": 5, "처리내용": "문장 분리", "목적": "한국어 kss(대체 정규식), 일본어 정규식 적용"},
    {"단계": 6, "처리내용": "ESG 키워드 매칭", "목적": "언어별 concept 사전(alias→canonical) 기반 span 매칭"},
    {"단계": 7, "처리내용": "longest-match 중복 제거", "목적": "겹치는 표현 중 긴 것만 selected로 채택"},
    {"단계": 8, "처리내용": "scope 부여", "목적": "page_type 기준 main/all 구분, 저신뢰검토 확정값 반영"},
])


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_scope_funnel_table() -> pd.DataFrame:
    """크롤링 원본 분류 → 저신뢰플래그 검토 반영 → 전처리(정제+길이필터) 최종까지, main scope
    문서수가 단계별로 어떻게 바뀌는지 보여주는 표.

    저신뢰검토로 분류가 바뀐 문서는 분류 전후 단계에 각각 반영한다."""
    univ_kr = pd.read_excel(cfg.UNIV_LIST_XLSX, sheet_name="univ_lits_kr")
    univ_jp = pd.read_excel(cfg.UNIV_LIST_XLSX, sheet_name="univ_list_jp")

    raw = io.load_final_dataset()
    raw_main = raw[~raw["page_type"].isin(cfg.EXCLUDED_FROM_MAIN_PAGE_TYPES)]

    reviewed = io.drop_invalid_documents(raw, verbose=False)
    reviewed = io.apply_review_overrides(reviewed)
    reviewed = io.add_scope_columns(reviewed)
    reviewed_main = reviewed[reviewed["is_main"]]

    final_docs = {lang: _read_csv(OUT / f"docs_{lang}.csv") for lang in ("kr", "jp")}

    rows = [
        {
            "단계": "0-1. 최초 조사대상 대학 수(univ_list.xlsx 설계 시점, 크롤링 이전)",
            "한국": int(univ_kr["University"].nunique()),
            "일본": int(univ_jp["University"].nunique()),
        },
        {
            "단계": "0-2. 최초 조사대상 학과(entity) 수(univ_list.xlsx 설계 시점, 크롤링 이전)",
            "한국": len(univ_kr),
            "일본": len(univ_jp),
        },
    ]
    for label, frame, col in [
        ("① 크롤링·원본 page_type 분류 후 main scope", raw_main, "lang"),
        ("② 저신뢰플래그 수동검토 반영 후 main scope(전처리 전)", reviewed_main, "lang"),
    ]:
        rows.append({
            "단계": label,
            "한국": int((frame[col] == "kr").sum()),
            "일본": int((frame[col] == "jp").sum()),
        })
    rows.append({
        "단계": "③ 전처리(UI정제+200자 미만 제외) 후 최종 main scope(=I_문서구성의 값, 실제 분석 대상)",
        "한국": int(final_docs["kr"]["is_main"].sum()) if not final_docs["kr"].empty else None,
        "일본": int(final_docs["jp"]["is_main"].sum()) if not final_docs["jp"].empty else None,
    })
    return pd.DataFrame(rows)


def build_doc_composition_table() -> pd.DataFrame:
    """의뢰자 docx "I. 분석 대상의 문서 구성" 섹션.
    docs_{lang}.csv(단계 I 완료 후 최종 분석 대상 문서, UI정제+길이필터 반영) 기준 산출.
    university(대학) 수와 entity(학과) 수는 다른 개념이라 별도 행으로 분리."""
    docs = {lang: _read_csv(cfg.OUTPUT_DIR / f"docs_{lang}.csv") for lang in ("kr", "jp")}

    def counts(cond_ko, cond_ja):
        d_ko, d_ja = docs["kr"], docs["jp"]
        n_ko = int(cond_ko(d_ko).sum()) if not d_ko.empty else None
        n_ja = int(cond_ja(d_ja).sum()) if not d_ja.empty else None
        return n_ko, n_ja

    rows = []
    n_ko, n_ja = len(docs["kr"]), len(docs["jp"])
    rows.append({"항목": "총 분석 대상 문서 수", "한국어 라벨": "", "일본어 라벨": "", "한국": n_ko, "일본": n_ja})
    rows.append({"항목": "대학(university) 수", "한국어 라벨": "", "일본어 라벨": "",
                 "한국": docs["kr"]["university"].nunique(), "일본": docs["jp"]["university"].nunique()})
    rows.append({"항목": "학과(entity_id, 최종 데이터셋 기준) 수", "한국어 라벨": "", "일본어 라벨": "",
                 "한국": docs["kr"]["entity_id"].nunique(), "일본": docs["jp"]["entity_id"].nunique()})
    rows.append({"항목": "main scope 문서 수", "한국어 라벨": "", "일본어 라벨": "",
                 "한국": int(docs["kr"]["is_main"].sum()), "일본": int(docs["jp"]["is_main"].sum())})
    rows.append({"항목": "all scope 문서 수(=main+제외분 전체)", "한국어 라벨": "", "일본어 라벨": "", "한국": n_ko, "일본": n_ja})

    for tf in cfg.TARGET_FIELD_ORDER:
        n_ko, n_ja = counts(
            lambda d: d["target_field_primary"] == tf, lambda d: d["target_field_primary"] == tf,
        )
        rows.append({
            "항목": f"target_field={tf}", "한국어 라벨": cfg.TARGET_FIELD_LABELS[tf]["kr"],
            "일본어 라벨": cfg.TARGET_FIELD_LABELS[tf]["jp"], "한국": n_ko, "일본": n_ja,
        })
    for pt in cfg.PAGE_TYPE_ORDER:
        n_ko, n_ja = counts(
            lambda d: d["page_type_reviewed"] == pt, lambda d: d["page_type_reviewed"] == pt,
        )
        rows.append({
            "항목": f"page_type={pt}", "한국어 라벨": cfg.PAGE_TYPE_LABELS[pt]["kr"],
            "일본어 라벨": cfg.PAGE_TYPE_LABELS[pt]["jp"], "한국": n_ko, "일본": n_ja,
        })
    return pd.DataFrame(rows)


def build_aggregation_criteria_table() -> pd.DataFrame:
    """"II. 집계기준 구분" — raw(longest-match 적용 전 전수) vs selected(적용 후, 본문 분석 기준)."""
    rows = []
    for lang in ("kr", "jp"):
        m = _read_csv(cfg.OUTPUT_DIR / f"mentions_{lang}_raw.csv")
        if m.empty:
            continue
        rows.append({"집계기준": "raw inclusive(longest-match 적용 전, 중복 포함)", "언어": lang, "건수": len(m)})
        rows.append({
            "집계기준": "selected(longest-match 적용 후, 본문 분석 기준)", "언어": lang,
            "건수": int(m["is_selected_longest"].sum()),
        })
    return pd.DataFrame(rows)


NOTES = pd.DataFrame([
    {"시트": "I_문서구성_퍼널", "주의사항": "①은 원본 페이지 유형 기준, ②는 수동검토 반영 후, "
                              "③은 전처리 완료 후 문서 수다. 실제 분석 대상은 ③을 사용한다."},
    {"시트": "II9_3중교차표", "주의사항": "sparse_flag=True(n<5) 셀은 해석 주의 — Cochran(1954) 기준"},
    {"시트": "II11_TFIDF특징어", "주의사항": "한국어·일본어 어휘집합이 원천적으로 달라 직접 통계비교가 아니라, "
                                    "각자 독립적으로 뽑은 특징어 목록을 나란히 놓고 보는 정성적 병렬대조임. "
                                    "ESG concept 토큰 한정으로 계산"},
    {"시트": "II11b_계열별TFIDF / II11c_ESG범주별TFIDF",
     "주의사항": "II11(page_type별)과 동일한 방식으로, 문서를 계열(target_field)/ESG범주 단위로 묶어 "
             "ESG concept 토큰 한정 TF-IDF를 계산한 표. mean_tfidf는 해당 그룹 문서들의 TF-IDF 평균이며, "
             "raw_count(총 등장수)·doc_freq(등장 문서수)를 함께 봐야 함. II11과 마찬가지로 한일 직접 "
             "통계비교가 아니라 각자 독립적으로 뽑은 특징어 목록의 정성적 병렬대조임"},
    {"시트": "II12_계열별네트워크요약", "주의사항": "문서 30건 미만 계열은 skipped=True(네트워크 미생성). "
                                          "개별 네트워크 그림은 output/{lang}/단계3_연관분석/D_network_*.png 참고"},
    {"시트": "II13a_카테고리구성비_기술통계", "주의사항": "ESG 카테고리 구성비(전체 형태) 비교는 검정하지 않고 "
                                          "기술통계로만 제시 — '국가 A 문서집합 vs 국가 B "
                                          "문서집합'이 아니라 '국가 A 프로필 전체 vs 국가 B 프로필 전체' 비교라 "
                                          "문서단위 검정으로 깔끔하게 안 바뀌기 때문. p-value 없음, density_ratio만 참고용."},
    {"시트": "II13_통계검정결과", "주의사항": "문서단위 Mann-Whitney U(mwu_p)와 Fisher-Pitman 순열검정(fp_p)을 사용한다. "
                                     "valid=False 행은 문서수 최소기준(양쪽 n>=5) "
                                     "미달로 검정 불가(기술통계만 유효). fdr_q_mwu/fdr_q_fp는 전체 후속검정을 "
                                     "하나의 family로 묶어 Benjamini-Hochberg 보정한 값. rank_biserial은 MWU "
                                     "효과크기(양수=한국 밀도가 대체로 더 큼), density_ratio(평균밀도비)도 병기. "
                                     "이상적으로는 mwu_p·fp_p 둘 다 유의미해야 하며, 한쪽만 유의미하면 해석 주의"},
    {"시트": "II14_국가내부통계검정", "주의사항": "한국·일본 각각 자기 나라 안에서만(계열별/page_type별 밀도 "
                                        "이질성 옴니버스 2개 + 쌍별 post-hoc) 검정. FDR은 국가별로 따로 보정 "
                                        "하며 옴니버스(kw_p)와 post-hoc(mwu_p/fp_p)도 검정 종류가 달라 FDR family를 "
                                        "분리(한국/일본, 그리고 kw/mwu/fp 각각). 옴니버스는 "
                                        "Kruskal-Wallis(kw_p, 문서단위 밀도의 k그룹 이질성), post-hoc은 문서단위 "
                                        "Mann-Whitney U + Fisher-Pitman을 사용한다. density_ratio(평균밀도비)를 "
                                        "기술통계로 함께 보고, epsilon_squared는 Kruskal-Wallis 효과크기를 나타낸다."},
])


def build():
    with pd.ExcelWriter(REPORT_PATH) as writer:
        PREPROCESSING_COMPARISON.to_excel(writer, sheet_name="I_전처리비교", index=False)
        PIPELINE_FLOW.to_excel(writer, sheet_name="I_전처리흐름", index=False)
        build_doc_composition_table().to_excel(writer, sheet_name="I_문서구성", index=False)
        build_scope_funnel_table().to_excel(writer, sheet_name="I_문서구성_퍼널", index=False)
        build_aggregation_criteria_table().to_excel(writer, sheet_name="I_집계기준", index=False)

        # II-1 ESG 범주별 개요
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"1_ESG범주별_개요_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II1_ESG범주개요_{lang}", index=False)

        # II-2 학문분야별 밀도
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"2_학문분야별_밀도_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II2_학문분야별밀도_{lang}", index=False)

        # II-3 히트맵
        for lang in ("kr", "jp"):
            path = OUT / lang / "단계2_빈도분석" / f"3_히트맵_{lang}.csv"
            if path.exists():
                pd.read_csv(path, index_col=0).to_excel(writer, sheet_name=f"II3_히트맵_{lang}")

        # II-4 대학별 개요
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"4_대학별_개요_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II4_대학별개요_{lang}", index=False)

        # II-5 페이지타입별 개요
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"5_페이지타입별_개요_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II5_페이지타입별개요_{lang}", index=False)

        # II-6 연관분석 (top50 등장횟수 + top50 ochiai)
        for lang in ("kr", "jp"):
            path = OUT / lang / "단계3_연관분석" / f"concept_edges_{lang}.csv"
            edges = _read_csv(path)
            if not edges.empty:
                edges.sort_values("co_sent_n", ascending=False).head(50).to_excel(
                    writer, sheet_name=f"II6_연관_등장횟수_{lang}", index=False)
                edges.sort_values("ochiai", ascending=False).head(50).to_excel(
                    writer, sheet_name=f"II6_연관_Ochiai_{lang}", index=False)

        # II-9 국가×계열×page_type 3중 교차표
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"A_3중교차표_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II9_3중교차표_{lang}", index=False)

        # II-10 ESG 하위범주(E/S/G)별 밀도
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"C_ESG하위범주별_밀도_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II10_ESG하위범주밀도_{lang}", index=False)

        # II-11 page_type별 ESG concept TF-IDF 특징어
        # 한국어·일본어 어휘집합이 달라 직접 통계비교가 아니라 정성적 병렬대조
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"B_page_type별_TFIDF_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II11_TFIDF특징어_{lang}", index=False)

        # II-11b 계열(target_field)별 ESG concept TF-IDF 특징어
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"D_계열별_TFIDF_{lang}.csv")
            if not t.empty:
                t.to_excel(writer, sheet_name=f"II11b_계열별TFIDF_{lang}", index=False)

        # II-11c ESG 범주(E/S/G 및 중복범주)별 concept TF-IDF 특징어
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계2_빈도분석" / f"F_ESG범주별_TFIDF_{lang}.csv")
            if not t.empty:
                t.to_excel(writer, sheet_name=f"II11c_ESG범주별TFIDF_{lang}", index=False)

        # II-12 계열별 핵심 연관어 네트워크 요약(개별 네트워크 그림은 PNG로 별도)
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계3_연관분석" / f"D_계열별네트워크_요약_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II12_계열별네트워크요약_{lang}", index=False)

        # II-13a 카테고리 구성비 — 검정 없이 기술통계만
        comp = _read_csv(OUT / "significance" / "category_composition_descriptive.csv")
        if not comp.empty:
            comp.to_excel(writer, sheet_name="II13a_카테고리구성비_기술통계", index=False)

        # II-13 통계적 검정 결과 — 문서단위 Mann-Whitney U + Fisher-Pitman,
        # 문서수 최소기준 통과분만, FDR(Benjamini-Hochberg) 보정값 포함
        sig = _read_csv(OUT / "significance" / "significance_tests_all.csv")
        if not sig.empty:
            sig.to_excel(writer, sheet_name="II13_통계검정결과", index=False)

        # II-14 한국·일본 각각 국가 내부 통계검정(계열별/page_type별 밀도 이질성 + post-hoc)
        sig_within = _read_csv(OUT / "significance" / "significance_within_country.csv")
        if not sig_within.empty:
            sig_within.to_excel(writer, sheet_name="II14_국가내부통계검정", index=False)

        NOTES.to_excel(writer, sheet_name="0_주의사항", index=False)

        # II-7 ESG 키워드 한정 LDA 요약
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계4_ESG_LDA" / f"lda_summary_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II7_ESG-LDA요약_{lang}", index=False)

        # II-8 전체어휘 LDA 요약(학과 계열이 토픽으로 잡히는 한계 있음, 별도 명시)
        for lang in ("kr", "jp"):
            t = _read_csv(OUT / lang / "단계4_전체어휘_LDA" / f"lda_full_summary_{lang}.csv")
            t.to_excel(writer, sheet_name=f"II8_전체어휘LDA요약_{lang}", index=False)
            terms = _read_csv(OUT / lang / "단계4_전체어휘_LDA" / f"topic_terms_{lang}.csv")
            if not terms.empty:
                terms.to_excel(writer, sheet_name=f"II8_전체어휘LDA_토픽어_{lang}", index=False)

    print(f"한일 비교결과 저장 완료: {REPORT_PATH}")


if __name__ == "__main__":
    build()
