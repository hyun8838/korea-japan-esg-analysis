# -*- coding: utf-8 -*-
"""전체 파이프라인 실행: 단계 I(언어별 전처리) → II(빈도) → III(연관) → IV(ESG LDA + 전체어휘 LDA), 한국·일본 순.

사용법: python run_analysis_pipeline.py
(kss/KoNLPy Okt/Janome/gensim/networkx/matplotlib/wordcloud/sklearn 설치 필요.
Okt는 전체어휘 LDA에 사용하며 JAVA_HOME은 실행 환경에 맞게 설정한다.)

언어 코드는 kr(한국)/jp(일본)로 통일 (common/config.py LANGS 참고).

폴더 구조:
  common/                 경로·라벨·IO 등 공용 유틸
  stage1_preprocess/      한국어·일본어 전처리(언어별 .py/.ipynb)
  stage2_frequency/       빈도분석
  stage3_association/     연관분석(Ochiai, 네트워크)
  stage4_lda/             ESG 키워드 한정·전체어휘 LDA
  output/                 산출물(언어 무관 공용 + output/{kr,jp}/단계N_*/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import config as cfg
from stage1_preprocess import preprocess_kr
from stage1_preprocess import preprocess_jp
from stage2_frequency import frequency as stage2
from stage2_frequency import tfidf as stage2_tfidf
from stage2_frequency import significance as stage2_significance
from stage3_association import association as stage3
from stage4_lda import lda_esg as stage4_esg
from stage4_lda import lda_full as stage4_full


def run_stage1():
    docs_kr, mentions_kr = preprocess_kr.run()
    docs_kr.drop(columns=["full_text"]).to_csv(cfg.OUTPUT_DIR / "docs_kr.csv", index=False, encoding="utf-8-sig")
    mentions_kr.to_csv(cfg.OUTPUT_DIR / "mentions_kr_raw.csv", index=False, encoding="utf-8-sig")

    docs_jp, mentions_jp = preprocess_jp.run()
    docs_jp.drop(columns=["full_text"]).to_csv(cfg.OUTPUT_DIR / "docs_jp.csv", index=False, encoding="utf-8-sig")
    mentions_jp.to_csv(cfg.OUTPUT_DIR / "mentions_jp_raw.csv", index=False, encoding="utf-8-sig")


def run_stage234():
    for lang in cfg.LANGS:
        stage2.build_stage2_report(lang)
        stage2_tfidf.build_stage2_tfidf_report(lang)
        stage3.build_stage3_report(lang)
        stage4_esg.build_stage4_report(lang)
        stage4_full.build_stage4_full_report(lang)
    # 한일 비교 문서단위 검정(Mann-Whitney U, Fisher-Pitman, FDR)은 두 언어 산출 후 실행
    stage2_significance.build_significance_report(cfg.OUTPUT_DIR / "significance")
    # 국가 내부 통계검정 — 한국·일본 각각 자기 안에서 계열별/page_type별 밀도 이질성
    stage2_significance.build_within_country_significance(cfg.OUTPUT_DIR / "significance")


if __name__ == "__main__":
    run_stage1()
    run_stage234()
    print("전체 파이프라인 실행 완료.")
