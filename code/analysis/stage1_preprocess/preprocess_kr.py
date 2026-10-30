# -*- coding: utf-8 -*-
"""단계 I(한국어 전처리): 한국어 문서를 문장 단위 ESG mention table로 변환한다.

텍스트 정제, kss 문장 분리(오류 시 정규식), 사전 기반 span 매칭과 longest-match를 수행한다."""
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common import io_common
from common import mention_common

ESG_CONCEPTS_KR_PATH = cfg.ESG_ROOT / "korea" / "preprocessing" / "esg_concepts_kr.json"

FP_EXCLUDE_KW = {"법", "물"}  # 단독 사용 시 FP 비율 높아 전면 제외. 'water'(영문)는 유지.


def load_esg_concepts_kr(path: Path = ESG_CONCEPTS_KR_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


ESG_CONCEPTS_KR = load_esg_concepts_kr()

# 행정 UI 불용어
ADMIN_STOPWORDS_RAW = [
    "홈", "동", "관", "상", "순", "외", "전", "중", "후", "형", "당",
    "공지사항", "공지", "안내", "알림", "게시판", "첨부", "첨부파일",
    "다운로드", "업로드", "제출", "제출기한", "제출방법", "신청",
    "신청기간", "신청방법", "신청서", "서류", "양식", "확인", "문의",
    "접수", "마감", "기간", "기한", "일정", "일자", "날짜", "시간",
    "일시", "장소", "방법", "절차", "기준", "대상", "해당", "내용",
    "사항", "경우", "관련", "수행", "운영", "진행", "실시", "추진",
    "개최", "시행", "적용", "해당자", "이상", "이하", "초과", "미만",
    "참고", "참조", "참조바람", "바람", "바랍니다", "합니다", "드립니다",
    "학과", "학부", "대학원", "학과장", "담당", "담당자", "교수님",
    "조교", "사무실", "사무처", "행정실", "부서", "팀", "실",
    "번호", "제목", "작성자", "작성일", "조회수", "조회", "댓글",
    "이전글", "다음글", "목록", "처음", "끝", "페이지", "검색",
    "등록", "수정", "삭제", "저장", "취소", "완료", "닫기",
    "학점", "이수", "수강", "수업", "강의", "강의계획", "계획",
    "성적", "출석", "시험", "과제", "레포트", "발표", "평가",
    "학번", "이름", "연락처", "이메일", "전화", "팩스",
    "및", "또는", "혹은", "등", "각", "본", "위한", "위해",
    "통해", "통한", "위하여", "으로", "에서", "에게", "에게서",
    "대하여", "대한", "관하여", "관한",
    "년도", "년", "월", "일", "분기", "학기", "회", "개",
    "명", "건", "원", "만원", "억원",
    "바로가기", "상세보기", "더보기", "전체보기", "링크", "클릭",
    "열기", "펼치기", "접기",
    "논문", "학술", "학술지", "학위", "일반", "학회", "춘계", "추계",
    "대회", "국제", "대한민국", "우수", "저서", "역서", "편집", "위원",
    "회지", "포스터", "주년", "기념",
    "전공", "공선택", "교과목", "학년", "필수", "과목",
    "분반", "교양", "이수구분", "학사", "석사", "박사", "통합과정",
    "소개", "명예", "취임", "퇴임", "관심", "분야", "박사과정",
    "석사졸업", "박사졸업", "졸업논문",
]
ADMIN_STOPWORDS_KR = set(ADMIN_STOPWORDS_RAW)


CHAR_LIMIT = 40_000
ADMIN_STOPWORDS_KR_PATH = Path(__file__).parent / "admin_stopwords_kr.txt"

_RE_PHONE = re.compile(r"\d{2,4}[-. ]\d{3,4}[-. ]\d{4}")
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_URL = re.compile(r"https?://\S+")
_RE_ENTITY = re.compile(r"&[a-zA-Z#0-9]+;")
_RE_REPEAT_SYM = re.compile(r"[-=_]{4,}")
_RE_SPACE = re.compile(r"[ \t]+")
_RE_NEWLINE = re.compile(r"\n{3,}")

# 문장 중간에 섞여 나오는 UI 잔재 (줄 단위로는 안 걸러짐)
# 예: "홍길동 조교수 홍길동 사이트로 이동 Gildong Hong ...", "COPYRIGHT (C)2024 Department ..."
_RE_INLINE_NOISE = [
    re.compile(r"[가-힣]{2,10}\s*사이트로\s*이동"),
    re.compile(r"COPYRIGHT\s*\(C\)\s*\d{4}[^\n]{0,80}", re.IGNORECASE),
    re.compile(r"번호\s*제목\s*작성자\s*조회\s*작성일"),
    re.compile(r"번호\s*제목\s*작성자\s*작성일\s*조회수"),
    re.compile(r"전체\s*[\d,]+\s*(건|개)?"),
    re.compile(r"상세보기|바로가기|더보기(?!\S)|전체보기"),
]


def _load_admin_stopwords_kr(path: Path = ADMIN_STOPWORDS_KR_PATH) -> set:
    sw = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sw.add(line)
    return sw


ADMIN_STOPWORDS_KR_LINES = _load_admin_stopwords_kr()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = _RE_PHONE.sub(" ", t)
    t = _RE_EMAIL.sub(" ", t)
    t = _RE_URL.sub(" ", t)
    t = _RE_ENTITY.sub(" ", t)
    for pat in _RE_INLINE_NOISE:
        t = pat.sub(" ", t)
    # 줄 단위 UI 잔재 제거
    lines = t.split("\n")
    t = "\n".join(ln for ln in lines if ln.strip() not in ADMIN_STOPWORDS_KR_LINES or not ln.strip())
    t = _RE_REPEAT_SYM.sub(" ", t)
    t = _RE_SPACE.sub(" ", t)
    t = _RE_NEWLINE.sub("\n\n", t)
    t = t.strip()
    if len(t) > CHAR_LIMIT:
        t = t[:CHAR_LIMIT]
    return t


_SENT_SPLIT_FALLBACK = re.compile(r"(?<=[.!?。])[\s\n]+")


def _make_sentence_splitter():
    try:
        import kss

        def split(text):
            if not text.strip():
                return []
            try:
                return [s.strip() for s in kss.split_sentences(text, backend="fast") if s.strip()]
            except Exception:
                return [s.strip() for s in _SENT_SPLIT_FALLBACK.split(text) if s.strip()]

        return split
    except ImportError:
        def split(text):
            return [s.strip() for s in _SENT_SPLIT_FALLBACK.split(text) if s.strip()]

        return split


def _track_of(alias: str) -> str:
    has_alpha = bool(re.search(r"[A-Za-z]", alias))
    if has_alpha and " " not in alias and "-" not in alias:
        return "A_en"
    if has_alpha:
        return "B"
    return "A_kr"


ALIAS_POOL_KR = mention_common.build_alias_pool(ESG_CONCEPTS_KR, FP_EXCLUDE_KW, _track_of)


def run(limit_docs: int | None = None, verbose: bool = True):
    df = io_common.load_scoped_dataset()
    df_kr = df[df["lang"] == "kr"].copy()
    df_kr["doc_id"] = df_kr.index.astype(str)
    if limit_docs:
        df_kr = df_kr.head(limit_docs)

    split_sentences = _make_sentence_splitter()

    if verbose:
        print(f"[KR 전처리] 대상 문서 {len(df_kr)}건, alias pool {len(ALIAS_POOL_KR)}개")

    cleaned_texts = {}
    all_rows = []
    for i, (doc_id, text) in enumerate(zip(df_kr["doc_id"], df_kr["full_text"])):
        cleaned = clean_text(text)
        cleaned_texts[doc_id] = cleaned
        sentences = split_sentences(cleaned)
        rows = mention_common.extract_doc_mentions(doc_id, sentences, ALIAS_POOL_KR)
        all_rows.extend(rows)
        if verbose and (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(df_kr)}건 처리, 누적 mention {len(all_rows)}건")

    df_kr["cleaned_text_len"] = df_kr["doc_id"].map(lambda d: len(cleaned_texts[d]))
    n_before = len(df_kr)
    short_ids = set(df_kr.loc[df_kr["cleaned_text_len"] < io_common.MIN_CLEANED_CHARS, "doc_id"])
    df_kr = df_kr[~df_kr["doc_id"].isin(short_ids)].copy()
    all_rows = [r for r in all_rows if r["doc_id"] not in short_ids]
    if verbose:
        print(
            f"[KR 전처리] UI 잔재 정제 후 {io_common.MIN_CLEANED_CHARS}자 미만 문서 제외: "
            f"{n_before}건 → {len(df_kr)}건 ({len(short_ids)}건 제외)"
        )

    mentions = pd.DataFrame(all_rows)
    if verbose:
        print(f"[KR 전처리] 완료: 문서 {len(df_kr)}건 → mention(raw) {len(mentions)}건")
    return df_kr, mentions


if __name__ == "__main__":
    docs, mentions = run()
    out_dir = cfg.OUTPUT_DIR
    docs.drop(columns=["full_text"]).to_csv(out_dir / "docs_kr.csv", index=False, encoding="utf-8-sig")
    mentions.to_csv(out_dir / "mentions_kr_raw.csv", index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_dir / 'docs_kr.csv'}, {out_dir / 'mentions_kr_raw.csv'}")
