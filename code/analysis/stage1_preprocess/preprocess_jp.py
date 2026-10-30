# -*- coding: utf-8 -*-
"""단계 I(일본어 전처리): 일본어 문서를 문장 단위 ESG mention table로 변환한다.

일본어 사전과 UI 불용어를 사용하고, 정규식 문장 분리·span 매칭·longest-match를 수행한다."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common import io_common
from common import mention_common

JAPAN_PREP_DIR = cfg.ESG_ROOT / "japan" / "preprocessing"
ESG_CONCEPTS_JA_PATH = JAPAN_PREP_DIR / "esg_concepts_ja.json"
ADMIN_STOPWORDS_JA_PATH = JAPAN_PREP_DIR / "admin_stopwords_ja.txt"

CHAR_LIMIT = 40_000

_RE_PHONE = re.compile(r"\d{2,4}[-−]\d{2,4}[-−]\d{4}")
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_URL = re.compile(r"https?://\S+")
_RE_REF = re.compile(r"(CiNii|J-STAGE|NDL|doi:|DOI:)[^\n]{0,200}", re.IGNORECASE)
_RE_DATE_JA = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日(?:更新|現在)?")
_RE_DATE_SL = re.compile(r"\d{4}/\d{2}/\d{2}")
_RE_NENDO = re.compile(r"\d{4}年度")
_RE_PDF_IND = re.compile(r"\[PDF\]|\[ファイル\s*[\d.]+MB\]")
_RE_COPYRIGHT = re.compile(r"(?:copyright[^\n]*|all\s+rights\s+reserved[^\n]*)", re.IGNORECASE)
_RE_SPACE = re.compile(r"[ \t]{2,}")
_RE_NEWLINE = re.compile(r"\n{3,}")
_RE_SENT_SPLIT = re.compile(r"(?<=[。！？])|(?:\n+)")


def load_stopwords_ja(path: Path = ADMIN_STOPWORDS_JA_PATH) -> set:
    sw = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sw.add(line)
    return sw


def load_esg_concepts_ja(path: Path = ESG_CONCEPTS_JA_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # 단독 숫자 alias는 ESG 맥락과 무관한 숫자도 매칭할 수 있어 제외.
    # 양국 사전의 Scope 1/2/3 포함 범위 차이는 결과 해석에 반영.
    # 카테고리 라벨을 한국어 사전과 통일: 'ESG'(총괄)→'ESG_META', 'DUP_GS'→'DUP_SG'
    _CATEGORY_RENAME = {"ESG": "ESG_META", "DUP_GS": "DUP_SG"}

    concepts = []
    for c in raw["concepts"]:
        aliases = [a for a in c["aliases"] if not re.fullmatch(r"[0-9]+", a)]
        if not aliases:
            continue  # concept 전체가 숫자뿐이면 통째로 제외
        category = _CATEGORY_RENAME.get(c["category"], c["category"])
        concepts.append({"canonical": c["canonical_ja"], "category": category, "aliases": aliases})

    alias_to_canonical = {
        k: v for k, v in raw["alias_to_canonical"].items() if not re.fullmatch(r"[0-9]+", k)
    }
    return {
        "concepts": concepts,
        "alias_to_canonical": alias_to_canonical,
        "canonical_categories": raw["canonical_categories"],
    }


ADMIN_STOPWORDS_JA = load_stopwords_ja()
ESG_CONCEPTS_JA = load_esg_concepts_ja()
FP_EXCLUDE_JA: set = set()  # 일본어 사전의 水는 유지


def clean_text_ja(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = _RE_PHONE.sub("", t)
    t = _RE_EMAIL.sub("", t)
    t = _RE_URL.sub("", t)
    t = _RE_REF.sub("", t)
    t = _RE_DATE_JA.sub("", t)
    t = _RE_DATE_SL.sub("", t)
    t = _RE_NENDO.sub("", t)
    t = _RE_PDF_IND.sub("", t)
    t = _RE_COPYRIGHT.sub("", t)

    lines = t.split("\n")
    t = "\n".join(ln for ln in lines if ln.strip() not in ADMIN_STOPWORDS_JA or not ln.strip())

    t = _RE_SPACE.sub(" ", t)
    t = _RE_NEWLINE.sub("\n\n", t)
    t = t.strip()
    if len(t) > CHAR_LIMIT:
        t = t[:CHAR_LIMIT]
    return t


def split_sentences_ja(text: str, min_len: int = 5) -> list:
    if not text:
        return []
    parts = _RE_SENT_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip() and len(s.strip()) >= min_len]


def _track_of_ja(alias: str) -> str:
    # get_keyword_track()과 동일 규칙. 일본어 문자가 하나도 없으면 Track C
    # (없으면 소문자 영단어가 Track A/B로 잘못 분류됨)
    if re.match(r"^[A-Z][A-Za-z0-9]{1,}$", alias):
        return "C"
    if not re.search(r"[一-鿿゠-ヿｦ-ﾟ]", alias):
        return "C"
    if len(alias) > 4:
        return "B"
    return "A"


ALIAS_POOL_JA = mention_common.build_alias_pool(ESG_CONCEPTS_JA, FP_EXCLUDE_JA, _track_of_ja)

# 단일 한자 alias(水/法 등)는 substring 매칭이라 고유명사 안에서도 걸림
# (예: お茶の水女子大学 → E(水) 오탐이 대학 전체 문서에 체계적 발생)
# 문장 매칭 전 해당 부분 문자열을 동일 길이 더미 문자로 마스킹해 위치는 보존하면서 오매칭만 차단
_JA_BLOCKED_CONTEXT = ["お茶の水"]


def _mask_blocked_context(sentence: str) -> str:
    for blocked in _JA_BLOCKED_CONTEXT:
        sentence = sentence.replace(blocked, "#" * len(blocked))
    return sentence


def run(limit_docs: int | None = None, verbose: bool = True):
    df = io_common.load_scoped_dataset()
    df_jp = df[df["lang"] == "jp"].copy()
    df_jp["doc_id"] = df_jp.index.astype(str)
    if limit_docs:
        df_jp = df_jp.head(limit_docs)

    if verbose:
        print(f"[JP 전처리] 대상 문서 {len(df_jp)}건, alias pool {len(ALIAS_POOL_JA)}개")

    cleaned_texts = {}
    all_rows = []
    for i, (doc_id, text) in enumerate(zip(df_jp["doc_id"], df_jp["full_text"])):
        cleaned = clean_text_ja(text)
        cleaned_texts[doc_id] = cleaned
        orig_sentences = split_sentences_ja(cleaned)
        masked_sentences = [_mask_blocked_context(s) for s in orig_sentences]
        rows = mention_common.extract_doc_mentions(doc_id, masked_sentences, ALIAS_POOL_JA)
        for r in rows:
            r["sentence_text"] = orig_sentences[r["sent_id"]]  # 보고서 표시용 원문 복원(매칭은 마스킹본 기준)
        all_rows.extend(rows)
        if verbose and (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(df_jp)}건 처리, 누적 mention {len(all_rows)}건")

    df_jp["cleaned_text_len"] = df_jp["doc_id"].map(lambda d: len(cleaned_texts[d]))
    n_before = len(df_jp)
    # japan/preprocessing/step1_preprocess.ipynb의 MIN_CHARS=200(정제 후 길이 기준)과 동일 적용
    short_ids = set(df_jp.loc[df_jp["cleaned_text_len"] < io_common.MIN_CLEANED_CHARS, "doc_id"])
    df_jp = df_jp[~df_jp["doc_id"].isin(short_ids)].copy()
    all_rows = [r for r in all_rows if r["doc_id"] not in short_ids]
    if verbose:
        print(
            f"[JP 전처리] UI 잔재 정제 후 {io_common.MIN_CLEANED_CHARS}자 미만 문서 제외: "
            f"{n_before}건 → {len(df_jp)}건 ({len(short_ids)}건 제외)"
        )

    mentions = pd.DataFrame(all_rows)
    if verbose:
        print(f"[JP 전처리] 완료: 문서 {len(df_jp)}건 → mention(raw) {len(mentions)}건")
    return df_jp, mentions


if __name__ == "__main__":
    docs, mentions = run()
    out_dir = cfg.OUTPUT_DIR
    docs.drop(columns=["full_text"]).to_csv(out_dir / "docs_jp.csv", index=False, encoding="utf-8-sig")
    mentions.to_csv(out_dir / "mentions_jp_raw.csv", index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_dir / 'docs_jp.csv'}, {out_dir / 'mentions_jp_raw.csv'}")
