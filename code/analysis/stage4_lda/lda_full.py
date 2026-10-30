# -*- coding: utf-8 -*-
"""단계 IV: 전체어휘 기반 일반 LDA — ESG 키워드 한정 LDA(lda_esg.py)와 별도로 병행 제시.
학과 계열이 토픽으로 잡히는 현상은 알려진 한계로 명시하되 결과 자체는 그대로 제시한다.

불용어: PROJECT_STOPWORDS와 EN_STOPWORDS(영어 학술 메타데이터·불용어)를
한국·일본에 공통 적용한다.

파라미터: 비교를 위해 한국·일본에 동일한 PARAMS를 적용한다.

토픽마다 원문 문장 예시(토픽당 최대 20개) — 그 토픽이 dominant_topic인 문서들의 원문을
문장 단위로 다시 나눠, 토픽 상위 키워드가 실제 등장한 문장을 추출한다(전체어휘 토큰은
concept 매핑이 없어 ESG-LDA와 달리 원문을 재분리해서 찾는다). 그림 chrome 텍스트(제목)는
영문으로 통일, 워드클라우드 단어 자체는 번역하지 않는다.
"""
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common import io_common
from common import plotting
from common import sample_text
from stage1_preprocess import preprocess_kr as pkr
from stage1_preprocess import preprocess_jp as pjp

# 한국·일본 공통 불용어
EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "will", "with", "without",
    "about", "into", "between", "within", "using", "based", "study",
    "studies", "research", "journal", "paper", "article", "abstract",
    "keyword", "keywords", "vol", "no", "pp", "doi", "issn", "isbn",
    "scie", "ssci", "ahci", "kci", "scopus", "orcid",
}

PROJECT_STOPWORDS_KR = {
    "대학", "대학교", "대학원", "학부", "학과", "전공", "교수", "학생", "학교",
    "교육", "연구", "과정", "교과", "교과목", "교육과정", "수업", "강의", "학점",
    "학사", "석사", "박사", "졸업", "입학", "입시", "모집", "지원", "신청",
    "안내", "소개", "공지", "공지사항", "게시판", "자료실", "문의", "번호",
    "제목", "작성자", "등록일", "조회수", "첨부", "파일", "다운로드",
    "홈페이지", "사이트", "메뉴", "바로가기", "로그인", "검색", "전체",
    "이전", "다음", "목록", "보기", "상세", "페이지", "콘텐츠", "내용",
    "관련", "운영", "관리", "센터", "사업", "프로그램", "행사",
    "활동", "분야", "정보", "기준", "현황", "개요", "목표", "비전",
    "연락처", "전화", "이메일", "주소", "위치", "캠퍼스",
    "학기", "년도", "연도", "월", "일", "차", "회", "급", "명",
    "신입생", "재학생", "졸업생", "수료", "장학", "장학생", "자격증",
    "취업", "진로", "인턴", "현장실습", "실습", "특강", "세미나",
    "layout", "unknown", "web", "jsp", "site", "inf", "artclview",
    "board", "notice", "view", "list", "contents", "content", "subview",
    "가능", "필요", "향상", "강화", "제공", "확인", "사용", "활용",
    "수행", "진행", "참여", "개최", "선발", "제출", "변경", "발급",
    "해당", "기타", "공통", "일반", "구분", "대상", "기간", "방법",
    "결과", "계획", "평가", "심사", "시험", "필독",
}

PROJECT_STOPWORDS_JP = {
    "大学", "大学院", "学部", "学科", "専攻", "課程", "教育課程", "授業", "講義", "科目",
    "学生", "教員", "教授", "研究", "研究室", "案内", "紹介", "詳細", "内容", "概要",
    "情報", "一覧", "検索", "ページ", "サイト", "リンク", "画面", "機能",
    "利用", "有効", "動作", "ブラウザ", "スキップ", "メイン", "フッター", "ヘッダー",
    "お知らせ", "新着", "更新", "資料", "問い合わせ", "電話", "住所", "所在地",
    "キャンパス", "施設", "設備", "図書館", "入試", "入学", "出願", "募集", "願書",
    "卒業", "修了", "資格", "取得", "免許", "試験", "単位", "履修", "時間割",
    "学年", "年次", "前期", "後期", "学期", "月", "日", "年度", "本学", "当学科",
    "皆さん", "皆様", "私たち", "私達", "以下", "以上", "場合", "対象", "目的",
    "特色", "特徴", "方針", "理念", "方法", "実施", "実現", "取り組み", "推進",
    "お願い", "ご覧", "アクセス", "アド", "アドミッションポリシー", "イベント", "インタビュー", "インターネット",
    "インターンシップ", "ウィンドウ", "オープン", "カテゴリ", "カリキュラム", "キャリア", "クリック", "グローバルナビゲーション",
    "サプリメントアドバイザー", "システム", "シラバス", "ステップ", "スペシャリスト", "ゼミ", "タップ", "テーマ",
    "ディプロマポリシー", "データ", "トップ", "パンフ", "パンフレット", "フード", "プロジェクト", "ホーム",
    "ホームページ", "ボタン", "ポリシー", "マップ", "ミッション", "メッセージ", "メディア", "モデル",
    "ローカルナビゲーション", "一般", "主事", "事業", "交通", "任用", "体制", "先輩",
    "先頭", "公式", "分野", "動画", "卒業生", "博物館", "印刷", "取扱い",
    "受け入れ", "受入", "受験", "受験生", "司書", "団体", "報告", "学び",
    "学位", "学外", "学校", "学芸", "寄付", "専門", "就職", "年間",
    "応援", "拡大", "授与", "採用", "推奨", "推薦", "支援", "教諭",
    "日本語", "本文", "活動", "流れ", "発行", "登録", "目次", "相談",
    "確認", "移動", "結果", "総合", "編入", "編成", "考え", "説明",
    "請求", "講師", "週間", "進学", "進路", "選抜", "開講", "領域",
    "高校",
}

STOPWORDS_BY_LANG = {"kr": PROJECT_STOPWORDS_KR, "jp": PROJECT_STOPWORDS_JP}

# 한국·일본 완전 동일 파라미터 (비교연구이므로 원칙 통일)
PARAMS = {"k_range": [4, 5, 6, 7, 8, 9, 10], "no_below": 5, "no_above": 0.5, "keep_n": 4000, "passes": 20}
SAMPLES_PER_TOPIC = 20
TOPIC_KEYWORD_TOP_N = 8

_RE_ONLY_NUM = re.compile(r"^\d+$")
_RE_HAS_DIGIT = re.compile(r"^\d+[^\d]*$|^[^\d]*\d+$")


def is_valid_token(t: str, lang: str) -> bool:
    t = str(t).strip()
    if len(t) < 2:
        return False
    if _RE_ONLY_NUM.match(t) or _RE_HAS_DIGIT.match(t):
        return False
    if t in STOPWORDS_BY_LANG[lang]:
        return False
    if t.lower() in EN_STOPWORDS:
        return False
    if re.fullmatch(r"[A-Za-z]+", t) and len(t) <= 2:
        return False
    return True


def _okt_nouns(texts: list) -> list:
    from konlpy.tag import Okt

    okt = Okt()
    out = []
    for t in texts:
        nouns = okt.nouns(t) if isinstance(t, str) and t.strip() else []
        out.append([n for n in nouns if is_valid_token(n, "kr")])
    return out


def _janome_nouns(texts: list) -> list:
    from janome.tokenizer import Tokenizer

    tok = Tokenizer()
    out = []
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            out.append([])
            continue
        nouns = []
        for token in tok.tokenize(t):
            surface = token.surface
            pos = token.part_of_speech.split(",")[0]
            if pos != "名詞":
                continue
            has_kanji = bool(re.search(r"[一-鿿]", surface))
            has_kata = bool(re.search(r"[゠-ヿｦ-ﾟ]", surface))
            if not (has_kanji or has_kata):
                continue
            if is_valid_token(surface, "jp"):
                nouns.append(surface)
        out.append(nouns)
    return out


def build_doc_tokens(lang: str, verbose: bool = True) -> pd.DataFrame:
    """docs_{lang}.csv(단계 I 최종 유효 문서)와 동일한 문서 집합에 대해, 전체 명사 어휘 토큰을 뽑는다.
    cleaned text는 저장돼 있지 않으므로 stage1 clean_text 함수로 재생성(원본 full_text 재로드).
    cleaned_text는 샘플 문장 추출에도 재사용."""
    docs_valid = pd.read_csv(cfg.OUTPUT_DIR / f"docs_{lang}.csv", dtype={"doc_id": str})
    valid_ids = set(docs_valid["doc_id"])

    df = io_common.load_scoped_dataset()
    df = df[df["lang"] == lang].copy()
    df["doc_id"] = df.index.astype(str)
    df = df[df["doc_id"].isin(valid_ids)]
    df = df[df["is_main"]].copy()  # ESG-LDA와 동일하게 main scope만 사용(두 LDA 결과를 나란히 비교하기 위함)

    if verbose:
        print(f"[{lang} 전체어휘 LDA] 대상 문서 {len(df)}건(main scope)")

    if lang == "kr":
        cleaned = [pkr.clean_text(t) for t in df["full_text"]]
        token_lists = _okt_nouns(cleaned)
    else:
        cleaned = [pjp.clean_text_ja(t) for t in df["full_text"]]
        token_lists = _janome_nouns(cleaned)

    out = pd.DataFrame({
        "doc_id": df["doc_id"].values,
        "target_field_primary": df["target_field_primary"].values,
        "tokens": token_lists,
        "cleaned_text": cleaned,
    })
    out = out[out["tokens"].apply(len) > 0]
    if verbose:
        avg_n = out["tokens"].apply(len).mean()
        print(f"[{lang} 전체어휘 LDA] 토큰 보유 문서 {len(out)}건, 문서당 평균 토큰수 {avg_n:.1f}개")
    return out


def run_lda(doc_tokens: pd.DataFrame, params: dict, seed: int = 42):
    texts = doc_tokens["tokens"].tolist()
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=params["no_below"], no_above=params["no_above"], keep_n=params["keep_n"])
    corpus = [dictionary.doc2bow(t) for t in texts]
    non_empty = [i for i, c in enumerate(corpus) if len(c) > 0]
    if len(non_empty) < 10 or len(dictionary) < 2:
        return None

    best = None
    coherence_rows = []
    for k in params["k_range"]:
        if k >= len(non_empty):
            continue
        model = LdaModel(
            corpus=[corpus[i] for i in non_empty], id2word=dictionary, num_topics=k,
            passes=params["passes"], alpha="auto", eta="auto", random_state=seed,
        )
        cm = CoherenceModel(
            model=model, texts=[texts[i] for i in non_empty], dictionary=dictionary, coherence="c_v",
        )
        c_v = cm.get_coherence()
        coherence_rows.append({"k": k, "c_v": c_v})
        if best is None or c_v > best["c_v"]:
            best = {"k": k, "c_v": c_v, "model": model, "corpus": corpus, "dictionary": dictionary,
                    "non_empty": non_empty, "doc_tokens": doc_tokens}
    if best is None:
        return None
    best["coherence_by_k"] = pd.DataFrame(coherence_rows)
    return best


def extract_topic_terms(result: dict, top_n: int = 15) -> pd.DataFrame:
    rows = []
    for topic_id in range(result["k"]):
        terms = result["model"].show_topic(topic_id, topn=top_n)
        for rank, (term, prob) in enumerate(terms, 1):
            rows.append({"topic": topic_id, "rank": rank, "term": term, "prob": round(prob, 4)})
    return pd.DataFrame(rows)


def extract_doc_dominant_topic(result: dict) -> pd.DataFrame:
    doc_tokens = result["doc_tokens"].reset_index(drop=True)
    rows = []
    for i in result["non_empty"]:
        bow = result["corpus"][i]
        topic_dist = result["model"].get_document_topics(bow)
        if not topic_dist:
            continue
        dom_topic, dom_prob = max(topic_dist, key=lambda x: x[1])
        rows.append({
            "doc_id": doc_tokens.loc[i, "doc_id"],
            "target_field_primary": doc_tokens.loc[i, "target_field_primary"],
            "dominant_topic": dom_topic, "prob": round(dom_prob, 4),
        })
    return pd.DataFrame(rows)


def sample_full_vocab_topic_sentences(
    doc_tokens: pd.DataFrame, docs_valid: pd.DataFrame, dom_df: pd.DataFrame, topic_terms: pd.DataFrame,
    lang: str, n: int = SAMPLES_PER_TOPIC, top_n_terms: int = TOPIC_KEYWORD_TOP_N,
) -> pd.DataFrame:
    """전체어휘 토큰은 concept 매핑이 없으므로, 그 토픽이 dominant_topic인 문서의 cleaned_text를
    문장 단위로 재분리해 토픽 상위 키워드가 실제 등장한 문장을 최대 n개 추출."""
    split_fn = pkr._make_sentence_splitter() if lang == "kr" else pjp.split_sentences_ja
    text_map = dict(zip(doc_tokens["doc_id"], doc_tokens["cleaned_text"]))
    meta = docs_valid.set_index("doc_id")
    all_rows = []
    for topic_id, grp in topic_terms.groupby("topic"):
        keywords = list(grp.sort_values("rank")["term"].head(top_n_terms))
        topic_doc_ids = dom_df.loc[dom_df["dominant_topic"] == topic_id, "doc_id"].tolist()
        matches = []
        for doc_id in topic_doc_ids:
            text = text_map.get(doc_id, "")
            if not text:
                continue
            for sent in split_fn(text):
                hit = next((kw for kw in keywords if kw in sent), None)
                if not hit:
                    continue
                row = {"doc_id": doc_id, "matched_keyword": hit, "sentence_text": sent}
                if doc_id in meta.index:
                    row["university"] = meta.loc[doc_id].get("university", "")
                    row["title"] = meta.loc[doc_id].get("title", "")
                    row["url"] = meta.loc[doc_id].get("url", "")
                matches.append(row)
        if not matches:
            continue
        picked = sample_text.sample_rows(pd.DataFrame(matches), n)
        for _, r in picked.iterrows():
            all_rows.append({
                "topic": topic_id, "matched_keyword": r.get("matched_keyword", ""),
                "university": r.get("university", ""), "title": r.get("title", ""),
                "url": r.get("url", ""), "sentence": r.get("sentence_text", ""),
            })
    return pd.DataFrame(all_rows)


def plot_coherence(coherence_df: pd.DataFrame, lang: str, out_path: Path):
    plotting.set_font(lang)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coherence_df["k"], coherence_df["c_v"], marker="o")
    ax.set_title("c_v Coherence by Number of Topics (k) — Full-Vocabulary LDA")
    ax.set_xlabel("k")
    ax.set_ylabel("c_v")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wordclouds(result: dict, lang: str, out_dir: Path, top_n: int = 30):
    plotting.set_font(lang)
    font_path = plotting._FONT_PATHS.get(lang)
    for topic_id in range(result["k"]):
        weights = dict(result["model"].show_topic(topic_id, topn=top_n))
        wc = WordCloud(
            font_path=font_path, background_color="white",
            width=900, height=650, prefer_horizontal=0.9, colormap="tab10",
        ).generate_from_frequencies(weights)
        fig, ax = plt.subplots(figsize=(6, 4.3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {topic_id} Word Cloud — Full-Vocabulary LDA", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"전체어휘_topic{topic_id}_wordcloud_{lang}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def build_stage4_full_report(lang: str):
    out_dir = cfg.OUTPUT_DIR / lang / "단계4_전체어휘_LDA"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_tokens = build_doc_tokens(lang)
    result = run_lda(doc_tokens, PARAMS)
    if not result:
        print(f"[{lang}] 전체어휘 LDA 실패(데이터 부족)")
        return None

    terms = extract_topic_terms(result)
    dom = extract_doc_dominant_topic(result)
    terms.to_csv(out_dir / f"topic_terms_{lang}.csv", index=False, encoding="utf-8-sig")
    dom.to_csv(out_dir / f"doc_dominant_topic_{lang}.csv", index=False, encoding="utf-8-sig")
    result["coherence_by_k"].to_csv(out_dir / f"coherence_by_k_{lang}.csv", index=False)
    plot_coherence(result["coherence_by_k"], lang, out_dir / f"coherence_{lang}.png")
    plot_wordclouds(result, lang, out_dir)

    docs_valid = pd.read_csv(cfg.OUTPUT_DIR / f"docs_{lang}.csv", dtype={"doc_id": str})
    samples = sample_full_vocab_topic_sentences(doc_tokens, docs_valid, dom, terms, lang)
    samples.to_csv(out_dir / f"topic_samples_{lang}.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "scope": "전체어휘(main scope)", "n_docs": len(doc_tokens),
        "best_k": result["k"], "c_v": round(result["c_v"], 4),
        "vocab_size": len(result["dictionary"]),
    }])
    summary.to_csv(out_dir / f"lda_full_summary_{lang}.csv", index=False, encoding="utf-8-sig")
    print(f"[{lang}] 전체어휘 LDA: k={result['k']}, c_v={result['c_v']:.4f}, 문서 {len(doc_tokens)}건, "
          f"어휘 {len(result['dictionary'])}개, 샘플 {len(samples)}건")
    print(f"[{lang}] 단계 IV 전체어휘 LDA 저장 완료: {out_dir}")
    return summary


if __name__ == "__main__":
    for _lang in cfg.LANGS:
        build_stage4_full_report(_lang)
