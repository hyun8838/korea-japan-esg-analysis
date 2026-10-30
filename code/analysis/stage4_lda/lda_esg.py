# -*- coding: utf-8 -*-
"""단계 IV: ESG 키워드 한정 LDA를 한국·일본 공용 모듈로 실행한다.

핵심 방법: 전체어휘가 아니라 '선택된(selected) ESG concept 토큰'만 문서별 bag-of-words로 사용.
전처리 기준:
  - canonical 병합(한/영, 한국은 이미 stage1에서 canonical로 병합됨 — 별도 처리 불필요)
  - ESG LDA 전용 추가 불용어(범용어 제거, 계열 판별력 없는 concept)
scope A(전체) + scope B(target_field별, 최소 30건) 둘 다 산출.

토픽마다 원문 문장 예시(토픽당 최대 20개, common/sample_text.py 기준)를 산출 — 그 토픽이
dominant_topic인 문서들 중, 그 토픽 상위 키워드가 실제로 등장한 문장을 뽑는다. 그림 chrome
텍스트(제목)는 영문으로 통일하고, 워드클라우드 자체의 단어(원문 concept)는 번역하지 않는다.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common import plotting
from common import sample_text
from common.io_common import load_enriched

# 범용어(계열 구분력 없음, ESG 사전에는 있으나 전 분야에 고르게 등장) 제외
ESG_LDA_STOPWORDS = {
    "kr": {"교육", "개발", "관리", "관계", "행동", "실천", "활동"},
    "jp": {"教育", "開発", "管理", "関係", "行動", "実践", "活動"},
}

PARAMS = {
    "A": {"k_range": [2, 3, 4, 5, 6], "no_below": 2, "no_above": 0.70, "keep_n": 300, "passes": 10},
    "B": {"k_range": [2, 3, 4], "no_below": 1, "no_above": 0.80, "keep_n": 100, "passes": 10},
}
MIN_DOCS_PER_FIELD = 30
SAMPLES_PER_TOPIC = 20
TOPIC_KEYWORD_TOP_N = 8  # 샘플 문장 매칭에 쓸 토픽 상위 키워드 개수


def build_doc_tokens(merged: pd.DataFrame, lang: str, extra_stopwords: bool = True) -> pd.DataFrame:
    """doc_id별 selected canonical 토큰 리스트(main scope)."""
    m_main = merged[merged["is_main"]].copy()
    stop = ESG_LDA_STOPWORDS[lang] if extra_stopwords else set()
    m_main = m_main[~m_main["canonical"].isin(stop)]
    grouped = (
        m_main.groupby(["doc_id", "target_field_primary"])["canonical"]
        .apply(list)
        .reset_index()
        .rename(columns={"canonical": "tokens"})
    )
    grouped = grouped[grouped["tokens"].apply(len) > 0]
    return grouped


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
                    "non_empty": non_empty}
    if best is None:
        return None
    best["coherence_by_k"] = pd.DataFrame(coherence_rows)
    best["doc_tokens"] = doc_tokens
    return best


def extract_topic_terms(result: dict, top_n: int = 12) -> pd.DataFrame:
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


def sample_topic_sentences(
    merged: pd.DataFrame, docs: pd.DataFrame, dom_df: pd.DataFrame, topic_terms: pd.DataFrame,
    n: int = SAMPLES_PER_TOPIC, top_n_terms: int = TOPIC_KEYWORD_TOP_N,
) -> pd.DataFrame:
    """토픽마다: 그 토픽이 dominant_topic인 문서들 중, 토픽 상위 키워드가 실제 등장한 문장을
    최대 n개 추출(대학 다양성 우선, common/sample_text.py 기준)."""
    m_main = merged[merged["is_main"]]
    all_rows = []
    for topic_id, grp in topic_terms.groupby("topic"):
        keywords = set(grp.sort_values("rank")["term"].head(top_n_terms))
        topic_doc_ids = set(dom_df.loc[dom_df["dominant_topic"] == topic_id, "doc_id"])
        if not topic_doc_ids or not keywords:
            continue
        sub = m_main[m_main["doc_id"].isin(topic_doc_ids) & m_main["canonical"].isin(keywords)]
        sub = sub.drop_duplicates(subset=["doc_id", "sent_id"])
        sub = sample_text.attach_doc_meta(sub, docs)
        picked = sample_text.sample_rows(sub, n, keyword_hint=list(keywords))
        for _, r in picked.iterrows():
            all_rows.append({
                "topic": topic_id, "matched_keyword": r.get("canonical", ""),
                "university": r.get("university", ""), "title": r.get("title", ""),
                "url": r.get("url", ""), "sentence": r.get("sentence_text", ""),
            })
    return pd.DataFrame(all_rows)


def plot_coherence(coherence_df: pd.DataFrame, lang: str, scope_label_en: str, out_path: Path):
    plotting.set_font(lang)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coherence_df["k"], coherence_df["c_v"], marker="o")
    ax.set_title(f"c_v Coherence by Number of Topics (k) — {scope_label_en}")
    ax.set_xlabel("k")
    ax.set_ylabel("c_v")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wordclouds(result: dict, lang: str, scope_label_en: str, out_dir: Path, prefix: str, top_n: int = 30):
    """토픽별 상위 단어를 워드클라우드로 표시한다."""
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
        ax.set_title(f"Topic {topic_id} Word Cloud — {scope_label_en}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_topic{topic_id}_wordcloud_{lang}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def build_stage4_report(lang: str):
    out_dir = cfg.OUTPUT_DIR / lang / "단계4_ESG_LDA"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged, docs = load_enriched(lang)
    doc_tokens_all = build_doc_tokens(merged, lang)

    summary_rows = []

    # ── scope A: 전체 ──
    result_a = run_lda(doc_tokens_all, PARAMS["A"])
    if result_a:
        terms_a = extract_topic_terms(result_a)
        dom_a = extract_doc_dominant_topic(result_a)
        terms_a.to_csv(out_dir / f"A_전체_topic_terms_{lang}.csv", index=False, encoding="utf-8-sig")
        dom_a.to_csv(out_dir / f"A_전체_doc_dominant_topic_{lang}.csv", index=False, encoding="utf-8-sig")
        result_a["coherence_by_k"].to_csv(out_dir / f"A_전체_coherence_by_k_{lang}.csv", index=False)
        plot_coherence(result_a["coherence_by_k"], lang, "A (All)", out_dir / f"A_전체_coherence_{lang}.png")
        plot_wordclouds(result_a, lang, "A (All)", out_dir, prefix="A_전체")
        samples_a = sample_topic_sentences(merged, docs, dom_a, terms_a)
        samples_a.to_csv(out_dir / f"A_전체_topic_samples_{lang}.csv", index=False, encoding="utf-8-sig")
        summary_rows.append({"scope": "A_전체", "target_field": "", "n_docs": len(doc_tokens_all),
                              "best_k": result_a["k"], "c_v": round(result_a["c_v"], 4)})
        print(f"[{lang}] A(전체) LDA: k={result_a['k']}, c_v={result_a['c_v']:.4f}, 문서 {len(doc_tokens_all)}건, "
              f"샘플 {len(samples_a)}건")

    # ── scope B: target_field별 (최소 30건) ──
    for tf in cfg.TARGET_FIELD_ORDER:
        sub = doc_tokens_all[doc_tokens_all["target_field_primary"] == tf]
        if len(sub) < MIN_DOCS_PER_FIELD:
            print(f"[{lang}] target_field={tf}: {len(sub)}건 < {MIN_DOCS_PER_FIELD}건 → SKIP")
            continue
        result_b = run_lda(sub, PARAMS["B"])
        label = cfg.TARGET_FIELD_LABELS[tf][lang]
        label_en = cfg.TARGET_FIELD_LABELS[tf]["en"]
        if not result_b:
            print(f"[{lang}] target_field={tf}({label}): LDA 실패/데이터 부족 → SKIP")
            continue
        terms_b = extract_topic_terms(result_b)
        dom_b = extract_doc_dominant_topic(result_b)
        terms_b.to_csv(out_dir / f"B_{label}_topic_terms_{lang}.csv", index=False, encoding="utf-8-sig")
        plot_wordclouds(result_b, lang, f"B ({label_en})", out_dir, prefix=f"B_{label}")
        samples_b = sample_topic_sentences(merged, docs, dom_b, terms_b)
        samples_b.to_csv(out_dir / f"B_{label}_topic_samples_{lang}.csv", index=False, encoding="utf-8-sig")
        summary_rows.append({"scope": f"B_{label}", "target_field": tf, "n_docs": len(sub),
                              "best_k": result_b["k"], "c_v": round(result_b["c_v"], 4)})
        print(f"[{lang}] B({label}) LDA: k={result_b['k']}, c_v={result_b['c_v']:.4f}, 문서 {len(sub)}건, "
              f"샘플 {len(samples_b)}건")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"lda_summary_{lang}.csv", index=False, encoding="utf-8-sig")
    print(f"[{lang}] 단계 IV ESG-LDA 저장 완료: {out_dir}")
    return summary


SEED = 42


def doc_topic_distribution(result: dict, docs: pd.DataFrame) -> pd.DataFrame:
    """문서별 토픽 확률분포(전 토픽) + dominant_topic + 토픽별 토큰 질량 기여율.

    컬럼 정의
      topic{t}_prob        : 문서의 토픽 t 확률 θ_dt (minimum_probability=0.0, 합=1)
      topic{t}_token_mass  : θ_dt × n_model_tokens = 그 문서가 토픽 t에 낸 기대 토큰 수
      topic{t}_mass_share  : topic{t}_token_mass ÷ (계열 전체 문서의 topic{t}_token_mass 합)
    보고서에 인용된 '토픽 확률질량 기여 %'는 topic{t}_mass_share(= 길이 가중) 기준이다.
    문서 확률 θ만 단순 평균하면 짧은 문서가 과대대표되므로 길이 가중을 쓴다.
    """
    doc_tokens = result["doc_tokens"].reset_index(drop=True)
    k = result["k"]
    rows = []
    for i in result["non_empty"]:
        bow = result["corpus"][i]
        dist = dict(result["model"].get_document_topics(bow, minimum_probability=0.0))
        if not dist:
            continue
        probs = [float(dist.get(t, 0.0)) for t in range(k)]
        dom = max(range(k), key=lambda t: probs[t])
        n_model_tokens = int(sum(c for _, c in bow))
        row = {
            "doc_id": doc_tokens.loc[i, "doc_id"],
            "target_field_primary": doc_tokens.loc[i, "target_field_primary"],
            "n_esg_tokens": len(doc_tokens.loc[i, "tokens"]),
            "n_model_tokens": n_model_tokens,
        }
        for t in range(k):
            row[f"topic{t}_prob"] = round(probs[t], 4)
        for t in range(k):
            row[f"topic{t}_token_mass"] = round(probs[t] * n_model_tokens, 4)
        row["dominant_topic"] = dom
        row["dominant_prob"] = round(probs[dom], 4)
        rows.append(row)
    df = pd.DataFrame(rows)

    for t in range(k):
        col = f"topic{t}_token_mass"
        total = df[col].sum()
        df[f"topic{t}_mass_share"] = (df[col] / total).round(4) if total > 0 else 0.0

    meta = docs[["doc_id", "entity_id", "university", "title", "url", "page_type_reviewed"]]
    df = df.merge(meta, on="doc_id", how="left")
    front = ["doc_id", "entity_id", "university", "title", "url", "page_type_reviewed",
             "target_field_primary", "n_esg_tokens", "n_model_tokens"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].sort_values("doc_id").reset_index(drop=True)


def export_scope_b_doc_topic_dist(lang: str, only_fields=None):
    out_dir = cfg.OUTPUT_DIR / lang / "단계4_ESG_LDA"
    merged, docs = load_enriched(lang)
    docs = docs.copy()
    docs["doc_id"] = docs["doc_id"].astype(str)
    doc_tokens_all = build_doc_tokens(merged, lang)

    summary_path = out_dir / f"lda_summary_{lang}.csv"
    stored = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    for tf in cfg.TARGET_FIELD_ORDER:
        label = cfg.TARGET_FIELD_LABELS[tf][lang]
        if only_fields and label not in only_fields:
            continue
        sub = doc_tokens_all[doc_tokens_all["target_field_primary"] == tf]
        if len(sub) < MIN_DOCS_PER_FIELD:
            continue
        result = run_lda(sub, PARAMS["B"], seed=SEED)
        if not result:
            continue
        dist = doc_topic_distribution(result, docs)
        # extract_doc_dominant_topic(파이프라인 본체 로직)과 dominant_topic 일치 확인
        ref = extract_doc_dominant_topic(result)
        chk = dist.merge(ref, on="doc_id", suffixes=("", "_ref"))
        assert (chk["dominant_topic"] == chk["dominant_topic_ref"]).all(), f"{label}: dominant_topic 불일치"
        path = out_dir / f"B_{label}_doc_topic_dist_{lang}.csv"
        dist.to_csv(path, index=False, encoding="utf-8-sig")

        check = ""
        if len(stored):
            row = stored[stored["scope"] == f"B_{label}"]
            if len(row):
                same_k = int(row.iloc[0]["best_k"]) == result["k"]
                same_cv = abs(float(row.iloc[0]["c_v"]) - round(result["c_v"], 4)) < 1e-9
                same_n = int(row.iloc[0]["n_docs"]) == len(sub)
                check = f" | 저장본 대조 k={same_k}, c_v={same_cv}, n_docs={same_n}"
        print(f"[{lang}] B({label}): n_docs={len(sub)}, k={result['k']}, "
              f"c_v={result['c_v']:.4f}, 저장 {len(dist)}행 → {path.name}{check}")


if __name__ == "__main__":
    if "--export-doc-topics" in sys.argv[1:]:
        langs = [arg for arg in sys.argv[1:] if arg != "--export-doc-topics"] or list(cfg.LANGS)
        for _lang in langs:
            export_scope_b_doc_topic_dist(_lang)
    else:
        for _lang in cfg.LANGS:
            build_stage4_report(_lang)
