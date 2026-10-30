# -*- coding: utf-8 -*-
"""단계 III 연관분석(mention 기반) — 한국·일본 공용 모듈.

문장 단위 selected mention 조합으로 edge를 생성한다(문자열 재탐색 없음).
네트워크 그림 + 단순 등장횟수 + Ochiai 계수를 한국·일본 동일 형식으로 산출.

- anchor_sent_n: 개념이 등장한 문장 수
- co_sent_n: 두 개념이 같은 문장에 함께 등장한 문장 수 (= 단순 등장횟수)
- ochiai = co_sent_n / sqrt(anchor_sent_n_a * anchor_sent_n_b)
- 내부 중첩 쌍(한 canonical이 다른 canonical의 부분 문자열) 자동 차단 — 수동 나열 대신 일반
  규칙으로 구현해 두 언어 모두에 동일하게 적용

그림/표에 실제로 등장하는 상위 쌍마다 원문 문장 예시(쌍당 최대 4개, common/sample_text.py 기준)를
산출. 그림 chrome 텍스트(제목)는 영문으로 통일하되, 네트워크 노드/막대그래프의 concept 자체
(원문 키워드)는 번역하지 않고 그대로 둔다.
"""
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common import plotting
from common import sample_text
from common.io_common import load_enriched

MIN_ANCHOR_SENT_N = 5
MIN_CO_SENT_N = 3
MIN_UNIV_N = 1
SAMPLES_PER_EDGE = 4


def _is_nested(a: str, b: str) -> bool:
    a_n, b_n = a.replace(" ", ""), b.replace(" ", "")
    return a_n != b_n and (a_n in b_n or b_n in a_n)


def build_sentence_concept_table(merged: pd.DataFrame) -> pd.DataFrame:
    """(doc_id, sent_id)별 등장 canonical 집합 + 대학명."""
    m_main = merged[merged["is_main"]]
    grouped = (
        m_main.groupby(["doc_id", "sent_id", "university"])["canonical"]
        .apply(lambda s: sorted(set(s)))
        .reset_index()
    )
    return grouped


def compute_anchor_sent_n(sentence_table: pd.DataFrame) -> dict:
    """개념(canonical)별 등장 문장 수(anchor_sent_n) — concept node 자체의 빈도.
    compute_edges의 쌍(edge) 필터링에도 쓰이고, build_top_concept_nodes의 순위 산정에도 쓰임."""
    anchor_sent = {}
    for concepts in sentence_table["canonical"]:
        for c in set(concepts):
            anchor_sent[c] = anchor_sent.get(c, 0) + 1
    return anchor_sent


def compute_edges(sentence_table: pd.DataFrame, category_of: dict) -> pd.DataFrame:
    anchor_sent = compute_anchor_sent_n(sentence_table)

    pair_co_sent = {}
    pair_univ = {}
    for concepts, univ in zip(sentence_table["canonical"], sentence_table["university"]):
        uniq = sorted(set(concepts))
        for a, b in itertools.combinations(uniq, 2):
            if _is_nested(a, b):
                continue
            key = (a, b)
            pair_co_sent[key] = pair_co_sent.get(key, 0) + 1
            pair_univ.setdefault(key, set()).add(univ)

    rows = []
    for (a, b), co_n in pair_co_sent.items():
        n_a, n_b = anchor_sent.get(a, 0), anchor_sent.get(b, 0)
        if n_a < MIN_ANCHOR_SENT_N or n_b < MIN_ANCHOR_SENT_N:
            continue
        if co_n < MIN_CO_SENT_N:
            continue
        univ_n = len(pair_univ[(a, b)])
        if univ_n < MIN_UNIV_N:
            continue
        ochiai = co_n / (n_a * n_b) ** 0.5
        rows.append({
            "concept_a": a, "concept_b": b,
            "category_a": category_of.get(a, ""), "category_b": category_of.get(b, ""),
            "anchor_sent_n_a": n_a, "anchor_sent_n_b": n_b,
            "co_sent_n": co_n, "univ_n": univ_n, "ochiai": round(ochiai, 4),
        })
    return pd.DataFrame(rows).sort_values("ochiai", ascending=False)


def _category_of_map(merged: pd.DataFrame) -> dict:
    return merged.drop_duplicates("canonical").set_index("canonical")["category"].to_dict()


def sample_edge_sentences(merged: pd.DataFrame, docs: pd.DataFrame, edges: pd.DataFrame, n: int = SAMPLES_PER_EDGE) -> pd.DataFrame:
    """edges(concept_a/concept_b 열 보유)의 각 쌍마다, 두 concept이 같은 문장에 함께 등장한
    원문 문장을 쌍당 최대 n개 추출한다."""
    m_main = merged[merged["is_main"]]
    all_rows = []
    for _, e in edges.iterrows():
        a, b = e["concept_a"], e["concept_b"]
        sub = m_main[m_main["canonical"].isin([a, b])]
        grp = sub.groupby(["doc_id", "sent_id"])["canonical"].apply(lambda s: set(s))
        valid = grp[grp.apply(lambda s: a in s and b in s)]
        if valid.empty:
            continue
        valid_idx = pd.MultiIndex.from_tuples(valid.index, names=["doc_id", "sent_id"])
        sub_indexed = sub.set_index(["doc_id", "sent_id"])
        rep = sub_indexed.loc[sub_indexed.index.isin(valid_idx)].reset_index().drop_duplicates(
            subset=["doc_id", "sent_id"]
        )
        rep = sample_text.attach_doc_meta(rep, docs)
        picked = sample_text.sample_rows(rep, n, keyword_hint=[a, b])
        for _, r in picked.iterrows():
            all_rows.append({
                "edge_pair": f"{a} x {b}", "concept_a": a, "concept_b": b,
                "university": r.get("university", ""), "title": r.get("title", ""),
                "url": r.get("url", ""), "sentence": r.get("sentence_text", ""),
            })
    return pd.DataFrame(all_rows)


def plot_network(edges: pd.DataFrame, lang: str, out_path: Path, top_n: int = 40, scope_label_en: str = ""):
    top = edges.sort_values("co_sent_n", ascending=False).head(top_n)
    g = nx.Graph()
    for _, r in top.iterrows():
        g.add_edge(r["concept_a"], r["concept_b"], weight=r["ochiai"], co_n=r["co_sent_n"])

    plotting.set_font(lang)  # 노드 라벨은 원문 concept 그대로 표시 — CJK 폰트 필요
    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(g, seed=42, k=0.6)
    weights = [g[u][v]["weight"] * 20 for u, v in g.edges()]
    node_sizes = [300 + 60 * g.degree(n) for n in g.nodes()]
    font_family = plt.rcParams["font.family"][0]
    nx.draw_networkx_edges(g, pos, width=weights, edge_color="gray", alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color="tab:blue", alpha=0.85, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=9, font_family=font_family, ax=ax)
    suffix = f" — {scope_label_en}" if scope_label_en else ""
    title = f"ESG Concept Co-occurrence Network (Top {top_n} pairs, main scope){suffix}"
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# D. 계열별 핵심 연관어 네트워크
def build_target_field_networks(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path, min_docs: int = 30):
    """target_field별로 슬라이싱해 기존 네트워크 로직 재사용. 문서 30건 미만 계열은 스킵(LDA scope B와 동일 기준)."""
    category_of = _category_of_map(merged)
    m_main = merged[merged["is_main"]]
    summary_rows = []
    for tf in cfg.TARGET_FIELD_ORDER:
        label = cfg.TARGET_FIELD_LABELS[tf][lang]
        label_en = cfg.TARGET_FIELD_LABELS[tf]["en"]
        sub = m_main[m_main["target_field_primary"] == tf]
        n_docs = sub["doc_id"].nunique()
        if n_docs < min_docs:
            print(f"[{lang}] D. target_field={tf}({label}): 문서 {n_docs}건 < {min_docs}건 → 스킵")
            summary_rows.append({"target_field": tf, "label": label, "n_docs": n_docs, "n_edges": None, "skipped": True})
            continue
        sentence_table = build_sentence_concept_table(sub)
        edges = compute_edges(sentence_table, category_of)
        if edges.empty:
            print(f"[{lang}] D. target_field={tf}({label}): edge 없음(문서 {n_docs}건)")
            summary_rows.append({"target_field": tf, "label": label, "n_docs": n_docs, "n_edges": 0, "skipped": False})
            continue
        edges.to_csv(out_dir / f"D_연관어_{label}_{lang}.csv", index=False, encoding="utf-8-sig")
        plot_network(edges, lang, out_dir / f"D_network_{label}_{lang}.png", top_n=25, scope_label_en=label_en)
        top25 = edges.sort_values("co_sent_n", ascending=False).head(25)
        samples = sample_edge_sentences(sub, docs, top25)
        samples.to_csv(out_dir / f"D_연관어_샘플_{label}_{lang}.csv", index=False, encoding="utf-8-sig")
        summary_rows.append({"target_field": tf, "label": label, "n_docs": n_docs, "n_edges": len(edges), "skipped": False})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"D_계열별네트워크_요약_{lang}.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_top_pairs_bar(edges: pd.DataFrame, value_col: str, lang: str, out_path: Path, top_n: int = 20):
    top = edges.sort_values(value_col, ascending=False).head(top_n).copy()
    top["pair_label"] = top["concept_a"] + " × " + top["concept_b"]
    top = top.sort_values(value_col, ascending=True)

    plotting.set_font(lang)  # 막대 y축 라벨(concept 원문) 표시에 CJK 폰트 필요
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(top) + 1.5))
    color = "tab:purple" if value_col == "ochiai" else "tab:blue"
    ax.barh(top["pair_label"], top[value_col], color=color)
    for i, v in enumerate(top[value_col]):
        ax.text(v, i, f" {v:.3f}" if value_col == "ochiai" else f" {int(v)}", va="center", fontsize=8)
    metric_label = {
        "co_sent_n": "Top Pairs by Raw Co-occurrence Count (sentences)",
        "ochiai": "Top Pairs by Ochiai Coefficient",
    }[value_col]
    ax.set_title(f"{metric_label} (main scope, n>={MIN_CO_SENT_N})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# E. 상위 N개 ESG concept node
# top_cooccurrence/top_ochiai는 쌍(edge) 순위이고, 이건 개별 키워드(node) 자체의
# 등장 문장 수(anchor_sent_n) 순위. 한일 양쪽 다 산출.
def build_top_concept_nodes(sentence_table: pd.DataFrame, category_of: dict, top_n: int = None) -> pd.DataFrame:
    """top_n=None이면 전체 concept 순위(상위 25개 CSV와 함께 전체 데이터도 첨부)."""
    anchor_sent = compute_anchor_sent_n(sentence_table)
    table = pd.DataFrame(
        [{"concept": c, "category": category_of.get(c, ""), "sent_n": n} for c, n in anchor_sent.items()]
    )
    table = table.sort_values("sent_n", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.head(top_n) if top_n is not None else table


CATEGORY_COLORS = {
    "S": "tab:blue", "E": "tab:green", "G": "tab:brown",
    "DUP_ES": "tab:purple", "DUP_SG": "tab:orange", "DUP_EG": "tab:red", "ESG_META": "tab:gray",
}


def plot_top_concept_nodes(table: pd.DataFrame, lang: str, out_path: Path, top_n: int = 25):
    top = table.sort_values("sent_n", ascending=True)  # 가로막대: 위쪽이 1위가 되도록
    plotting.set_font(lang)  # 막대 y축 라벨(concept 원문) 표시에 CJK 폰트 필요
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(top) + 1.5))
    colors = [CATEGORY_COLORS.get(c, "tab:gray") for c in top["category"]]
    ax.barh(top["concept"], top["sent_n"], color=colors)
    for i, v in enumerate(top["sent_n"]):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
    ax.set_title(f"Top {top_n} ESG Concept Nodes (Sentence Count, Main Scope)")
    ax.set_xlabel("Sentence Count")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c]) for c in dict.fromkeys(top["category"])]
    ax.legend(handles, list(dict.fromkeys(top["category"])), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_stage3_report(lang: str):
    out_dir = cfg.OUTPUT_DIR / lang / "단계3_연관분석"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged, docs = load_enriched(lang)

    sentence_table = build_sentence_concept_table(merged)
    category_of = _category_of_map(merged)
    edges = compute_edges(sentence_table, category_of)
    edges.to_csv(out_dir / f"concept_edges_{lang}.csv", index=False, encoding="utf-8-sig")

    plot_network(edges, lang, out_dir / f"network_{lang}.png")
    plot_top_pairs_bar(edges, "co_sent_n", lang, out_dir / f"top_cooccurrence_{lang}.png")
    plot_top_pairs_bar(edges, "ochiai", lang, out_dir / f"top_ochiai_{lang}.png")

    all_nodes = build_top_concept_nodes(sentence_table, category_of, top_n=None)
    all_nodes.to_csv(out_dir / f"E_전체_concept_node_{lang}.csv", index=False, encoding="utf-8-sig")
    top_nodes = all_nodes.head(25)
    top_nodes.to_csv(out_dir / f"E_상위25_concept_node_{lang}.csv", index=False, encoding="utf-8-sig")
    plot_top_concept_nodes(top_nodes, lang, out_dir / f"top_concept_nodes_{lang}.png", top_n=25)

    tf_network_summary = build_target_field_networks(merged, docs, lang, out_dir)

    # 상위 쌍(막대그래프에 실제로 표시되는 top20 x 2지표, 중복 제거) 원문 샘플
    shown_pairs = pd.concat([
        edges.sort_values("co_sent_n", ascending=False).head(20),
        edges.sort_values("ochiai", ascending=False).head(20),
    ]).drop_duplicates(subset=["concept_a", "concept_b"])
    edge_samples = sample_edge_sentences(merged, docs, shown_pairs)
    edge_samples.to_csv(out_dir / f"edge_samples_{lang}.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_dir / f"연관분석결과_{lang}.xlsx") as writer:
        edges.to_excel(writer, sheet_name="concept_edges", index=False)
        edges.sort_values("co_sent_n", ascending=False).head(50).to_excel(
            writer, sheet_name="top50_등장횟수", index=False)
        edges.sort_values("ochiai", ascending=False).head(50).to_excel(
            writer, sheet_name="top50_ochiai", index=False)
        tf_network_summary.to_excel(writer, sheet_name="D_계열별네트워크_요약", index=False)
        all_nodes.to_excel(writer, sheet_name="E_전체_concept_node", index=False)
        top_nodes.to_excel(writer, sheet_name="E_상위25_concept_node", index=False)

    print(f"[{lang}] 단계 III 연관분석 저장 완료: {out_dir} (edge {len(edges)}개, 샘플 {len(edge_samples)}건)")
    return edges


if __name__ == "__main__":
    for _lang in cfg.LANGS:
        build_stage3_report(_lang)
