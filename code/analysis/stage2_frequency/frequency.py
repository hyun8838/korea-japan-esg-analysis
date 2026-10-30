# -*- coding: utf-8 -*-
"""단계 II 빈도분석 — 한국·일본 동일 코드로 결과표와 그림을 산출한다.

산출 5종:
  1. ESG 범주별 개요 (막대 main/all + 파이 2개)
  2. 학문분야별(target_field) ESG 언급밀도 3패널
  3. target_field × 상위 15 concept 빈도 히트맵
  4. 대학별 ESG 언급 문서 비율(%)
  5. page_type별 ESG mention 밀도(1,000자당)

한국(kr)·일본(jp) 모두 이 모듈 하나로 처리 — 언어별로 다른 것은 CSV 원문 라벨과 폰트뿐이고,
그림(PNG) 안의 제목·축·범례는 영문(en 라벨)으로 통일. 단, 히트맵 키워드 축은 분석 데이터
자체이므로 번역하지 않고 원문 그대로 둔다."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config as cfg
from common import plotting
from common.io_common import load_enriched


def _label(code, lang: str, table: dict) -> str:
    """CSV 등 원문 표기가 필요한 곳에 쓰는 라벨(한국어/일본어 원문)."""
    entry = table.get(code)
    return entry[lang] if entry else str(code)


def _label_en(code, table: dict) -> str:
    """그림(PNG) chrome 텍스트에 쓰는 영문 라벨(두 교수 공통으로 읽을 수 있게 통일)."""
    entry = table.get(code)
    return entry["en"] if entry else str(code)


# 1. ESG 범주별 개요
def build_category_overview(merged: pd.DataFrame, lang: str, out_dir: Path):
    cats_present = [c for c in cfg.ESG_CATEGORY_ORDER if c in merged["category"].unique()]
    color_map = {c: f"C{i}" for i, c in enumerate(cfg.ESG_CATEGORY_ORDER)}

    main_counts = merged[merged["is_main"]]["category"].value_counts().reindex(cats_present).fillna(0)
    all_counts = merged["category"].value_counts().reindex(cats_present).fillna(0)

    table = pd.DataFrame({"category": cats_present, "main": main_counts.values, "all": all_counts.values})
    table.to_csv(out_dir / f"1_ESG범주별_개요_{lang}.csv", index=False, encoding="utf-8-sig")

    plotting.set_font(lang)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = range(len(cats_present))
    w = 0.35
    axes[0].bar([i - w / 2 for i in x], table["main"], width=w, label=f"main (n={int(table['main'].sum()):,})")
    axes[0].bar([i + w / 2 for i in x], table["all"], width=w, label=f"all (n={int(table['all'].sum()):,})")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(cats_present, rotation=30)
    axes[0].set_ylabel("Count (concept-deduplicated)")
    axes[0].set_title("ESG Category Frequency\nMain vs All Scope")
    axes[0].legend()
    for i, v in enumerate(table["main"]):
        axes[0].text(i - w / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(table["all"]):
        axes[0].text(i + w / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)

    for ax, col in zip(axes[1:], ["main", "all"]):
        vals = table[col]
        colors = [color_map[c] for c in cats_present]
        total = int(vals.sum())
        wedges, _, autotexts = ax.pie(
            vals, autopct=lambda p: f"{p:.1f}%" if p >= 2 else "", colors=colors, startangle=90,
            pctdistance=0.75,
        )
        legend_labels = [f"{c} ({int(v):,})" for c, v in zip(cats_present, vals)]
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        ax.set_title(f"ESG Category Share -- {col} scope\n(Total {total:,})")

    fig.text(
        0.5, -0.02,
        "Note: S-category dominance reflects lexical overlap between field vocabulary and S keywords; "
        "not a measure of ESG practice level.",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"1_ESG범주별_개요_{lang}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


# 2. 학문분야별(target_field) ESG 언급밀도 3패널
def build_target_field_density(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path):
    m_main = merged[merged["is_main"]]
    d_main = docs[docs["is_main"]]

    rows = []
    for tf in cfg.TARGET_FIELD_ORDER:
        tf_docs = d_main[d_main["target_field_primary"] == tf]
        tf_mentions = m_main[m_main["target_field_primary"] == tf]
        n_docs = len(tf_docs)
        if n_docs == 0:
            continue
        n_mentions = len(tf_mentions)
        n_chars = tf_docs["full_text_len"].sum()
        n_esg_docs = tf_docs["doc_id"].isin(tf_mentions["doc_id"]).sum()
        rows.append({
            "target_field": tf,
            "label": _label(tf, lang, cfg.TARGET_FIELD_LABELS),
            "label_en": _label_en(tf, cfg.TARGET_FIELD_LABELS),
            "mention_per_doc": n_mentions / n_docs,
            "mention_per_1000char": n_mentions / n_chars * 1000 if n_chars else 0,
            "esg_doc_ratio_pct": n_esg_docs / n_docs * 100,
            "n_docs": n_docs,
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"2_학문분야별_밀도_{lang}.csv", index=False, encoding="utf-8-sig")

    plotting.set_font(lang)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        ("mention_per_doc", "tab:blue", "Mentions per Document by Academic Field\n(main scope)"),
        ("mention_per_1000char", "tab:orange", "Mentions per 1,000 Characters by Academic Field\n(main scope)"),
        ("esg_doc_ratio_pct", "tab:green", "Share of Documents with ESG Mentions by Academic Field (%)\n(main scope)"),
    ]
    for ax, (col, color, title) in zip(axes, panels):
        srt = table.sort_values(col, ascending=True)
        ax.barh(srt["label_en"], srt[col], color=color)
        for i, v in enumerate(srt[col]):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"2_학문분야별_밀도_{lang}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


# 3. target_field × 상위 15 concept 히트맵
def build_heatmap(merged: pd.DataFrame, lang: str, out_dir: Path, top_n: int = 15):
    m_main = merged[merged["is_main"]]
    top_concepts = m_main["canonical"].value_counts().head(top_n).index.tolist()

    matrix = pd.DataFrame(index=cfg.TARGET_FIELD_ORDER, columns=top_concepts, dtype=int).fillna(0)
    for tf in cfg.TARGET_FIELD_ORDER:
        sub = m_main[m_main["target_field_primary"] == tf]
        counts = sub["canonical"].value_counts()
        for c in top_concepts:
            matrix.loc[tf, c] = int(counts.get(c, 0))
    matrix.index = [_label(tf, lang, cfg.TARGET_FIELD_LABELS) for tf in cfg.TARGET_FIELD_ORDER]
    matrix = matrix[(matrix.sum(axis=1) > 0) | True]  # 표시는 전체 유지(0행도 "데이터 없음"으로 보여줌)
    matrix.to_csv(out_dir / f"3_히트맵_{lang}.csv", encoding="utf-8-sig")

    row_labels_en = [_label_en(tf, cfg.TARGET_FIELD_LABELS) for tf in cfg.TARGET_FIELD_ORDER]

    plotting.set_font(lang)  # 히트맵 x축(top_concepts)은 원문 키워드 그대로 표시 — CJK 폰트 필요
    fig, ax = plt.subplots(figsize=(1.1 * len(top_concepts) + 2, 5))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(top_concepts)))
    ax.set_xticklabels(top_concepts, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels_en)))
    ax.set_yticklabels(row_labels_en)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > matrix.values.max() * 0.5 else "black", fontsize=8)
    title = f"Top {top_n} ESG Keyword Frequency Heatmap by Academic Field\n(concept-merged, main scope)"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"3_히트맵_{lang}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return matrix


# A. 국가×계열×page_type 3중 교차표
def build_crosstab_density_heatmap(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path):
    """국가×계열×page_type 3중 교차표. 셀 값=밀도(1,000자당 mention 수), (n=문서수) 표기.
    Cochran(1954) 기준(n<5) 셀은 회색 처리 + "해석 주의" 표시."""
    d_main = docs[docs["is_main"]]
    m_main = merged[merged["is_main"]]
    # main scope는 정의상 page_type I·J가 존재하지 않으므로(config.EXCLUDED_FROM_MAIN_PAGE_TYPES),
    # 그 두 열을 넣으면 항상 n=0으로만 나와 의미가 없다 — main에 실제로 존재하는 page_type만 사용.
    page_types_in_scope = [pt for pt in cfg.PAGE_TYPE_ORDER if pt not in cfg.EXCLUDED_FROM_MAIN_PAGE_TYPES]

    rows = []
    matrix = pd.DataFrame(index=cfg.TARGET_FIELD_ORDER, columns=page_types_in_scope, dtype=float)
    n_docs_matrix = pd.DataFrame(index=cfg.TARGET_FIELD_ORDER, columns=page_types_in_scope, dtype=int)
    for tf in cfg.TARGET_FIELD_ORDER:
        for pt in page_types_in_scope:
            cell_docs = d_main[(d_main["target_field_primary"] == tf) & (d_main["page_type_reviewed"] == pt)]
            n_docs = len(cell_docs)
            n_chars = cell_docs["full_text_len"].sum()
            n_mentions = len(
                m_main[(m_main["target_field_primary"] == tf) & (m_main["page_type_reviewed"] == pt)]
            )
            density = n_mentions / n_chars * 1000 if n_chars else 0.0
            matrix.loc[tf, pt] = density
            n_docs_matrix.loc[tf, pt] = n_docs
            rows.append({
                "target_field": tf, "page_type": pt, "n_docs": n_docs, "n_chars": int(n_chars),
                "n_mentions": n_mentions, "density_per_1000char": density, "sparse_flag": n_docs < 5,
            })
    long_table = pd.DataFrame(rows)
    long_table.to_csv(out_dir / f"A_3중교차표_{lang}.csv", index=False, encoding="utf-8-sig")

    row_labels_en = [_label_en(tf, cfg.TARGET_FIELD_LABELS) for tf in cfg.TARGET_FIELD_ORDER]
    col_labels_en = [_label_en(pt, cfg.PAGE_TYPE_LABELS) for pt in page_types_in_scope]

    plotting.set_font(lang)
    fig, ax = plt.subplots(figsize=(1.1 * len(col_labels_en) + 2, 0.6 * len(row_labels_en) + 2))
    vals = matrix.values.astype(float)
    im = ax.imshow(vals, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(col_labels_en)))
    ax.set_xticklabels(col_labels_en, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels_en)))
    ax.set_yticklabels(row_labels_en)
    vmax = vals.max() if vals.size else 1
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            n_docs = int(n_docs_matrix.values[i, j])
            sparse = n_docs < 5
            if sparse:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=True, color="lightgray", alpha=0.9, zorder=2))
                label = f"n={n_docs}\nCaution: small n"
            else:
                label = f"{vals[i, j]:.1f}\n(n={n_docs})"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                    color="black" if sparse else ("white" if vals[i, j] > vmax * 0.5 else "black"), zorder=3)
    title = "Country x Field x Page-Type Cross-Tabulation\n(Density: mentions per 1,000 characters, main scope)"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    footnote = "Note: Gray cells (n<5) have too small a sample for reliable interpretation (Cochran 1954 criterion)."
    fig.text(0.5, -0.02, footnote, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"A_3중교차표_{lang}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return long_table


# C. ESG 하위범주(E/S/G)별 언급 밀도
# DUP 카테고리는 관련된 두 범주 모두에 중복 포함
ESG_SUBCATEGORY_MEMBERSHIP = {
    "E": {"E", "DUP_ES", "DUP_EG"},
    "S": {"S", "DUP_ES", "DUP_SG"},
    "G": {"G", "DUP_SG", "DUP_EG"},
}


def build_esg_subcategory_density(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path):
    d_main = docs[docs["is_main"]]
    m_main = merged[merged["is_main"]]
    total_chars = d_main["full_text_len"].sum()

    rows = []
    for sub, cats in ESG_SUBCATEGORY_MEMBERSHIP.items():
        n_mentions = len(m_main[m_main["category"].isin(cats)])
        rows.append({
            "subcategory": sub, "n_mentions": n_mentions, "n_chars": int(total_chars),
            "density_per_1000char": n_mentions / total_chars * 1000 if total_chars else 0,
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"C_ESG하위범주별_밀도_{lang}.csv", index=False, encoding="utf-8-sig")

    plotting.set_font(lang)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"E": "tab:green", "S": "tab:blue", "G": "tab:brown"}
    ax.bar(table["subcategory"], table["density_per_1000char"], color=[colors[s] for s in table["subcategory"]])
    for i, v in enumerate(table["density_per_1000char"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    title = "ESG Subcategory (E/S/G) Mention Density\n(per 1,000 characters, main scope, DUP overlaps included)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"C_ESG하위범주별_밀도_{lang}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


# 4. 대학별 ESG 언급 문서 비율(%)
def build_university_overview(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path):
    """4. 대학별 개요 + 대학 규모 대비 정규화 밀도: 페이지당 평균 mention 수, 학과(entity)당
    평균 페이지 수. 문서당 언급 수는 문서 길이 보정이 안 돼 대학 간 비교시 오해 소지가 있어
    1,000자당 밀도 열도 함께 추가."""
    d_main = docs[docs["is_main"]]
    m_main = merged[merged["is_main"]]
    mention_counts = m_main.groupby("doc_id").size()
    rows = []
    for univ, grp in d_main.groupby("university"):
        n_docs = len(grp)
        n_esg_docs = grp["doc_id"].isin(m_main["doc_id"]).sum()
        n_mentions = mention_counts.reindex(grp["doc_id"]).fillna(0).sum()
        n_chars = grp["full_text_len"].sum()
        n_entities = grp["entity_id"].nunique() if "entity_id" in grp.columns else None
        rows.append({
            "university": univ, "n_docs": n_docs,
            "esg_doc_ratio_pct": n_esg_docs / n_docs * 100,
            "mentions_per_doc": n_mentions / n_docs if n_docs else 0,
            "density_per_1000char": n_mentions / n_chars * 1000 if n_chars else 0,
            "n_entities": n_entities,
            "pages_per_entity": n_docs / n_entities if n_entities else None,
        })
    table = pd.DataFrame(rows).sort_values("esg_doc_ratio_pct", ascending=False)
    table.to_csv(out_dir / f"4_대학별_개요_{lang}.csv", index=False, encoding="utf-8-sig")

    plotting.set_font(lang)
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(table) + 1.5))
    srt = table.sort_values("esg_doc_ratio_pct", ascending=True)
    ax.barh(srt["university"], srt["esg_doc_ratio_pct"], color="tab:blue")
    for i, v in enumerate(srt["esg_doc_ratio_pct"]):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    ax.set_title("Share of Documents with ESG Mentions by University (main scope)")
    ax.set_xlabel("Share of Documents with ESG Mentions (%)")
    fig.text(0.5, -0.01,
              "Note: Share of documents containing at least one ESG keyword; distinct from mention frequency.",
              ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"4_대학별_개요_{lang}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


# 5. page_type별 ESG mention 밀도(1,000자당)
def build_pagetype_density(merged: pd.DataFrame, docs: pd.DataFrame, lang: str, out_dir: Path):
    d_main = docs[docs["is_main"]]
    m_main = merged[merged["is_main"]]
    rows = []
    for pt in cfg.PAGE_TYPE_ORDER:
        pt_docs = d_main[d_main["page_type_reviewed"] == pt]
        if len(pt_docs) == 0:
            continue
        n_chars = pt_docs["full_text_len"].sum()
        n_mentions = len(m_main[m_main["page_type_reviewed"] == pt])
        rows.append({
            "page_type": pt,
            "label": _label(pt, lang, cfg.PAGE_TYPE_LABELS),
            "label_en": _label_en(pt, cfg.PAGE_TYPE_LABELS),
            "density_per_1000char": n_mentions / n_chars * 1000 if n_chars else 0,
            "n_docs": len(pt_docs),
        })
    table = pd.DataFrame(rows).sort_values("density_per_1000char", ascending=False)
    table.to_csv(out_dir / f"5_페이지타입별_개요_{lang}.csv", index=False, encoding="utf-8-sig")

    plotting.set_font(lang)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1.5))
    srt = table.sort_values("density_per_1000char", ascending=True)
    ax.barh(srt["label_en"], srt["density_per_1000char"], color="tab:green")
    for i, v in enumerate(srt["density_per_1000char"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    title = "ESG Mention Density by Page Type\n(per 1,000 characters, main scope)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"5_페이지타입별_개요_{lang}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


def build_stage2_report(lang: str):
    out_dir = cfg.OUTPUT_DIR / lang / "단계2_빈도분석"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged, docs = load_enriched(lang)

    t1 = build_category_overview(merged, lang, out_dir)
    t2 = build_target_field_density(merged, docs, lang, out_dir)
    t3 = build_heatmap(merged, lang, out_dir)
    t4 = build_university_overview(merged, docs, lang, out_dir)
    t5 = build_pagetype_density(merged, docs, lang, out_dir)
    tA = build_crosstab_density_heatmap(merged, docs, lang, out_dir)
    tC = build_esg_subcategory_density(merged, docs, lang, out_dir)

    with pd.ExcelWriter(out_dir / f"빈도분석결과_{lang}.xlsx") as writer:
        t1.to_excel(writer, sheet_name="1_ESG범주별개요", index=False)
        t2.to_excel(writer, sheet_name="2_학문분야별밀도", index=False)
        t3.to_excel(writer, sheet_name="3_히트맵")
        t4.to_excel(writer, sheet_name="4_대학별개요", index=False)
        t5.to_excel(writer, sheet_name="5_페이지타입별개요", index=False)
        tA.to_excel(writer, sheet_name="A_3중교차표", index=False)
        tC.to_excel(writer, sheet_name="C_ESG하위범주별밀도", index=False)

    print(f"[{lang}] 단계 II 빈도분석 저장 완료: {out_dir}")
    return {
        "category": t1, "target_field_density": t2, "heatmap": t3, "university": t4, "page_type": t5,
        "crosstab": tA, "esg_subcategory": tC,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from common import config as _cfg
    for _lang in _cfg.LANGS:
        build_stage2_report(_lang)
