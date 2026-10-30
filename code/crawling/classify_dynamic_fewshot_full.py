# -*- coding: utf-8 -*-
"""
규칙/정적few-shot 없이, 검색 기반 동적 few-shot(retrieval-augmented in-context learning)
하나로 전체 3,102건(status=200)을 일괄 재분류.
GT 풀(gt_pool_v3.xlsx, 364건)을 임베딩해 검색 대상으로 삼고, 페이지마다 top-5 유사 예시를
동적으로 검색해 프롬프트에 삽입 (Liu 2021 / Rubin et al. 2022).
"""
from pathlib import Path
import os
import re
import math
import json
import time
import openpyxl
from openpyxl.styles import Font, PatternFill

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

ROOT = f"{COMP_ROOT}/0_crawling"
SRC = f"{ROOT}/phase4_url_inventory/url_inventory_full_trafilatura.xlsx"
GT_POOL = f"{ROOT}/phase5_page_type/gt_pool_v3.xlsx"
OUT = f"{ROOT}/phase5_page_type/page_type_classified_v3.xlsx"
CHECKPOINT_JSONL = f"{ROOT}/phase5_page_type/classify_v3_checkpoint.jsonl"

EMBED_MODEL = "text-embedding-3-small"
K = 5
LOW_CONFIDENCE_TYPES = {"G", "H"}
ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

PAGE_TYPE_DEF = """
A 학과소개: 교육목표·비전·소개, 안내/홍보 포스터·리플렛·소개영상
B 커리큘럼: 교과과정·교과목·실라버스, 연계/복수전공·협동과정, 실습과목 소개
C 교수/연구: 교수·연구실 소개 전체, 연구분야·프로젝트·논문·학위논문·도서, 조교(助手) 소개
D 학과활동: 학과·기관 주체 심포지엄·동향·소식, 산학협력/지역연계활동, 학과 주최 설명회·멘토링데이·오픈캠퍼스
E 학생/졸업생 활동: 학생활동·학생회·동아리·동창회, 학생/졸업생 소개·인터뷰, 졸업프로젝트, 공모전 수상, 졸업전시회, 대학원생 논문현황, 학생실험/블로그 기록
F 진로/취업: 자격증·면허 취득, 졸업 후 진로 정보
G 관련 협회/기관: 협력기관·산학협력기관, 학과 소속/운영 학회, 부설연구소 공식소개·정관, 협약기관 목록
H 국제교류: 교환유학, 협정교, 국제교류 절차 안내
I 커뮤니티: 게시판, 공지사항, Q&A
J 기타: 분류 불명확·소수 페이지, 순수 시설/캠퍼스 안내, 장학금 안내
""".strip()


def sanitize(v):
    return ILLEGAL_XML_RE.sub("", v) if isinstance(v, str) else v


def embed(texts):
    out = []
    B = 100
    for i in range(0, len(texts), B):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + B])
        out.extend([d.embedding for d in resp.data])
    return out


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def gpt_classify_dynamic(title, url, snippet, fewshot_examples):
    fewshot_text = "\n\n".join(
        f"[예시{i+1} - {ex['label']}로 분류] 제목: {ex['title']}\n본문: {ex['snippet'][:400]}\n-> {ex['label']}"
        for i, ex in enumerate(fewshot_examples)
    )
    prompt = f"""다음은 대학 학과 홈페이지에서 크롤링한 페이지 정보다. page_type을 아래 정의 중 하나의 코드(A~J)로만 답하라.

{PAGE_TYPE_DEF}

아래는 이 페이지와 의미적으로 유사한, 사람이 직접 라벨링한 예시들이다. 참고하라.

{fewshot_text}

이제 아래 페이지를 분류하라.
제목: {title}
URL: {url}
본문 발췌: {(snippet or '')[:1200]}

답은 코드 한 글자(A/B/C/D/E/F/G/H/I/J)만 출력하라."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=5,
    )
    out = resp.choices[0].message.content.strip().upper()
    for c in "ABCDEFGHIJ":
        if c in out:
            return c
    return "J"


def main():
    wb = openpyxl.load_workbook(SRC, data_only=True)
    ws = wb["url_inventory"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]
    target_rows = [r for r in rows if r["status"] in (200, "200")]
    non_target = [r for r in rows if r["status"] not in (200, "200")]
    print(f"분류 대상(status=200): {len(target_rows)} / 전체: {len(rows)}", flush=True)

    wbg = openpyxl.load_workbook(GT_POOL, data_only=True)
    wsg = wbg.active
    gh = [c.value for c in next(wsg.iter_rows(min_row=1, max_row=1))]
    gt_rows = [dict(zip(gh, r)) for r in wsg.iter_rows(min_row=2, max_row=wsg.max_row, values_only=True)]
    gt_texts = [(r.get("title") or "") + " " + (r.get("snippet") or "")[:800] for r in gt_rows]
    gt_embeds = embed(gt_texts)
    for r, e in zip(gt_rows, gt_embeds):
        r["_emb"] = e
    print(f"GT 풀 임베딩 완료: {len(gt_rows)}건", flush=True)

    open(CHECKPOINT_JSONL, "w", encoding="utf-8").close()
    t0 = time.time()
    for i, r in enumerate(target_rows, 1):
        title = r.get("title") or ""
        snippet = r.get("snippet_trafilatura") or ""
        url = r["url"]
        qemb = embed([title + " " + snippet[:800]])[0]
        candidates = sorted(((g, cosine(qemb, g["_emb"])) for g in gt_rows if g["url"] != url), key=lambda x: -x[1])
        topk = [g for g, s in candidates[:K]]
        fewshot_examples = [{"title": g.get("title") or "", "snippet": g.get("snippet") or "",
                              "label": g.get("gt_label")} for g in topk]
        page_type = gpt_classify_dynamic(title, url, snippet, fewshot_examples)

        r["page_type"] = page_type
        r["classify_source"] = "dynamic_fewshot"
        r["needs_review_priority"] = page_type in LOW_CONFIDENCE_TYPES

        with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps({"url": url, "page_type": page_type}, ensure_ascii=False) + "\n")
        if i % 100 == 0 or i == len(target_rows):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(target_rows) - i)
            print(f"[{i}/{len(target_rows)}] done | 경과 {elapsed/60:.1f}분 | 예상잔여 {eta/60:.1f}분", flush=True)

    for r in non_target:
        r["page_type"] = ""
        r["classify_source"] = "skipped_not_200"
        r["needs_review_priority"] = False

    all_rows = target_rows + non_target
    from collections import Counter
    print("\npage_type 분포:", Counter(r["page_type"] for r in target_rows), flush=True)
    print("저신뢰 플래그:", sum(1 for r in target_rows if r["needs_review_priority"]), flush=True)

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "page_type_classified"
    fields = headers + ["page_type", "classify_source", "needs_review_priority"]
    ws2.append(fields)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    review_fill = PatternFill("solid", fgColor="FFF2CC")
    for c in range(1, len(fields) + 1):
        cell = ws2.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for r in all_rows:
        row = [sanitize(r.get(f, "")) for f in fields]
        ws2.append(row)
        if r.get("needs_review_priority"):
            for c in range(1, len(fields) + 1):
                ws2.cell(row=ws2.max_row, column=c).fill = review_fill
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = ws2.dimensions
    wb2.save(OUT)
    print("saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
