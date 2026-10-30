# -*- coding: utf-8 -*-
"""GT 364건 leave-one-out 검증과 저장된 예측의 정확도·Cohen's κ 계산."""
from pathlib import Path
import os
import math
import json
import sys
from collections import Counter
import openpyxl

client = None

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

ROOT = f"{COMP_ROOT}/0_crawling/phase5_page_type"
GT_POOL = f"{ROOT}/gt_pool_v3.xlsx"
OUT_JSON = f"{ROOT}/gt364_loo_results.json"

EMBED_MODEL = "text-embedding-3-small"
K = 5

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


def embed(texts):
    out = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 100])
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


def score_results(results):
    """같은 행의 수동 라벨과 예측 라벨로 정확도와 Cohen's κ를 계산한다."""
    if not results:
        raise ValueError("평가할 예측이 없습니다.")
    n = len(results)
    actual = Counter(r["gt_label"] for r in results)
    predicted = Counter(r["pred_label"] for r in results)
    correct = sum(r["gt_label"] == r["pred_label"] for r in results)
    observed = correct / n
    expected = sum(actual[label] * predicted[label] for label in actual.keys() | predicted.keys()) / n**2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return correct, n, observed, kappa


def print_score(results):
    correct, n, accuracy, kappa = score_results(results)
    print(f"정확도: {correct}/{n} = {accuracy * 100:.1f}%")
    print(f"Cohen's κ: {kappa:.3f}")


def main():
    global client
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wb = openpyxl.load_workbook(GT_POOL, data_only=True)
    ws = wb.active
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    rows = [dict(zip(headers, r)) for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True)]
    print(f"GT 총 {len(rows)}건")

    texts = [(r.get("title") or "") + " " + (r.get("snippet") or "")[:800] for r in rows]
    embs = embed(texts)
    for r, e in zip(rows, embs):
        r["_emb"] = e
    print("임베딩 완료")

    results = []
    for i, r in enumerate(rows, 1):
        candidates = sorted(
            ((g, cosine(r["_emb"], g["_emb"])) for g in rows if g["url"] != r["url"]),
            key=lambda x: -x[1],
        )
        topk = [g for g, s in candidates[:K]]
        fewshot_examples = [{"title": g.get("title") or "", "snippet": g.get("snippet") or "",
                              "label": g.get("gt_label")} for g in topk]
        pred = gpt_classify_dynamic(r.get("title") or "", r["url"], r.get("snippet") or "", fewshot_examples)
        results.append({"url": r["url"], "gt_label": r["gt_label"], "pred_label": pred})
        if i % 40 == 0 or i == len(rows):
            correct = sum(1 for x in results if x["gt_label"] == x["pred_label"])
            print(f"[{i}/{len(rows)}] 누적 정확도: {correct}/{i} = {correct/i*100:.1f}%")

    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print_score(results)
    print("저장:", OUT_JSON)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--score":
        with open(sys.argv[2], encoding="utf-8") as f:
            print_score(json.load(f))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("사용법: python gt364_leave_one_out_validation.py [--score 결과.json]")
