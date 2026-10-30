# -*- coding: utf-8 -*-
"""
Seed Registry의 165개 entity(KR 82, JP 83)를 수집한다.
이메일 형태 링크를 제외하고, PDF 텍스트와 이미지 기반 PDF의 OCR 결과를 수집한다.
이미지 중심 HTML은 별도로 표시하며, 게시판 페이지 수집 수를 제한한다.
"""
import os
from pathlib import Path
import io
import re
import time
import csv
from collections import deque
from urllib.parse import urlsplit, urljoin
from urllib import robotparser

import requests
from bs4 import BeautifulSoup
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

RESEARCH_ROOT = os.environ.get(
    "ESG_RESEARCH_ROOT",
    str(Path.home() / "Desktop" / "JHI" / "Lab Project" / "ESG"),
)
COMP_ROOT = f"{RESEARCH_ROOT}/comp"

SEED_XLSX = f"{COMP_ROOT}/0_crawling/phase2_seed_registry/seed_registry.xlsx"
OUT_XLSX = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/url_inventory_full.xlsx"
CHECKPOINT_JSONL = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/crawl_checkpoint.jsonl"
CHECKPOINT_FULL_JSON = f"{COMP_ROOT}/0_crawling/phase4_url_inventory/crawl_checkpoint_full.json"

ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize(v):
    if isinstance(v, str):
        return ILLEGAL_XML_RE.sub("", v)
    return v

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": BROWSER_UA}
TIMEOUT = 10
DELAY = 0.4
MAX_PAGES_PER_SEED = 60      # seed별 최대 수집 페이지 수
BOARD_CAP_PER_SEED = 10
BOARD_KEYWORDS = ["board", "bbs", "articleno", "mode=list", "mode=view", "mode=download", "offset="]

NON_HEADLESS_DOMAINS = {"www.thu.ac.jp", "www.tokyo-kasei.ac.jp"}
_selenium_driver = None

EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")

PDF_MAX_BYTES = 20 * 1024 * 1024  # 20MB 넘는 PDF는 스킵(안전장치)
OCR_TEXT_THRESHOLD = 50           # 이만큼도 텍스트가 안 나오면 이미지기반으로 간주
OCR_MAX_PAGES = 5                 # OCR은 PDF 앞 5페이지까지만


def is_board_like(url):
    low = url.lower()
    return any(k in low for k in BOARD_KEYWORDS)


def get_selenium_driver():
    global _selenium_driver
    if _selenium_driver is None:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1400,1000")
        if os.environ.get("CHROME_BINARY"):
            options.binary_location = os.environ["CHROME_BINARY"]
        _selenium_driver = webdriver.Chrome(options=options)
        _selenium_driver.set_page_load_timeout(35)
    return _selenium_driver


def quit_driver():
    global _selenium_driver
    if _selenium_driver is not None:
        _selenium_driver.quit()
        _selenium_driver = None


robots_cache = {}


def get_robots(netloc, scheme):
    if netloc in robots_cache:
        return robots_cache[netloc]
    rp = robotparser.RobotFileParser()
    try:
        resp = requests.get(f"{scheme}://{netloc}/robots.txt", headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp = None
    except requests.RequestException:
        rp = None
    robots_cache[netloc] = rp
    return rp


def robots_allowed(url):
    parts = urlsplit(url)
    rp = get_robots(parts.netloc, parts.scheme or "https")
    if rp is None:
        return True
    try:
        return rp.can_fetch(BROWSER_UA, url)
    except Exception:
        return True


class SeleniumResp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text
        self.apparent_encoding = None
        self.encoding = "utf-8"
        self.headers = {"Content-Type": "text/html; charset=utf-8"}


def fetch_selenium(url):
    try:
        driver = get_selenium_driver()
        driver.get(url)
        title = driver.title or ""
        html = driver.page_source or ""
        if title.strip() == "Access Denied" or len(html) < 1000:
            return SeleniumResp(403, html), None
        return SeleniumResp(200, html), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def fetch(url):
    domain = urlsplit(url).netloc
    if domain in NON_HEADLESS_DOMAINS:
        time.sleep(1.0)
        return fetch_selenium(url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        return resp, None
    except requests.RequestException as e:
        return None, f"{type(e).__name__}: {e}"


def extract_page(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # 링크/이미지는 정제 전 원본에서 뽑는다(메뉴 링크도 스코프 판단엔 필요).
    links, pdfs, images = [], [], []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        if EMAIL_RE.match(href):  # mailto 없이 적힌 이메일 링크도 제외
            continue
        abs_url = urljoin(base_url, href)
        abs_url = abs_url.split("#")[0]
        if abs_url in seen:
            continue
        seen.add(abs_url)
        if abs_url.lower().endswith(".pdf"):
            pdfs.append(abs_url)
        else:
            links.append(abs_url)

    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        if not src or src.startswith("data:"):
            continue
        low = src.lower()
        if any(k in low for k in ("logo", "icon", "btn", "banner", "spacer", "bullet")):
            continue
        images.append(urljoin(base_url, src))

    # 링크/이미지 다 뽑은 뒤 구조적 노이즈(메뉴/헤더/푸터/스크립트) 제거하고 본문 텍스트 추출.
    # 완전한 본문(main_text) 추출은 별도 전처리 단계(Trafilatura)에서 하고, 여기서는
    # 명백한 비-본문 태그만 제거하는 가벼운 정제만 수행한다.
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    body_text = soup.get_text(separator=" ", strip=True)

    return title, body_text, links, pdfs, list(dict.fromkeys(images))


def extract_pdf_text(pdf_bytes):
    """PyMuPDF로 텍스트 추출. 텍스트가 거의 없으면 앞 N페이지에 OCR 적용."""
    try:
        import fitz
    except ImportError:
        return "", False, "fitz_not_installed"
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return "", False, f"pdf_open_error:{e}"

    text = "\n".join(page.get_text() for page in doc).strip()
    if len(text) >= OCR_TEXT_THRESHOLD or len(doc) == 0:
        return text, False, ""

    try:
        import pytesseract
        from PIL import Image
        ocr_parts = []
        for page in doc[:OCR_MAX_PAGES]:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_parts.append(pytesseract.image_to_string(img, lang="kor+jpn+eng"))
        ocr_text = "\n".join(ocr_parts).strip()
        return ocr_text, True, ""
    except Exception as e:
        return text, False, f"ocr_error:{e}"


def fetch_pdf_and_extract(pdf_url):
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=TIMEOUT * 2, stream=True)
        if resp.status_code >= 400:
            return {"status": resp.status_code, "text_len": 0, "ocr_applied": False, "note": "HTTP_ERROR"}
        content = resp.content
        if len(content) > PDF_MAX_BYTES:
            return {"status": resp.status_code, "text_len": 0, "ocr_applied": False, "note": "TOO_LARGE_SKIPPED"}
        text, ocr_applied, note = extract_pdf_text(content)
        return {"status": resp.status_code, "text_len": len(text), "ocr_applied": ocr_applied,
                "note": note, "text_sample": text[:300]}
    except requests.RequestException as e:
        return {"status": "ERROR", "text_len": 0, "ocr_applied": False, "note": str(e)}


def in_scope(url, domain, path_prefix, deny_urls, base_deny_keywords):
    p = urlsplit(url)
    if p.netloc != domain:
        return False
    if not p.path.startswith(path_prefix):
        return False
    for d in deny_urls:
        if url.startswith(d):
            return False
    low = url.lower()
    if any(k in low for k in base_deny_keywords):
        return False
    return True


def crawl_one(seed_row, do_pdf_extract=True):
    instr = seed_row["crawl_instruction"]
    seed_url = seed_row["seed_url"]
    domain = seed_row["allowed_domains"]
    path_prefix = seed_row["allowed_path_prefix"]
    max_depth = int(seed_row["max_depth"])
    manual_urls = [u.strip() for u in seed_row["manual_urls"].split(";") if u.strip()]
    base_deny_keywords = [k for k in seed_row["deny_patterns"].split("|") if k and not k.startswith("http")]

    results = []
    pdf_cache = {}

    def process_pdfs(pdf_urls):
        out = []
        for pu in pdf_urls:
            if pu not in pdf_cache and do_pdf_extract:
                pdf_cache[pu] = fetch_pdf_and_extract(pu)
                time.sleep(0.1)
            out.append(pu)
        return out

    def do_fetch(url, depth):
        if not robots_allowed(url):
            results.append({"url": url, "depth": depth, "status": "", "title": "",
                             "n_links": 0, "n_pdf": 0, "n_images": 0, "note": "ROBOTS_DISALLOWED",
                             "pdf_urls": [], "possible_image_content": False})
            return None
        resp, err = fetch(url)
        time.sleep(DELAY)
        if err:
            results.append({"url": url, "depth": depth, "status": "ERROR", "title": "",
                             "n_links": 0, "n_pdf": 0, "n_images": 0, "note": err,
                             "pdf_urls": [], "possible_image_content": False})
            return None
        status = resp.status_code
        if status >= 400:
            results.append({"url": url, "depth": depth, "status": status, "title": "",
                             "n_links": 0, "n_pdf": 0, "n_images": 0, "note": "HTTP_ERROR",
                             "pdf_urls": [], "possible_image_content": False})
            return None
        # Content-Type에 charset이 명시돼 있으면 그대로 신뢰(requests가 이미 정확히 파싱함).
        # charset이 없을 때만 chardet(apparent_encoding) 추정치로 보완
        # (무조건 덮어쓰면 UTF-8 declared 페이지도 오탐될 수 있음, 예: 한양대 -> Windows-1254).
        if "charset" not in resp.headers.get("Content-Type", "").lower():
            resp.encoding = resp.apparent_encoding or resp.encoding
        title, body_text, links, pdfs, images = extract_page(resp.text, url)
        process_pdfs(pdfs)
        possible_image_content = len(body_text) < 200 and len(images) > 0
        results.append({"url": url, "depth": depth, "status": status, "title": title,
                         "n_links": len(links), "n_pdf": len(pdfs), "n_images": len(images),
                         "note": "", "pdf_urls": pdfs, "text_len": len(body_text),
                         "text_snippet": body_text[:300],
                         "possible_image_content": possible_image_content})
        return links

    if instr == "manual_collect":
        results.append({"url": seed_url, "depth": 0, "status": "", "title": "",
                         "n_links": 0, "n_pdf": 0, "n_images": 0, "note": "SKIPPED_manual_collect",
                         "pdf_urls": [], "possible_image_content": False})
        return results, pdf_cache

    if instr == "single_page":
        do_fetch(seed_url, 0)
        return results, pdf_cache

    if instr == "manual_url_list":
        do_fetch(seed_url, 0)
        for u in manual_urls:
            do_fetch(u, 0)
        return results, pdf_cache

    visited = set()
    queue = deque([(seed_url, 0)])
    board_queue = deque()
    board_visited = 0

    while (queue or board_queue) and len(visited) < MAX_PAGES_PER_SEED:
        if queue:
            url, depth = queue.popleft()
        else:
            if board_visited >= BOARD_CAP_PER_SEED:
                break
            url, depth = board_queue.popleft()
            board_visited += 1
        if url in visited:
            continue
        visited.add(url)
        links = do_fetch(url, depth)
        if links is None or depth >= max_depth:
            continue
        for link in links:
            if link in visited:
                continue
            if not in_scope(link, domain, path_prefix, manual_urls, base_deny_keywords):
                continue
            if is_board_like(link):
                board_queue.append((link, depth + 1))
            else:
                queue.append((link, depth + 1))
    return results, pdf_cache


def load_seed_rows():
    wb = openpyxl.load_workbook(SEED_XLSX, data_only=True)
    rows = {}
    order = []
    for sheet in ("seed_registry_kr", "seed_registry_jp"):
        ws = wb[sheet]
        headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        idx = {h: i for i, h in enumerate(headers)}
        for r in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
            eid = r[idx["entity_id (Phase1 연결)"]]
            order.append(eid)
            rows[eid] = {
                "seed_id": r[idx["seed_id"]],
                "university": r[idx["대학"]],
                "college": r[idx["학부/College"]],
                "entity_name": r[idx["학과/전공/과정명"]],
                "org_level": r[idx["org_level"]],
                "seed_url": r[idx["seed_url (root URL)"]],
                "manual_urls": r[idx["manual_urls (추가/예외 URL)"]] or "",
                "crawl_instruction": r[idx["crawl_instruction (담당자 지정)"]],
                "allowed_domains": r[idx["allowed_domains"]],
                "allowed_path_prefix": r[idx["allowed_path_prefix (제안값)"]],
                "deny_patterns": r[idx["deny_patterns"]] or "",
                "max_depth": r[idx["max_depth (제안값)"]],
            }
    return rows, order


def main():
    import json
    all_rows, order = load_seed_rows()
    out_records = []
    all_pdf_cache = {}
    total = len(order)

    # 전체 수집 결과를 기록할 체크포인트를 초기화한다.
    open(CHECKPOINT_JSONL, "w", encoding="utf-8").close()

    for i, eid in enumerate(order, 1):
        seed_row = all_rows[eid]
        print(f"[{i}/{total}] {eid} {seed_row['university']} {seed_row['entity_name']} "
              f"[{seed_row['crawl_instruction']}] {seed_row['seed_url']}", flush=True)
        results, pdf_cache = crawl_one(seed_row)
        all_pdf_cache.update(pdf_cache)
        entity_records = []
        for r in results:
            entity_records.append({**{"entity_id": eid, "university": seed_row["university"],
                                       "entity_name": seed_row["entity_name"],
                                       "crawl_instruction": seed_row["crawl_instruction"]}, **r})
        out_records.extend(entity_records)
        # 엔티티 단위 증분 체크포인트 - 중간에 죽어도 여기까지는 살아남는다.
        with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
            for rec in entity_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for pu, info in pdf_cache.items():
                f.write(json.dumps({"__pdf__": pu, **info}, ensure_ascii=False) + "\n")
        print(f"    -> {len(results)} pages", flush=True)

    # 전체 크롤링 끝난 직후, 엑셀 저장 전에 한 번 더 통째로 체크포인트(안전장치 이중화)
    with open(CHECKPOINT_FULL_JSON, "w", encoding="utf-8") as f:
        json.dump({"out_records": out_records, "pdf_cache": all_pdf_cache}, f, ensure_ascii=False)
    print("full checkpoint saved:", CHECKPOINT_FULL_JSON, flush=True)

    build_excel(out_records, all_pdf_cache)


def build_excel(out_records, all_pdf_cache):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "url_inventory"
    fields = ["entity_id", "university", "entity_name", "crawl_instruction",
              "depth", "status", "note", "n_links", "n_pdf", "n_images",
              "possible_image_content", "text_len", "title", "url", "text_snippet"]
    ws.append(fields + ["비고"])
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    err_fill = PatternFill("solid", fgColor="FCE4D6")
    img_fill = PatternFill("solid", fgColor="D9D2E9")
    for c in range(1, len(fields) + 2):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for rec in out_records:
        row = [sanitize(rec.get(f, "")) for f in fields] + [""]
        ws.append(row)
        rn = ws.max_row
        if rec.get("note"):
            for c in range(1, len(fields) + 1):
                ws.cell(row=rn, column=c).fill = err_fill
        elif rec.get("possible_image_content"):
            for c in range(1, len(fields) + 1):
                ws.cell(row=rn, column=c).fill = img_fill
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    widths = [12, 16, 26, 20, 8, 10, 22, 8, 8, 9, 16, 9, 30, 60, 50, 40]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws2 = wb.create_sheet("pdf_extraction")
    ws2.append(["entity_id", "university", "entity_name", "source_page_url", "pdf_url",
                "status", "text_len", "ocr_applied", "note", "text_sample"])
    for c in range(1, 11):
        cell = ws2.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for rec in out_records:
        for pdf_url in rec.get("pdf_urls", []) or []:
            info = all_pdf_cache.get(pdf_url, {})
            ws2.append([sanitize(v) for v in [
                rec["entity_id"], rec["university"], rec["entity_name"], rec["url"], pdf_url,
                info.get("status", ""), info.get("text_len", 0), info.get("ocr_applied", False),
                info.get("note", ""), info.get("text_sample", "")]])
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = ws2.dimensions
    for i, w in enumerate([12, 16, 26, 50, 50, 10, 10, 12, 20, 50], start=1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    wb.save(OUT_XLSX)
    print("\nsaved:", OUT_XLSX, "total records:", len(out_records), "pdfs processed:", len(all_pdf_cache))


if __name__ == "__main__":
    main()
    quit_driver()
