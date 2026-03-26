import time
import re
import random
import os
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

BASE = "https://www.topcv.vn"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/145.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.topcv.vn/",
    "Connection": "keep-alive",
}

# ── Hỗ trợ cơ bản ────────────────────────────────────────────────

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)

    retry = Retry(
        total=6,
        connect=3,
        read=3,
        status=6,
        backoff_factor=1.2,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    try:
        s.get(BASE, timeout=20)
        time.sleep(1.0)
    except requests.RequestException:
        pass
    return s

def text(el):
    if not el:
        return None
    t = el.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", t) if t else None

def smart_sleep(min_s=1.2, max_s=2.8):
    time.sleep(random.uniform(min_s, max_s))

def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    for attempt in range(1, 6):
        r = session.get(url, timeout=30)
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = int(retry_after)
                except ValueError:
                    wait = 6 * attempt
            else:
                wait = 6 * attempt
            wait = wait + random.uniform(0.5, 2.0)
            print(f"[WARN] 429 tại {url} → ngủ {wait:.1f}s (attempt {attempt})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    r.raise_for_status()
    return BeautifulSoup("", "lxml")

# ── Parse trang tìm kiếm ─────────────────────────────────────────

def parse_search_page(session: requests.Session, url: str) -> List[Dict]:
    soup = get_soup(session, url)
    jobs = []
    for job in soup.select("div.job-item-search-result"):
        a_title = job.select_one("h3.title a[href]")
        if not a_title:
            continue
        title = text(a_title)
        job_url = urljoin(BASE, a_title.get("href"))

        comp_a = job.select_one("a.company[href]")
        company_name = text(job.select_one("a.company .company-name"))
        company_url = urljoin(BASE, comp_a.get("href")) if comp_a else None

        salary = text(job.select_one("label.title-salary"))
        address = text(job.select_one("label.address .city-text"))
        exp = text(job.select_one("label.exp span"))

        jobs.append({
            "title": title,
            "job_url": job_url,
            "company_name": company_name,
            "company_url": company_url,
            "salary_list": salary,
            "address_list": address,
            "exp_list": exp,
        })
    return jobs

# ── Các pattern và helper dùng chung cho detail & company ────────

COMPANY_INFO_PATTERNS = [
    {
        "item": ".job-detail__company--information-item",
        "title": ".company-title",
        "value": ".company-value",
        "match": "contains",
        "fallback_text": True,
    }
]

GENERAL_INFO_PATTERNS = [
    {
        "item": ".box-general-group",
        "title": ".box-general-group-info-title",
        "value": ".box-general-group-info-value",
        "match": "contains",
        "fallback_text": True,
    },
    {
        "item": ".box-main .box-item",
        "title": "strong",
        "value": "span",
        "match": "contains",
        "fallback_text": True,
    },
    {
        "item": ".premium-job-general-information__content--row",
        "title": ".general-information-data__label",
        "value": ".general-information-data__value",
        "match": "contains",
        "fallback_text": True,
    }
]

INFO_PATTERNS = [
    {
        "item": ".job-detail__info--section",
        "title": ".job-detail__info--section-content-title",
        "value": ".job-detail__info--section-content-value",
        "match": "exact",
        "fallback_text": True,
    },
    {
        "item": ".box-main .box-item",
        "title": "strong",
        "value": "span",
        "match": "contains",
        "fallback_text": True,
    },
    {
        "item": ".basic-information-item",
        "title": ".basic-information-item__data--label",
        "value": ".basic-information-item__data--value",
        "match": "contains",
        "fallback_text": True,
    }
]

PATTERN_GROUPS = {
    "info": INFO_PATTERNS,
    "general": GENERAL_INFO_PATTERNS,
    "company": COMPANY_INFO_PATTERNS,
}

def pick_value_by_title(soup: BeautifulSoup, title: str, patterns: list[dict]) -> Optional[str]:
    target = title.lower().strip()

    for pattern in patterns:
        item_selector = pattern["item"]
        title_selector = pattern["title"]
        value_selector = pattern.get("value")
        match_mode = pattern.get("match", "contains")
        fallback_text = pattern.get("fallback_text", False)

        for item in soup.select(item_selector):
            title_el = item.select_one(title_selector)
            if not title_el:
                continue

            label_text = text(title_el)
            if not label_text:
                continue

            label_text = label_text.strip()
            label_lower = label_text.lower()

            if match_mode == "exact":
                matched = (label_lower == target)
            else:
                matched = (target in label_lower or label_lower in target)

            if not matched:
                continue

            if value_selector:
                value_el = item.select_one(value_selector)
                if value_el:
                    val = text(value_el)
                    if val:
                        val = val.strip()
                        if val:
                            return val

            if fallback_text:
                full_text = text(item)
                if full_text:
                    value = full_text.replace(label_text, "", 1).strip(" :-–—").strip()
                    if value:
                        return value

    return None

def pick_value(soup, title, group):
    return pick_value_by_title(soup, title, PATTERN_GROUPS[group])

# ── Trích xuất chi tiết job ──────────────────────────────────────

def extract_deadline(soup: BeautifulSoup) -> Optional[str]:
    selectors = [
        ".job-detail__info--deadline-date",
        ".job-detail__information-detail--actions-label",
        ".deadline",
        "[class*='deadline']",
        ".desc",
    ]

    for sel in selectors:
        for el in soup.select(sel):
            t = text(el)
            if not t:
                continue

            m_date = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})", t)
            if m_date:
                return m_date.group(1)

            m_days = re.search(r"Còn\s*(\d+)\s*ngày\s*(để ứng tuyển|để nộp hồ sơ)?", t, re.IGNORECASE)
            if m_days:
                days = m_days.group(1)
                return f"Còn {days} ngày để ứng tuyển"

            if "Hạn nộp" in t or "Hạn ứng tuyển" in t:
                return t.strip()

    return None

def extract_tags(soup: BeautifulSoup):
    tags = []

    for group in soup.select('.job-tags__group'):
        group_name_el = group.select_one('.job-tags__group-name')
        if not group_name_el:
            continue

        group_name_text = text(group_name_el).strip().lower()
        
        if any(kw in group_name_text for kw in [
            "chuyên môn", "kỹ năng chuyên môn", "chuyên ngành", "kỹ năng", "skill"
        ]):
            for a in group.select('a.item.search-from-tag'):
                tag_text = text(a)
                if tag_text and tag_text.strip():
                    tags.append(tag_text.strip())

    return tags

DESC_LAYOUTS = [
    {
        "name": "basic_company",
        "section": ".job-description__item",
        "title": "h3",
        "content_mode": "wrapper",
        "content": ".job-description__item--content",
    },
    {
        "name": "box_info",
        "section": ".box-info",
        "title": "h2.title, .title",
        "content_mode": "children",
        "content": "p, li, span, div",
    },
    {
        "name": "box_address",
        "section": ".box-address",
        "title": "h2.title, .title",
        "content_mode": "children",
        "content": "div, p, li, span, strong",
    },
    {
        "name": "premium_company",
        "section": ".premium-job-description__box",
        "title": "h3, h2, .title, .premium-job-description__box-title, .premium-job-description__box--title",
        "content_mode": "children",
        "content": "p, li, span, div",
    },
]

def find_block_by_title(blocks, *titles):
    if not blocks:
        return None

    for expected in titles:
        expected_norm = re.sub(r"\s+", " ", expected).strip().lower()

        for key, value in blocks.items():
            if not key or not value:
                continue

            key_norm = re.sub(r"\s+", " ", key).strip().lower()

            if expected_norm in key_norm:
                return value

    return None

def extract_desc_blocks(soup):
    data = {}

    def clean_title_text(title_el):
        if not title_el:
            return None
        t = title_el.get_text(" ", strip=True)
        t = re.sub(r"\s+", " ", t).strip()
        t = t.rstrip(":").strip()
        return t or None

    def extract_section_content(section, layout, title_text):
        content_parts = []

        if layout["content_mode"] == "wrapper":
            content_wrap = section.select_one(layout["content"])
            if content_wrap:
                raw = content_wrap.get_text(separator="\n", strip=True)
                if raw:
                    content_parts.extend([x.strip() for x in raw.split("\n") if x.strip()])

        elif layout["content_mode"] == "children":
            for el in section.select(layout["content"]):
                t = el.get_text(" ", strip=True)
                t = re.sub(r"\s+", " ", t).strip()
                if t:
                    content_parts.append(t)

            if not content_parts:
                for child in section.find_all(recursive=False):
                    if child == section.select_one(layout["title"]):
                        continue
                    t = child.get_text("\n", strip=True)
                    t = re.sub(r"\n+", "\n", t).strip()
                    if t:
                        content_parts.extend([x.strip() for x in t.split("\n") if x.strip()])

            if not content_parts:
                raw = section.get_text(separator="\n", strip=True)
                raw = raw.replace(title_text, "", 1).strip()
                if raw:
                    content_parts.extend([x.strip() for x in raw.split("\n") if x.strip()])

        seen = set()
        cleaned = []
        for x in content_parts:
            x = re.sub(r"\s+", " ", x).strip()
            if not x:
                continue
            if x == title_text:
                continue
            if x not in seen:
                seen.add(x)
                cleaned.append(x)

        return "\n".join(cleaned) if cleaned else None

    for layout in DESC_LAYOUTS:
        for section in soup.select(layout["section"]):
            title_el = section.select_one(layout["title"])
            if not title_el:
                continue

            title = clean_title_text(title_el)
            if not title:
                continue

            content = extract_section_content(section, layout, title)
            if content and title not in data:
                data[title] = content

    return data

def extract_company_link_from_job(soup: BeautifulSoup) -> Optional[str]:
    cand = (
        soup.select_one("a.company-logo[href]") or
        soup.select_one("a.name[href]") or
        soup.select_one(".job-detail_company--link a[href]") or
        soup.select_one(".navbar-diamond-company-menu a[href]") or
        soup.select_one(".premium-job-header__company--tab-list a[href]")
    )
    return urljoin(BASE, cand["href"]) if cand and cand.has_attr("href") else None

def scrape_job_detail(session: requests.Session, job_url: str) -> Dict:
    soup = get_soup(session, job_url)
    smart_sleep()

    title = (
        text(soup.select_one(".job-detail__info--title"))
        or text(soup.select_one("h2.title"))
        or text(soup.select_one("h2.premium-job-basic-information__content--title"))
    )

    salary = pick_value(soup, "Thu nhập", "info") or pick_value(soup, "Mức lương", "info")
    location = pick_value(soup, "Địa điểm", "info")
    experience = pick_value(soup, "Kinh nghiệm", "info")

    deadline = extract_deadline(soup)
    tags = extract_tags(soup)
    desc_blocks = extract_desc_blocks(soup)

    job_level = pick_value(soup, "Cấp bậc", "general")
    education_level = pick_value(soup, "Học vấn", "general")
    job_quantity = pick_value(soup, "Số lượng tuyển", "general")
    employment_type = pick_value(soup, "Hình thức làm việc", "general")

    company_url_detail = extract_company_link_from_job(soup)

    company_scale = pick_value(soup, "Quy mô", "company")
    company_address = pick_value(soup, "Địa điểm", "company")
    company_field = pick_value(soup, "Lĩnh vực", "company")

    return {
        "detail_title": title,
        "detail_salary": salary,
        "detail_location": location,
        "detail_experience": experience,
        "deadline": deadline,
        "tags": "; ".join(tags) if tags else None,
        "job_level": job_level,
        "education_level": education_level,
        "job_quantity": job_quantity,
        "employment_type": employment_type,
        "desc_mota": find_block_by_title(desc_blocks, "Mô tả công việc", "Job Description"),
        "desc_yeucau": find_block_by_title(desc_blocks, "Yêu cầu ứng viên", "Yêu cầu", "Requirement"),
        "desc_quyenloi": find_block_by_title(desc_blocks, "Quyền lợi", "Quyền lợi được hưởng", "Phúc lợi", "Benefit"),
        "working_addresses": find_block_by_title(desc_blocks, "Địa điểm làm việc", "Địa điểm"),
        "working_times": find_block_by_title(desc_blocks, "Thời gian làm việc", "Thời gian"),
        "company_url_from_job": company_url_detail,
        "company_scale_from_job": company_scale,
        "company_address_from_job": company_address,
        "company_field_from_job": company_field,
    }

# ── Trích xuất trang công ty ─────────────────────────────────────

COMPANY_NAME_LAYOUTS = [
    {"name": "basic_company",   "selector": "h1.company-detail-name.text-highlight"},
    {"name": "basic_company",   "selector": "h1.company-name"},
    {"name": "premium_company", "selector": "#company-name h1"},
    {"name": "premium_company", "selector": "#cover-body h1"},
]

def extract_company_name(soup: BeautifulSoup) -> Optional[str]:
    for layout in COMPANY_NAME_LAYOUTS:
        el = soup.select_one(layout["selector"])
        if not el:
            continue

        if el.name == "meta":
            company_name = (el.get("content") or "").strip()
        else:
            company_name = text(el)

        if not company_name:
            continue

        company_name = re.sub(r"\s*\|\s*TopCV.*$", "", company_name, flags=re.I).strip()
        if company_name:
            return company_name

    return None

COMPANY_WEBSITE_LAYOUTS = [
    {"name": "basic_company",   "selector": "a.company-subdetail-info-text[href]"},
    {"name": "basic_company",   "selector": "a.website-link[href]"},
    {"name": "premium_company", "selector": ".content-contact .info-line a[href]"},
    {"name": "premium_company", "selector": ".info-line a.color-premium[href]"},
]

def extract_company_website(soup: BeautifulSoup) -> Optional[str]:
    for layout in COMPANY_WEBSITE_LAYOUTS:
        for el in soup.select(layout["selector"]):
            href = (el.get("href") or "").strip()
            txt = text(el) or ""

            candidate = href or txt
            if not candidate:
                continue

            candidate = candidate.strip()

            if re.match(r"^https?://", candidate, re.I):
                return candidate

    return None

COMPANY_ADDRESS_LAYOUTS = [
    {
        "name": "premium_contact_address",
        "container": ".content-contact .info-line",
        "icon": "fa-location-dot",
        "value": "span",
    }
]

def extract_company_address(soup: BeautifulSoup) -> Optional[str]:
    for layout in COMPANY_ADDRESS_LAYOUTS:
        for line in soup.select(layout["container"]):
            icon = line.select_one(f"i[class*='{layout['icon']}']")
            if not icon:
                continue

            value_el = line.select_one(layout["value"])
            if not value_el:
                continue

            addr = text(value_el)
            if addr:
                return addr

    return None

COMPANY_SCALE_LAYOUTS = [
    {
        "name": "company_information_section",
        "container": ".information-section .box-items",
        "title": ".title-block",
        "value": ".value-block",
    }
]

def extract_company_scale(soup: BeautifulSoup) -> Optional[str]:
    for layout in COMPANY_SCALE_LAYOUTS:
        for item in soup.select(layout["container"]):
            title_el = item.select_one(layout["title"])
            if not title_el:
                continue

            title_text = text(title_el)
            if not title_text:
                continue

            if "quy mô" not in title_text.lower():
                continue

            value_el = item.select_one(layout["value"])
            if not value_el:
                continue

            value = text(value_el)
            if value:
                return value

    return None

COMPANY_DESCRIPTION_LAYOUTS = [
    {
        "name": "basic_company",
        "container": "#section-introduce .content",
        "paragraphs": "p",
    },
    {
        "name": "premium_company",
        "container": ".intro-section .intro-content",
        "paragraphs": "p",
    },
]

def extract_company_description(soup: BeautifulSoup) -> Optional[str]:
    for layout in COMPANY_DESCRIPTION_LAYOUTS:
        container = soup.select_one(layout["container"])
        if not container:
            continue

        paragraphs = []
        for p in container.select(layout.get("paragraphs", "p")):
            t = p.get_text(" ", strip=True)
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                paragraphs.append(t)

        if paragraphs:
            return "\n".join(paragraphs)

        raw = container.get_text("\n", strip=True)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n+", "\n", raw).strip()
        if raw:
            return raw

    return None

def scrape_company(session, company_url):
    if not company_url:
        return {
            "company_name_full": None,
            "company_website": None,
            "company_address": None,
            "company_scale": None,
            "company_description": None,
        }

    try:
        soup = get_soup(session, company_url)
        smart_sleep()

        company_name = extract_company_name(soup)
        company_website = extract_company_website(soup)
        company_address = extract_company_address(soup)
        company_scale = extract_company_scale(soup)
        company_description = extract_company_description(soup)

        return {
            "company_name_full": company_name,
            "company_website": company_website,
            "company_address": company_address,
            "company_scale": company_scale,
            "company_description": company_description,
        }

    except Exception as e:
        print(f"[WARN] scrape_company failed for {company_url}: {e}")
        return {
            "company_name_full": None,
            "company_website": None,
            "company_address": None,
            "company_scale": None,
            "company_description": None,
        }

# ── Pipeline chính ───────────────────────────────────────────────

def crawl_to_dataframe(session, query_url_template: str, start_page: int = 1, end_page: int = 1,
                       delay_between_pages=(0.5 , 1)) -> pd.DataFrame:
    rows: List[Dict] = []
    seen_jobs = set()

    for page in range(start_page, end_page + 1):
        url = query_url_template.format(page=page)
        print(f"[INFO] Crawling search page {page}: {url}")
        jobs = parse_search_page(session, url)

        if not jobs:
            print(f"[INFO] Trang {page} không còn job — dừng sớm.")
            break

        for j in jobs:
            job_url = j["job_url"]
            job_id = urlparse(job_url).path
            if job_id in seen_jobs:
                continue
            seen_jobs.add(job_id)

            try:
                detail = scrape_job_detail(session, job_url)
            except Exception as e:
                print(f"[WARN] Lỗi job detail {job_url}: {e}")
                detail = {k: None for k in [
                    "detail_title", "detail_salary", "detail_location", "detail_experience", 
                    "deadline", "tags", 
                    "job_level", "education_level", "job_quantity", "employment_type",
                    "desc_mota", "desc_yeucau", "desc_quyenloi", "working_addresses", "working_times", 
                    "company_url_from_job", "company_scale_from_job", "company_address_from_job", "company_field_from_job"
                ]}

            company_url = detail.get("company_url_from_job") or j.get("company_url")

            try:
                comp = scrape_company(session, company_url)
            except Exception as e:
                print(f"[WARN] Lỗi company {company_url}: {e}")
                comp = {k: None for k in [
                    "company_name_full", "company_website", "company_description",
                    "company_address", "company_scale"
                ]}

            row = {**j, **detail, **comp}
            rows.append(row)

        smart_sleep(*delay_between_pages)

    df = pd.DataFrame(rows)
    cols = [
        "title", "detail_title",
        "job_url",
        "company_name", "company_name_full",
        "company_url", "company_url_from_job",
        "salary_list", "detail_salary",
        "address_list", "detail_location",
        "exp_list", "detail_experience",
        "deadline", "tags", 
        "job_level", "education_level", "job_quantity", "employment_type",
        "working_addresses", "working_times",
        "desc_mota", "desc_yeucau", "desc_quyenloi",
        "company_website", 
        "company_scale_from_job","company_scale", "company_field_from_job",
        "company_address_from_job", "company_address", 
        "company_description",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.loc[:, cols] if cols else df

# ── Xử lý nhiều lĩnh vực + gộp + lưu ─────────────────────────────

def normalize_job_url(url):
    if pd.isna(url):
        return None
    url = str(url).strip()
    if not url:
        return None
    url = url.split("?")[0]
    url = url.rstrip("/")
    return url

def first_non_empty(series):
    for val in series:
        if pd.notna(val):
            val_str = str(val).strip()
            if val_str and val_str.lower() != "nan":
                return val
    return None

def join_unique(series, sep=" | "):
    values = []
    seen = set()
    for val in series:
        if pd.notna(val):
            val_str = str(val).strip()
            if val_str and val_str.lower() != "nan" and val_str not in seen:
                seen.add(val_str)
                values.append(val_str)
    return sep.join(values) if values else None

def crawl_all_fields(fields, session):
    all_dfs = []
    for field in fields:
        slug = field["slug"]
        name = field["name_vn"]
        max_page = field.get("end_page", 5)

        print(f"\n{'═'*60}")
        print(f" BẮT ĐẦU CÀO: {name} ({slug}) — tối đa {max_page} trang")
        print(f"{'═'*60}\n")

        url_template = f"https://www.topcv.vn/tim-viec-lam-{slug}?type_keyword=1&page={{page}}&sba=1"

        try:
            df = crawl_to_dataframe(
                session=session,
                query_url_template=url_template,
                start_page=1,
                end_page=max_page,
                delay_between_pages=(1.0, 2.2),
            )

            if df.empty:
                print(f"→ Không có dữ liệu cho {name}")
                continue

            df["source_field_name"] = name
            print(f"  Thu được {len(df)} jobs từ {name}")
            all_dfs.append(df)

        except Exception as e:
            print(f"[LỖI] {name} thất bại: {e}")

    return all_dfs

def merge_jobs_by_url(all_dfs):
    if not all_dfs:
        return pd.DataFrame()

    final_df = pd.concat(all_dfs, ignore_index=True)
    total_before = len(final_df)

    if "job_url" not in final_df.columns:
        raise ValueError("Không tìm thấy cột 'job_url' để gộp job trùng.")

    final_df["job_url"] = final_df["job_url"].apply(normalize_job_url)
    missing_url_count = final_df["job_url"].isna().sum()
    if missing_url_count > 0:
        print(f"[CẢNH BÁO] Có {missing_url_count} dòng không có job_url, sẽ bị loại bỏ.")
        final_df = final_df[final_df["job_url"].notna()].copy()

    agg_dict = {}
    for col in final_df.columns:
        if col == "job_url":
            continue
        elif col == "source_field_name":
            agg_dict[col] = join_unique
        else:
            agg_dict[col] = first_non_empty

    merged_df = (
        final_df
        .groupby("job_url", as_index=False)
        .agg(agg_dict)
    )

    if "source_field_name" in merged_df.columns:
        merged_df["field_count"] = merged_df["source_field_name"].apply(
            lambda x: len(str(x).split(" | ")) if pd.notna(x) and str(x).strip() else 0
        )

    total_after = len(merged_df)
    merged_count = total_before - total_after

    print(f"Tổng số dòng trước khi gộp: {total_before}")
    print(f"Tổng số job sau khi gộp : {total_after}")
    print(f"Số dòng được gộp lại    : {merged_count}")

    cols = merged_df.columns.tolist()
    preferred_order = ["job_url", "source_field_name", "field_count"]
    ordered_cols = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
    merged_df = merged_df[ordered_cols]

    return merged_df

def save_output_files(df, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    xlsx_path = os.path.join(output_dir, f"{base_name}.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu CSV  → {csv_path}")

    try:
        df.to_excel(xlsx_path, index=False)
        print(f"Đã lưu Excel → {xlsx_path}")
    except Exception as e:
        print(f"Lỗi lưu Excel: {e}")

    return csv_path, xlsx_path

def build_fields_config(session: requests.Session):
    """
    Trả về danh sách cấu hình các lĩnh vực cần cào.
    end_page của từng slug sẽ được tự động tính bằng:
        tổng số job lấy từ title trang search / 51, rồi làm tròn lên.
    """
    fields = [
        {"slug": "data-analyst", "name_vn": "Data Analyst"},
        {"slug": "data-engineer", "name_vn": "Data Engineer"},
        {"slug": "data-scientist", "name_vn": "Data Scientist"},
        {"slug": "ai-engineer", "name_vn": "AI Engineer"},
        {"slug": "ai-researcher", "name_vn": "AI Researcher"},
        {"slug": "data-labeling-gan-nhan-du-lieu", "name_vn": "Data Labeling (Gán nhãn dữ liệu)"},
    ]

    for field in fields:
        slug = field["slug"]
        url = f"https://www.topcv.vn/tim-viec-lam-{slug}?type_keyword=1&page=1&sba=1"

        try:
            soup = get_soup(session, url)
            page_title = soup.title.get_text(" ", strip=True) if soup.title else ""

            # Ví dụ bắt các dạng:
            # "123 việc làm ..."
            # "1.234 việc làm ..."
            m = re.search(r"([\d\.,]+)\s+việc làm", page_title, re.IGNORECASE)

            if m:
                total_jobs = int(m.group(1).replace(".", "").replace(",", ""))
                end_page = (total_jobs + 50) // 51   # chia 51 và làm tròn lên
            else:
                end_page = 1

        except Exception as e:
            print(f"[WARN] Không tính được end_page cho slug={slug}: {e}")
            end_page = 1

        field["end_page"] = max(end_page, 1)
        print(f"[INFO] {slug}: end_page = {field['end_page']}")

    return fields

if __name__ == "__main__":

    session = build_session()              # tạo session trước
    fields = build_fields_config(session)  # truyền session vào đây

    # Resolve project root from src/crawl/topcv_crawler.py
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(BASE_DIR, "data", "raw", "jobs")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"topcv_all_fields_merged_{timestamp}"

    all_dfs = crawl_all_fields(fields, session)

    if not all_dfs:
        print("\nKHÔNG THU ĐƯỢC DỮ LIỆU NÀO!")
    else:

        print(f"\n{'═' * 60}")
        print(" XỬ LÝ DỮ LIỆU TRƯỚC KHI XUẤT FILE ")
        print(f"{'═' * 60}")

        merged_df = merge_jobs_by_url(all_dfs)

        if merged_df.empty:
            print("Không có dữ liệu hợp lệ để lưu file.")
        else:
            print(f"\n{'═' * 60}")
            print(" LƯU FILE OUTPUT ")
            print(f"{'═' * 60}")

            save_output_files(merged_df, output_dir, base_name)
            print("\nHOÀN TẤT TOÀN BỘ CÁC LĨNH VỰC!")
