#!/usr/bin/env python3
"""
Extract the bilingual Egyptian Civil Code PDF into one JSON record per article.

Pipeline
    pdftotext -layout  ->  logical lines (English | Arabic)
                       ->  heading detection (lookahead, so headings never leak)
                       ->  article state machine
                       ->  repeal handling / normalisation / validation
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ══════════════════════════════════════════════════════════════════════════
# Constants / regexes
# ══════════════════════════════════════════════════════════════════════════

BIDI_RE = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
SPLIT_RE = re.compile(r"\s{2,}")  # pdftotext -layout column separator

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EASTERN_ARABIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
WESTERN_DIGITS = "0123456789"
DIGIT_TRANSLATION = str.maketrans(
    ARABIC_INDIC_DIGITS + EASTERN_ARABIC_DIGITS, WESTERN_DIGITS * 2
)
TO_ARABIC_INDIC = str.maketrans(WESTERN_DIGITS, ARABIC_INDIC_DIGITS)
TO_EASTERN = str.maketrans(WESTERN_DIGITS, EASTERN_ARABIC_DIGITS)

NUM = "[0-9٠-٩۰-۹]"
AR_LETTERS = "\u0621-\u064A"

MONTHS = (
    "يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|سبتمبر|"
    "أكتوبر|اكتوبر|نوفمبر|ديسمبر|رمضان|محرم|صفر|ربيع|جمادى|شوال|"
    "ذو القعدة|ذو الحجة"
)
DAY_RE = re.compile(rf"(?<!{NUM})({NUM}{{1,2}})(?!{NUM})(?=\s*(?:{MONTHS}))")
YEAR_RE = re.compile(
    rf"((?:لسنة|بسنة|سنة|لعام|عام|{MONTHS})\s*)({NUM}{{4}})(?!{NUM})"
)

# --- articles ---------------------------------------------------------------
ARTICLE_EN_RE = re.compile(r"^A?rticle\s*(\d+)\b(.*)$", re.IGNORECASE)
ARTICLE_AR_MARKER_RE = re.compile(r"(?:مادة|المادة|المواد)")
ARTICLE_AR_HEADING_RE = re.compile(
    rf"^\s*(?:(?:مادة|المادة|المواد)\s*[\(\-]?\s*{NUM}+"
    rf"(?:\s*(?:الى|إلى|و|-)\s*{NUM}+)*"
    rf"|{NUM}+\s*[\)\-]?\s*(?:مادة|المادة))\s*[\)\-.:]?\s*"
)
ARTICLE_AR_NUMBER_RE = re.compile(
    rf"(?:مادة|المادة|المواد)\s*[\(\-]?\s*({NUM}+)|({NUM}+)\s*[\)\-]?\s*(?:مادة|المادة)"
)

REPEALED_RANGE_RE = re.compile(
    r"Articles?\s+(\d+)\s*(?:-|–|—|to)\s*(\d+)\s+"
    r"(?:(?:have|has)\s+been\s+|are\s+|is\s+)?repealed",
    re.IGNORECASE,
)
REPEALED_SINGLE_RE = re.compile(
    r"^Article\s+(\d+)\s*[-–—:.]?\s*\(?(?:(?:is|has\s+been)\s+)?repealed\)?",
    re.IGNORECASE,
)

# --- hierarchy --------------------------------------------------------------
BOOK_RE = re.compile(r"^BOOK\s+([IVXLCDM]+|\d+)\b\.?\s*(.*)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^Chapter\s+([IVXLCDM]+|\d+)\b\.?\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^Section\s+([IVXLCDM]+|\d+)\b\.?\s*(.*)$", re.IGNORECASE)
TOPIC_RE = re.compile(r"^(\d{1,2})\s*\.\s+(\S.*)$")
HIERARCHY_AR_MARKER_RE = re.compile(r"(?:الكتاب|الباب|الفصل|الفرع|القسم)")
ARABIC_NUMBERED_HEADING_RE = re.compile(rf"^\s*[-–—]?\s*{NUM}{{1,2}}\s*[-–—.)]?\s+\S")

MAX_HEADING_CHARS = 160
MAX_HEADING_WORDS = 22
MAX_CONT_LINES = 3          # subtitle lines allowed between heading and next anchor
MAX_FORWARD_SKIP = 5        # allowed jump when an Arabic marker confirms an anchor
EDGE_ROWS = 2               # first/last rows of a page considered for running headers

NOISE_PATTERNS = [re.compile(r"^page\s+\d+(\s+of\s+\d+)?$", re.IGNORECASE)]

EN_TAIL_HEADING_RE = re.compile(r"(?:^|\s)\d{1,2}\.\s+[A-Z][^.!?;]{0,150}:?$")
AR_TAIL_HEADING_RE = re.compile(rf"(?:^|\s)[-–—]?{NUM}{{1,2}}[-–—.)]?\s+[^.؛؟]{{0,150}}$")


# ══════════════════════════════════════════════════════════════════════════
# Text cleaning / normalisation
# ══════════════════════════════════════════════════════════════════════════

def clean(text: str) -> str:
    """Strip bidi controls, normalise NFC / spaces. Never reverses text."""
    if not text:
        return ""
    text = BIDI_RE.sub("", text)
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_digits(text: str) -> str:
    return text.translate(DIGIT_TRANSLATION)


def _same_digits(token: str, western: str) -> str:
    """Return ``western`` in the same digit family as ``token``."""
    if any(c in ARABIC_INDIC_DIGITS for c in token):
        return western.translate(TO_ARABIC_INDIC)
    if any(c in EASTERN_ARABIC_DIGITS for c in token):
        return western.translate(TO_EASTERN)
    return western


def _plausible_year(value: int) -> bool:
    return 1750 <= value <= 2100 or 1250 <= value <= 1500


def repair_reversed_numeric_context(text: str) -> str:
    """
    Very conservative digit-direction repair (text_ar only, text_ar_raw is kept).

      * a day before an Arabic month is reversed only if impossible (51 -> 15);
      * a 4-digit year is reversed only in a *date context* (after سنة/لسنة/عام or
        a month name) and only if the original is implausible and the reversal
        is plausible. Article references such as "المادة ١٠٢١" are never touched.
    """
    if not text:
        return text

    def fix_day(m: re.Match[str]) -> str:
        token = m.group(1)
        western = normalize_digits(token)
        value = int(western)
        if 1 <= value <= 31:
            return token
        rev = western[::-1]
        if len(rev) == 2 and 1 <= int(rev) <= 31:
            return _same_digits(token, str(int(rev)))
        return token

    def fix_year(m: re.Match[str]) -> str:
        prefix, token = m.group(1), m.group(2)
        western = normalize_digits(token)
        if _plausible_year(int(western)):
            return m.group(0)
        rev = western[::-1]
        if _plausible_year(int(rev)):
            return prefix + _same_digits(token, rev)
        return m.group(0)

    text = DAY_RE.sub(fix_day, text)
    return YEAR_RE.sub(fix_year, text)


LAM_ALEF_STANDALONE_RE = re.compile(rf"(?<![{AR_LETTERS}])([وف]?)ال(?![{AR_LETTERS}])")


def repair_lam_alef(text: str) -> str:
    """
    pdftotext frequently flips the lam-alef ligature ("لا") into "ال".
    Only the *unambiguous* cases are repaired:
        standalone "ال" / "وال" / "فال"  -> "لا" / "ولا" / "فلا"
        "اال" (double alef, never valid)  -> "الا"   (االحوال, استعماال, إال ...)
    Ambiguous cases (ثلاث vs ثالث, ...) cannot be fixed without a dictionary.
    """
    text = LAM_ALEF_STANDALONE_RE.sub(lambda m: (m.group(1) or "") + "لا", text)
    return text.replace("اال", "الا")


def normalize_arabic(text: str, repair_ligatures: bool = True) -> str:
    """Normalise Arabic for retrieval. ``text_ar_raw`` is always kept separately."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", clean(text))
    text = text.replace("ـ", "")                         # tatweel
    text = re.sub(r"[إأآٱ]", "ا", text)                  # alef variants
    text = text.replace("ک", "ك").replace("ی", "ي")      # Persian variants
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)    # tashkeel
    if repair_ligatures:
        text = repair_lam_alef(text)
    text = repair_reversed_numeric_context(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_english(text: str) -> str:
    return clean(text)


# ══════════════════════════════════════════════════════════════════════════
# Physical line -> (english, arabic)
# ══════════════════════════════════════════════════════════════════════════

def classify_fragment(text: str) -> str:
    if not text:
        return "unknown"
    arabic = len(ARABIC_RE.findall(text))
    latin = len(LATIN_LETTER_RE.findall(text))
    if arabic and arabic >= latin:
        return "ar"
    if latin:
        return "en"
    return "unknown"


def _is_mixed(fragment: str) -> bool:
    return len(ARABIC_RE.findall(fragment)) >= 3 and len(LATIN_LETTER_RE.findall(fragment)) >= 5


def split_by_script(fragment: str) -> list[tuple[str, str]]:
    """Split a fragment that mixes both scripts into per-language runs."""
    runs: list[list[Any]] = []
    pending: list[str] = []
    for token in fragment.split():
        if ARABIC_RE.search(token):
            lang = "ar"
        elif LATIN_WORD_RE.search(token):
            lang = "en"
        else:
            lang = None  # digits / punctuation / single letters: stick to neighbours
        if lang is None:
            (runs[-1][1] if runs else pending).append(token)
        elif runs and runs[-1][0] == lang:
            runs[-1][1].append(token)
        else:
            runs.append([lang, pending + [token]])
            pending = []
    if not runs:
        return [("unknown", " ".join(pending))]
    return [(lang, " ".join(tokens)) for lang, tokens in runs]


def split_line(line: str) -> tuple[str, str]:
    """
    Return (english, arabic). Whitespace is NOT collapsed before splitting
    because the multi-space gap is the column separator.
    """
    line = unicodedata.normalize("NFC", BIDI_RE.sub("", line)).replace("\u00a0", " ")
    if not line.strip():
        return "", ""

    english: list[str] = []
    arabic: list[str] = []
    for fragment in SPLIT_RE.split(line.strip()):
        fragment = clean(fragment)
        if not fragment:
            continue
        parts = (
            split_by_script(fragment)
            if _is_mixed(fragment)
            else [(classify_fragment(fragment), fragment)]
        )
        for lang, text in parts:
            (arabic if lang == "ar" else english).append(text)
    return " ".join(english).strip(), " ".join(arabic).strip()


# ══════════════════════════════════════════════════════════════════════════
# PDF -> layout text -> logical lines
# ══════════════════════════════════════════════════════════════════════════

def pdf_to_layout_text(pdf_path: Path) -> str:
    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError("pdftotext not found in PATH (install poppler-utils).")
    result = subprocess.run(
        [executable, "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pdftotext failed ({result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout.decode("utf-8", errors="replace")


def iter_pages(layout_text: str):
    for index, text in enumerate(layout_text.split("\f"), start=1):
        if text.strip():
            yield index, text


@dataclass
class Line:
    page: int
    en: str
    ar: str
    role: str = "body"                       # body | heading | heading_cont
    heading: tuple[str, str] | None = None   # (type, label) when role == "heading"


def is_noise_line(en: str, ar: str) -> bool:
    combined = clean(f"{en} {ar}")
    return not combined or any(p.match(combined) for p in NOISE_PATTERNS)


def header_key(en: str, ar: str) -> str:
    """Key used to detect running headers. Article/hierarchy lines are never keys."""
    if ARTICLE_EN_RE.match(en) or parse_heading(en) is not None:
        return ""
    return clean(f"{en} | {ar}")


def detect_running_headers(pages: list[tuple[int, list[tuple[str, str]]]]) -> set[str]:
    """Lines repeated verbatim at page edges on >=40% of pages are page furniture."""
    if len(pages) < 10:
        return set()
    counter: Counter[str] = Counter()
    for _, rows in pages:
        edge = rows[:EDGE_ROWS] + rows[-EDGE_ROWS:]
        counter.update({k for k in (header_key(e, a) for e, a in edge) if k})
    threshold = max(5, int(len(pages) * 0.4))
    return {key for key, count in counter.items() if count >= threshold}


def build_lines(layout_text: str, stats: Counter) -> list[Line]:
    pages: list[tuple[int, list[tuple[str, str]]]] = []
    for page_no, page_text in iter_pages(layout_text):
        rows = []
        for raw in page_text.splitlines():
            if raw.strip():
                en, ar = split_line(raw)
                if en or ar:
                    rows.append((en, ar))
        pages.append((page_no, rows))

    running = detect_running_headers(pages)
    stats["running_header_keys"] = len(running)

    lines: list[Line] = []
    for page_no, rows in pages:
        last = len(rows) - 1
        for idx, (en, ar) in enumerate(rows):
            on_edge = idx < EDGE_ROWS or idx > last - EDGE_ROWS
            if is_noise_line(en, ar) or (on_edge and header_key(en, ar) in running):
                stats["noise_lines_removed"] += 1
                continue
            lines.append(Line(page_no, en, ar))
    stats["page_count"] = len(pages)
    return lines


# ══════════════════════════════════════════════════════════════════════════
# Article helpers
# ══════════════════════════════════════════════════════════════════════════

def article_number_from_english(english: str) -> int | None:
    m = ARTICLE_EN_RE.match(clean(english))
    return int(m.group(1)) if m else None


def article_body_after_heading(english: str) -> str:
    m = ARTICLE_EN_RE.match(clean(english))
    return m.group(2).strip() if m else ""


def article_same_line_body(english: str) -> str:
    return article_body_after_heading(english).lstrip(" .:-–—").strip()


def is_article_heading(english: str, arabic: str) -> bool:
    """Real bilingual heading (Arabic marker) or a bare English 'Article N' line."""
    if article_number_from_english(english) is None:
        return False
    if ARTICLE_AR_MARKER_RE.search(arabic):
        return True
    return article_body_after_heading(english) == ""


def arabic_article_number(arabic: str) -> int | None:
    m = ARTICLE_AR_NUMBER_RE.search(arabic)
    if not m:
        return None
    token = m.group(1) or m.group(2)
    try:
        return int(normalize_digits(token))
    except ValueError:
        return None


def article_arabic_remainder(arabic: str) -> str:
    """Arabic text on the heading line after 'مادة N' (dropped if pattern differs)."""
    stripped, count = ARTICLE_AR_HEADING_RE.subn("", arabic, count=1)
    if count and ARABIC_RE.search(stripped):
        return clean(stripped)
    return ""


def detect_repealed_range(english: str) -> tuple[int, int] | None:
    m = REPEALED_RANGE_RE.search(clean(english))
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi) if hi - lo <= 500 else None


def detect_repealed_single(english: str) -> int | None:
    m = REPEALED_SINGLE_RE.match(clean(english))
    return int(m.group(1)) if m else None


def next_expected(last: int | None, repealed: set[int]) -> int:
    n = 1 if last is None else last + 1
    while n in repealed:
        n += 1
    return n


# ══════════════════════════════════════════════════════════════════════════
# Heading detection (lookahead based: this is what stops the leakage)
# ══════════════════════════════════════════════════════════════════════════

def looks_like_heading_text(text: str) -> bool:
    text = clean(text)
    return (
        bool(text)
        and len(text) <= MAX_HEADING_CHARS
        and len(text.split()) <= MAX_HEADING_WORDS
        and not text.endswith((".", ";", "؛", "؟", "?"))
    )


def parse_heading(english: str) -> tuple[str, str, str] | None:
    """Shape-only test. Returns (type, number, title) or None."""
    english = clean(english)
    if not english:
        return None
    for htype, regex in (("book", BOOK_RE), ("chapter", CHAPTER_RE), ("section", SECTION_RE)):
        m = regex.match(english)
        if m:
            title = m.group(2).strip(" -–—:")
            if len(title) > MAX_HEADING_CHARS:
                return None
            return htype, m.group(1).upper(), title
    m = TOPIC_RE.match(english)
    if m and re.match(r"[A-Z]", m.group(2)) and looks_like_heading_text(m.group(2)):
        return "topic", m.group(1), m.group(2).strip()
    return None


def is_article_shape(en: str, ar: str) -> bool:
    return article_number_from_english(en) is not None and (
        bool(ARTICLE_AR_MARKER_RE.search(ar)) or article_body_after_heading(en) == ""
    )


def is_terminal(line: Line) -> bool:
    """A line that can legitimately follow a heading block."""
    return is_article_shape(line.en, line.ar) or parse_heading(line.en) is not None


def is_continuation(line: Line) -> bool:
    """Short subtitle line that may belong to a heading block."""
    if line.en:
        return looks_like_heading_text(line.en) and len(line.en) <= 100
    return bool(line.ar) and len(line.ar) <= 140 and not line.ar.endswith((".", "،"))


def heading_span_end(lines: list[Line], start: int) -> int | None:
    """Index of the first line AFTER the heading block, or None if not a heading."""
    upper = min(len(lines), start + MAX_CONT_LINES + 2)
    for j in range(start + 1, upper):
        if lines[j].role != "body":
            return None
        if is_terminal(lines[j]):
            return j
        if not is_continuation(lines[j]):
            return None
        if (detect_repealed_range(lines[j].en) is not None or detect_repealed_single(lines[j].en) is not None):
            return j
    return None


def strong_hierarchy_signal(line: Line, number: str, title: str) -> bool:
    """Accept Book/Chapter/Section without lookahead only on strong evidence."""
    if HIERARCHY_AR_MARKER_RE.search(line.ar):
        return True
    return not title and bool(re.fullmatch(r"[IVXLCDM]+", number))


def build_label(htype: str, number: str, title: str, subtitles: list[str]) -> str:
    if htype == "topic":
        pieces = [title] + subtitles
        return " - ".join(p.strip().rstrip(":").strip() for p in pieces if p.strip())
    extra = " ".join(p for p in [title] + subtitles if p).strip().rstrip(":")
    prefix = f"{htype.capitalize()} {number}"
    return f"{prefix} - {extra}" if extra else prefix


def mark_headings(lines: list[Line], stats: Counter) -> list[dict[str, Any]]:
    """
    Mark heading lines in place. Returns a sample of rejected candidates (debug).

    A numbered line such as "2. Juristic persons" is a topic heading ONLY when the
    lines that follow (<= MAX_CONT_LINES short subtitle lines) end at an
    'Article N' line or another heading. A numbered paragraph inside an article
    body is followed by more body text, so it is never mistaken for a heading.
    """
    rejected: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        parsed = parse_heading(line.en) if line.role == "body" else None
        if parsed is None:
            i += 1
            continue

        htype, number, title = parsed
        end = heading_span_end(lines, i)
        if end is None:
            if htype != "topic" and strong_hierarchy_signal(line, number, title):
                end = i + 1
            else:
                stats["rejected_heading_candidates"] += 1
                if len(rejected) < 50:
                    rejected.append({"page": line.page, "text": line.en[:120]})
                i += 1
                continue

        span = lines[i:end]
        subtitles = [l.en for l in span[1:] if l.en]
        line.role = "heading"
        line.heading = (htype, build_label(htype, number, title, subtitles))
        for extra in span[1:]:
            extra.role = "heading_cont"

        # Arabic heading text sometimes sits one physical line above the English one.
        if htype == "topic" and i > 0 and not any(l.ar for l in span):
            prev = lines[i - 1]
            if (
                prev.role == "body"
                and not prev.en
                and len(prev.ar) <= 140
                and not prev.ar.endswith((".", "،"))
                and ARABIC_NUMBERED_HEADING_RE.match(prev.ar)
            ):
                prev.role = "heading_cont"
                stats["heading_arabic_prelines"] += 1

        stats[f"{htype}_headings"] += 1
        i = end
    return rejected


def update_heading_stack(stack: dict[str, str | None], htype: str, value: str) -> None:
    order = ["book", "chapter", "section", "topic"]
    stack[htype] = value
    for lower in order[order.index(htype) + 1:]:
        stack[lower] = None


# ══════════════════════════════════════════════════════════════════════════
# Records
# ══════════════════════════════════════════════════════════════════════════

def new_article(number: int, stack: dict[str, str | None], page: int) -> dict[str, Any]:
    return {
        "article_number": number,
        "book": stack["book"],
        "chapter": stack["chapter"],
        "section": stack["section"],
        "topic": stack["topic"],
        "text_ar": "",
        "text_en": "",
        "text_ar_raw": "",
        "is_repealed": False,
        "source_page": page,
        "citation": f"Egyptian Civil Code, Article {number}",
        "source_status": "normal",
    }


def append_article_text(article: dict[str, Any], english: str, arabic: str) -> None:
    if english:
        article["text_en"] = f"{article['text_en']} {normalize_english(english)}".strip()
    if arabic:
        article["text_ar_raw"] = f"{article['text_ar_raw']} {clean(arabic)}".strip()


def finalize_article(article: dict[str, Any], repair_ligatures: bool) -> None:
    article["text_en"] = normalize_english(article.get("text_en", ""))
    article["text_ar_raw"] = clean(article.get("text_ar_raw", ""))
    article["text_ar"] = normalize_arabic(article["text_ar_raw"], repair_ligatures)


def annotate_source_status(records: list[dict[str, Any]]) -> None:
    for r in records:
        if r["source_status"] != "normal":
            continue
        has_ar, has_en = bool(r["text_ar"].strip()), bool(r["text_en"].strip())
        if not has_ar and not has_en:
            r["source_status"] = "empty_in_source_pdf"
        elif not has_ar:
            r["source_status"] = "missing_arabic_text_in_source_pdf"
        elif not has_en:
            r["source_status"] = "missing_english_text_in_source_pdf"


# ══════════════════════════════════════════════════════════════════════════
# Main parser
# ══════════════════════════════════════════════════════════════════════════

def parse_layout_text(
    layout_text: str, repair_ligatures: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: Counter = Counter()
    lines = build_lines(layout_text, stats)
    rejected_headings = mark_headings(lines, stats)

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    stack: dict[str, str | None] = {"book": None, "chapter": None, "section": None, "topic": None}

    repealed_set: set[int] = set()
    repeal_notes: list[dict[str, Any]] = []
    uncertain_lines: list[dict[str, Any]] = []
    rejected_anchors: list[dict[str, Any]] = []
    number_mismatches: list[dict[str, Any]] = []
    last_number: int | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            finalize_article(current, repair_ligatures)
            records.append(current)
            current = None

    for line in lines:
        en, ar = line.en, line.ar
        
        # ---- repeal notices -------------------------------------------------
        rng = detect_repealed_range(en)
        if rng is not None:
            flush()
            lo, hi = rng
            repealed_set.update(range(lo, hi + 1))
            repeal_notes.append({
                "start_article": lo, "end_article": hi, "source_page": line.page,
                "text_en": clean(en), "text_ar": normalize_arabic(ar, repair_ligatures),
                "text_ar_raw": clean(ar), "hierarchy": dict(stack),
            })
            continue

        single = detect_repealed_single(en)
        if single is not None:
            repealed_set.add(single)
            repeal_notes.append({
                "start_article": single, "end_article": single, "source_page": line.page,
                "text_en": clean(en), "text_ar": normalize_arabic(ar, repair_ligatures),
                "text_ar_raw": clean(ar), "hierarchy": dict(stack),
            })
            if current is not None and current["article_number"] == single:
                current["is_repealed"] = True
            flush()  # the notice must never be glued to another article
            continue

        # ---- hierarchy headings (already resolved by lookahead) -------------
        if line.role == "heading":
            flush()
            update_heading_stack(stack, *line.heading)  # type: ignore[misc]
            continue
        if line.role == "heading_cont":
            continue

        # ---- article anchor -------------------------------------------------
        number = article_number_from_english(en)
        if number is not None and is_article_heading(en, ar):
            has_marker = bool(ARTICLE_AR_MARKER_RE.search(ar))
            expected = next_expected(last_number, repealed_set)
            accept = number == expected or (
                has_marker
                and (
                    expected < number <= expected + MAX_FORWARD_SKIP
                    or (number in repealed_set and (last_number is None or number > last_number))
                )
            )
            if accept:
                flush()
                current = new_article(number, stack, line.page)
                last_number = number
                append_article_text(current, article_same_line_body(en), article_arabic_remainder(ar))
                ar_no = arabic_article_number(ar)
                if ar_no is not None and ar_no != number:
                    number_mismatches.append(
                        {"english": number, "arabic": ar_no, "source_page": line.page}
                    )
                continue
            rejected_anchors.append(
                {"candidate": number, "expected": expected, "source_page": line.page, "text": en[:100]}
            )
            # fall through: treated as ordinary body text (an in-body citation)

        # ---- body text ------------------------------------------------------
        if current is not None:
            append_article_text(current, en, ar)
        else:
            uncertain_lines.append({"source_page": line.page, "english": en, "arabic": ar})

    flush()

    # ---- repealed handling --------------------------------------------------
    note_by_number: dict[int, dict[str, Any]] = {}
    for note in repeal_notes:
        for n in range(note["start_article"], note["end_article"] + 1):
            note_by_number[n] = note

    for r in records:
        n = r["article_number"]
        if n in repealed_set:
            r["is_repealed"] = True
            if not r["text_en"].strip() and not r["text_ar"].strip():
                r["source_status"] = "repealed_no_individual_body_in_source"
                note = note_by_number.get(n)
                if note:
                    r["repeal_note_en"] = note["text_en"]
                    r["repeal_note_ar"] = note["text_ar"]

    # Placeholders: explicit source fact "Articles X-Y repealed", no invented text.
    existing = {r["article_number"] for r in records}
    for n in sorted(repealed_set - existing):
        note = note_by_number.get(n)
        h = (note or {}).get("hierarchy") or {}
        records.append({
            "article_number": n,
            "book": h.get("book"), "chapter": h.get("chapter"),
            "section": h.get("section"), "topic": h.get("topic"),
            "text_ar": "", "text_en": "", "text_ar_raw": "",
            "is_repealed": True,
            "source_page": note["source_page"] if note else None,
            "citation": f"Egyptian Civil Code, Article {n}",
            "source_status": "repealed_no_individual_body_in_source",
            "repeal_note_en": note["text_en"] if note else "",
            "repeal_note_ar": note["text_ar"] if note else "",
        })

    records.sort(key=lambda r: r["article_number"])
    annotate_source_status(records)

    metadata = {
        "page_count": stats.pop("page_count", 0),
        "parser_stats": dict(stats),
        "repealed_ranges": [
            {k: n[k] for k in ("start_article", "end_article", "source_page")}
            for n in repeal_notes
        ],
        "rejected_heading_candidates": rejected_headings,
        "rejected_article_anchors": rejected_anchors[:100],
        "arabic_number_mismatches": number_mismatches[:100],
        "uncertain_line_count": len(uncertain_lines),
        "uncertain_lines": uncertain_lines[:100],
        "ligature_repair_enabled": repair_ligatures,
    }
    return records, metadata


# ══════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════

def validate(
    records: list[dict[str, Any]], metadata: dict[str, Any], expected_max: int | None
) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []

    numbers = [r["article_number"] for r in records]
    counts = Counter(numbers)
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    if duplicates:
        problems.append(f"duplicate article numbers: {duplicates}")

    sorted_numbers = sorted(set(numbers))
    gaps = [
        [a + 1, b - 1] for a, b in zip(sorted_numbers, sorted_numbers[1:]) if b > a + 1
    ]
    if gaps:
        problems.append(f"unexplained article gaps: {gaps}")

    if expected_max:
        missing = sorted(set(range(1, expected_max + 1)) - set(numbers))
        extra = sorted(set(numbers) - set(range(1, expected_max + 1)))
        if missing:
            problems.append(f"missing articles 1..{expected_max}: {missing[:30]}")
        if extra:
            problems.append(f"articles outside 1..{expected_max}: {extra[:30]}")

    repealed_numbers = sorted({r["article_number"] for r in records if r["is_repealed"]})
    missing_repealed: list[int] = []
    for rng in metadata.get("repealed_ranges", []):
        for n in range(rng["start_article"], rng["end_article"] + 1):
            if n not in repealed_numbers:
                missing_repealed.append(n)
    if missing_repealed:
        problems.append(f"declared repealed articles not represented: {missing_repealed[:30]}")

    def real(r: dict[str, Any]) -> bool:
        return r["source_status"] != "repealed_no_individual_body_in_source"

    empty_ar = [r["article_number"] for r in records if real(r) and not r["text_ar"].strip()]
    empty_en = [r["article_number"] for r in records if real(r) and not r["text_en"].strip()]
    if empty_ar:
        warnings.append(f"records with empty text_ar: {empty_ar[:20]}")
    if empty_en:
        warnings.append(f"records with empty text_en: {empty_en[:20]}")

    huge_en = [r["article_number"] for r in records if len(r["text_en"]) > 10000]
    huge_ar = [r["article_number"] for r in records if len(r["text_ar"]) > 10000]
    if huge_en:
        problems.append(f"suspiciously long English records: {huge_en}")
    if huge_ar:
        problems.append(f"suspiciously long Arabic records: {huge_ar}")

    # Heading-leakage detectors (should be empty after the fix).
    suspect_en = [r["article_number"] for r in records if EN_TAIL_HEADING_RE.search(r["text_en"])]
    suspect_ar = [r["article_number"] for r in records if AR_TAIL_HEADING_RE.search(r["text_ar_raw"])]
    if suspect_en:
        warnings.append(f"English text ends like a numbered heading: {suspect_en[:30]}")
    if suspect_ar:
        warnings.append(f"Arabic text ends like a numbered heading: {suspect_ar[:30]}")

    unterminated = [
        r["article_number"] for r in records
        if r["text_en"] and not r["text_en"].rstrip().endswith((".", ":", ";", ")", '"', "”", "?"))
    ]
    if unterminated:
        warnings.append(f"{len(unterminated)} English texts lack final punctuation: {unterminated[:20]}")

    if metadata.get("arabic_number_mismatches"):
        warnings.append(
            f"{len(metadata['arabic_number_mismatches'])} Arabic/English article-number mismatches "
            "(see extraction_metadata)"
        )
    if metadata.get("rejected_article_anchors"):
        warnings.append(
            f"{len(metadata['rejected_article_anchors'])} article-like lines rejected as citations "
            "(see extraction_metadata)"
        )

    status_counts = Counter(r.get("source_status", "unknown") for r in records)
    hierarchy_nulls = {
        key: sum(1 for r in records if r[key] is None)
        for key in ("book", "chapter", "section", "topic")
    }

    return {
        "ok": not problems,
        "total_records": len(records),
        "unique_article_numbers": len(set(numbers)),
        "min_article": min(numbers) if numbers else None,
        "max_article": max(numbers) if numbers else None,
        "duplicate_article_numbers": duplicates,
        "empty_arabic_records": empty_ar,
        "empty_english_records": empty_en,
        "huge_english_records": huge_en,
        "huge_arabic_records": huge_ar,
        "all_gaps": gaps,
        "unexplained_gaps": gaps,
        "repealed_articles": repealed_numbers,
        "missing_repealed_articles": missing_repealed,
        "suspect_trailing_heading_en": suspect_en,
        "suspect_trailing_heading_ar": suspect_ar,
        "hierarchy_null_counts": hierarchy_nulls,
        "source_status_counts": dict(status_counts),
        "warnings": warnings,
        "problems": problems,
    }


# ══════════════════════════════════════════════════════════════════════════
# Output / reporting
# ══════════════════════════════════════════════════════════════════════════

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_validation(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72 + "\nVALIDATION REPORT\n" + "=" * 72)
    print(f"Total records:          {report['total_records']}")
    print(f"Unique article numbers: {report['unique_article_numbers']}")
    print(f"Article range:          {report['min_article']} -> {report['max_article']}")
    print(f"Duplicates:             {len(report['duplicate_article_numbers'])}")
    print(f"Unexplained gaps:       {len(report['unexplained_gaps'])}")
    print(f"Repealed represented:   {len(report['repealed_articles'])}")
    print(f"Hierarchy nulls:        {report['hierarchy_null_counts']}")
    print(f"Source status:          {report['source_status_counts']}")
    if report["problems"]:
        print("Problems:")
        for p in report["problems"]:
            print(f"  - {p}")
    else:
        print("Validation: OK")
    for w in report["warnings"]:
        print(f"  warning: {w}")
    print("=" * 72)


def _print_record(r: dict[str, Any], tail: int = 250) -> None:
    print(f"\nArticle {r['article_number']} | page {r.get('source_page')} | {r['source_status']}")
    print(f"Book: {r['book']} | Chapter: {r['chapter']} | Section: {r['section']} | Topic: {r['topic']}")
    for label, key in (("AR", "text_ar"), ("EN", "text_en")):
        text = r.get(key, "")
        head = text[:300]
        end = f"\n   ... ENDS: {text[-tail:]}" if len(text) > 300 else ""
        print(f"{label}: {head}{end}")
    print("-" * 72)


def print_manual_sample(
    records: list[dict[str, Any]], spot: list[int], count: int, seed: int
) -> None:
    by_number = {r["article_number"]: r for r in records}
    print("\n" + "=" * 72 + "\nSPOT CHECK (look at the ENDS: lines for leaked headings)\n" + "=" * 72)
    for n in spot:
        if n in by_number:
            _print_record(by_number[n])

    candidates = [r for r in records if r["text_ar"] and r["text_en"] and r["article_number"] not in spot]
    sample = random.Random(seed).sample(candidates, min(count, len(candidates)))
    sample.sort(key=lambda r: r["article_number"])
    print("\n" + "=" * 72 + f"\nRANDOM SAMPLE ({len(sample)} records)\n" + "=" * 72)
    for r in sample:
        _print_record(r)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the bilingual Egyptian Civil Code PDF into per-article JSON."
    )
    parser.add_argument("--pdf", type=Path, help="Input PDF path.")
    parser.add_argument("--input", type=Path, help="Existing `pdftotext -layout` dump.")
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    parser.add_argument("--debug-layout", type=Path, default=None)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--expected-max", type=int, default=None,
                        help="Expected highest article number (0 disables the check).")
    parser.add_argument("--spot-check", default="5,6,51,52,53,554,556,988,1022",
                        help="Comma-separated article numbers to print.")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ligature-repair", action="store_true",
                        help="Disable the safe lam-alef repair in text_ar.")
    args = parser.parse_args()

    if bool(args.pdf) == bool(args.input):
        parser.error("Provide exactly one of --pdf or --input.")

    if args.pdf:
        if not args.pdf.exists():
            raise FileNotFoundError(f"PDF not found: {args.pdf}")
        print(f"Extracting with pdftotext -layout: {args.pdf}")
        layout_text = pdf_to_layout_text(args.pdf)
    else:
        if not args.input.exists():
            raise FileNotFoundError(f"Input text not found: {args.input}")
        layout_text = args.input.read_text(encoding="utf-8")

    if args.debug_layout:
        args.debug_layout.parent.mkdir(parents=True, exist_ok=True)
        args.debug_layout.write_text(layout_text, encoding="utf-8")
        print(f"Saved layout text: {args.debug_layout}")

    records, metadata = parse_layout_text(layout_text, repair_ligatures=not args.no_ligature_repair)
    report = validate(records, metadata, args.expected_max or None)
    report["extraction_metadata"] = metadata

    save_json(records, args.output)
    print(f"Parsed {len(records)} records -> {args.output}")

    validation_path = args.validation or args.output.parent / "validation_report.json"
    save_json(report, validation_path)
    print(f"Validation report -> {validation_path}")

    print(f"Parser stats: {metadata['parser_stats']}")
    print_validation(report)
    spot = [int(x) for x in args.spot_check.split(",") if x.strip().isdigit()]
    print_manual_sample(records, spot, args.sample_size, args.seed)


if __name__ == "__main__":
    main()