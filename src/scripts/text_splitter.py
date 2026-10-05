#!/usr/bin/env python3
"""
Article-aware bilingual splitter for the Egyptian Civil Code.

    articles.json  ->  chunks.json  (+ optional stats.json for MLflow)

Design
------
1. The ARTICLE is the unit. An article that fits the token budget is ONE chunk.
2. Only oversized articles are split, at the coarsest boundary that works:
   sentence -> clause (; :) -> comma -> word. Never mid-word, never across articles.
   Marker-based (a)/(b)/(1) splitting is deliberately NOT used: on this corpus the
   markers are scrambled by the PDF extraction (see README notes), so sentence
   boundaries are the reliable signal.
3. The budget covers what the embedding model actually sees:
   doc_prefix + context header + body + special tokens. Not just the body.
4. `text` is the display/citation text (untouched). `embed_text` is what you embed
   (context header + cleaned Arabic). Store `text`, embed `embed_text`.
5. Repealed articles get an embeddable notice chunk, so "what is Article 60?"
   retrieves "repealed" instead of the retriever hallucinating around the gap.
6. Every chunk carries article_number + flat, vector-DB-safe metadata.

Strategies (for the MLflow chunking ablation)
---------------------------------------------
    article : sentence-aware, balanced packing (the recommended default)
    window  : naive word-window baseline (cuts mid-sentence) - the "wrong" approach
              you compare against, so your report can show the numbers.

Usage
-----
    python text_splitter.py --input data/articles.json --output data/chunks.json \
        --strategy article --max-tokens 256 --overlap 0 \
        --tokenizer intfloat/multilingual-e5-base --doc-prefix "passage: " \
        --stats data/chunk_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "article"        # "article" | "window"
    max_tokens: int = 512            # HARD ceiling on the full embedded string
    overlap_tokens: int = 0          # carried between parts of ONE article only
    context_header: str = "full"     # "none" | "article" | "full"
    doc_prefix: str = ""             # e.g. "passage: " for E5 models
    special_tokens: int = 2          # [CLS]/[SEP] or <s>/</s> reserve
    balanced: bool = True            # even out parts instead of 256 + 12
    min_tail_tokens: int = 24        # merge a tiny final part back if it fits

    def validate(self) -> None:
        if self.strategy not in {"article", "window"}:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.context_header not in {"none", "article", "full"}:
            raise ValueError(f"unknown context_header {self.context_header!r}")
        if self.max_tokens < 48:
            raise ValueError("max_tokens must be >= 48")
        if not 0 <= self.overlap_tokens < self.max_tokens // 2:
            raise ValueError("overlap_tokens must be in [0, max_tokens/2)")


# ══════════════════════════════════════════════════════════════════════════
# Text helpers
# ══════════════════════════════════════════════════════════════════════════

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
BIDI_RE = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
INDIC = "٠١٢٣٤٥٦٧٨٩"


def clean_text(text: str | None) -> str:
    """Strip bidi controls and collapse whitespace. Never reorders or rewrites words."""
    if not text:
        return ""
    text = BIDI_RE.sub("", str(text)).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


_MARKER_AFTER_PAREN = re.compile(rf"\(\s*\)?\s*[{INDIC}]{{1,2}}(?![{INDIC}])")
_MARKER_BEFORE_PAREN = re.compile(rf"(?<![{INDIC}])[{INDIC}]{{1,2}}\s*\(\s*\)?")


def arabic_for_embedding(text: str) -> str:
    """
    Remove PDF paragraph-marker debris such as ') (١' and 'الرقابة١ (' from Arabic.
    Only 1-2 digit numerals ADJACENT to a parenthesis are removed, so real numbers
    ('سنة 1948', 'خمس عشرة') survive. Used for embed_text only; `text` is untouched.
    """
    text = _MARKER_AFTER_PAREN.sub(" ", text)
    text = _MARKER_BEFORE_PAREN.sub(" ", text)
    text = re.sub(r"[()]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════
# Token counting
# ══════════════════════════════════════════════════════════════════════════


class TokenCounter:
    """
    Real tokenizer when --tokenizer is given (strongly recommended: token counts
    are only meaningful for the model you actually embed with). Otherwise a
    conservative estimate, good enough for a first pass but NOT for final numbers.
    """

    def __init__(self, tokenizer_name: str | None = None):
        self.name = tokenizer_name
        self._tok = None
        self._cache: dict[str, int] = {}
        if tokenizer_name:
            from transformers import AutoTokenizer  # fail loudly: silent fallback hides errors

            self._tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        if self._tok is not None:
            n = len(self._tok.encode(text, add_special_tokens=False))
        else:
            words = len(text.split())
            n = max(1, math.ceil(words * (2.2 if ARABIC_RE.search(text) else 1.4)))
        self._cache[text] = n
        return n


# ══════════════════════════════════════════════════════════════════════════
# Unit splitting (coarse -> fine)
# ══════════════════════════════════════════════════════════════════════════
# The Arabic column is visually ordered in the PDF, so punctuation often sticks
# to the FOLLOWING word ("...المشروع .هذا") - hence the lookahead variants.

SEPARATORS: dict[str, list[re.Pattern[str]]] = {
    "en": [
        re.compile(r"(?<=[.?!])\s+"),                 # sentence
        re.compile(r"(?<=[;:])\s+|\s+(?=;)"),         # clause
        re.compile(r"(?<=,)\s+"),                     # comma
    ],
    "ar": [
        re.compile(r"(?<=[.؟!])\s+|\s+(?=\.)"),
        re.compile(r"(?<=[؛;:])\s+|\s+(?=[؛;])"),
        re.compile(r"(?<=[،,])\s+|\s+(?=،)"),
    ],
}
WORD_SEP = re.compile(r"\s+")


def _split(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [p.strip() for p in pattern.split(text) if p and p.strip()]


def sentence_units(text: str, lang: str, budget: int, counter: TokenCounter) -> list[str]:
    """Sentences; any sentence over budget is recursively split at finer separators."""
    seps = SEPARATORS[lang]

    def fit(piece: str, level: int) -> list[str]:
        if counter.count(piece) <= budget:
            return [piece]
        while level < len(seps):
            parts = _split(piece, seps[level])
            level += 1
            if len(parts) > 1:
                return [u for p in parts for u in fit(p, level)]
        # last resort: words (a single word over budget is left whole; validation flags it)
        return _split(piece, WORD_SEP)

    return [u for s in _split(text, seps[0]) for u in fit(s, 1)]


def word_units(text: str) -> list[str]:
    return _split(text, WORD_SEP)


# ══════════════════════════════════════════════════════════════════════════
# Packing (exact: every candidate is re-counted as a joined string)
# ══════════════════════════════════════════════════════════════════════════


def pack(
    units: list[str], limit: int, overlap: int, counter: TokenCounter, min_tail: int = 24
) -> list[str]:
    chunks: list[list[str]] = []
    cur: list[str] = []

    def joined(us: list[str]) -> str:
        return " ".join(us)

    for u in units:
        if cur and counter.count(joined(cur + [u])) > limit:
            chunks.append(cur)
            carry: list[str] = []
            if overlap:
                for x in reversed(cur):
                    if counter.count(joined([x] + carry)) > overlap:
                        break
                    carry.insert(0, x)
            while carry and counter.count(joined(carry + [u])) > limit:
                carry.pop(0)
            cur = carry
        cur.append(u)
    if cur:
        chunks.append(cur)

    # merge a tiny tail back into its predecessor when that still fits (no-overlap case)
    if len(chunks) > 1 and overlap == 0:
        tail, prev = chunks[-1], chunks[-2]
        if counter.count(joined(tail)) < min_tail and counter.count(joined(prev + tail)) <= limit:
            chunks[-2] = prev + tail
            chunks.pop()
    return [joined(c) for c in chunks]


def split_body(text: str, lang: str, budget: int, cfg: ChunkConfig, counter: TokenCounter) -> list[str]:
    total = counter.count(text)
    if total <= budget:
        return [text]

    if cfg.strategy == "window":
        return pack(word_units(text), budget, cfg.overlap_tokens, counter, cfg.min_tail_tokens)

    units = sentence_units(text, lang, budget, counter)
    if cfg.balanced and cfg.overlap_tokens == 0:
        n = math.ceil(total / budget)
        longest = max(counter.count(u) for u in units)
        target = min(budget, max(longest, math.ceil(total / n * 1.15)))
        balanced = pack(units, target, 0, counter, cfg.min_tail_tokens)
        if len(balanced) <= n + 1:
            return balanced
    return pack(units, budget, cfg.overlap_tokens, counter, cfg.min_tail_tokens)


# ══════════════════════════════════════════════════════════════════════════
# Headers, metadata, records
# ══════════════════════════════════════════════════════════════════════════


def build_header(article: dict[str, Any], lang: str, mode: str) -> str:
    if mode == "none":
        return ""
    n = article["article_number"]
    head = f"المادة {n}" if lang == "ar" else f"Egyptian Civil Code, Article {n}"
    if mode == "article":
        return head
    crumbs = [article.get(k) for k in ("book", "chapter", "section", "topic")]
    crumb = " > ".join(c.strip() for c in crumbs if c and c.strip())
    return f"{head} | {crumb}" if crumb else head


def flat_metadata(article: dict[str, Any], lang: str, index: int, total: int, ctype: str) -> dict[str, Any]:
    """Flat, None-free metadata: safe for Chroma / Qdrant / pgvector filters."""
    page = article.get("source_page")
    return {
        "article_number": int(article["article_number"]),
        "language": lang,
        "book": article.get("book") or "",
        "chapter": article.get("chapter") or "",
        "section": article.get("section") or "",
        "topic": article.get("topic") or "",
        "is_repealed": bool(article.get("is_repealed", False)),
        "source_page": int(page) if page is not None else -1,
        "citation": article.get("citation") or f"Egyptian Civil Code, Article {article['article_number']}",
        "source_status": article.get("source_status") or "normal",
        "chunk_index": index,
        "n_chunks": total,
        "chunk_type": ctype,
    }


def make_chunk(
    article: dict[str, Any], lang: str, body: str, index: int, total: int, ctype: str,
    cfg: ChunkConfig, counter: TokenCounter,
) -> dict[str, Any]:
    n = article["article_number"]
    header = build_header(article, lang, cfg.context_header)
    embed_body = arabic_for_embedding(body) if lang == "ar" and ctype != "repealed_notice" else body
    embed_text = f"{cfg.doc_prefix}{header}\n{embed_body}" if header else f"{cfg.doc_prefix}{embed_body}"
    return {
        "chunk_id": f"art{n:04d}-{lang}-{index:02d}",
        "article_number": n,
        "language": lang,
        "chunk_index": index,
        "n_chunks": total,
        "chunk_type": ctype,
        "text": body,                        # display / citation / LLM context
        "embed_text": embed_text,            # what you embed
        "token_count": counter.count(embed_text) + cfg.special_tokens,
        "metadata": flat_metadata(article, lang, index, total, ctype),
    }


def repealed_notice_text(article: dict[str, Any], lang: str) -> str:
    n = article["article_number"]
    if lang == "ar":
        return f"المادة {n} ملغاة ولم تعد سارية."
    note = clean_text(article.get("repeal_note_en"))
    base = f"Article {n} of the Egyptian Civil Code has been repealed and is no longer in force."
    return f"{base} Source note: {note}" if note else base


def process_article(article: dict[str, Any], cfg: ChunkConfig, counter: TokenCounter) -> list[dict[str, Any]]:
    text_by_lang = {"ar": clean_text(article.get("text_ar")), "en": clean_text(article.get("text_en"))}
    chunks: list[dict[str, Any]] = []

    # Repealed with no body: emit an embeddable notice in both languages.
    if article.get("is_repealed") and not any(text_by_lang.values()):
        for lang in ("en", "ar"):
            body = repealed_notice_text(article, lang)
            chunks.append(make_chunk(article, lang, body, 0, 1, "repealed_notice", cfg, counter))
        return chunks

    for lang, text in text_by_lang.items():
        if not text:
            continue
        header = build_header(article, lang, cfg.context_header)
        overhead = counter.count(f"{cfg.doc_prefix}{header}\n") + cfg.special_tokens
        budget = cfg.max_tokens - overhead
        if budget < 32:
            raise ValueError(
                f"article {article['article_number']}: header/prefix eat {overhead} tokens; "
                f"raise max_tokens or use a shorter context_header"
            )
        parts = split_body(text, lang, budget, cfg, counter)
        ctype = "article" if len(parts) == 1 else "article_part"
        for i, part in enumerate(parts):
            chunks.append(make_chunk(article, lang, part, i, len(parts), ctype, cfg, counter))
    return chunks


def split_articles(
    articles: list[dict[str, Any]], cfg: ChunkConfig, counter: TokenCounter
) -> list[dict[str, Any]]:
    """Importable entry point: use this from src/rag.py and your MLflow loop."""
    cfg.validate()
    out: list[dict[str, Any]] = []
    for a in articles:
        if "article_number" not in a:
            raise ValueError(f"record without article_number: {str(a)[:80]}")
        out.extend(process_article(a, cfg, counter))
    return out


# ══════════════════════════════════════════════════════════════════════════
# Validation (fails the build - a silent chunking bug becomes a hallucination later)
# ══════════════════════════════════════════════════════════════════════════


def validate_chunks(
    chunks: list[dict[str, Any]], articles: list[dict[str, Any]], cfg: ChunkConfig
) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    ids = Counter(c["chunk_id"] for c in chunks)
    errors += [f"duplicate chunk_id {i}" for i, k in ids.items() if k > 1]

    for c in chunks:
        if not c["text"].strip():
            errors.append(f"{c['chunk_id']}: empty text")
        if c["token_count"] > cfg.max_tokens:
            errors.append(f"{c['chunk_id']}: {c['token_count']} tokens > {cfg.max_tokens}")
        if c["language"] not in {"ar", "en"}:
            errors.append(f"{c['chunk_id']}: bad language")

    by_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for c in chunks:
        by_key.setdefault((c["article_number"], c["language"]), []).append(c)

    for a in articles:
        n = a["article_number"]
        for lang, field in (("ar", "text_ar"), ("en", "text_en")):
            src = clean_text(a.get(field))
            got = sorted(by_key.get((n, lang), []), key=lambda c: c["chunk_index"])
            if a.get("is_repealed") and not src:
                if not got or got[0]["chunk_type"] != "repealed_notice":
                    errors.append(f"article {n} ({lang}): repealed but no notice chunk")
                continue
            if not src:
                continue
            if not got:
                errors.append(f"article {n} ({lang}): no chunks for non-empty text")
                continue
            if cfg.overlap_tokens == 0:
                rebuilt = " ".join(c["text"] for c in got).split()
                if rebuilt != src.split():
                    errors.append(f"article {n} ({lang}): chunks do not reconstruct the source text")
            elif not set(src.split()) <= set(" ".join(c["text"] for c in got).split()):
                errors.append(f"article {n} ({lang}): words lost during splitting")

    nums = sorted({a["article_number"] for a in articles})
    gaps = [(x + 1, y - 1) for x, y in zip(nums, nums[1:]) if y > x + 1]
    if gaps:
        warnings.append(f"input has article-number gaps (upstream extraction issue): {gaps[:5]}")

    for w in warnings:
        print(f"[WARN] {w}")
    return errors


# ══════════════════════════════════════════════════════════════════════════
# Stats (log these to MLflow next to chunk_size / overlap / embedding_model)
# ══════════════════════════════════════════════════════════════════════════


def compute_stats(chunks: list[dict[str, Any]], cfg: ChunkConfig, counter: TokenCounter) -> dict[str, Any]:
    tok = sorted(c["token_count"] for c in chunks if c["chunk_type"] != "repealed_notice")
    pct = lambda p: tok[min(len(tok) - 1, int(len(tok) * p))] if tok else 0  # noqa: E731
    split_articles_ = {
        (c["article_number"], c["language"]) for c in chunks if c["chunk_type"] == "article_part"
    }
    return {
        "config": asdict(cfg),
        "tokenizer": counter.name or "APPROXIMATE (pass --tokenizer for real numbers)",
        "tokenizer_exact": counter.exact,
        "total_chunks": len(chunks),
        "by_language": dict(Counter(c["language"] for c in chunks)),
        "by_type": dict(Counter(c["chunk_type"] for c in chunks)),
        "articles_split": len(split_articles_),
        "tokens": {
            "min": tok[0] if tok else 0,
            "median": statistics.median(tok) if tok else 0,
            "mean": round(sum(tok) / len(tok), 1) if tok else 0,
            "p95": pct(0.95),
            "max": tok[-1] if tok else 0,
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    p = argparse.ArgumentParser(description="Article-aware bilingual splitter (Egyptian Civil Code).")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--stats", type=Path, default=None, help="Write config + stats JSON (for MLflow).")
    p.add_argument("--strategy", choices=["article", "window"], default="article")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--overlap", type=int, default=0, help="Overlap tokens between parts of one article.")
    p.add_argument("--context-header", choices=["none", "article", "full"], default="full")
    p.add_argument("--doc-prefix", default="", help='E5 models: "passage: "')
    p.add_argument("--tokenizer", default=None, help="HF tokenizer of your embedding model.")
    p.add_argument("--no-balance", action="store_true")
    args = p.parse_args()

    cfg = ChunkConfig(
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        context_header=args.context_header,
        doc_prefix=args.doc_prefix,
        balanced=not args.no_balance,
    )
    counter = TokenCounter(args.tokenizer)
    if not counter.exact:
        print("[WARN] No --tokenizer given: token counts are estimates. Fine for a dry run, "
              "not for reported results.")

    articles = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(articles, list):
        raise ValueError("input must be a JSON list of article records")

    chunks = split_articles(articles, cfg, counter)
    errors = validate_chunks(chunks, articles, cfg)
    if errors:
        print("[ERROR] chunk validation failed:")
        for e in errors[:40]:
            print("  -", e)
        raise SystemExit(f"{len(errors)} validation error(s)")

    stats = compute_stats(chunks, cfg, counter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {len(chunks)} chunks -> {args.output}")
    print(json.dumps({k: stats[k] for k in ("by_language", "by_type", "articles_split", "tokens")}, indent=2))


if __name__ == "__main__":
    main()
