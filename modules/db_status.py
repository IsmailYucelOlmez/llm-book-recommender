"""Veritabani durum ozeti: Google Books ile eklenen kitaplar."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from modules.config import (
    BOOKS_CSV_PATH,
    CACHE_DB_PATH,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    TAGGED_DESCRIPTION_PATH,
)

BASELINE_LOCAL_BOOKS = 5197


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _count_tagged_descriptions() -> int:
    path = Path(TAGGED_DESCRIPTION_PATH)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_sqlite_books() -> list[dict]:
    path = Path(CACHE_DB_PATH)
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT isbn13, book_json FROM book_cache ORDER BY isbn13").fetchall()
    conn.close()
    return [{"isbn13": isbn, **json.loads(book_json)} for isbn, book_json in rows]


def _load_sqlite_queries() -> list[dict]:
    path = Path(CACHE_DB_PATH)
    if not path.exists():
        return []

    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT query_hash, isbns_json, created_at FROM query_cache ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    results = []
    for query_hash, isbns_json, created_at in rows:
        results.append(
            {
                "query_hash": query_hash,
                "isbns": json.loads(isbns_json),
                "created_at": datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return results


def _load_csv_google_books() -> pd.DataFrame:
    path = Path(BOOKS_CSV_PATH)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "source" not in df.columns:
        return pd.DataFrame()
    return df[df["source"] == "google_books"].copy()


def _load_chroma_google_isbns() -> tuple[int, list[str]]:
    path = Path(CHROMA_DIR)
    if not path.exists() or not any(path.iterdir()):
        return 0, []

    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    from modules.embeddings_utils import get_collection

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
    )
    collection = get_collection(db)
    total = collection.count()
    if total == 0:
        return 0, []

    data = collection.get(include=["metadatas"])
    google_ids = [
        book_id
        for book_id, meta in zip(data["ids"], data["metadatas"], strict=False)
        if meta.get("source") == "google_books"
    ]
    return total, sorted(google_ids)


def print_status(include_chroma: bool = True) -> None:
    _configure_stdout()
    load_dotenv()

    print("=" * 60)
    print("Google Books Dinamik Modul — Veritabani Durumu")
    print("=" * 60)

    sqlite_books = _load_sqlite_books()
    csv_books = _load_csv_google_books()
    tagged_count = _count_tagged_descriptions()
    csv_total = len(pd.read_csv(BOOKS_CSV_PATH)) if Path(BOOKS_CSV_PATH).exists() else 0

    chroma_total = 0
    chroma_google_ids: list[str] = []
    if include_chroma:
        try:
            chroma_total, chroma_google_ids = _load_chroma_google_isbns()
        except Exception as error:
            print(f"\nChroma okunamadi: {error}")
            print("(GOOGLE_API_KEY gerekli olabilir)\n")

    sqlite_isbns = {book["isbn13"] for book in sqlite_books}
    csv_isbns = {str(isbn) for isbn in csv_books["isbn13"].astype(str)} if not csv_books.empty else set()
    chroma_isbns = set(chroma_google_ids)

    print("\n--- Ozet ---")
    print(f"  CSV toplam kitap        : {csv_total}")
    print(f"  CSV google_books        : {len(csv_isbns)}")
    print(f"  SQLite book_cache       : {len(sqlite_isbns)}")
    print(f"  Chroma toplam vektor    : {chroma_total or '-'}")
    print(f"  Chroma google_books     : {len(chroma_isbns) or '-'}")
    print(f"  tagged_description satir: {tagged_count}")
    print(f"  Tahmini yeni kitap      : {max(0, csv_total - BASELINE_LOCAL_BOOKS)}")

    print("\n--- Tutarlilik ---")
    if sqlite_isbns == csv_isbns == chroma_isbns and sqlite_isbns:
        print("  OK — uc katman uyumlu")
    elif not sqlite_isbns and not csv_isbns:
        print("  Henuz Google Books kitabi eklenmemis")
    else:
        only_sqlite = sqlite_isbns - csv_isbns - chroma_isbns
        only_csv = csv_isbns - sqlite_isbns - chroma_isbns
        only_chroma = chroma_isbns - sqlite_isbns - csv_isbns
        if only_sqlite:
            print(f"  Sadece SQLite : {sorted(only_sqlite)}")
        if only_csv:
            print(f"  Sadece CSV    : {sorted(only_csv)}")
        if only_chroma:
            print(f"  Sadece Chroma : {sorted(only_chroma)}")

    if sqlite_books:
        print("\n--- SQLite / CSV Kitaplar ---")
        for book in sqlite_books:
            title = book.get("title", "?")
            category = book.get("simple_categories", "?")
            print(f"  {book['isbn13']} | {title} [{category}]")
    elif not csv_books.empty:
        print("\n--- CSV Kitaplar ---")
        for _, row in csv_books.iterrows():
            print(f"  {row['isbn13']} | {row['title']}")

    queries = _load_sqlite_queries()
    if queries:
        print("\n--- Onbelleklenmis Sorgular ---")
        for item in queries:
            isbns = item["isbns"] or "(bos)"
            print(f"  {item['created_at']} | ISBN: {isbns} | hash: {item['query_hash'][:12]}...")

    print("\n" + "=" * 60)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Google Books modulu veritabani durumu")
    parser.add_argument(
        "--no-chroma",
        action="store_true",
        help="Chroma kontrolunu atla (API anahtari gerektirmez)",
    )
    args = parser.parse_args()
    print_status(include_chroma=not args.no_chroma)


if __name__ == "__main__":
    main()
