"""Yazari olmayan Google Books kitaplarini tum katmanlardan temizler."""

from __future__ import annotations

import sys

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from modules.book_cache import BookCache
from modules.config import CHROMA_DIR, EMBEDDING_MODEL
from modules.persistence import (
    find_google_books_without_authors,
    remove_books_from_csv,
    remove_books_from_tagged_descriptions,
)
from modules.vector_ingester import VectorIngester


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cleanup_books_without_authors() -> list[str]:
    invalid = find_google_books_without_authors()
    if invalid.empty:
        return []

    isbn13s = [str(isbn) for isbn in invalid["isbn13"].astype(str)]
    cache = BookCache()

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
    )
    ingester = VectorIngester(db, db._embedding_function)

    csv_removed = remove_books_from_csv(isbn13s)
    tagged_removed = remove_books_from_tagged_descriptions(isbn13s)
    chroma_removed = ingester.remove_books(isbn13s)
    cache_removed = cache.remove_books(isbn13s)

    print(f"Silinen ISBN'ler: {isbn13s}")
    print(f"  CSV satir       : {csv_removed}")
    print(f"  tagged_desc satir: {tagged_removed}")
    print(f"  Chroma vektor   : {chroma_removed}")
    print(f"  SQLite kayit    : {cache_removed}")
    return isbn13s


def main() -> None:
    _configure_stdout()
    load_dotenv()

    invalid = find_google_books_without_authors()
    if invalid.empty:
        print("Yazari olmayan Google Books kitabi bulunamadi.")
        return

    print("Yazari olmayan kitaplar:")
    for _, row in invalid.iterrows():
        print(f"  {row['isbn13']} | {row['title']} | authors={row['authors']!r}")

    cleanup_books_without_authors()


if __name__ == "__main__":
    main()
