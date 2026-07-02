import logging
import threading
from pathlib import Path

import pandas as pd

from modules.book_normalizer import has_valid_authors, parse_tagged_isbn
from modules.config import BOOKS_CSV_PATH, TAGGED_DESCRIPTION_PATH
from modules.types import BookRecord

logger = logging.getLogger(__name__)

EMOTION_COLUMNS = ["anger", "disgust", "neutral", "fear", "surprise", "joy", "sadness"]

# Serializes read-modify-write access to the CSV / tagged-description files so
# concurrent Gradio request threads cannot corrupt them or lose each other's
# writes. A single process-wide lock is enough for Gradio's threaded server.
_FILE_LOCK = threading.Lock()


def _category_column(df: pd.DataFrame) -> str:
    return "simple_categories" if "simple_categories" in df.columns else "simple_category"


def _build_row(book: BookRecord, columns: pd.Index, category_col: str) -> dict:
    """Build a single CSV row dict from a book, padding all expected columns.

    Shared by both the single- and batch-append paths so the row schema stays
    defined in exactly one place.
    """
    row = {
        "isbn13": int(book.isbn13),
        "isbn10": book.isbn10 or "",
        "title": book.title,
        "authors": book.authors,
        "categories": book.categories,
        "thumbnail": book.thumbnail or "",
        "description": book.description,
        "tagged_description": book.tagged_description,
        category_col: book.simple_categories or "Unknown",
        "source": book.source,
    }
    for col in EMOTION_COLUMNS:
        row[col] = None
    for col in columns:
        row.setdefault(col, None)
    return row


def _book_to_row(book: BookRecord, books_df: pd.DataFrame) -> dict | None:
    if not has_valid_authors(book.authors):
        logger.warning("Skipping book without author: %s (%s)", book.title, book.isbn13)
        return None

    if int(book.isbn13) in books_df["isbn13"].values:
        return None

    return _build_row(book, books_df.columns, _category_column(books_df))


def append_book_to_csv(book: BookRecord, books_df: pd.DataFrame) -> pd.DataFrame:
    return append_books_to_csv([book], books_df)


def append_books_to_csv(books: list[BookRecord], books_df: pd.DataFrame) -> pd.DataFrame:
    if not books:
        return books_df

    existing_isbns = set(books_df["isbn13"].values)
    category_col = _category_column(books_df)
    rows: list[dict] = []
    tagged_descriptions: list[str] = []

    for book in books:
        if not has_valid_authors(book.authors):
            logger.warning("Skipping book without author: %s (%s)", book.title, book.isbn13)
            continue

        isbn = int(book.isbn13)
        if isbn in existing_isbns:
            continue

        rows.append(_build_row(book, books_df.columns, category_col))
        existing_isbns.add(isbn)
        tagged_descriptions.append(book.tagged_description)

    if not rows:
        return books_df

    new_df = pd.concat([books_df, pd.DataFrame(rows)], ignore_index=True)
    with _FILE_LOCK:
        try:
            new_df.to_csv(BOOKS_CSV_PATH, index=False)
        except OSError as error:
            logger.warning("Could not persist books to CSV: %s", error)
            return books_df

        _append_tagged_descriptions(tagged_descriptions)
    return new_df


def _append_tagged_descriptions(tagged_descriptions: list[str]) -> None:
    """Append new tagged descriptions. Caller must hold ``_FILE_LOCK``."""
    if not tagged_descriptions:
        return

    path = Path(TAGGED_DESCRIPTION_PATH)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines_to_append = [
        tagged_description
        for tagged_description in tagged_descriptions
        if tagged_description and tagged_description not in existing
    ]
    if not lines_to_append:
        return

    with path.open("a", encoding="utf-8") as handle:
        for tagged_description in lines_to_append:
            handle.write(tagged_description + "\n")


def find_google_books_without_authors(books_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = books_df if books_df is not None else pd.read_csv(BOOKS_CSV_PATH)
    if "source" not in df.columns:
        return pd.DataFrame()

    google_books = df[df["source"] == "google_books"].copy()
    mask = ~google_books["authors"].apply(has_valid_authors)
    return google_books[mask]


def remove_books_from_csv(isbn13s: list[str]) -> int:
    if not isbn13s:
        return 0

    path = Path(BOOKS_CSV_PATH)
    if not path.exists():
        return 0

    isbn_set = {int(isbn) for isbn in isbn13s}
    with _FILE_LOCK:
        df = pd.read_csv(path)
        before = len(df)
        df = df[~df["isbn13"].isin(isbn_set)]
        removed = before - len(df)
        if removed:
            df.to_csv(path, index=False)
    return removed


def remove_books_from_tagged_descriptions(isbn13s: list[str]) -> int:
    path = Path(TAGGED_DESCRIPTION_PATH)
    if not path.exists() or not isbn13s:
        return 0

    isbn_set = set(isbn13s)
    with _FILE_LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()
        kept: list[str] = []
        removed = 0

        for line in lines:
            if not line.strip():
                continue
            line_isbn = parse_tagged_isbn(line)
            if line_isbn in isbn_set:
                removed += 1
                continue
            kept.append(line)

        if removed:
            path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed
