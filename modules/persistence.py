import logging
from pathlib import Path

import pandas as pd

from modules.book_normalizer import has_valid_authors
from modules.config import BOOKS_CSV_PATH, TAGGED_DESCRIPTION_PATH
from modules.types import BookRecord

logger = logging.getLogger(__name__)

EMOTION_COLUMNS = ["anger", "disgust", "neutral", "fear", "surprise", "joy", "sadness"]


def _book_to_row(book: BookRecord, books_df: pd.DataFrame) -> dict | None:
    if not has_valid_authors(book.authors):
        logger.warning("Skipping book without author: %s (%s)", book.title, book.isbn13)
        return None

    if int(book.isbn13) in books_df["isbn13"].values:
        return None

    category_col = "simple_categories" if "simple_categories" in books_df.columns else "simple_category"
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

    for col in books_df.columns:
        row.setdefault(col, None)

    return row


def append_book_to_csv(book: BookRecord, books_df: pd.DataFrame) -> pd.DataFrame:
    row = _book_to_row(book, books_df)
    if row is None:
        return books_df

    new_df = pd.concat([books_df, pd.DataFrame([row])], ignore_index=True)
    try:
        new_df.to_csv(BOOKS_CSV_PATH, index=False)
    except OSError as error:
        logger.warning("Could not persist book to CSV: %s", error)
        return books_df

    _append_tagged_description(book.tagged_description)
    return new_df


def append_books_to_csv(books: list[BookRecord], books_df: pd.DataFrame) -> pd.DataFrame:
    if not books:
        return books_df

    existing_isbns = set(books_df["isbn13"].values)
    rows: list[dict] = []
    tagged_descriptions: list[str] = []
    category_col = "simple_categories" if "simple_categories" in books_df.columns else "simple_category"

    for book in books:
        if not has_valid_authors(book.authors):
            logger.warning("Skipping book without author: %s (%s)", book.title, book.isbn13)
            continue

        isbn = int(book.isbn13)
        if isbn in existing_isbns:
            continue

        row = {
            "isbn13": isbn,
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
        for col in books_df.columns:
            row.setdefault(col, None)

        existing_isbns.add(isbn)
        rows.append(row)
        tagged_descriptions.append(book.tagged_description)

    if not rows:
        return books_df

    new_df = pd.concat([books_df, pd.DataFrame(rows)], ignore_index=True)
    try:
        new_df.to_csv(BOOKS_CSV_PATH, index=False)
    except OSError as error:
        logger.warning("Could not persist books to CSV: %s", error)
        return books_df

    _append_tagged_descriptions(tagged_descriptions)
    return new_df


def _append_tagged_description(tagged_description: str) -> None:
    _append_tagged_descriptions([tagged_description])


def _append_tagged_descriptions(tagged_descriptions: list[str]) -> None:
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

    df = pd.read_csv(path)
    isbn_set = {int(isbn) for isbn in isbn13s}
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
    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed = 0

    for line in lines:
        if not line.strip():
            continue
        line_isbn = line.strip('"').split()[0]
        if line_isbn in isbn_set:
            removed += 1
            continue
        kept.append(line)

    if removed:
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed
