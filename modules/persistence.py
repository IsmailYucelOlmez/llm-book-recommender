import logging
from pathlib import Path

import pandas as pd

from modules.book_normalizer import has_valid_authors
from modules.config import BOOKS_CSV_PATH, TAGGED_DESCRIPTION_PATH
from modules.types import BookRecord

logger = logging.getLogger(__name__)

EMOTION_COLUMNS = ["anger", "disgust", "neutral", "fear", "surprise", "joy", "sadness"]


def append_book_to_csv(book: BookRecord, books_df: pd.DataFrame) -> pd.DataFrame:
    if not has_valid_authors(book.authors):
        logger.warning("Skipping book without author: %s (%s)", book.title, book.isbn13)
        return books_df

    if int(book.isbn13) in books_df["isbn13"].values:
        return books_df

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

    new_df = pd.concat([books_df, pd.DataFrame([row])], ignore_index=True)
    try:
        new_df.to_csv(BOOKS_CSV_PATH, index=False)
    except OSError as error:
        logger.warning("Could not persist book to CSV: %s", error)
        return books_df

    _append_tagged_description(book.tagged_description)
    return new_df


def _append_tagged_description(tagged_description: str) -> None:
    path = Path(TAGGED_DESCRIPTION_PATH)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if tagged_description in existing:
        return
    with path.open("a", encoding="utf-8") as handle:
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
