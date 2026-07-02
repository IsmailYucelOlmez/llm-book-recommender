"""Manual test script for the Google Books dynamic module."""

import logging

from dotenv import load_dotenv

from modules.book_cache import BookCache
from modules.book_normalizer import normalize_volumes
from modules.google_books_client import GoogleBooksClient

logging.basicConfig(level=logging.INFO)
load_dotenv()


def main() -> None:
    query = "A story about forgiveness"
    client = GoogleBooksClient()
    cache = BookCache()

    print(f"Searching Google Books for: {query!r}")
    volumes = client.search(query)
    print(f"Raw results: {len(volumes)}")

    books = normalize_volumes(volumes, existing_isbns=set(), max_books=5)
    print(f"Normalized books: {len(books)}")

    for book in books:
        print(f"  - {book.title} ({book.isbn13}) [{book.simple_categories}]")
        cache.set_book(book)

    cache.set_query_isbns(query, [book.isbn13 for book in books])
    cached = cache.get_query_isbns(query)
    print(f"Cached ISBNs: {cached}")


if __name__ == "__main__":
    main()
