import hashlib
import json
import sqlite3
import time

from modules.config import CACHE_DB_PATH, QUERY_CACHE_TTL_SECONDS, ensure_data_dir
from modules.types import BookRecord


class BookCache:
    def __init__(self, db_path: str = CACHE_DB_PATH):
        ensure_data_dir()
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    isbns_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    rewrite_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS book_cache (
                    isbn13 TEXT PRIMARY KEY,
                    book_json TEXT NOT NULL
                )
                """
            )
            self._ensure_rewrite_column(conn)

    @staticmethod
    def _ensure_rewrite_column(conn: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(query_cache)").fetchall()
        }
        if "rewrite_json" not in columns:
            conn.execute("ALTER TABLE query_cache ADD COLUMN rewrite_json TEXT")

    @staticmethod
    def _query_hash(query: str, category: str = "All") -> str:
        normalized = f"{query.strip().lower()}|{category.strip().lower()}"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get_query_cache(
        self,
        query: str,
        category: str = "All",
    ) -> tuple[list[str], dict | None] | None:
        query_hash = self._query_hash(query, category)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT isbns_json, created_at, rewrite_json
                FROM query_cache
                WHERE query_hash = ?
                """,
                (query_hash,),
            ).fetchone()
        if not row:
            return None

        isbns_json, created_at, rewrite_json = row
        if time.time() - created_at > QUERY_CACHE_TTL_SECONDS:
            return None

        rewrite = json.loads(rewrite_json) if rewrite_json else None
        return json.loads(isbns_json), rewrite

    def get_query_isbns(self, query: str, category: str = "All") -> list[str] | None:
        cached = self.get_query_cache(query, category)
        if cached is None:
            return None
        isbns, _rewrite = cached
        return isbns

    def set_query_isbns(
        self,
        query: str,
        isbns: list[str],
        category: str = "All",
        rewrite: dict | None = None,
    ) -> None:
        query_hash = self._query_hash(query, category)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_cache (query_hash, isbns_json, created_at, rewrite_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    query_hash,
                    json.dumps(isbns),
                    time.time(),
                    json.dumps(rewrite) if rewrite else None,
                ),
            )

    def get_book(self, isbn13: str) -> BookRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT book_json FROM book_cache WHERE isbn13 = ?",
                (isbn13,),
            ).fetchone()
        if not row:
            return None
        return BookRecord.from_dict(json.loads(row[0]))

    def set_book(self, book: BookRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO book_cache (isbn13, book_json)
                VALUES (?, ?)
                """,
                (book.isbn13, json.dumps(book.to_dict())),
            )

    def is_known_isbn(self, isbn13: str) -> bool:
        return self.get_book(isbn13) is not None

    def remove_books(self, isbn13s: list[str]) -> int:
        if not isbn13s:
            return 0

        isbn_set = set(isbn13s)
        with self._connect() as conn:
            removed = 0
            for isbn in isbn_set:
                cursor = conn.execute("DELETE FROM book_cache WHERE isbn13 = ?", (isbn,))
                removed += cursor.rowcount

            rows = conn.execute("SELECT query_hash, isbns_json FROM query_cache").fetchall()
            for query_hash, isbns_json in rows:
                isbns = json.loads(isbns_json)
                updated = [isbn for isbn in isbns if isbn not in isbn_set]
                if updated != isbns:
                    conn.execute(
                        "UPDATE query_cache SET isbns_json = ? WHERE query_hash = ?",
                        (json.dumps(updated), query_hash),
                    )
        return removed
