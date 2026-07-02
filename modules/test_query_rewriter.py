import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from modules.book_cache import BookCache
from modules.google_books_client import GoogleBooksClient
from modules.persistence import append_books_to_csv
from modules.query_rewriter import QueryRewriter, RewriteResult, _extract_json_payload
from modules.types import BookRecord


class RewriteResultTests(unittest.TestCase):
    def test_effective_api_queries_prefers_google_books_queries(self) -> None:
        result = RewriteResult(
            detected_language="tr",
            english_summary="A thriller about family secrets",
            keywords=["family", "secrets"],
            google_books_queries=["family secrets thriller", "subject:family secrets"],
            genre_hint="fiction",
        )
        self.assertEqual(
            result.effective_api_queries("fallback"),
            ["family secrets thriller", "subject:family secrets"],
        )

    def test_effective_api_queries_falls_back_to_keywords(self) -> None:
        result = RewriteResult(
            detected_language="en",
            english_summary="Forgiveness story",
            keywords=["forgiveness", "redemption", "family"],
            google_books_queries=[],
            genre_hint="fiction",
        )
        self.assertEqual(result.effective_api_queries("fallback"), ["forgiveness redemption family"])

    def test_effective_local_query_uses_summary(self) -> None:
        result = RewriteResult(
            detected_language="tr",
            english_summary="A missing child and family secrets",
            keywords=["missing child"],
            google_books_queries=["missing child mystery fiction"],
            genre_hint="fiction",
        )
        self.assertEqual(
            result.effective_local_query("Kayıp bir çocuk"),
            "A missing child and family secrets",
        )

    def test_extract_json_payload_strips_markdown_fence(self) -> None:
        payload = _extract_json_payload(
            '```json\n{"detected_language":"tr","english_summary":"test"}\n```'
        )
        self.assertEqual(payload["detected_language"], "tr")


class BookCacheRewriteTests(unittest.TestCase):
    def test_rewrite_roundtrip_in_query_cache(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            cache = BookCache(db_path=db_path)
            rewrite = {
                "detected_language": "tr",
                "english_summary": "A thriller about family secrets",
                "keywords": ["family secrets"],
                "google_books_queries": ["family secrets thriller fiction"],
                "genre_hint": "fiction",
            }

            cache.set_query_isbns("Kayıp çocuk", ["9780000000000"], "Fiction", rewrite=rewrite)
            cached = cache.get_query_cache("Kayıp çocuk", "Fiction")

            self.assertIsNotNone(cached)
            isbns, rewrite_payload = cached
            self.assertEqual(isbns, ["9780000000000"])
            self.assertEqual(rewrite_payload, rewrite)

            restored = QueryRewriter.from_cache_dict(rewrite_payload)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.english_summary, "A thriller about family secrets")
        finally:
            if os.path.exists(db_path):
                try:
                    os.unlink(db_path)
                except PermissionError:
                    pass


class GoogleBooksClientTests(unittest.TestCase):
    def test_search_many_deduplicates_volumes(self) -> None:
        client = GoogleBooksClient(api_key="test")
        shared_volume = {
            "title": "Example Book",
            "authors": ["Jane Doe"],
            "description": "A long enough description for testing duplicate handling in search_many.",
        }
        other_volume = {
            "title": "Another Book",
            "authors": ["John Smith"],
            "description": "Another long enough description for testing duplicate handling in search_many.",
        }

        with patch.object(client, "search", side_effect=[[shared_volume, other_volume], [shared_volume]]):
            volumes = client.search_many(["query one", "query two"])

        self.assertEqual(len(volumes), 2)
        titles = {volume["title"] for volume in volumes}
        self.assertEqual(titles, {"Example Book", "Another Book"})


class PersistenceBatchTests(unittest.TestCase):
    def test_append_books_to_csv_writes_once(self) -> None:
        books_df = pd.DataFrame(
            {
                "isbn13": [111],
                "isbn10": ["0111"],
                "title": ["Existing"],
                "authors": ["Author One"],
                "categories": ["Fiction"],
                "thumbnail": [""],
                "description": ["Existing description"],
                "tagged_description": ['111 "Existing description"'],
                "simple_categories": ["Fiction"],
                "source": ["local"],
            }
        )
        books = [
            BookRecord(
                isbn13="222",
                isbn10="0222",
                title="New Book",
                authors="Author Two",
                description="A long enough description for batch CSV persistence testing.",
                thumbnail=None,
                categories="Fiction",
                tagged_description='222 "A long enough description for batch CSV persistence testing."',
                source="google_books",
                simple_categories="Fiction",
            ),
            BookRecord(
                isbn13="333",
                isbn10="0333",
                title="Another Book",
                authors="Author Three",
                description="Another long enough description for batch CSV persistence testing.",
                thumbnail=None,
                categories="Fiction",
                tagged_description='333 "Another long enough description for batch CSV persistence testing."',
                source="google_books",
                simple_categories="Fiction",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "books.csv")
            tagged_path = os.path.join(tmp_dir, "tagged.txt")
            with patch("modules.persistence.BOOKS_CSV_PATH", csv_path), patch(
                "modules.persistence.TAGGED_DESCRIPTION_PATH", tagged_path
            ):
                updated_df = append_books_to_csv(books, books_df)

            self.assertEqual(len(updated_df), 3)
            saved = pd.read_csv(csv_path)
            self.assertEqual(len(saved), 3)
            self.assertIn("222", saved["isbn13"].astype(str).values)
            self.assertIn("333", saved["isbn13"].astype(str).values)


if __name__ == "__main__":
    unittest.main()
