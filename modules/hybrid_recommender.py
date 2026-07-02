import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from modules.book_cache import BookCache
from modules.book_normalizer import normalize_volumes, parse_tagged_isbn
from modules.config import MAX_NEW_BOOKS
from modules.google_books_client import GoogleBooksClient
from modules.persistence import append_books_to_csv
from modules.query_rewriter import QueryRewriter, RewriteResult
from modules.types import BookRecord
from modules.vector_ingester import VectorIngester

logger = logging.getLogger(__name__)

TONE_COLUMNS = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


class HybridRecommender:
    def __init__(
        self,
        db_books: Chroma,
        embeddings: GoogleGenerativeAIEmbeddings,
        books_df: pd.DataFrame,
        google_client: GoogleBooksClient | None = None,
        cache: BookCache | None = None,
        query_rewriter: QueryRewriter | None = None,
    ):
        self.db = db_books
        self.embeddings = embeddings
        self.books_df = books_df
        self.google_client = google_client or GoogleBooksClient()
        self.cache = cache or BookCache()
        self.query_rewriter = query_rewriter if query_rewriter is not None else QueryRewriter()
        self.ingester = VectorIngester(db_books, embeddings)

    def recommend(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        max_new_books: int = MAX_NEW_BOOKS,
        final_top_k: int = 16,
        fetch_external: bool = True,
    ) -> pd.DataFrame:
        search_query = query.strip()
        query_vector: list[float] | None = None

        if fetch_external and search_query:
            rewrite_result, query_vector = self._fetch_and_ingest(
                query,
                max_new_books,
                category,
                tone,
            )
            if rewrite_result:
                search_query = rewrite_result.effective_local_query(query)

        return self._local_search(
            search_query,
            category,
            tone,
            initial_top_k,
            final_top_k,
            query_vector=query_vector,
        )

    def _local_search(
        self,
        query: str,
        category: str,
        tone: str,
        initial_top_k: int,
        final_top_k: int,
        query_vector: list[float] | None = None,
    ) -> pd.DataFrame:
        if query_vector is not None:
            recs = self.db.similarity_search_by_vector(query_vector, k=initial_top_k)
        else:
            recs = self.db.similarity_search(query, k=initial_top_k)

        books_list: list[int] = []
        for rec in recs:
            isbn = parse_tagged_isbn(rec.page_content)
            if isbn and isbn.isdigit():
                books_list.append(int(isbn))
        book_recs = self.books_df[self.books_df["isbn13"].isin(books_list)].copy()

        if category != "All":
            category_col = (
                "simple_categories"
                if "simple_categories" in book_recs.columns
                else "simple_category"
            )
            book_recs = book_recs[book_recs[category_col] == category]

        book_recs = self._apply_tone_sort(book_recs, tone)
        book_recs = book_recs.head(final_top_k)

        if "source" not in book_recs.columns:
            book_recs["source"] = "local"
        else:
            book_recs["source"] = book_recs["source"].fillna("local")

        book_recs["large_thumbnail"] = np.where(
            book_recs["thumbnail"].notna() & (book_recs["thumbnail"] != ""),
            book_recs["thumbnail"] + "&fife=w800",
            "cover-not-found.jpg",
        )
        return book_recs

    def _fetch_and_ingest(
        self,
        query: str,
        max_new_books: int,
        category: str = "All",
        tone: str = "All",
    ) -> tuple[RewriteResult | None, list[float] | None]:
        if not query.strip():
            return None, None

        cached = self.cache.get_query_cache(query, category)
        if cached is not None:
            cached_isbns, cached_rewrite = cached
            self._books_from_cache(cached_isbns, category)
            rewrite_result = QueryRewriter.from_cache_dict(cached_rewrite)
            return rewrite_result, None

        with ThreadPoolExecutor(max_workers=2) as executor:
            rewrite_future = executor.submit(
                self.query_rewriter.rewrite,
                query,
                category,
                tone,
            )
            existing_isbns_future = executor.submit(self._existing_isbn_set)
            rewrite_result = rewrite_future.result()
            existing_isbns = existing_isbns_future.result()

        api_queries = (
            rewrite_result.effective_api_queries(query)
            if rewrite_result
            else [query.strip()]
        )
        local_query = (
            rewrite_result.effective_local_query(query)
            if rewrite_result
            else query.strip()
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            volumes_future = executor.submit(self.google_client.search_many, api_queries)
            query_vector_future = executor.submit(self._embed_query, local_query)
            volumes = volumes_future.result()
            query_vector = query_vector_future.result()

        books = normalize_volumes(
            volumes,
            existing_isbns,
            max_books=max_new_books,
            category=category,
        )

        rewrite_cache = (
            QueryRewriter.to_cache_dict(rewrite_result) if rewrite_result else None
        )

        if not books:
            self.cache.set_query_isbns(query, [], category, rewrite=rewrite_cache)
            return rewrite_result, query_vector

        ingested = self.ingester.ingest_books(books)
        for book in ingested:
            self.cache.set_book(book)

        if ingested:
            self.books_df = append_books_to_csv(ingested, self.books_df)

        self.cache.set_query_isbns(
            query,
            [book.isbn13 for book in books],
            category,
            rewrite=rewrite_cache,
        )
        return rewrite_result, query_vector

    def _embed_query(self, query: str) -> list[float] | None:
        query = query.strip()
        if not query:
            return None
        try:
            return self.embeddings.embed_query(query)
        except Exception as error:
            logger.warning("Query embedding failed for %r: %s", query[:80], error)
            return None

    def _books_from_cache(self, isbns: list[str], category: str) -> list[BookRecord]:
        books: list[BookRecord] = []
        for isbn in isbns:
            book = self.cache.get_book(isbn)
            if not book:
                continue
            if category != "All" and book.simple_categories != category:
                continue
            books.append(book)
        return books

    def _existing_isbn_set(self) -> set[str]:
        csv_isbns = {str(isbn) for isbn in self.books_df["isbn13"].astype(str)}
        chroma_isbns = self.ingester.get_existing_ids()
        return csv_isbns | chroma_isbns

    def _apply_tone_sort(self, df: pd.DataFrame, tone: str) -> pd.DataFrame:
        if df.empty or tone == "All":
            return df

        column = TONE_COLUMNS.get(tone)
        if not column or column not in df.columns:
            return df

        result = df.copy()
        result["_has_tone"] = result[column].notna().astype(int)
        result = result.sort_values(by=["_has_tone", column], ascending=[False, False])
        return result.drop(columns=["_has_tone"])
