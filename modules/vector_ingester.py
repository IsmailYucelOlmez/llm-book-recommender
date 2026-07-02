import logging

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from modules.embeddings_utils import embed_with_retry, get_existing_ids
from modules.types import BookRecord

logger = logging.getLogger(__name__)


class VectorIngester:
    def __init__(self, db: Chroma, embeddings: GoogleGenerativeAIEmbeddings):
        self.db = db
        self.embeddings = embeddings

    def get_existing_ids(self) -> set[str]:
        return get_existing_ids(self.db)

    def ingest_books(self, books: list[BookRecord]) -> list[BookRecord]:
        existing = self.get_existing_ids()
        pending = [book for book in books if book.isbn13 not in existing]
        if not pending:
            return []

        texts = [book.tagged_description for book in pending]
        try:
            vectors = embed_with_retry(self.embeddings, texts)
        except Exception as error:
            logger.warning("Embedding failed for dynamic books: %s", error)
            return []

        try:
            self.db._collection.add(
                ids=[book.isbn13 for book in pending],
                embeddings=vectors,
                documents=texts,
                metadatas=[{"source": "google_books"} for _ in pending],
            )
        except Exception as error:
            logger.warning("Chroma add failed for dynamic books: %s", error)
            return []

        return pending

    def remove_books(self, isbn13s: list[str]) -> int:
        if not isbn13s:
            return 0

        existing = self.get_existing_ids()
        to_remove = [isbn for isbn in isbn13s if isbn in existing]
        if not to_remove:
            return 0

        try:
            self.db._collection.delete(ids=to_remove)
        except Exception as error:
            logger.warning("Chroma delete failed: %s", error)
            return 0
        return len(to_remove)
