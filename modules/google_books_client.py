import logging
import os
from typing import Any

import requests

from modules.config import (
    GOOGLE_BOOKS_BASE_URL,
    GOOGLE_BOOKS_MAX_RESULTS,
    GOOGLE_BOOKS_TIMEOUT_SECONDS,
    MAX_QUERY_LENGTH,
)

logger = logging.getLogger(__name__)


class GoogleBooksClient:
    def __init__(self, api_key: str | None = None, timeout: int = GOOGLE_BOOKS_TIMEOUT_SECONDS):
        self.api_key = api_key if api_key is not None else os.getenv("GOOGLE_BOOKS_API_KEY")
        self.timeout = timeout

    def search(self, query: str, max_results: int = GOOGLE_BOOKS_MAX_RESULTS) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]

        params: dict[str, Any] = {
            "q": query,
            "maxResults": max_results,
            "printType": "books",
            "langRestrict": "en",
        }
        if self.api_key:
            params["key"] = self.api_key

        try:
            response = requests.get(
                GOOGLE_BOOKS_BASE_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout:
            logger.warning("Google Books API timed out for query: %s", query[:80])
            return []
        except requests.RequestException as error:
            logger.warning("Google Books API error: %s", error)
            return []

        payload = response.json()
        items = payload.get("items") or []
        return [item.get("volumeInfo", {}) for item in items if item.get("volumeInfo")]

    def search_many(self, queries: list[str], max_results: int = GOOGLE_BOOKS_MAX_RESULTS) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for query in queries:
            for volume in self.search(query, max_results=max_results):
                dedupe_key = self._volume_dedupe_key(volume)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                merged.append(volume)

        return merged

    @staticmethod
    def _volume_dedupe_key(volume: dict[str, Any]) -> str:
        title = (volume.get("title") or "").strip().lower()
        authors = volume.get("authors") or []
        first_author = authors[0].strip().lower() if authors else ""
        return f"{title}|{first_author}"
