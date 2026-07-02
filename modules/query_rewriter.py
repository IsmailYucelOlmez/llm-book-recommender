import json
import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

from modules.config import (
    MAX_GOOGLE_BOOKS_QUERIES,
    MAX_REWRITE_QUERY_CHARS,
    REWRITE_MAX_OUTPUT_TOKENS,
    REWRITE_MAX_RETRIES,
    REWRITE_MODEL,
    REWRITE_THINKING_BUDGET,
    REWRITE_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

TONE_SEARCH_HINTS = {
    "Happy": "uplifting joyful heartwarming",
    "Surprising": "twist unexpected plot",
    "Angry": "revenge justice intense",
    "Suspenseful": "thriller suspense mystery",
    "Sad": "tragedy grief loss emotional",
}

CATEGORY_SEARCH_HINTS = {
    "Fiction": "fiction novel",
    "Nonfiction": "nonfiction",
}

REWRITE_PROMPT = """You prepare Google Books API search queries from a user's book description.

Tasks (single step):
1. Detect the input language.
2. If not English, interpret the meaning in English.
3. Extract 3-6 English search themes (nouns, genres, settings, moods).
4. Build {max_queries} short Google Books "q" strings (max {max_chars} chars each).
   - Use plain keywords, not full sentences.
   - Add genre words when implied (fiction, mystery, biography, etc.).
   - Use subject: only for clear nonfiction themes.
   - Do NOT invent book titles or authors not mentioned by the user.
   - Ignore filler like "a story about", "kitap öner", "benzeri".

User category hint: {category}
User tone hint: {tone}
Tone search hints (optional): {tone_hints}
Category search hints (optional): {category_hints}

Input: {user_query}

Respond with JSON only, no markdown fences."""


class RewriteResult(BaseModel):
    detected_language: str = Field(description="ISO 639-1 code or language name")
    english_summary: str = Field(description="Short English paraphrase of user intent")
    keywords: list[str] = Field(default_factory=list, description="English keyword phrases")
    google_books_queries: list[str] = Field(
        default_factory=list,
        description="Ready-to-use Google Books q parameter strings",
    )
    genre_hint: Literal["fiction", "nonfiction", "unknown"] = "unknown"

    @field_validator("keywords", "google_books_queries", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def effective_local_query(self, fallback: str) -> str:
        summary = self.english_summary.strip()
        return summary if summary else fallback.strip()

    def effective_api_queries(self, fallback: str) -> list[str]:
        queries = [query.strip() for query in self.google_books_queries if query.strip()]
        if queries:
            return queries[:MAX_GOOGLE_BOOKS_QUERIES]

        keywords = [keyword.strip() for keyword in self.keywords if keyword.strip()]
        if keywords:
            return [" ".join(keywords[:6])]

        fallback = fallback.strip()
        return [fallback] if fallback else []


def _is_quota_error(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _extract_json_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class QueryRewriter:
    def __init__(
        self,
        model: str = REWRITE_MODEL,
        temperature: float = 0.0,
        timeout: int = REWRITE_TIMEOUT_SECONDS,
        max_retries: int = REWRITE_MAX_RETRIES,
    ):
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            max_output_tokens=REWRITE_MAX_OUTPUT_TOKENS,
            thinking_budget=REWRITE_THINKING_BUDGET,
        )
        self._structured = self._llm.with_structured_output(RewriteResult)

    def rewrite(
        self,
        user_query: str,
        category: str = "All",
        tone: str = "All",
    ) -> RewriteResult | None:
        user_query = user_query.strip()
        if not user_query:
            return None

        tone_hints = TONE_SEARCH_HINTS.get(tone, "") if tone != "All" else ""
        category_hints = CATEGORY_SEARCH_HINTS.get(category, "") if category != "All" else ""

        prompt = REWRITE_PROMPT.format(
            max_queries=MAX_GOOGLE_BOOKS_QUERIES,
            max_chars=MAX_REWRITE_QUERY_CHARS,
            category=category,
            tone=tone,
            tone_hints=tone_hints or "none",
            category_hints=category_hints or "none",
            user_query=user_query,
        )

        try:
            result = self._structured.invoke([HumanMessage(content=prompt)])
            if isinstance(result, RewriteResult):
                return result
            if isinstance(result, dict):
                return RewriteResult.model_validate(result)
            return RewriteResult.model_validate(result)
        except Exception as structured_error:
            if _is_quota_error(structured_error):
                logger.warning(
                    "Query rewrite skipped (quota/rate limit) for %r: %s",
                    user_query[:80],
                    structured_error,
                )
                return None
            logger.warning("Structured rewrite failed, trying raw JSON parse: %s", structured_error)

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            payload = _extract_json_payload(str(content))
            return RewriteResult.model_validate(payload)
        except Exception as error:
            logger.warning("Query rewrite failed for %r: %s", user_query[:80], error)
            return None

    @staticmethod
    def to_cache_dict(result: RewriteResult) -> dict:
        return result.model_dump()

    @staticmethod
    def from_cache_dict(data: dict | None) -> RewriteResult | None:
        if not data:
            return None
        try:
            return RewriteResult.model_validate(data)
        except Exception as error:
            logger.warning("Invalid cached rewrite payload: %s", error)
            return None
