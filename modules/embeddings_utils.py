import logging
import re
import time

logger = logging.getLogger(__name__)

PAUSE_SECONDS = 65
# Bound the retry loop so a stuck request can never block a web worker forever.
MAX_RETRIES = 5
MAX_TOTAL_WAIT_SECONDS = 300


def parse_retry_seconds(error: Exception) -> int:
    match = re.search(r"retry in ([0-9.]+)s", str(error), re.IGNORECASE)
    if match:
        return max(int(float(match.group(1))) + 1, PAUSE_SECONDS)
    return PAUSE_SECONDS


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def embed_with_retry(
    embeddings,
    texts: list[str],
    max_retries: int = MAX_RETRIES,
    max_total_wait: int = MAX_TOTAL_WAIT_SECONDS,
) -> list[list[float]]:
    """Embed ``texts``, retrying on rate limits with a hard cap on retries/wait.

    Unlike an unbounded ``while True`` loop, this gives up after ``max_retries``
    attempts or once cumulative sleep exceeds ``max_total_wait`` seconds, so a
    Gradio request thread can never be blocked indefinitely. The final rate-limit
    error is re-raised for the caller to handle (e.g. surface a user message).
    """
    attempts = 0
    waited = 0
    while True:
        try:
            return embeddings.embed_documents(texts, batch_size=min(len(texts), 100))
        except Exception as error:
            if not _is_rate_limit_error(error):
                raise
            attempts += 1
            wait = parse_retry_seconds(error)
            if attempts > max_retries or waited + wait > max_total_wait:
                logger.warning(
                    "Giving up embedding after %d rate-limited attempt(s) (~%ds waited).",
                    attempts,
                    waited,
                )
                raise
            logger.warning(
                "Rate limited — waiting %ds before retry (%d/%d)...",
                wait,
                attempts,
                max_retries,
            )
            time.sleep(wait)
            waited += wait


def get_collection(db):
    """Return the underlying Chroma collection behind a LangChain wrapper.

    LangChain's ``Chroma`` does not expose a public API for adding precomputed
    embeddings with explicit ids, so we reach into ``_collection``. Centralizing
    that access here means a library change only needs a fix in one place, and we
    fail loudly with an actionable message instead of an opaque AttributeError.
    """
    collection = getattr(db, "_collection", None)
    if collection is None:
        raise AttributeError(
            "Chroma wrapper exposes no '_collection'; the langchain-chroma private "
            "API may have changed. Update modules/embeddings_utils.get_collection()."
        )
    return collection


def get_existing_ids(db) -> set[str]:
    collection = get_collection(db)
    if collection.count() == 0:
        return set()
    return set(collection.get(include=[])["ids"])
