import re
import time

PAUSE_SECONDS = 65


def parse_retry_seconds(error: Exception) -> int:
    match = re.search(r"retry in ([0-9.]+)s", str(error), re.IGNORECASE)
    if match:
        return max(int(float(match.group(1))) + 1, PAUSE_SECONDS)
    return PAUSE_SECONDS


def embed_with_retry(embeddings, texts: list[str]) -> list[list[float]]:
    while True:
        try:
            return embeddings.embed_documents(texts, batch_size=min(len(texts), 100))
        except Exception as error:
            if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
                wait = parse_retry_seconds(error)
                print(f"Rate limited — waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise


def get_existing_ids(db) -> set[str]:
    if db._collection.count() == 0:
        return set()
    return set(db._collection.get(include=[])["ids"])
