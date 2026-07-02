from pathlib import Path

MAX_NEW_BOOKS = 5
GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1/volumes"
GOOGLE_BOOKS_MAX_RESULTS = 20
GOOGLE_BOOKS_TIMEOUT_SECONDS = 10
MAX_QUERY_LENGTH = 500
MAX_GOOGLE_BOOKS_QUERIES = 2
MAX_REWRITE_QUERY_CHARS = 80

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001"
REWRITE_MODEL = "gemini-2.5-flash"
REWRITE_TIMEOUT_SECONDS = 15
REWRITE_MAX_RETRIES = 1
CACHE_DB_PATH = "data/dynamic_books_cache.db"
QUERY_CACHE_TTL_SECONDS = 3600

BOOKS_CSV_PATH = "books_with_emotions.csv"
TAGGED_DESCRIPTION_PATH = "tagged_description.txt"

MIN_DESCRIPTION_LENGTH = 50


def ensure_data_dir() -> Path:
    data_dir = Path(CACHE_DB_PATH).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
