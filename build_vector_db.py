"""
Build the Chroma vector database from tagged_description.txt using Gemini embeddings.

Gemini free tier allows ~100 embedding requests per minute. This script batches
documents, waits between batches, retries on 429 errors, and resumes if interrupted.
"""
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from modules.config import CHROMA_DIR, EMBEDDING_MODEL
from modules.embeddings_utils import embed_with_retry, get_existing_ids

load_dotenv()

BATCH_SIZE = 90
PAUSE_SECONDS = 65


def load_tagged_documents(path="tagged_description.txt"):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Document(page_content=line, metadata={"source": path}) for line in lines if line.strip()]


def document_id(doc: Document) -> str:
    return doc.page_content.strip('"').split()[0]


def main():
    if not Path("tagged_description.txt").exists():
        raise FileNotFoundError(
            "tagged_description.txt not found. Run data-exploration.ipynb and vector-search.ipynb first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    documents = load_tagged_documents()
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    existing_ids = get_existing_ids(db)
    pending = [doc for doc in documents if document_id(doc) not in existing_ids]
    total = len(documents)

    if not pending:
        print(f"Vector database already complete ({total}/{total} documents).")
        return

    print(
        f"Building vector database in {CHROMA_DIR}: "
        f"{len(existing_ids)} done, {len(pending)} remaining ({total} total)."
    )
    print(f"Free tier: ~{BATCH_SIZE} docs per batch, {PAUSE_SECONDS}s pause (~{len(pending) // BATCH_SIZE + 1} min estimated).")

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        ids = [document_id(doc) for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        vectors = embed_with_retry(embeddings, texts)
        db._collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

        done = len(existing_ids) + start + len(batch)
        print(f"Progress: {done}/{total}")

        if start + BATCH_SIZE < len(pending):
            time.sleep(PAUSE_SECONDS)

    print("Done.")


if __name__ == "__main__":
    main()
