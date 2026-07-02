from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr

from modules.hybrid_recommender import HybridRecommender

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001"
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

books = pd.read_csv("books_with_emotions.csv")
if "simple_categories" not in books.columns and "simple_category" in books.columns:
    books = books.rename(columns={"simple_category": "simple_categories"})
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

lines = Path("tagged_description.txt").read_text(encoding="utf-8").splitlines()
documents = [
    Document(page_content=line, metadata={"source": "tagged_description.txt"})
    for line in lines
    if line.strip()
]

if Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir()):
    db_books = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    db_books = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=CHROMA_DIR,
    )

recommender = HybridRecommender(
    db_books=db_books,
    embeddings=embeddings,
    books_df=books,
)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        search_mode: str = "Simple",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    return recommender.recommend(
        query=query,
        category=category or "All",
        tone=tone or "All",
        initial_top_k=initial_top_k,
        final_top_k=final_top_k,
        fetch_external=search_mode == "Advanced",
    )


def recommend_books(
        query: str,
        category: str,
        tone: str,
        search_mode: str,
):
    recommendations = retrieve_semantic_recommendations(
        query, category, tone, search_mode
    )
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        if row.get("source") == "google_books":
            caption = f"[Google Books] {caption}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness / Kayıp bir çocuk ve aile sırları",
        )
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")

    with gr.Row():
        search_mode = gr.Radio(
            choices=["Simple", "Advanced"],
            value="Simple",
            label="Search mode",
            info="Simple: local database only. Advanced: LLM rewrite + Google Books fetch (supports Turkish input).",
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown, search_mode],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()