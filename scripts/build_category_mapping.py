"""Build comprehensive Google Books category -> simple_categories mapping."""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "modules" / "category_mapping.json"

FICTION_PATTERNS = [
    r"fictitious character",
    r"imaginary place",
    r"\bfiction\b",
    r"stories",
    r"novel",
    r"mystery",
    r"horror",
    r"fantasy",
    r"romance",
    r"thriller",
    r"suspense",
    r"comic",
    r"graphic novel",
    r"adventure fiction",
    r"ghost stories",
    r"science fiction",
    r"detective",
    r"political fiction",
    r"domestic fiction",
    r"experimental fiction",
    r"humorous fiction",
    r"diary fiction",
    r"occult fiction",
    r"christian fiction",
    r"arabic fiction",
    r"chick lit",
    r"dystopias",
    r"espionage",
    r"conspiracies",
    r"murder",
    r"young adult fiction",
    r"allegories",
    r"epic literature",
    r"sea stories",
    r"short stories",
    r"plays",
    r"poetry",
    r"drama",
]

CHILDREN_FICTION_PATTERNS = [
    r"juvenile fiction",
    r"children's stories",
    r"children's plays",
    r"children's poetry",
    r"children's fiction",
]

CHILDREN_NONFICTION_PATTERNS = [
    r"juvenile nonfiction",
    r"children's nonfiction",
]

NONFICTION_PATTERNS = [
    r"nonfiction",
    r"biography",
    r"autobiography",
    r"history",
    r"\bscience\b",
    r"religion",
    r"philosophy",
    r"criticism",
    r"reference",
    r"self-help",
    r"business",
    r"health",
    r"cookery",
    r"cooking",
    r"political science",
    r"social science",
    r"psychology",
    r"education",
    r"study aids",
    r"true crime",
    r"\bart\b",
    r"\bmusic\b",
    r"\blaw\b",
    r"medical",
    r"mathematics",
    r"computers",
    r"engineering",
    r"gardening",
    r"travel",
    r"performing arts",
    r"photography",
    r"architecture",
    r"crafts",
    r"antiques",
    r"language arts",
    r"transportation",
    r"family & relationships",
    r"body, mind",
    r"essays",
    r"theology",
    r"spiritual life",
    r"christian life",
    r"bible",
    r"bibles",
]

BISAC_MAPPING = {
    "Fiction": "Fiction",
    "FICTION": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "JUVENILE FICTION": "Children's Fiction",
    "Young Adult Fiction": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Biography & Autobiography": "Nonfiction",
    "BIOGRAPHY & AUTOBIOGRAPHY": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "LITERARY CRITICISM": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
    "Self-Help": "Nonfiction",
    "Business & Economics": "Nonfiction",
    "Health & Fitness": "Nonfiction",
    "Travel": "Nonfiction",
    "Cooking": "Nonfiction",
    "Cookery": "Nonfiction",
    "Sports & Recreation": "Nonfiction",
    "Political Science": "Nonfiction",
    "Social Science": "Nonfiction",
    "Psychology": "Nonfiction",
    "Education": "Nonfiction",
    "Reference": "Nonfiction",
    "Study Aids": "Nonfiction",
    "True Crime": "Nonfiction",
    "Art": "Nonfiction",
    "Music": "Nonfiction",
    "Nature": "Nonfiction",
    "Gardening": "Nonfiction",
    "Crafts & Hobbies": "Nonfiction",
    "House & Home": "Nonfiction",
    "Antiques & Collectibles": "Nonfiction",
    "Foreign Language Study": "Nonfiction",
    "Mathematics": "Nonfiction",
    "Computers": "Nonfiction",
    "Technology & Engineering": "Nonfiction",
    "Medical": "Nonfiction",
    "Law": "Nonfiction",
    "Body, Mind & Spirit": "Nonfiction",
    "Games & Activities": "Nonfiction",
    "Language Arts & Disciplines": "Nonfiction",
    "Transportation": "Nonfiction",
    "Family & Relationships": "Nonfiction",
    "Performing Arts": "Nonfiction",
    "Photography": "Nonfiction",
    "Design": "Nonfiction",
    "Literary Collections": "Fiction",
    "Literary Criticism & Collections": "Nonfiction",
    "Christian life": "Nonfiction",
    "Thriller": "Fiction",
    "Thrillers": "Fiction",
    "Suspense": "Fiction",
    "Horror": "Fiction",
    "Mystery": "Fiction",
    "Romance": "Fiction",
    "Fantasy": "Fiction",
    "Science Fiction": "Fiction",
    "Humor": "Nonfiction",
    "Games": "Nonfiction",
}


def classify_by_patterns(text: str) -> str | None:
    lowered = text.lower()
    for pattern in CHILDREN_NONFICTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Children's Nonfiction"
    for pattern in CHILDREN_FICTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Children's Fiction"
    for pattern in FICTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Fiction"
    for pattern in NONFICTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Nonfiction"
    return None


def main() -> None:
    df = pd.read_csv(ROOT / "books_with_emotions.csv")
    csv_mapping = (
        df.groupby("categories")["simple_categories"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    mapping: dict[str, str] = {}
    mapping.update(BISAC_MAPPING)

    for category, simple in csv_mapping.items():
        classified = classify_by_patterns(category) or simple
        mapping[category] = classified

    for category, simple in BISAC_MAPPING.items():
        mapping[category] = simple

    OUTPUT.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(mapping)} mappings to {OUTPUT}")
    print(pd.Series(mapping.values()).value_counts().to_dict())


if __name__ == "__main__":
    main()
