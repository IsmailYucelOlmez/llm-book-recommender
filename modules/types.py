from dataclasses import dataclass, asdict
from typing import Any, Optional


@dataclass
class BookRecord:
    isbn13: str
    isbn10: Optional[str]
    title: str
    authors: str
    description: str
    thumbnail: Optional[str]
    categories: str
    tagged_description: str
    source: str
    simple_categories: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BookRecord":
        return cls(**data)
