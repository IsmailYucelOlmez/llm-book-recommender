"""Google Books dynamic module for hybrid semantic book recommendations."""

__all__ = ["HybridRecommender"]


def __getattr__(name: str):
    if name == "HybridRecommender":
        from modules.hybrid_recommender import HybridRecommender

        return HybridRecommender
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
