import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pipeline.config import EMBEDDING_MODEL_NAME

class PageEmbedder:
    """Generates normalized dense semantic embeddings for document pages and text chunks."""
    _instance: Optional["PageEmbedder"] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    @classmethod
    def get_instance(cls) -> "PageEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of text strings into normalized 384-dimensional embeddings.
        Empty texts receive a zero vector.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        cleaned_texts = [t.strip() if t.strip() else "[EMPTY_PAGE]" for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(embeddings, dtype=np.float32)

    def compute_adjacent_similarities(self, embeddings: np.ndarray) -> List[float]:
        """
        Computes cosine similarity between adjacent page embeddings: S_i = E_i · E_{i+1}.
        Returns list of similarities for transitions (length = len(embeddings) - 1).
        """
        n = len(embeddings)
        if n < 2:
            return []

        similarities = []
        for i in range(n - 1):
            e_curr = embeddings[i]
            e_next = embeddings[i + 1]

            # For normalized embeddings, cosine similarity is simply the dot product
            sim = float(np.dot(e_curr, e_next))
            # Clamp between -1.0 and 1.0 to guard against numerical float errors
            sim = max(-1.0, min(1.0, sim))
            similarities.append(round(sim, 4))

        return similarities
