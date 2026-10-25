import re
import numpy as np
from typing import List, Dict, Any, Tuple
from pipeline.page_embedder import PageEmbedder
from pipeline.document_assembler import LogicalDocument

class DocumentEmbedder:
    """
    Creates normalized 384-dimensional document-level semantic embeddings
    using chunking and weighted position aggregation (giving priority to the title/first page).
    """

    def __init__(self):
        self.embedder = PageEmbedder.get_instance()

    def clean_text(self, text: str) -> str:
        """Cleans extracted text, removing line noise and normalizing whitespace."""
        # Remove repeated page break markers
        text = re.sub(r'--- PAGE BREAK ---', ' ', text)
        text = re.sub(r'\[Page \d+\]', ' ', text)
        # Collapse excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chunk_text(self, text: str, max_words: int = 180, overlap: int = 30) -> List[str]:
        """Chunks long document text into overlapping token windows for transformer encoding."""
        words = text.split()
        if not words:
            return ["[EMPTY_DOCUMENT]"]
        if len(words) <= max_words:
            return [text]

        chunks = []
        step = max(1, max_words - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + max_words])
            chunks.append(chunk)
            if i + max_words >= len(words):
                break
        return chunks

    def embed_document(self, doc: LogicalDocument) -> np.ndarray:
        """
        Embeds a single LogicalDocument.
        Weights the first page/chunk higher (40%) and remaining chunks (60%).
        """
        cleaned = self.clean_text(doc.combined_text)
        chunks = self.chunk_text(cleaned)

        # Encode chunks
        chunk_embeddings = self.embedder.embed_texts(chunks)  # (K, 384)

        if len(chunk_embeddings) == 1:
            return chunk_embeddings[0]

        # Weighted aggregation: 40% for the lead chunk (title/summary), 60% split across the rest
        w_first = 0.40
        w_rest = 0.60 / float(len(chunk_embeddings) - 1)
        weights = np.array([w_first] + [w_rest] * (len(chunk_embeddings) - 1), dtype=np.float32)
        
        aggregated = np.sum(chunk_embeddings * weights[:, np.newaxis], axis=0)

        # Re-normalize to unit length
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm

        return aggregated.astype(np.float32)

    def embed_documents(self, documents: List[LogicalDocument]) -> Tuple[np.ndarray, List[str]]:
        """
        Embeds a collection of N documents.
        Returns:
            - embedding_matrix: np.ndarray of shape (N, 384)
            - doc_ids: list of document IDs corresponding to each row
        """
        if not documents:
            return np.empty((0, 384), dtype=np.float32), []

        embeddings = []
        doc_ids = []

        for doc in documents:
            emb = self.embed_document(doc)
            embeddings.append(emb)
            doc_ids.append(doc.doc_id)

        matrix = np.vstack(embeddings).astype(np.float32)
        return matrix, doc_ids
