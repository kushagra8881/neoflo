import re
import numpy as np
from typing import List, Dict, Any, Tuple
from pipeline.config import BOUNDARY_WEIGHTS, BOUNDARY_DECISION_THRESHOLD
from pipeline.page_embedder import PageEmbedder

class BoundaryDetector:
    """
    Implements multi-signal document boundary detection across PDF pages:
    B_i = w_s * (1 - S_i) + w_h * H_i + w_m * M_i + w_v * V_i
    
    Distinguishes where one logical document ends and another begins (Scenarios B & C).
    """

    def __init__(self, weights: Dict[str, float] = None, threshold: float = BOUNDARY_DECISION_THRESHOLD):
        self.weights = weights or BOUNDARY_WEIGHTS
        self.threshold = threshold
        self.embedder = PageEmbedder.get_instance()

    def detect_boundaries(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes page objects with text, lines, and signals.
        Returns:
            - page_embeddings: np.ndarray of shape (N, 384)
            - transitions: list of transition diagnostics
            - boundary_indices: list of page indices where a NEW document begins
            - document_ranges: list of (start_page_idx, end_page_idx) tuples
        """
        n_pages = len(pages)
        if n_pages == 0:
            return {
                "page_embeddings": np.empty((0, 384)),
                "transitions": [],
                "boundary_indices": [],
                "document_ranges": []
            }

        if n_pages == 1:
            embeddings = self.embedder.embed_texts([pages[0]["text"]])
            return {
                "page_embeddings": embeddings,
                "transitions": [],
                "boundary_indices": [0],
                "document_ranges": [(0, 0)]
            }

        # 1. Generate page-level embeddings
        page_texts = [p["text"] for p in pages]
        embeddings = self.embedder.embed_texts(page_texts)

        # 2. Compute adjacent similarities: S_i = E_i · E_{i+1}
        similarities = self.embedder.compute_adjacent_similarities(embeddings)

        transitions = []
        boundary_indices = [0]  # First page is always the start of the first document

        for i in range(n_pages - 1):
            curr_p = pages[i]
            next_p = pages[i + 1]
            sim = similarities[i]

            # Signal 1: Semantic dissimilarity (1 - S_i)
            # Clip between 0.0 and 1.0
            sem_dissim = max(0.0, min(1.0, 1.0 - sim))

            # Signal 2: Heading & Title cues (H_i)
            h_score, h_reason = self._compute_heading_score(curr_p, next_p)

            # Signal 3: Identifier / Metadata mismatch (M_i)
            m_score, m_reason = self._compute_metadata_score(curr_p, next_p)

            # Signal 4: Visual & Layout transition (V_i)
            v_score, v_reason = self._compute_visual_score(curr_p, next_p)

            # Weighted boundary score
            ws = self.weights["semantic"]
            wh = self.weights["heading"]
            wm = self.weights["identifier"]
            wv = self.weights["visual"]

            b_score = (ws * sem_dissim) + (wh * h_score) + (wm * m_score) + (wv * v_score)
            b_score = max(0.0, min(1.0, b_score))

            # Definite overrides:
            # 1. If explicit "Page 1 of N" or "Page 1" is detected on next page -> Boundary
            if next_p["signals"].get("page_number") == 1:
                b_score = max(b_score, 0.85)
            # 2. If consecutive page numbering detected (e.g. Page 2 follows Page 1) -> Continuation
            curr_pg = curr_p["signals"].get("page_number")
            next_pg = next_p["signals"].get("page_number")
            if curr_pg is not None and next_pg is not None and next_pg == curr_pg + 1:
                b_score = min(b_score, 0.20)

            is_boundary = b_score >= self.threshold

            rationales = []
            if sem_dissim > 0.4:
                rationales.append(f"Semantic shift (sim={sim:.2f})")
            if h_reason:
                rationales.append(h_reason)
            if m_reason:
                rationales.append(m_reason)
            if not rationales:
                rationales.append("Continuous content and layout")

            transition_info = {
                "from_page": curr_p["page_number"],
                "to_page": next_p["page_number"],
                "similarity": sim,
                "semantic_dissimilarity": round(sem_dissim, 4),
                "heading_score": round(h_score, 4),
                "metadata_score": round(m_score, 4),
                "visual_score": round(v_score, 4),
                "boundary_score": round(b_score, 4),
                "is_boundary": bool(is_boundary),
                "rationale": "; ".join(rationales)
            }
            transitions.append(transition_info)

            if is_boundary:
                boundary_indices.append(i + 1)

        # Build contiguous document ranges
        document_ranges = []
        for b_idx in range(len(boundary_indices)):
            start_p = boundary_indices[b_idx]
            end_p = boundary_indices[b_idx + 1] - 1 if b_idx + 1 < len(boundary_indices) else n_pages - 1
            document_ranges.append((start_p, end_p))

        return {
            "page_embeddings": embeddings,
            "transitions": transitions,
            "boundary_indices": boundary_indices,
            "document_ranges": document_ranges
        }

    def _compute_heading_score(self, curr_p: Dict[str, Any], next_p: Dict[str, Any]) -> Tuple[float, str]:
        """Detects whether next page begins with a fresh document title."""
        next_text_start = next_p["text"][:300].lower()
        curr_text_end = curr_p["text"][-300:].lower()

        # Check if current page ends with continuation markers
        if any(w in curr_text_end for w in ["continued on", "cont...", "turn over", "please see next page"]):
            return 0.0, "Current page has continuation footer"

        # Strong new document headers at top of next page
        new_doc_patterns = [
            (r'\b(tax invoice|commercial invoice|proforma invoice|invoice)\b', "New invoice header detected"),
            (r'\b(statement of account|bank statement|account summary)\b', "New bank statement header detected"),
            (r'\b(customer kyc|identity verification|kyc document|kyc form)\b', "New KYC/ID header detected"),
            (r'\b(receipt|cash receipt|sales receipt)\b', "New receipt header detected"),
            (r'\b(employment contract|agreement|terms of service)\b', "New contract header detected")
        ]

        for pattern, desc in new_doc_patterns:
            if re.search(pattern, next_text_start):
                return 1.0, desc

        return 0.2, ""

    def _compute_metadata_score(self, curr_p: Dict[str, Any], next_p: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluates invoice IDs, account IDs, and page numbers."""
        c_sig = curr_p["signals"]
        n_sig = next_p["signals"]

        # 1. Invoice ID change
        c_inv = c_sig.get("invoice_id")
        n_inv = n_sig.get("invoice_id")
        if c_inv and n_inv:
            if c_inv != n_inv:
                return 1.0, f"Invoice ID changed from {c_inv} to {n_inv}"
            else:
                return 0.0, f"Matching invoice ID: {c_inv}"

        # 2. Account ID continuity
        c_acc = c_sig.get("account_id")
        n_acc = n_sig.get("account_id")
        if c_acc and n_acc:
            if c_acc != n_acc:
                return 0.9, f"Account ID changed from {c_acc} to {n_acc}"
            else:
                return 0.0, f"Matching account ID: {c_acc}"

        # 3. Document type cue transition (e.g. invoice cues to bank statement cues)
        c_is_inv = c_sig.get("has_invoice_cues")
        n_is_bank = n_sig.get("has_bank_cues")
        c_is_bank = c_sig.get("has_bank_cues")
        n_is_inv = n_sig.get("has_invoice_cues")
        if (c_is_inv and n_is_bank) or (c_is_bank and n_is_inv):
            return 0.8, "Transition between distinct document cue types"

        return 0.3, ""

    def _compute_visual_score(self, curr_p: Dict[str, Any], next_p: Dict[str, Any]) -> Tuple[float, str]:
        """Compares word count and line density shifts."""
        w1 = max(1, curr_p.get("word_count", 1))
        w2 = max(1, next_p.get("word_count", 1))

        ratio = min(w1, w2) / float(max(w1, w2))
        # Severe density imbalance can suggest transition
        if ratio < 0.20:
            return 0.6, "Significant visual density/length shift"
        return 0.2, ""
