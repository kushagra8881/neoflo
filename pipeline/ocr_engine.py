import re
import numpy as np
from typing import Dict, Any, List, Optional
from rapidocr_onnxruntime import RapidOCR

class OCREngine:
    """
    Local OCR engine leveraging PaddleOCR models (via RapidOCR ONNX runtime)
    for high accuracy, zero-GPU-overhead text and layout coordinate extraction.
    """
    _instance: Optional["OCREngine"] = None

    def __init__(self):
        # Initialize RapidOCR with default PP-OCRv4 detection & recognition models
        self.engine = RapidOCR()

    @classmethod
    def get_instance(cls) -> "OCREngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def extract_text(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs OCR on an image and returns structured text, lines, bounding boxes, and document signals.
        """
        if img_bgr is None or img_bgr.size == 0:
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "signals": self._extract_signals("")
            }

        result, elapse_list = self.engine(img_bgr)
        if not result:
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "signals": self._extract_signals("")
            }

        lines = []
        full_text_chunks = []
        confidences = []

        for item in result:
            # item format: [box_points, text, confidence]
            box_points, text, score = item
            clean_text = text.strip()
            if not clean_text:
                continue

            full_text_chunks.append(clean_text)
            confidences.append(float(score))

            # Convert 4-point polygon to [x1, y1, x2, y2]
            pts = np.array(box_points, dtype=np.int32)
            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)

            lines.append({
                "text": clean_text,
                "confidence": round(float(score), 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "polygon": pts.tolist()
            })

        combined_text = "\n".join(full_text_chunks)
        avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0

        return {
            "text": combined_text,
            "lines": lines,
            "confidence": avg_conf,
            "word_count": len(combined_text.split()),
            "signals": self._extract_signals(combined_text)
        }

    def _extract_signals(self, text: str) -> Dict[str, Any]:
        """Deterministic signal extractor to assist boundary detection."""
        text_lower = text.lower()

        # Identifiers
        inv_match = re.search(r'\b(inv(?:oice)?[\s#:\-]*[a-z0-9\-]+)\b', text_lower)
        acct_match = re.search(r'\b(account|acct|acc)[\s#:\-]*([0-9x\-]{4,})\b', text_lower)
        page_num_match = re.search(r'\bpage\s+(\d+)\s*(?:of|/)\s*(\d+)\b', text_lower)
        page_single_match = re.search(r'\bpage\s+(\d+)\b', text_lower)

        # Keyword sets
        has_invoice_cues = bool(re.search(r'\b(invoice|subtotal|amount due|tax invoice|bill to|unit price)\b', text_lower))
        has_bank_cues = bool(re.search(r'\b(statement of account|balance|deposit|withdrawal|ledger|closing balance)\b', text_lower))
        has_id_cues = bool(re.search(r'\b(driving license|identity card|passport|national id|date of birth|valid until)\b', text_lower))
        has_receipt_cues = bool(re.search(r'\b(receipt|cashier|cash receipt|pos terminal|card payment)\b', text_lower))
        has_signature_cue = bool(re.search(r'\b(authorized signature|signed by|signature of|date:)\b', text_lower))

        current_page = None
        total_pages = None
        if page_num_match:
            current_page = int(page_num_match.group(1))
            total_pages = int(page_num_match.group(2))
        elif page_single_match:
            current_page = int(page_single_match.group(1))

        return {
            "invoice_id": inv_match.group(1) if inv_match else None,
            "account_id": acct_match.group(2) if acct_match else None,
            "page_number": current_page,
            "total_pages": total_pages,
            "has_invoice_cues": has_invoice_cues,
            "has_bank_cues": has_bank_cues,
            "has_id_cues": has_id_cues,
            "has_receipt_cues": has_receipt_cues,
            "has_signature_cue": has_signature_cue
        }
