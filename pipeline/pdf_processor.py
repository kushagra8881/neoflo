import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Union
from pipeline.config import PROCESSED_DIR, PDF_RENDER_DPI
from pipeline.ocr_engine import OCREngine
from pipeline.image_preprocessor import ImagePreprocessor

class PDFProcessor:
    """
    Renders and extracts text from PDFs.
    Uses digital text fast-path for digital PDFs, and routes scanned pages to OCR.
    """

    def __init__(self, render_dpi: int = PDF_RENDER_DPI):
        self.render_dpi = render_dpi
        self.ocr_engine = OCREngine.get_instance()

    def process_pdf(self, pdf_path: Union[str, Path], doc_id: str) -> List[Dict[str, Any]]:
        path = Path(pdf_path)
        doc = fitz.open(str(path))
        pages_data = []

        zoom = self.render_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Render page to high-res image
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            
            # Save preview image for UI and visual processing
            preview_filename = f"{doc_id}_page_{page_num}.png"
            preview_path = PROCESSED_DIR / preview_filename
            with open(preview_path, "wb") as f:
                f.write(img_bytes)

            img_bgr = ImagePreprocessor.load_image(img_bytes)

            # 2. Check for embedded digital text
            digital_text = page.get_text("text").strip()
            digital_blocks = page.get_text("blocks")
            is_digital = len(digital_text.split()) >= 30

            if is_digital:
                # Digital text fast-path: construct lines from digital text blocks
                lines = []
                for b in digital_blocks:
                    # block: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(b) >= 5 and b[4].strip():
                        lines.append({
                            "text": b[4].strip(),
                            "confidence": 1.0,
                            "bbox": [int(b[0] * zoom), int(b[1] * zoom), int(b[2] * zoom), int(b[3] * zoom)]
                        })
                
                signals = self.ocr_engine._extract_signals(digital_text)
                pages_data.append({
                    "page_index": page_idx,
                    "page_number": page_num,
                    "text": digital_text,
                    "lines": lines,
                    "preview_image_path": str(preview_path),
                    "is_scanned": False,
                    "confidence": 1.0,
                    "word_count": len(digital_text.split()),
                    "signals": signals,
                    "width": pix.width,
                    "height": pix.height
                })
            else:
                # Scanned path: run local PaddleOCR
                ocr_res = self.ocr_engine.extract_text(img_bgr)
                pages_data.append({
                    "page_index": page_idx,
                    "page_number": page_num,
                    "text": ocr_res["text"],
                    "lines": ocr_res["lines"],
                    "preview_image_path": str(preview_path),
                    "is_scanned": True,
                    "confidence": ocr_res["confidence"],
                    "word_count": ocr_res["word_count"],
                    "signals": ocr_res["signals"],
                    "width": pix.width,
                    "height": pix.height
                })

        doc.close()
        return pages_data

    def process_standalone_image(self, img_path: Union[str, Path], doc_id: str) -> Dict[str, Any]:
        """Processes a standalone image file (JPG, PNG, TIFF) as a 1-page document."""
        path = Path(img_path)
        img_bgr = ImagePreprocessor.load_image(path)
        
        # Deskew if needed
        deskewed_bgr, angle = ImagePreprocessor.deskew(img_bgr)
        h, w = deskewed_bgr.shape[:2]

        # Save preview image
        preview_filename = f"{doc_id}_page_1.png"
        preview_path = PROCESSED_DIR / preview_filename
        cv2.imwrite(str(preview_path), deskewed_bgr)

        # Run OCR
        ocr_res = self.ocr_engine.extract_text(deskewed_bgr)

        return {
            "page_index": 0,
            "page_number": 1,
            "text": ocr_res["text"],
            "lines": ocr_res["lines"],
            "preview_image_path": str(preview_path),
            "is_scanned": True,
            "confidence": ocr_res["confidence"],
            "word_count": ocr_res["word_count"],
            "signals": ocr_res["signals"],
            "deskew_angle": angle,
            "width": w,
            "height": h
        }
