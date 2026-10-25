import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pipeline.config import PROCESSED_DIR, MIN_DOC_REGION_AREA_RATIO, MAX_DOC_REGION_AREA_RATIO, REGION_PADDING_PX
from pipeline.image_preprocessor import ImagePreprocessor
from pipeline.ocr_engine import OCREngine

class RegionDetector:
    """
    Solves Scenario A: Detects and crops multiple distinct documents arranged on a single scanned page
    (e.g., an invoice, a receipt, and an ID card scanned on the same flatbed).
    Uses a hybrid approach combining Canny edge card boundary detection with layout spatial grouping.
    """

    def __init__(self):
        self.ocr_engine = OCREngine.get_instance()

    def detect_regions(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects candidate document bounding boxes within a single page image.
        Returns list of bboxes: [{"bbox": [x1, y1, x2, y2], "area": int, "ratio": float}]
        """
        h, w = img_bgr.shape[:2]
        total_area = w * h
        min_area = total_area * MIN_DOC_REGION_AREA_RATIO
        max_area = total_area * MAX_DOC_REGION_AREA_RATIO

        # 1. Edge & Contour Detection for Document/Card Borders
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / float(ch) if ch > 0 else 0
                if 0.15 <= aspect <= 6.0:
                    candidates.append({
                        "bbox": [x, y, x + cw, y + ch],
                        "area": area,
                        "ratio": round(area / total_area, 4)
                    })

        # Apply Non-Maximum Suppression (NMS) to eliminate nested inner border boxes
        filtered = self._apply_nms(candidates, iou_thresh=0.35)

        # 2. If card border detection found >= 2 documents, return them
        if len(filtered) >= 2:
            filtered.sort(key=lambda c: (c["bbox"][1] // 100, c["bbox"][0]))
            return filtered

        # 3. Fallback: Text-block clustering using OCR line spatial distribution
        ocr_res = self.ocr_engine.extract_text(img_bgr)
        lines = ocr_res.get("lines", [])
        if len(lines) >= 4:
            spatial_candidates = self._cluster_text_lines(lines, w, h, min_area, max_area)
            if len(spatial_candidates) >= 2:
                spatial_candidates.sort(key=lambda c: (c["bbox"][1] // 100, c["bbox"][0]))
                return spatial_candidates

        return filtered

    def _cluster_text_lines(
        self, 
        lines: List[Dict], 
        img_w: int, 
        img_h: int, 
        min_area: float, 
        max_area: float
    ) -> List[Dict[str, Any]]:
        """Groups text lines by spatial proximity to isolate separate document blocks."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line["bbox"]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Morphological dilation to bridge intra-document lines
        k_w = max(20, int(img_w * 0.05))
        k_h = max(20, int(img_h * 0.04))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        dilated = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        total_area = img_w * img_h

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                candidates.append({
                    "bbox": [x, y, x + cw, y + ch],
                    "area": area,
                    "ratio": round(area / total_area, 4)
                })

        return self._apply_nms(candidates, iou_thresh=0.35)

    def _apply_nms(self, candidates: List[Dict[str, Any]], iou_thresh: float = 0.35) -> List[Dict[str, Any]]:
        """Suppresses heavily overlapping bounding boxes, keeping the larger dominant region."""
        if not candidates:
            return []

        # Sort by area descending
        sorted_boxes = sorted(candidates, key=lambda c: c["area"], reverse=True)
        keep = []

        for item in sorted_boxes:
            b1 = item["bbox"]
            overlap = False
            for kept in keep:
                b2 = kept["bbox"]
                xx1 = max(b1[0], b2[0])
                yy1 = max(b1[1], b2[1])
                xx2 = min(b1[2], b2[2])
                yy2 = min(b1[3], b2[3])

                w_inter = max(0, xx2 - xx1)
                h_inter = max(0, yy2 - yy1)
                inter_area = w_inter * h_inter

                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter_area

                iou = inter_area / float(union) if union > 0 else 0
                contained_ratio = inter_area / float(area1) if area1 > 0 else 0

                if iou > iou_thresh or contained_ratio > 0.65:
                    overlap = True
                    break

            if not overlap:
                keep.append(item)

        return keep

    def extract_document_crops(
        self, 
        img_bgr: np.ndarray, 
        parent_id: str, 
        page_num: int
    ) -> List[Dict[str, Any]]:
        """
        Detects document regions, crops them, runs OCR independently on each,
        and returns candidate documents.
        """
        regions = self.detect_regions(img_bgr)

        # If fewer than 2 regions found, treat the whole page as 1 document candidate
        if len(regions) < 2:
            return []

        crops = []
        for idx, reg in enumerate(regions):
            bbox = reg["bbox"]
            crop_bgr = ImagePreprocessor.crop_region(img_bgr, bbox, padding=REGION_PADDING_PX)
            
            crop_id = f"{parent_id}_p{page_num}_crop{idx + 1}"
            crop_filename = f"{crop_id}.png"
            crop_path = PROCESSED_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop_bgr)

            # Run OCR on this cropped document region independently
            ocr_res = self.ocr_engine.extract_text(crop_bgr)
            
            # Only keep crop if it contains meaningful text
            if len(ocr_res["text"].strip()) >= 5:
                crops.append({
                    "crop_id": crop_id,
                    "crop_index": idx,
                    "bbox": bbox,
                    "preview_image_path": str(crop_path),
                    "text": ocr_res["text"],
                    "lines": ocr_res["lines"],
                    "confidence": ocr_res["confidence"],
                    "word_count": ocr_res["word_count"],
                    "signals": ocr_res["signals"],
                    "width": crop_bgr.shape[1],
                    "height": crop_bgr.shape[0]
                })

        return crops
