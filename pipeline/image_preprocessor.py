import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict

class ImagePreprocessor:
    """Provides OpenCV & Pillow utilities for image cleaning, deskewing, and cropping."""

    @staticmethod
    def load_image(source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Standardizes any input image into a BGR OpenCV numpy array."""
        if isinstance(source, np.ndarray):
            return source.copy()
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                # Fallback to Pillow in case OpenCV has issues with format
                pil_img = Image.open(source).convert("RGB")
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        if isinstance(source, bytes):
            nparr = np.frombuffer(source, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if isinstance(source, Image.Image):
            rgb = np.array(source.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        raise ValueError(f"Unsupported image source type: {type(source)}")

    @staticmethod
    def to_pillow(img_bgr: np.ndarray) -> Image.Image:
        """Converts BGR numpy image to PIL Image."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @classmethod
    def deskew(cls, img_bgr: np.ndarray, max_angle: float = 45.0) -> Tuple[np.ndarray, float]:
        """Detects text orientation skew using OpenCV and deskews the image."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Invert colors: text becomes white, background black
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return img_bgr, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # If angle is minor (< 0.5 degrees) or wildly skewed (> max_angle), avoid over-correcting
        if abs(angle) < 0.5 or abs(angle) > max_angle:
            return img_bgr, 0.0

        (h, w) = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img_bgr, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated, float(angle)

    @classmethod
    def enhance_contrast(cls, img_bgr: np.ndarray) -> np.ndarray:
        """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for faint scans."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    @classmethod
    def crop_region(
        cls, 
        img_bgr: np.ndarray, 
        bbox: List[int], 
        padding: int = 10
    ) -> np.ndarray:
        """
        Crops a region from the image with boundary clamping and padding.
        bbox format: [x1, y1, x2, y2]
        """
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Apply padding safely
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        if x2 <= x1 or y2 <= y1:
            return img_bgr.copy()
            
        return img_bgr[y1:y2, x1:x2].copy()

    @classmethod
    def draw_bboxes(
        cls, 
        img_bgr: np.ndarray, 
        bboxes: List[Dict], 
        color: Tuple[int, int, int] = (0, 165, 255),
        thickness: int = 3
    ) -> np.ndarray:
        """Draws bounding boxes and labels for visualization."""
        vis = img_bgr.copy()
        for idx, item in enumerate(bboxes):
            x1, y1, x2, y2 = item["bbox"]
            label = item.get("label", f"Region {idx+1}")
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Label background box
            cv2.rectangle(vis, (x1, max(0, y1 - 25)), (x1 + len(label) * 11, max(25, y1)), color, -1)
            cv2.putText(
                vis, label, (x1 + 4, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )
        return vis
