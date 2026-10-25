import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

class LogicalDocument:
    """Represents a coherent extracted document (single-page, multi-page, or sub-page crop)."""

    def __init__(
        self,
        doc_id: str,
        source_filename: str,
        source_filepath: str,
        pages: List[Dict[str, Any]],
        page_range: str,
        is_crop: bool = False,
        crop_bbox: Optional[List[int]] = None,
        crop_id: Optional[str] = None
    ):
        self.doc_id = doc_id
        self.source_filename = source_filename
        self.source_filepath = source_filepath
        self.pages = pages
        self.page_range = page_range
        self.page_count = len(pages)
        self.is_crop = is_crop
        self.crop_bbox = crop_bbox
        self.crop_id = crop_id

        # Aggregate combined text
        self.combined_text = "\n\n--- PAGE BREAK ---\n\n".join(
            [f"[Page {p.get('page_number', i+1)}]\n{p.get('text', '')}" for i, p in enumerate(pages)]
        ).strip()
        self.word_count = len(self.combined_text.split())

        # Collect preview images
        self.preview_image_paths = [p["preview_image_path"] for p in pages if "preview_image_path" in p]

        # Aggregate signals
        self.primary_invoice_id = None
        self.primary_account_id = None
        for p in pages:
            sig = p.get("signals", {})
            if not self.primary_invoice_id and sig.get("invoice_id"):
                self.primary_invoice_id = sig["invoice_id"]
            if not self.primary_account_id and sig.get("account_id"):
                self.primary_account_id = sig["account_id"]

        # Fine-grained Document Type & Metadata
        self.document_type: str = "Unclassified Document"
        self.category: str = "General"
        self.type_confidence: str = "pending"
        self.type_summary: str = ""
        self.key_fields: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "source_filepath": self.source_filepath,
            "page_range": self.page_range,
            "page_count": self.page_count,
            "is_crop": self.is_crop,
            "crop_bbox": self.crop_bbox,
            "crop_id": self.crop_id,
            "document_type": self.document_type,
            "category": self.category,
            "type_confidence": self.type_confidence,
            "type_summary": self.type_summary,
            "key_fields": self.key_fields,
            "text": self.combined_text,
            "word_count": self.word_count,
            "preview_image_paths": self.preview_image_paths,
            "primary_invoice_id": self.primary_invoice_id,
            "primary_account_id": self.primary_account_id,
        }


class DocumentAssembler:
    """Assembles segmented pages and crops into standardized logical documents."""

    @staticmethod
    def assemble_from_ranges(
        source_meta: Dict[str, Any],
        pages: List[Dict[str, Any]],
        document_ranges: List[tuple]
    ) -> List[LogicalDocument]:
        """Assembles multi-page documents according to detected page ranges."""
        docs = []
        filename = source_meta["filename"]
        filepath = source_meta["filepath"]

        for idx, (start_p, end_p) in enumerate(document_ranges):
            doc_pages = pages[start_p : end_p + 1]
            p_start_num = doc_pages[0]["page_number"]
            p_end_num = doc_pages[-1]["page_number"]
            
            p_range_str = f"Pages {p_start_num}–{p_end_num}" if p_start_num != p_end_num else f"Page {p_start_num}"
            doc_id = f"{source_meta.get('file_hash', 'doc')}_seg{idx + 1}"

            doc = LogicalDocument(
                doc_id=doc_id,
                source_filename=filename,
                source_filepath=filepath,
                pages=doc_pages,
                page_range=p_range_str,
                is_crop=False
            )
            docs.append(doc)

        return docs

    @staticmethod
    def assemble_from_crops(
        source_meta: Dict[str, Any],
        crops: List[Dict[str, Any]]
    ) -> List[LogicalDocument]:
        """Assembles candidate documents from cropped regions on a single page (Scenario A)."""
        docs = []
        filename = source_meta["filename"]
        filepath = source_meta["filepath"]

        for idx, crop in enumerate(crops):
            pseudo_page = {
                "page_index": 0,
                "page_number": 1,
                "text": crop["text"],
                "lines": crop["lines"],
                "preview_image_path": crop["preview_image_path"],
                "confidence": crop["confidence"],
                "word_count": crop["word_count"],
                "signals": crop["signals"]
            }

            doc_id = crop["crop_id"]
            doc = LogicalDocument(
                doc_id=doc_id,
                source_filename=filename,
                source_filepath=filepath,
                pages=[pseudo_page],
                page_range=f"Page 1 (Region {idx + 1})",
                is_crop=True,
                crop_bbox=crop["bbox"],
                crop_id=crop["crop_id"]
            )
            docs.append(doc)

        return docs

    @staticmethod
    def manual_merge(doc_a: LogicalDocument, doc_b: LogicalDocument) -> LogicalDocument:
        """Merges two logical documents into one."""
        merged_pages = doc_a.pages + doc_b.pages
        p_start_num = merged_pages[0]["page_number"]
        p_end_num = merged_pages[-1]["page_number"]
        p_range_str = f"Pages {p_start_num}–{p_end_num}" if p_start_num != p_end_num else f"Page {p_start_num}"

        return LogicalDocument(
            doc_id=f"{doc_a.doc_id}_merged",
            source_filename=doc_a.source_filename,
            source_filepath=doc_a.source_filepath,
            pages=merged_pages,
            page_range=p_range_str,
            is_crop=False
        )

    @staticmethod
    def manual_split(doc: LogicalDocument, split_at_page_num: int) -> List[LogicalDocument]:
        """Splits a multi-page logical document at a specific page number."""
        part_1_pages = [p for p in doc.pages if p["page_number"] < split_at_page_num]
        part_2_pages = [p for p in doc.pages if p["page_number"] >= split_at_page_num]

        if not part_1_pages or not part_2_pages:
            return [doc]

        doc1 = LogicalDocument(
            doc_id=f"{doc.doc_id}_part1",
            source_filename=doc.source_filename,
            source_filepath=doc.source_filepath,
            pages=part_1_pages,
            page_range=f"Pages {part_1_pages[0]['page_number']}–{part_1_pages[-1]['page_number']}",
            is_crop=False
        )
        doc2 = LogicalDocument(
            doc_id=f"{doc.doc_id}_part2",
            source_filename=doc.source_filename,
            source_filepath=doc.source_filepath,
            pages=part_2_pages,
            page_range=f"Pages {part_2_pages[0]['page_number']}–{part_2_pages[-1]['page_number']}",
            is_crop=False
        )
        return [doc1, doc2]
