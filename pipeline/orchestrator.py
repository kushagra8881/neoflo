import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from pipeline.file_loader import FileLoader
from pipeline.pdf_processor import PDFProcessor
from pipeline.region_detector import RegionDetector
from pipeline.boundary_detector import BoundaryDetector
from pipeline.document_assembler import DocumentAssembler, LogicalDocument
from pipeline.document_embedder import DocumentEmbedder
from pipeline.cluster_engine import ClusterEngine
from services.gemini_service import GeminiService
from pipeline.image_preprocessor import ImagePreprocessor

class PipelineOrchestrator:
    """
    End-to-end processing pipeline connecting:
    File Ingestion -> OCR -> Boundary/Region Segmentation -> Embeddings -> K-Means -> Gemini Labeling.
    """

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.region_detector = RegionDetector()
        self.boundary_detector = BoundaryDetector()
        self.document_embedder = DocumentEmbedder()
        self.cluster_engine = ClusterEngine()
        self.gemini_service = GeminiService.get_instance()

    def process_batch(
        self,
        file_paths: List[Path],
        manual_k: Optional[int] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes the entire hybrid pipeline across multiple uploaded files.
        Yields live progress updates to progress_cb(fraction, status_message).
        """
        def update_progress(pct: float, msg: str):
            if progress_cb:
                progress_cb(pct, msg)

        update_progress(0.05, "Validating uploaded files and checksums...")
        validated_files = FileLoader.load_directory_or_files(file_paths)
        if not validated_files:
            return {"error": "No valid files found to process."}

        # -------------------------------------------------------------
        # PHASE 1 & 2: INGESTION, OCR, REGION & BOUNDARY SEGMENTATION
        # -------------------------------------------------------------
        update_progress(0.15, "Rendering pages and running local PaddleOCR...")
        all_logical_docs: List[LogicalDocument] = []
        transitions_by_file: Dict[str, List[Dict[str, Any]]] = {}
        total_pages_count = 0

        for f_idx, file_info in enumerate(validated_files):
            fpath = Path(file_info["filepath"])
            fhash = file_info["file_hash"]
            fname = file_info["filename"]

            update_progress(
                0.15 + (0.35 * (f_idx / len(validated_files))),
                f"Processing '{fname}' (OCR & Segmentation)..."
            )

            if file_info["is_pdf"]:
                # Multi-page PDF workflow
                pages = self.pdf_processor.process_pdf(fpath, doc_id=fhash)
                total_pages_count += len(pages)

                if len(pages) == 1:
                    # Single-page PDF: Check for Scenario A (multiple documents on 1 page)
                    page_img_bgr = ImagePreprocessor.load_image(pages[0]["preview_image_path"])
                    crops = self.region_detector.extract_document_crops(page_img_bgr, parent_id=fhash, page_num=1)
                    if len(crops) >= 2:
                        docs = DocumentAssembler.assemble_from_crops(file_info, crops)
                        all_logical_docs.extend(docs)
                    else:
                        doc = LogicalDocument(
                            doc_id=f"{fhash}_p1",
                            source_filename=fname,
                            source_filepath=str(fpath),
                            pages=pages,
                            page_range="Page 1",
                            is_crop=False
                        )
                        all_logical_docs.append(doc)
                else:
                    # Multi-page PDF: Run boundary detection (Scenarios B & C)
                    b_res = self.boundary_detector.detect_boundaries(pages)
                    transitions_by_file[fname] = b_res["transitions"]
                    docs = DocumentAssembler.assemble_from_ranges(file_info, pages, b_res["document_ranges"])
                    all_logical_docs.extend(docs)

            else:
                # Standalone Image workflow (JPG, PNG, TIFF)
                total_pages_count += 1
                img_data = self.pdf_processor.process_standalone_image(fpath, doc_id=fhash)
                img_bgr = ImagePreprocessor.load_image(fpath)

                # Check for Scenario A: Multiple document cards on single image
                crops = self.region_detector.extract_document_crops(img_bgr, parent_id=fhash, page_num=1)
                if len(crops) >= 2:
                    docs = DocumentAssembler.assemble_from_crops(file_info, crops)
                    all_logical_docs.extend(docs)
                else:
                    doc = LogicalDocument(
                        doc_id=f"{fhash}_doc",
                        source_filename=fname,
                        source_filepath=str(fpath),
                        pages=[img_data],
                        page_range="Page 1",
                        is_crop=False
                    )
                    all_logical_docs.append(doc)

        # -------------------------------------------------------------
        # PHASE 3: SEMANTIC DOCUMENT EMBEDDINGS
        # -------------------------------------------------------------
        update_progress(0.60, f"Generating 384-D semantic embeddings for {len(all_logical_docs)} documents...")
        embedding_matrix, doc_ids = self.document_embedder.embed_documents(all_logical_docs)

        # -------------------------------------------------------------
        # PHASE 4: UNSUPERVISED K-MEANS CLUSTERING
        # -------------------------------------------------------------
        update_progress(0.75, "Running K-Means & evaluating Silhouette Score...")
        cluster_results = self.cluster_engine.cluster(embedding_matrix, k=manual_k)

        # -------------------------------------------------------------
        # PHASE 5: GEMINI SEMANTIC REASONING & CLUSTER NAMING
        # -------------------------------------------------------------
        update_progress(0.85, "Querying Gemini API for structured cluster labels & descriptions...")
        gemini_labels = {}
        for c_id, c_data in cluster_results.get("clusters", {}).items():
            doc_indices = c_data["indices"]
            # Pick representative document texts from this cluster
            sample_texts = [all_logical_docs[idx].combined_text for idx in doc_indices[:3]]
            sample_signals = [all_logical_docs[idx].pages[0].get("signals", {}) for idx in doc_indices[:3]]
            
            label_info = self.gemini_service.describe_cluster(
                cluster_id=c_id,
                sample_texts=sample_texts,
                sample_signals=sample_signals
            )
            gemini_labels[c_id] = label_info

        # Fine-grained Document Type Classification for each document
        update_progress(0.92, "Classifying fine-grained document types (Invoices, Passports, Statements, etc.)...")
        for doc in all_logical_docs:
            sig = doc.pages[0].get("signals", {}) if doc.pages else {}
            type_info = self.gemini_service.classify_document(
                doc_text=doc.combined_text,
                signals=sig
            )
            doc.document_type = type_info.get("document_type", "General Document")
            doc.category = type_info.get("category", "General")
            doc.type_confidence = type_info.get("confidence", "medium")
            doc.type_summary = type_info.get("summary", "")
            doc.key_fields = type_info.get("key_fields", {})

        update_progress(1.0, "Processing complete!")

        return {
            "total_files": len(validated_files),
            "total_pages": total_pages_count,
            "total_logical_documents": len(all_logical_docs),
            "documents": all_logical_docs,
            "embedding_matrix": embedding_matrix,
            "cluster_results": cluster_results,
            "gemini_labels": gemini_labels,
            "transitions_by_file": transitions_by_file
        }
