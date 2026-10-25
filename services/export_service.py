import json
import csv
import io
import zipfile
from pathlib import Path
from typing import List, Dict, Any
import pymupdf as fitz
from pipeline.config import PROCESSED_DIR
from pipeline.document_assembler import LogicalDocument

class ExportService:
    """Handles structured data export: CSV summaries, full JSON schemas, and segmented PDF bundles."""

    @staticmethod
    def generate_csv(documents: List[LogicalDocument], cluster_results: Dict[str, Any], gemini_labels: Dict[int, Dict[str, Any]]) -> str:
        """Generates a comprehensive CSV string of segmented documents and their semantic clusters."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Document ID",
            "Source File",
            "Page Range",
            "Page Count",
            "Is Sub-Page Crop",
            "Document Type",
            "Document Category",
            "Cluster ID",
            "Cluster Name",
            "Cluster Category",
            "Distance to Centroid",
            "Ambiguous / Flagged",
            "Primary Invoice ID",
            "Primary Account ID",
            "Word Count",
            "Document Summary",
            "Text Snippet"
        ])

        labels = cluster_results.get("labels", [])
        distances = cluster_results.get("distances", [])
        ambiguities = cluster_results.get("is_ambiguous", [])

        for i, doc in enumerate(documents):
            c_id = labels[i] if i < len(labels) else -1
            c_meta = gemini_labels.get(c_id, {})
            c_name = c_meta.get("label", f"Cluster {c_id}")
            c_cat = c_meta.get("document_category", "N/A")
            dist = distances[i] if i < len(distances) else 0.0
            ambig = "YES" if (i < len(ambiguities) and ambiguities[i]) else "NO"

            snippet = " ".join(doc.combined_text.split()[:25])

            writer.writerow([
                doc.doc_id,
                doc.source_filename,
                doc.page_range,
                doc.page_count,
                "YES" if doc.is_crop else "NO",
                getattr(doc, "document_type", "General Document"),
                getattr(doc, "category", c_cat),
                c_id,
                c_name,
                c_cat,
                dist,
                ambig,
                doc.primary_invoice_id or "",
                doc.primary_account_id or "",
                doc.word_count,
                getattr(doc, "type_summary", ""),
                snippet
            ])

        return output.getvalue()

    @staticmethod
    def generate_json(
        documents: List[LogicalDocument], 
        cluster_results: Dict[str, Any], 
        gemini_labels: Dict[int, Dict[str, Any]],
        transitions_by_file: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Generates full machine-readable JSON schema containing the complete processing tree."""
        labels = cluster_results.get("labels", [])
        distances = cluster_results.get("distances", [])
        ambiguities = cluster_results.get("is_ambiguous", [])
        pca_coords = cluster_results.get("pca_coords", [])

        doc_entries = []
        for i, doc in enumerate(documents):
            c_id = labels[i] if i < len(labels) else -1
            c_meta = gemini_labels.get(c_id, {})
            
            entry = {
                "document_id": doc.doc_id,
                "source_filename": doc.source_filename,
                "source_filepath": doc.source_filepath,
                "page_range": doc.page_range,
                "page_count": doc.page_count,
                "is_crop": doc.is_crop,
                "crop_bbox": doc.crop_bbox,
                "identified_type": {
                    "document_type": getattr(doc, "document_type", "General Document"),
                    "category": getattr(doc, "category", "General"),
                    "confidence": getattr(doc, "type_confidence", "medium"),
                    "summary": getattr(doc, "type_summary", ""),
                    "key_fields": getattr(doc, "key_fields", {})
                },
                "cluster_assignment": {
                    "cluster_id": c_id,
                    "label": c_meta.get("label", f"Cluster {c_id}"),
                    "description": c_meta.get("description", ""),
                    "category": c_meta.get("document_category", ""),
                    "distance_to_centroid": distances[i] if i < len(distances) else 0.0,
                    "is_ambiguous": ambiguities[i] if i < len(ambiguities) else False,
                    "pca_coordinates": pca_coords[i] if i < len(pca_coords) else [0.0, 0.0]
                },
                "metadata": {
                    "invoice_id": doc.primary_invoice_id,
                    "account_id": doc.primary_account_id,
                    "word_count": doc.word_count
                },
                "text": doc.combined_text
            }
            doc_entries.append(entry)

        export_data = {
            "application": "Neoflio Intelligent Document Classifier",
            "version": "1.0.0",
            "clustering_summary": {
                "total_documents": len(documents),
                "total_clusters": cluster_results.get("k", 0),
                "quality_metrics": cluster_results.get("quality", {}),
                "cluster_catalog": gemini_labels
            },
            "document_boundary_transitions": transitions_by_file,
            "logical_documents": doc_entries
        }

        return json.dumps(export_data, indent=2)

    @staticmethod
    def extract_document_pdf(doc: LogicalDocument, output_dir: Path) -> Path:
        """Extracts the exact pages of a LogicalDocument from source PDF into a standalone PDF."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"{doc.doc_id}.pdf"
        out_path = output_dir / out_filename

        src_path = Path(doc.source_filepath)
        if not src_path.exists() or src_path.suffix.lower() != ".pdf":
            # If source was image or crop, create a 1-page PDF from image
            if doc.preview_image_paths and Path(doc.preview_image_paths[0]).exists():
                img_doc = fitz.open()
                pix = fitz.Pixmap(doc.preview_image_paths[0])
                pdf_page = img_doc.new_page(width=pix.width, height=pix.height)
                pdf_page.insert_image(pdf_page.rect, filename=doc.preview_image_paths[0])
                img_doc.save(str(out_path))
                img_doc.close()
            return out_path

        # Extract specific pages from source PDF
        src_pdf = fitz.open(str(src_path))
        dest_pdf = fitz.open()

        for p in doc.pages:
            p_idx = p["page_index"]
            if 0 <= p_idx < len(src_pdf):
                dest_pdf.insert_pdf(src_pdf, from_page=p_idx, to_page=p_idx)

        dest_pdf.save(str(out_path))
        dest_pdf.close()
        src_pdf.close()
        return out_path

    @classmethod
    def create_export_zip(
        cls, 
        documents: List[LogicalDocument], 
        cluster_results: Dict[str, Any], 
        gemini_labels: Dict[int, Dict[str, Any]],
        transitions_by_file: Dict[str, List[Dict[str, Any]]]
    ) -> bytes:
        """Packages CSV, JSON, and all segmented individual document PDFs into a single ZIP archive."""
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Add CSV
            csv_str = cls.generate_csv(documents, cluster_results, gemini_labels)
            zf.writestr("document_clusters.csv", csv_str)

            # 2. Add JSON
            json_str = cls.generate_json(documents, cluster_results, gemini_labels, transitions_by_file)
            zf.writestr("classification_results.json", json_str)

            # 3. Add extracted PDF documents
            pdf_temp_dir = PROCESSED_DIR / "segmented_pdfs"
            for doc in documents:
                extracted_path = cls.extract_document_pdf(doc, pdf_temp_dir)
                if extracted_path.exists():
                    zf.write(extracted_path, arcname=f"documents/{extracted_path.name}")

        zip_buffer.seek(0)
        return zip_buffer.getvalue()
