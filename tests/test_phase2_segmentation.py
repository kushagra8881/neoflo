import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.file_loader import FileLoader
from pipeline.pdf_processor import PDFProcessor
from pipeline.region_detector import RegionDetector
from pipeline.boundary_detector import BoundaryDetector
from pipeline.document_assembler import DocumentAssembler, LogicalDocument
from pipeline.image_preprocessor import ImagePreprocessor
from pipeline.config import TEST_SAMPLES_DIR

def run_phase2_tests():
    print("=" * 70)
    print("PHASE 2 TESTING: Document Segmentation & Multi-Signal Boundary Detection")
    print("=" * 70)

    pdf_processor = PDFProcessor()
    region_detector = RegionDetector()
    boundary_detector = BoundaryDetector()

    # -------------------------------------------------------------
    # TEST 1: SCENARIO A - MULTIPLE DOCUMENTS ON ONE PAGE
    # -------------------------------------------------------------
    print("\n[Scenario A] Detecting separate document cards on a single scanned page...")
    scen_a_file = TEST_SAMPLES_DIR / "scenario_a_multi_doc_on_single_page.png"
    img_bgr = ImagePreprocessor.load_image(scen_a_file)
    meta_a = FileLoader.validate_file(scen_a_file)

    detected_regions = region_detector.detect_regions(img_bgr)
    print(f"  ✓ Candidate regions detected on page: {len(detected_regions)}")
    for i, r in enumerate(detected_regions):
        print(f"    Region {i+1}: bbox={r['bbox']}, area={r['area']}, area_ratio={r['ratio']}")
    assert len(detected_regions) >= 2, f"Expected at least 2 distinct regions, got {len(detected_regions)}"

    crops = region_detector.extract_document_crops(img_bgr, parent_id="test_scen_a", page_num=1)
    print(f"  ✓ Cropped documents extracted & independently OCR'd: {len(crops)}")
    for c in crops:
        print(f"    - Crop '{c['crop_id']}': {c['word_count']} words | snippet: {repr(c['text'][:50])}...")

    docs_a = DocumentAssembler.assemble_from_crops(meta_a, crops)
    print(f"  ✓ Assembled {len(docs_a)} logical sub-documents from single page scan.")
    assert len(docs_a) >= 2, "Expected multiple logical documents assembled from single image."

    # -------------------------------------------------------------
    # TEST 2: SCENARIO B - COMPOUND MULTI-DOCUMENT PDF
    # -------------------------------------------------------------
    print("\n[Scenario B] Detecting document transitions across compound 9-page PDF...")
    scen_b_file = TEST_SAMPLES_DIR / "scenario_b_multi_doc_compound.pdf"
    meta_b = FileLoader.validate_file(scen_b_file)
    pages_b = pdf_processor.process_pdf(scen_b_file, doc_id="scen_b")

    b_res = boundary_detector.detect_boundaries(pages_b)
    print(f"  ✓ Total pages analyzed: {len(pages_b)}")
    print(f"  ✓ Adjacent page transitions evaluated:")
    for t in b_res["transitions"]:
        marker = ">>> BOUNDARY DETECTED <<<" if t["is_boundary"] else "continuation"
        print(f"    Page {t['from_page']} -> Page {t['to_page']} | Sim: {t['similarity']:.2f} | B-Score: {t['boundary_score']:.2f} | {marker} ({t['rationale']})")

    print(f"  ✓ Discovered document page ranges (0-indexed): {b_res['document_ranges']}")
    docs_b = DocumentAssembler.assemble_from_ranges(meta_b, pages_b, b_res['document_ranges'])
    print(f"  ✓ Assembled {len(docs_b)} logical documents from compound PDF:")
    for i, d in enumerate(docs_b):
        print(f"    Document {i+1} [{d.doc_id}]: {d.page_range} ({d.page_count} pages) | snippet: {repr(d.combined_text[:50])}...")

    # We expect 4 logical documents:
    # Doc 1: Invoice A (Pages 1-2)
    # Doc 2: KYC Form (Pages 3-4)
    # Doc 3: Invoice B (Pages 5-6)
    # Doc 4: Bank Statement (Pages 7-9)
    assert len(docs_b) == 4, f"Expected 4 logical documents in Scenario B, got {len(docs_b)}"
    assert docs_b[0].page_count == 2, f"Invoice A should have 2 pages, got {docs_b[0].page_count}"
    assert docs_b[1].page_count == 2, f"KYC form should have 2 pages, got {docs_b[1].page_count}"
    assert docs_b[2].page_count == 2, f"Invoice B should have 2 pages, got {docs_b[2].page_count}"
    assert docs_b[3].page_count == 3, f"Bank statement should have 3 pages, got {docs_b[3].page_count}"

    # -------------------------------------------------------------
    # TEST 3: SCENARIO C - MULTI-PAGE SINGLE DOCUMENT CONTINUATION
    # -------------------------------------------------------------
    print("\n[Scenario C] Verifying single multi-page document is NOT split (4-page statement)...")
    scen_c_file = TEST_SAMPLES_DIR / "scenario_c_multipage_bank_statement.pdf"
    meta_c = FileLoader.validate_file(scen_c_file)
    pages_c = pdf_processor.process_pdf(scen_c_file, doc_id="scen_c")

    c_res = boundary_detector.detect_boundaries(pages_c)
    print(f"  ✓ Transitions for 4-page statement:")
    for t in c_res["transitions"]:
        marker = "BOUNDARY" if t["is_boundary"] else "continuation"
        print(f"    Page {t['from_page']} -> Page {t['to_page']} | Sim: {t['similarity']:.2f} | B-Score: {t['boundary_score']:.2f} | {marker}")

    docs_c = DocumentAssembler.assemble_from_ranges(meta_c, pages_c, c_res['document_ranges'])
    print(f"  ✓ Assembled {len(docs_c)} logical document(s). Total pages: {docs_c[0].page_count}")
    assert len(docs_c) == 1, f"Scenario C should remain 1 unified document, got {len(docs_c)}"
    assert docs_c[0].page_count == 4, f"Expected 4 pages in unified document, got {docs_c[0].page_count}"

    # -------------------------------------------------------------
    # TEST 4: MANUAL INTERACTIVE REVIEW ACTIONS (SPLIT & MERGE)
    # -------------------------------------------------------------
    print("\n[Test 4] Verifying manual user split and merge actions...")
    # Test split
    split_parts = DocumentAssembler.manual_split(docs_c[0], split_at_page_num=3)
    print(f"  ✓ Split 4-page doc at Page 3 -> {len(split_parts)} parts:")
    print(f"    Part 1: {split_parts[0].page_range} ({split_parts[0].page_count} pages)")
    print(f"    Part 2: {split_parts[1].page_range} ({split_parts[1].page_count} pages)")
    assert len(split_parts) == 2 and split_parts[0].page_count == 2 and split_parts[1].page_count == 2

    # Test merge back
    merged_back = DocumentAssembler.manual_merge(split_parts[0], split_parts[1])
    print(f"  ✓ Merged parts back -> {merged_back.page_range} ({merged_back.page_count} pages)")
    assert merged_back.page_count == 4

    print("\n" + "=" * 70)
    print("PHASE 2 TESTING COMPLETED SUCCESSFULLY: ALL CHECKS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_phase2_tests()
