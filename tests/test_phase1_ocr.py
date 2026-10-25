import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.file_loader import FileLoader
from pipeline.pdf_processor import PDFProcessor
from pipeline.ocr_engine import OCREngine
from pipeline.image_preprocessor import ImagePreprocessor
from pipeline.config import TEST_SAMPLES_DIR

def run_phase1_tests():
    print("=" * 70)
    print("PHASE 1 TESTING: Input Handling, File Validation, and OCR Extraction")
    print("=" * 70)

    # 1. Test FileLoader
    sample_files = list(TEST_SAMPLES_DIR.glob("*"))
    print(f"\n[Test 1] Validating {len(sample_files)} sample files via FileLoader...")
    validated = FileLoader.load_directory_or_files(sample_files)
    assert len(validated) == len(sample_files), f"Expected {len(sample_files)} valid files, got {len(validated)}"
    for item in validated:
        print(f"  ✓ {item['filename']} ({item['size_bytes']} bytes, hash={item['file_hash']}, is_pdf={item['is_pdf']})")

    # 2. Test PDFProcessor on digital compound PDF (Scenario B)
    pdf_processor = PDFProcessor()
    scenario_b_file = TEST_SAMPLES_DIR / "scenario_b_multi_doc_compound.pdf"
    print(f"\n[Test 2] Processing compound PDF (Scenario B): {scenario_b_file.name}")
    pages_b = pdf_processor.process_pdf(scenario_b_file, doc_id="test_scen_b")
    print(f"  ✓ Extracted {len(pages_b)} pages from PDF.")
    assert len(pages_b) == 9, f"Expected 9 pages, got {len(pages_b)}"
    
    # Check page 1 text (Invoice A)
    p1 = pages_b[0]
    print(f"  ✓ Page 1 text snippet: {repr(p1['text'][:70])}...")
    print(f"  ✓ Page 1 signals: {p1['signals']}")
    assert "INV-A-101" in p1['text'] or "acme" in p1['text'].lower()

    # Check page 7 text (Bank statement)
    p7 = pages_b[6]
    print(f"  ✓ Page 7 (Statement) text snippet: {repr(p7['text'][:70])}...")
    print(f"  ✓ Page 7 signals: {p7['signals']}")
    assert "4401-2290-88" in p7['text'] or "statement" in p7['text'].lower()

    # 3. Test standalone image OCR on Scenario A
    scenario_a_file = TEST_SAMPLES_DIR / "scenario_a_multi_doc_on_single_page.png"
    print(f"\n[Test 3] Processing standalone image with RapidOCR (PaddleOCR models): {scenario_a_file.name}")
    img_data = pdf_processor.process_standalone_image(scenario_a_file, doc_id="test_scen_a")
    print(f"  ✓ OCR finished with average line confidence: {img_data['confidence']}")
    print(f"  ✓ Total words extracted: {img_data['word_count']}")
    print(f"  ✓ Extracted lines count: {len(img_data['lines'])}")
    print(f"  ✓ Extracted OCR text snippet:\n    {repr(img_data['text'][:150])}...")
    
    # Verify key words extracted across the image
    extracted_lower = img_data['text'].lower()
    has_invoice = "invoice" in extracted_lower or "88201" in extracted_lower
    has_receipt = "coffee" in extracted_lower or "receipt" in extracted_lower or "bakery" in extracted_lower
    has_id = "identity" in extracted_lower or "emma" in extracted_lower or "watson" in extracted_lower

    print(f"  ✓ Found Invoice cues: {has_invoice}")
    print(f"  ✓ Found Receipt cues: {has_receipt}")
    print(f"  ✓ Found ID Card cues: {has_id}")
    assert has_invoice or has_receipt or has_id, "OCR should detect text elements"

    print("\n" + "=" * 70)
    print("PHASE 1 TESTING COMPLETED SUCCESSFULLY: ALL CHECKS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_phase1_tests()
