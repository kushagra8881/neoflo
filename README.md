# Neoflio.ai Intelligent Document Classification & Clustering Web App

An AI-powered, dataset-independent document understanding and semantic clustering web application that accepts mixed PDFs, scans, and multi-format images, segments documents across single-page and multi-page boundaries, and automatically clusters them using local OCR, dense transformer embeddings, K-Means, and Google Gemini.

---

## 🚀 Key Highlights & Solved Scenarios

| Scenario | Challenge | How It Is Solved |
| :--- | :--- | :--- |
| **Scenario A** | **Multiple documents on one page** (e.g., an invoice, receipt, and ID card scanned together) | Computer Vision Canny card boundary contour detection + aspect-ratio filtering + Non-Maximum Suppression (NMS) crops each document independently and runs OCR per crop to eliminate text contamination. |
| **Scenario B** | **Multiple documents throughout one PDF** (e.g., compound 9-page PDF with 2 invoices, KYC form, bank statement) | Evaluates adjacent page transitions using a composite multi-signal boundary equation: $$B_i = w_s(1 - S_i) + w_h H_i + w_m M_i + w_v V_i$$ to detect transition boundaries. |
| **Scenario C** | **One document spanning multiple pages** (e.g., 4-page bank statement) | Tracks semantic cosine similarity $S_i$, continuous page numbers ("Page 1 of 4", "Page 2 of 4"), and persistent account IDs to keep all pages in a unified logical document without unwanted splitting. |
| **Scenario D** | **Unseen document types** (e.g., ISO environmental compliance audit) | Zero-shot generalization via pretrained Sentence Transformers (`all-MiniLM-L6-v2`) and unsupervised K-Means, without requiring class-specific retraining. |

---

## 🏗️ Architecture & Technology Stack

1. **Fast-path Digital Extraction & Page Rasterization**: PyMuPDF (`fitz`) renders 200–300 DPI preview images and directly extracts embedded digital blocks.
2. **Local OCR Engine**: RapidOCR (PaddleOCR PP-OCRv4 ONNX model) extracts text lines, bounding boxes, and word confidences locally.
3. **Semantic Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) creates normalized 384-dimensional vectors with title/first-page weighted chunk aggregation.
4. **Unsupervised Clustering**: Scikit-Learn K-Means with automatic $K$-estimation (Silhouette score & Davies-Bouldin index) and 2D PCA projection coordinates.
5. **AI Semantic Reasoning**: Google Gemini 3.8 Flash provides structured cluster naming, business explanations, category badges, and human-in-the-loop review flags.
6. **Web Application**: Streamlit dashboard featuring 5 dedicated views:
   * **Upload & Processing**: Batch drag & drop with live progress bars.
   * **Segmentation & Boundary Review**: Page-to-page signal inspection with interactive split & merge tools.
   * **Semantic Clusters & 2D PCA**: Interactive scatter plot and cluster cards.
   * **Document Inspector**: Detailed preview of OCR text, bounding boxes, and identifiers.
   * **Export Center**: Instant downloads for CSV, structured JSON, and ZIP bundle containing segmented PDFs.

---

## 📦 Project Structure

```text
intern/
├── app.py                         # Streamlit interactive web application
├── requirements.txt               # Dependencies
├── .env                           # API keys (GEMINI_API_KEY)
├── .gitignore
├── README.md
├── pipeline/
│   ├── config.py                  # Thresholds, weights, and paths
│   ├── file_loader.py             # File validation & SHA-256 deduplication
│   ├── image_preprocessor.py      # OpenCV deskewing, CLAHE, and cropping
│   ├── ocr_engine.py              # RapidOCR / PaddleOCR wrapper & cues
│   ├── pdf_processor.py           # Digital fast-path & page rasterizer
│   ├── region_detector.py         # Scenario A: single-page multi-doc isolation
│   ├── page_embedder.py           # Page-level transformer embeddings
│   ├── boundary_detector.py       # Scenarios B & C: multi-page boundary engine
│   ├── document_assembler.py      # Logical document assembly & split/merge
│   ├── document_embedder.py       # Weighted chunking & document-level vectors
│   ├── cluster_engine.py          # K-Means, auto-K, silhouette, and 2D PCA
│   └── orchestrator.py            # End-to-end pipeline connector
├── services/
│   ├── gemini_service.py          # Structured Gemini 3.8 Flash reasoning
│   └── export_service.py          # CSV, JSON, and segmented PDF ZIP generator
├── tests/
│   ├── generate_test_data.py      # Generates synthetic data for Scenarios A, B, C, D
│   ├── test_phase1_ocr.py         # Phase 1 test suite
│   ├── test_phase2_segmentation.py# Phase 2 boundary & region test suite
│   └── test_end_to_end.py         # Full Phase 1–6 end-to-end pipeline benchmark
└── data/
    ├── uploads/                   # Staged user uploads
    ├── processed/                 # Rendered page images & crops
    └── test_samples/              # Benchmark files for Scenarios A, B, C, D
```

---

## 🛠️ How to Run

### 1. Run the Test Suites
To verify each phase and the full end-to-end pipeline:
```bash
# Set PYTHONPATH to project root
export PYTHONPATH=/home/kushagra/Documents/PROJECTS/intern
PYTHON_BIN=/home/kushagra/Documents/code/AI/.venv/bin/python

# 1. Phase 1 Test (OCR & PyMuPDF)
$PYTHON_BIN tests/test_phase1_ocr.py

# 2. Phase 2 Test (Scenario A, B, C Segmentation & Split/Merge)
$PYTHON_BIN tests/test_phase2_segmentation.py

# 3. Full End-to-End Benchmark (Phases 1 through 6)
$PYTHON_BIN tests/test_end_to_end.py
```

### 2. Launch the Streamlit Web Application
```bash
$PYTHON_BIN -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.
Click **"🚀 Load Benchmark Test Dataset (Scenarios A, B, C, D)"** to immediately run and view the entire pipeline in action!
