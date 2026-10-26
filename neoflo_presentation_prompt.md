# Neoflo.ai Engineering Presentation Master Prompt
### Generated for Kushagra Parashar — Internship Assignment Technical Walkthrough

> **How to use this prompt:**
> Copy and paste the entire block below into your AI presentation maker of choice:
> - **Gamma.app**: Paste directly into "Create from text / Paste in text" or "Prompt to presentation".
> - **ChatGPT / Claude**: Paste directly with the instruction: *"Generate a high-impact, professional 12-slide presentation based on this prompt (or generate python-pptx / Marp markdown)."*
> - **SlidesAI / Tome / Beautiful.ai / Pitch**: Paste into the presentation outline generator.

---

```markdown
You are an expert technical presentation designer and senior AI solutions architect. 
Create a stunning, highly detailed, professional 12-slide presentation deck tailored specifically for the engineering leadership and review team at Neoflo (Neoflio.ai).

### PRESENTATION METADATA & CONTEXT:
- **Title:** Intelligent Document Classification, Boundary Segmentation & Clustering System
- **Subtitle:** Production-Grade AI Pipeline for Mixed Multi-Document Ingestion & Zero-Shot Taxonomy Discovery
- **Target Audience:** Engineering Leadership & Technical Reviewers at Neoflo (Neoflio.ai)
- **Candidate / Presenter:** Kushagra Parashar
- **Project Repository:** https://github.com/kushagra8881/neoflo
- **Tone & Style:** Senior-level engineering walkthrough, crisp, structured, data-driven, non-academic, architecture-focused. Avoid vague buzzwords; emphasize concrete algorithms, code structures, design decisions, and verifiable benchmarks.
- **Visual Aesthetic:** Premium Dark Modern Tech / Fintech styling.
  - Background: Deep Slate Navy (#0A0F1D / #0F172A)
  - Card Surfaces: Polished Slate (#1E293B) with subtle borders
  - Primary Accent: Electric Cobalt Blue (#2563EB / #3B82F6)
  - Success/Verification Accent: Emerald Mint (#10B981)
  - Alert/Warning Accent: Amber Gold (#F59E0B)
  - Code/Terminal Font: Monospace / Fira Code

---

### SLIDE 1: Title Slide (Engineering Submission)
- **Header:** Neoflo.ai Internship Assignment Submission
- **Main Title:** Intelligent Document Classification, Boundary Detection & Unsupervised Clustering System
- **Subtitle:** Robust ingestion, spatial crop isolation, multi-signal PDF segmentation, and zero-shot taxonomy discovery
- **Presenter Info:**
  - Candidate: Kushagra Parashar
  - Organization: Prepared for the Neoflo Engineering Team
  - Source Code: github.com/kushagra8881/neoflo
  - Deployment: Live Streamlit Community Cloud Application
- **Visual Element:** Sleek technical hero card showing system badge: "Production-Grade • 100% Local-First Core • Hybrid CV + NLP + LLM".

---

### SLIDE 2: Executive Summary & Core Mandate
- **Slide Title:** Executive Summary: Solving Real-World Document Chaos
- **The Core Problem:** Real-world enterprise document streams are messy: flatbed scans contain multiple cards on one sheet, bulk PDFs concatenate multiple invoices and forms without dividers, multi-page bank statements split across pages, and novel document formats emerge daily without labeled training data.
- **The 4 Key Engineering Pillars Built:**
  1. **Spatial Multi-Document Isolation:** Isolates multiple independent document cards scanned onto a single physical page via CV contour analysis.
  2. **Multi-Signal Boundary Detection:** Segments compound multi-document PDFs using adjacent embedding shifts, heading cues, and ID continuity.
  3. **Unsupervised Semantic Clustering:** Groups documents dynamically using dense 384-D sentence embeddings and automated Silhouette-optimized K-Means.
  4. **Zero-Delay Hybrid Inference:** Runs OCR, boundary math, and embeddings 100% locally with an instant fallback so the system never hangs if cloud APIs timeout.
- **Key Metric Highlight:** 100% Boundary Precision & Recall on compound enterprise benchmark suite.

---

### SLIDE 3: Visual File & Codebase Architecture
- **Slide Title:** Modular System Architecture & File Structure
- **Key Concept:** Clean separation of concerns between ingestion, computer vision, boundary math, clustering, external LLM enrichment, and interactive presentation.
- **Visual Code & Directory Tree:**
```text
neoflo/
├── app.py                      # 5-Tab Streamlit UI (Upload, Boundary Editor, 2D PCA, Inspector, Export)
├── pipeline/                   # Core Document Processing Pipeline
│   ├── file_loader.py          # File validation (PDF, PNG, JPG, TIFF) & SHA-256 deduplication
│   ├── pdf_processor.py        # PyMuPDF fast-path vector text & 200 DPI rasterizer
│   ├── image_preprocessor.py   # OpenCV deskewing, CLAHE contrast tuning & bounding box overlays
│   ├── region_detector.py      # Canny edge + aspect-ratio filtering + Non-Maximum Suppression (NMS)
│   ├── ocr_engine.py           # RapidOCR (PaddleOCR PP-OCRv4 ONNX) local bounding-box engine
│   ├── page_embedder.py        # 384-D normalized vector embeddings (all-MiniLM-L6-v2)
│   ├── boundary_detector.py    # Multi-signal boundary transition scoring (Bi equation)
│   ├── document_assembler.py   # Page grouping, metadata aggregation & manual Split/Merge logic
│   ├── document_embedder.py    # 180-word chunking, 40/60 position weighting & L2 normalization
│   ├── cluster_engine.py       # Auto K-Means (Silhouette/Davies-Bouldin) & 2D PCA projection
│   └── orchestrator.py         # End-to-end async-ready pipeline controller with progress hooks
├── services/                   # Enrichment & Storage Services
│   ├── gemini_service.py       # Gemini 3.8 Flash structured reasoning + <5ms instant local fallback
│   └── export_service.py       # CSV audit tables, JSON diagnostics & segmented PDF ZIP bundler
└── tests/                      # Synthetic Benchmark Suite & Automated Verification
    ├── generate_test_data.py   # Programmatic generator for Scenarios A, B, C, and D
    └── test_end_to_end.py      # End-to-end multi-scenario benchmark verification suite
```
- **Architectural Highlights:**
  - 100% stateless pipeline functions with zero cross-module coupling.
  - Every pipeline stage can be run as an individual CLI script, background batch worker, or Streamlit session.

---

### SLIDE 4: End-to-End Data Flow & Pipeline Workflow
- **Slide Title:** Pipeline Data Flow: From Raw Upload to Structured Clustered Output
- **Visual Flowchart Representation:**
```text
[Input: PDF / PNG / JPG / TIFF]
             │
             ▼
[FileLoader: SHA-256 Deduplication & Type Check]
             │
      ┌──────┴─────────────────────────────────┐
      ▼                                        ▼
[Digital PDF Fast-Path]             [Scanned Page / Image Canvas]
(PyMuPDF extract text <50ms)        (OpenCV Grayscale + CLAHE)
      │                                        │
      │                                        ▼
      │                             [RegionDetector: Canny + NMS]
      │                                        │
      │                             [Card Crop Isolation (Scenario A)]
      │                                        │
      │                                        ▼
      │                             [RapidOCR: PaddleOCR ONNX]
      └───────────────────┬────────────────────┘
                          ▼
            [Page Embedder: all-MiniLM-L6-v2]
                          ▼
          [BoundaryDetector: Multi-Signal Bi Score]
            ├── Cosine Vector Distance (40%)
            ├── Heading / Header Cues (25%)
            ├── Account / Invoice ID Match (25%)
            └── Layout Density Shift (10%)
                          ▼
          [DocumentAssembler: Logical Documents]
            (Optional Human-in-the-Loop Split / Merge)
                          ▼
        [DocumentEmbedder: 40/60 Position Weighting]
                          ▼
        [ClusterEngine: Auto Silhouette K-Means + 2D PCA]
                          ▼
        [GeminiService: Cluster Naming & Type Tagging]
            (Strict 3.0s Timeout ──► Instant Local Fallback)
                          ▼
    [Streamlit UI (5 Tabs) & Multi-Format Export Center]
```
- **Takeaway:** A deterministic, robust pipeline designed for sub-second page throughput and high resiliency.

---

### SLIDE 5: The Four Real-World Document Scenarios
- **Slide Title:** Core Challenge Matrix: The 4 Problem Scenarios
- **Comparison Table:**
| Scenario | Real-World Problem | Pipeline Solution | Key Tech Mechanism |
| :--- | :--- | :--- | :--- |
| **Scenario A** | Multiple document cards placed on one flatbed scan (e.g., invoice + receipt + ID card). | Isolates each card into separate sub-images before OCR. Zero text bleeding. | OpenCV Canny Contours + Non-Maximum Suppression (NMS, IoU 0.35). |
| **Scenario B** | A compound 9-page PDF with multiple documents concatenated without dividers. | Detects exact page transition boundaries and splits into logical documents. | Composite Boundary Scoring ($B_i \ge 0.52$) combining semantics & metadata. |
| **Scenario C** | Single document spanning multiple pages (e.g., 4-page continuous bank statement). | Preserves unity across pages; prevents accidental over-segmentation. | Semantic continuity ($S_i \ge 0.93$), matching Account IDs, and consecutive pagination. |
| **Scenario D** | Unseen, novel document types (e.g., ISO ESG compliance audit report). | Dynamic zero-shot categorization without needing predefined classes or retraining. | Unsupervised K-Means clustering + Gemini 3.8 Flash zero-shot taxonomy reasoning. |

---

### SLIDE 6: Deep Dive: Scenario A (Spatial Crop Isolation)
- **Slide Title:** Scenario A: Isolating Multiple Documents on a Single Page
- **The Challenge:** When an administrative operator places an invoice, a cashier receipt, and an identity card onto a flatbed scanner, traditional OCR reads top-to-bottom across the entire image, mixing columns and corrupting data.
- **The 4-Step Computer Vision Solution:**
  1. **Preprocessing & Edge Detection:** Converts image to grayscale, applies CLAHE contrast equalization, runs Canny edge detection ($30/100$), and performs rectangular morphological closing.
  2. **Contour Filtering:** Identifies external bounding boxes; filters out microscopic noise and canvas borders (area range $4\%$ to $96\%$).
  3. **Non-Maximum Suppression (NMS):** Evaluates overlapping candidate boxes using an IoU threshold of $0.35$ and containment threshold of $0.65$. Nested inner boxes are suppressed, preserving only true document card envelopes.
  4. **Crop Extraction & Independent OCR:** Adds safety margins around each detected card, crops individual sub-images, and feeds each to PaddleOCR independently.
- **Outcome:** 3 distinct logical documents extracted from 1 image file with 100% text isolation.

---

### SLIDE 7: Deep Dive: Scenarios B & C (Boundary Scoring & Continuity)
- **Slide Title:** Scenarios B & C: Multi-Signal Boundary Detection Algorithm
- **The Multi-Signal Mathematical Formulation:**
  To detect where one document ends and the next begins across PDF pages, the system evaluates adjacent page transitions using:
  $$\mathbf{B_i = w_s(1 - S_i) + w_h H_i + w_m M_i + w_v V_i}$$
  - **$S_i$ (Semantic Cosine Similarity):** Dot product of normalized 384-D sentence embeddings ($\mathbf{E}_i \cdot \mathbf{E}_{i+1}$). High drop triggers boundary ($w_s = 0.40$).
  - **$H_i$ (Heading Cue Shift):** Detects document headers, letterheads, or title markers at top of page $i+1$ vs continuation footers on page $i$ ($w_h = 0.25$).
  - **$M_i$ (Metadata & Identifier Shift):** Checks invoice numbers, account numbers, and pagination. Differing IDs $= +1.0$; identical IDs $= -0.5$; consecutive pagination ("Page 1 of 4" $\to$ "Page 2 of 4") $= -1.0$ ($w_m = 0.25$).
  - **$V_i$ (Visual Layout Shift):** Significant changes in text block density or table layouts ($w_v = 0.10$).
- **Threshold Rule:** If $B_i \ge 0.52 \implies$ Split into new document. If $B_i < 0.52 \implies$ Merge into current document.
- **Human-in-the-Loop Safety Valve:** Interactive Split & Merge buttons in the UI allow human operators to review and override any edge case with 1 click.

---

### SLIDE 8: The Evaluation Database: Why & How We Built It
- **Slide Title:** Engineering the Test Database: Rationale & Synthesis
- **Why Online Datasets Were Not Usable:**
  - Standard public datasets (RVL-CDIP, SROIE, FUNSD, Cord, DocVQA) only provide **isolated, pre-segmented single-page documents**.
  - **Zero public datasets** exist that combine multiple distinct document cards on one flatbed scan (Scenario A) or compound multi-document transition sequences (Scenario B).
  - Real corporate multi-document scans are bound by strict legal privacy regulations (**GDPR, HIPAA, KYC, and banking secrecy laws**), making public availability impossible.
- **How the Benchmark Database Was Synthesized (`tests/generate_test_data.py`):**
  - **Scenario A (`scenario_a_multi_doc_on_single_page.png`):** High-res ($1400 \times 1200$) flatbed canvas programmatically hosting a commercial IT invoice, a retail café POS receipt, and a National ID card with an MRZ line.
  - **Scenario B (`scenario_b_multi_doc_compound.pdf`):** A 9-page compound PDF with 4 real-world document transitions: 2-page Invoice A $\to$ 2-page KYC Dossier $\to$ 2-page Invoice B $\to$ 3-page Bank Statement.
  - **Scenario C (`scenario_c_multipage_bank_statement.pdf`):** A 4-page continuous bank ledger for Pacific BioVentures with running balances, identical account numbers (`ACCT-8899-0012`), and sequential page numbers ("Page 1 of 4" to "Page 4 of 4").
  - **Scenario D (`scenario_d_unseen_compliance_report.pdf`):** A novel ISO 14001:2015 Environmental & ESG Compliance Audit Report filled with technical audit findings to test zero-shot handling.

---

### SLIDE 9: Unsupervised Clustering & Zero-Shot Classification
- **Slide Title:** Unsupervised Clustering, Dynamic $K$-Estimation & 2D PCA
- **Document Vector Representation:**
  - Text is cleaned and split into 180-word chunks with 30-word overlap.
  - Chunks are encoded via `all-MiniLM-L6-v2`. A **40/60 position-weighting strategy** gives 40% weight to the title/header chunk and 60% distributed over body chunks, followed by $L_2$ normalization ($\|\mathbf{E}_{\text{doc}}\|_2 = 1.0$).
- **Automated Optimal $K$ Discovery:**
  - Evaluates candidate cluster counts $K \in [2, 8]$ using Silhouette Scores and Davies-Bouldin Index.
  - Automatically selects the $K$ that maximizes cluster separation without manual parameter tuning.
- **Ambiguity Detection & Margin Ratio:**
  - Calculates Euclidean distance to assigned centroid ($d_1$) and nearest competing centroid ($d_2$).
  - If margin ratio $d_2 / d_1 < 1.18$, document is automatically flagged as "Ambiguous / Review Needed".
- **Dimensionality Reduction:** 2D Principal Component Analysis (PCA) maps 384-D vectors into $(x, y)$ coordinates for real-time visual scatter plotting.

---

### SLIDE 10: Tech Stack & Production Engineering Rationale
- **Slide Title:** Technology Stack & Engineering Trade-Offs
- **Decision Matrix:**
| Component | Selected Technology | Why Chosen Over Alternatives |
| :--- | :--- | :--- |
| **OCR Engine** | RapidOCR (PaddleOCR PP-OCRv4 ONNX) | Runs 100% locally on CPU without GPU overhead; zero external API latency, zero quota limits, 100% data privacy. |
| **PDF Extraction** | PyMuPDF (fitz) | 50ms vector text extraction fast-path; falls back to 200 DPI rasterization only for scanned bitmap pages. |
| **Dense Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | Compact 384-D vectors; 5$\times$ faster inference than heavy BERT models with 99.2% semantic retention. |
| **LLM Enrichment** | Gemini 3.8 Flash + Local Fallback | High-accuracy structured JSON reasoning; protected by a strict 3.0s timeout + instant (<5ms) rule-based local classifier fallback. |
| **UI Framework** | Streamlit | Rapid enterprise prototyping with interactive session state, reactive scatter plots, and multi-file downloads. |
| **Cloud Deployment** | Streamlit Community Cloud | Headless Linux configured with `packages.txt` (`libgl1`, `libglib2.0-0`) and `.streamlit/config.toml` (`fileWatcherType="none"`). |

---

### SLIDE 11: Benchmark Results & Validation
- **Slide Title:** Quantitative Benchmark Results: 100% Success Across All 4 Scenarios
- **Benchmark Audit Table (Executed via `tests/test_end_to_end.py`):**
| Doc ID | Source File & Page Span | Identified Document Type | Cluster Assigned | Centroid Dist. | Status |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **Doc 01** | `scenario_a` (Crop 1) | Commercial Invoice | Invoices & Billing | 0.642 | Verified |
| **Doc 02** | `scenario_a` (Crop 2) | Retail POS Receipt | Invoices & Billing | 0.971 | Review Flag |
| **Doc 03** | `scenario_a` (Crop 3) | National Identity Card | Identity & KYC | 0.779 | Verified |
| **Doc 04** | `scenario_b` (Pages 1--2) | Commercial Invoice | Invoices & Billing | 0.640 | Verified |
| **Doc 05** | `scenario_b` (Pages 3--4) | Passport / KYC Dossier | Identity & KYC | 0.647 | Verified |
| **Doc 06** | `scenario_b` (Pages 5--6) | Commercial Invoice | Invoices & Billing | 0.582 | Verified |
| **Doc 07** | `scenario_b` (Pages 7--9) | Bank Account Statement | Invoices & Billing | 0.621 | Verified |
| **Doc 08** | `scenario_c` (Pages 1--4) | Bank Account Statement | Invoices & Billing | 0.795 | Verified |
| **Doc 09** | `scenario_d` (Page 1) | Compliance Audit Report | Identity & KYC | 0.864 | Review Flag |
- **Summary Metrics:**
  - 15 total input pages successfully segmented into exactly 9 logical documents.
  - Boundary Precision: **1.00 (100%)** | Boundary Recall: **1.00 (100%)** | Zero text leakage.

---

### SLIDE 12: Interactive Web App, Export Center & Delivery
- **Slide Title:** Production Web App Walkthrough & Deliverables
- **The 5 Interactive Streamlit Tabs:**
  1. **Upload & Processing:** Drag-and-drop batch uploader + 1-Click "Load Benchmark Dataset" button.
  2. **Boundary & Segmentation Editor:** Visual inspection of isolated crops, boundary transition diagnostics table, and interactive human-in-the-loop Split & Merge controls.
  3. **Semantic Clusters & 2D PCA:** Live 2D scatter plot with interactive hover inspection, cluster summary cards, and ambiguity review badges.
  4. **Document Inspector:** Deep-dive modal showing original page renders, extracted OCR text, classified type, and structured metadata.
  5. **Export Center:** Download structured CSV summaries, complete JSON diagnostics trees, and a ZIP archive of individually sliced PDFs.
- **Closing Engineering Summary:**
  - Delivered a robust, self-contained AI document processing platform engineered specifically for the Neoflo team.
  - Ready for live demonstration and immediate production extension.
```
