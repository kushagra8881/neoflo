import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_SAMPLES_DIR = DATA_DIR / "test_samples"

# Ensure runtime directories exist
for p in [UPLOAD_DIR, PROCESSED_DIR, TEST_SAMPLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Support Streamlit Cloud Secrets (st.secrets)
try:
    import streamlit as st
    if not GEMINI_API_KEY and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# Pipeline parameters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDF_RENDER_DPI = 200

# Boundary scoring weights
# Bi = ws*(1 - Si) + wh*Hi + wm*Mi + wv*Vi
BOUNDARY_WEIGHTS = {
    "semantic": 0.40,      # weight ws: drop in semantic embedding similarity
    "heading": 0.25,       # weight wh: new document title/heading structure detected
    "identifier": 0.25,    # weight wm: new ID / invoice number / account number transition
    "visual": 0.10,        # weight wv: layout transition / bounding box distribution change
}

# Thresholds
BOUNDARY_DECISION_THRESHOLD = 0.52   # Score >= threshold indicates start of a new logical document
ADJACENT_SIMILARITY_DROP_THRESHOLD = 0.60

# Region detection parameters for Scenario A (multiple documents on 1 page)
MIN_DOC_REGION_AREA_RATIO = 0.04    # At least 4% of total page area
MAX_DOC_REGION_AREA_RATIO = 0.96    # Less than 96% of page area (otherwise whole page)
REGION_PADDING_PX = 15

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
