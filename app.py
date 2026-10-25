import os
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config import TEST_SAMPLES_DIR, UPLOAD_DIR
from pipeline.document_assembler import DocumentAssembler
from services.export_service import ExportService

# Page configuration
st.set_page_config(
    page_title="Neoflio - Intelligent Document Classification",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .cluster-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    }
    .badge-good {
        background-color: #10b981;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-ambig {
        background-color: #f59e0b;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-category {
        background-color: #6366f1;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "results" not in st.session_state:
    st.session_state.results = None
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = PipelineOrchestrator()

# Sidebar
with st.sidebar:
    st.title("📑 Neoflio.ai")
    st.caption("Intelligent Document Classification & Clustering")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        [
            "📤 Upload & Processing",
            "🔍 Segmentation & Boundaries",
            "🧠 Semantic Clusters & PCA",
            "📄 Document Detail Inspector",
            "💾 Export Center"
        ]
    )

    st.markdown("---")
    st.subheader("⚙️ Pipeline Configuration")
    k_mode = st.radio("Clustering Strategy", ["Auto-estimate K (Silhouette)", "Manual K"])
    manual_k = None
    if k_mode == "Manual K":
        manual_k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=8, value=3)

    boundary_thresh = st.slider(
        "Boundary Sensitivity Threshold (Bi)",
        min_value=0.35, max_value=0.75, value=0.52, step=0.01,
        help="Higher values require stronger evidence to split documents."
    )
    st.session_state.orchestrator.boundary_detector.threshold = boundary_thresh

    st.markdown("---")
    st.caption("Powered by PaddleOCR, Sentence-Transformers, K-Means & Gemini 3.8 Flash.")

# -------------------------------------------------------------
# PAGE 1: UPLOAD & PROCESSING
# -------------------------------------------------------------
if nav_choice == "📤 Upload & Processing":
    st.title("📤 Batch Document Upload & AI Pipeline")
    st.write(
        "Upload any combination of PDFs and scanned images. The system automatically segments "
        "multiple documents per page (**Scenario A**), multi-page PDF boundaries (**Scenarios B & C**), "
        "and groups previously unseen types without dataset retraining (**Scenario D**)."
    )

    col_bench, col_upload = st.columns([1, 2])

    with col_bench:
        st.subheader("⚡ Quick Benchmark")
        st.write("Load the verified evaluation dataset containing all 4 assignment scenarios:")
        if st.button("🚀 Load Benchmark Test Dataset (Scenarios A, B, C, D)", use_container_width=True, type="primary"):
            sample_files = [
                TEST_SAMPLES_DIR / "scenario_a_multi_doc_on_single_page.png",
                TEST_SAMPLES_DIR / "scenario_b_multi_doc_compound.pdf",
                TEST_SAMPLES_DIR / "scenario_c_multipage_bank_statement.pdf",
                TEST_SAMPLES_DIR / "scenario_d_unseen_compliance_report.pdf",
            ]
            prog_bar = st.progress(0.0)
            status_text = st.empty()

            def update_ui(pct, msg):
                prog_bar.progress(pct)
                status_text.info(msg)

            with st.spinner("Processing benchmark batch..."):
                results = st.session_state.orchestrator.process_batch(
                    file_paths=sample_files,
                    manual_k=manual_k,
                    progress_cb=update_ui
                )
                st.session_state.results = results
                prog_bar.progress(1.0)
                status_text.success("Processing Complete! Navigate to other tabs to explore results.")

    with col_upload:
        st.subheader("📁 Custom Uploads")
        uploaded_files = st.file_uploader(
            "Drop PDFs or Images (JPG, PNG, TIFF)",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Uploaded Files", use_container_width=True):
            saved_paths = []
            for uf in uploaded_files:
                save_path = UPLOAD_DIR / uf.name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_paths.append(save_path)

            prog_bar = st.progress(0.0)
            status_text = st.empty()

            def update_ui(pct, msg):
                prog_bar.progress(pct)
                status_text.info(msg)

            with st.spinner(f"Processing {len(saved_paths)} files..."):
                results = st.session_state.orchestrator.process_batch(
                    file_paths=saved_paths,
                    manual_k=manual_k,
                    progress_cb=update_ui
                )
                st.session_state.results = results
                prog_bar.progress(1.0)
                status_text.success(f"Processed {len(saved_paths)} files successfully!")

    # Summary metrics if results exist
    if st.session_state.results and "error" not in st.session_state.results:
        res = st.session_state.results
        st.markdown("---")
        st.subheader("📊 Batch Extraction Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Input Files", res["total_files"])
        m2.metric("Total Pages Processed", res["total_pages"])
        m3.metric("Logical Documents Found", res["total_logical_documents"])
        m4.metric("Discovered Clusters (K)", res["cluster_results"]["k"])

# -------------------------------------------------------------
# PAGE 2: SEGMENTATION & BOUNDARY REVIEW
# -------------------------------------------------------------
elif nav_choice == "🔍 Segmentation & Boundaries":
    st.title("🔍 Document Segmentation & Boundary Review")
    st.write(
        "Inspect how multi-page PDFs and single-page scans were segmented into logical documents. "
        "Review boundary transition diagnostics or make manual adjustments (split / merge)."
    )

    if not st.session_state.results:
        st.warning("Please process a batch of documents in the 'Upload & Processing' tab first.")
    else:
        res = st.session_state.results
        tabs = st.tabs(["📄 Logical Documents Extracted", "📐 Page Transition Signals", "🛠️ Manual Boundary Editor"])

        with tabs[0]:
            st.subheader(f"Extracted Documents ({len(res['documents'])})")
            for i, doc in enumerate(res["documents"]):
                doc_type_title = getattr(doc, "document_type", "Document")
                with st.expander(f"Document {i+1} [{doc_type_title}]: {doc.source_filename} | {doc.page_range} ({doc.page_count} pages)"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if doc.preview_image_paths and Path(doc.preview_image_paths[0]).exists():
                            st.image(doc.preview_image_paths[0], caption="Page 1 Preview", use_column_width=True)
                    with c2:
                        st.markdown(f"### 🏷️ `{doc_type_title}`")
                        if getattr(doc, "type_summary", ""):
                            st.info(f"💡 **AI Summary:** {doc.type_summary}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Doc ID:** `{doc.doc_id}`")
                            st.write(f"**Source File:** `{doc.source_filename}`")
                            st.write(f"**Page Range:** {doc.page_range}")
                        with col_b:
                            st.write(f"**Category:** `{getattr(doc, 'category', 'General')}`")
                            st.write(f"**Is Sub-Page Crop:** {doc.is_crop}")
                            if doc.primary_invoice_id:
                                st.write(f"**Invoice ID:** `{doc.primary_invoice_id}`")
                            elif doc.primary_account_id:
                                st.write(f"**Account ID:** `{doc.primary_account_id}`")

                        st.text_area("Extracted Text Snippet", doc.combined_text[:500] + "...", height=140, key=f"txt_{doc.doc_id}")

        with tabs[1]:
            st.subheader("Multi-Signal Transition Diagnostics")
            for fname, transitions in res.get("transitions_by_file", {}).items():
                st.markdown(f"#### File: `{fname}`")
                if not transitions:
                    st.write("Single-page file (no transitions).")
                    continue

                t_df = pd.DataFrame([
                    {
                        "Transition": f"Page {t['from_page']} ➔ Page {t['to_page']}",
                        "Semantic Sim (Si)": t["similarity"],
                        "Heading Score (Hi)": t["heading_score"],
                        "Metadata Score (Mi)": t["metadata_score"],
                        "Boundary Score (Bi)": t["boundary_score"],
                        "Decision": "✂️ NEW DOCUMENT" if t["is_boundary"] else "🔗 CONTINUATION",
                        "Rationale": t["rationale"]
                    }
                    for t in transitions
                ])
                st.dataframe(t_df, use_container_width=True)

        with tabs[2]:
            st.subheader("Manual Review & Adjustment (Human-in-the-Loop)")
            col_split, col_merge = st.columns(2)

            with col_split:
                st.markdown("##### ✂️ Split a Document")
                multi_page_docs = [d for d in res["documents"] if d.page_count > 1]
                if multi_page_docs:
                    split_target_id = st.selectbox(
                        "Select multi-page document to split",
                        [d.doc_id for d in multi_page_docs],
                        key="split_sel"
                    )
                    target_doc = next(d for d in multi_page_docs if d.doc_id == split_target_id)
                    split_page = st.number_input(
                        "Split at page number",
                        min_value=target_doc.pages[1]["page_number"],
                        max_value=target_doc.pages[-1]["page_number"],
                        value=target_doc.pages[1]["page_number"]
                    )
                    if st.button("Apply Split"):
                        new_parts = DocumentAssembler.manual_split(target_doc, split_at_page_num=int(split_page))
                        # Update documents in session
                        idx = res["documents"].index(target_doc)
                        res["documents"].pop(idx)
                        for part in reversed(new_parts):
                            res["documents"].insert(idx, part)
                        st.success(f"Split document into {len(new_parts)} segments! Re-run clustering to update.")
                        st.rerun()
                else:
                    st.info("No multi-page documents available to split.")

            with col_merge:
                st.markdown("##### 🔗 Merge Two Documents")
                if len(res["documents"]) >= 2:
                    doc_ids = [d.doc_id for d in res["documents"]]
                    m_doc1_id = st.selectbox("First Document", doc_ids, index=0, key="m1")
                    m_doc2_id = st.selectbox("Second Document", doc_ids, index=min(1, len(doc_ids)-1), key="m2")

                    if m_doc1_id != m_doc2_id and st.button("Merge Selected Documents"):
                        d1 = next(d for d in res["documents"] if d.doc_id == m_doc1_id)
                        d2 = next(d for d in res["documents"] if d.doc_id == m_doc2_id)
                        merged = DocumentAssembler.manual_merge(d1, d2)
                        res["documents"].remove(d1)
                        res["documents"].remove(d2)
                        res["documents"].append(merged)
                        st.success(f"Merged into `{merged.doc_id}`! Re-run clustering to update.")
                        st.rerun()

# -------------------------------------------------------------
# PAGE 3: CLUSTER VISUALIZATION & 2D PCA
# -------------------------------------------------------------
elif nav_choice == "🧠 Semantic Clusters & PCA":
    st.title("🧠 Semantic Clusters & Vector Space")
    st.write(
        "Unsupervised K-Means clustering of 384-dimensional dense transformer embeddings. "
        "Each cluster's title and description are generated by Gemini API reasoning."
    )

    if not st.session_state.results:
        st.warning("Please process a batch of documents in the 'Upload & Processing' tab first.")
    else:
        res = st.session_state.results
        cr = res["cluster_results"]
        gemini = res["gemini_labels"]
        pca_coords = cr.get("pca_coords", [])
        labels = cr.get("labels", [])

        # Cluster Quality Banner
        q = cr.get("quality", {})
        st.info(
            f"**Clustering Health:** Status = `{q.get('status', 'N/A')}` | "
            f"Silhouette Score = `{q.get('silhouette_score', 'N/A')}` | "
            f"Davies-Bouldin Index = `{q.get('davies_bouldin_score', 'N/A')}`"
        )

        # 2D PCA Scatter Plot
        st.subheader("2D Semantic Vector Projection (PCA)")
        if len(pca_coords) >= 2:
            plot_df = pd.DataFrame({
                "PCA Dim 1 (x)": [pt[0] for pt in pca_coords],
                "PCA Dim 2 (y)": [pt[1] for pt in pca_coords],
                "Cluster": [f"Cluster {l}: {gemini.get(l, {}).get('label', 'Unknown')}" for l in labels],
                "Document": [f"{d.doc_id} ({d.page_range})" for d in res["documents"]],
                "Distance": cr.get("distances", [0.0]*len(labels))
            })
            st.scatter_chart(plot_df, x="PCA Dim 1 (x)", y="PCA Dim 2 (y)", color="Cluster", size="Distance")

        # Cluster Cards
        st.subheader(f"Discovered Clusters ({cr['k']})")
        for c_id, c_data in cr.get("clusters", {}).items():
            c_meta = gemini.get(c_id, {})
            label = c_meta.get("label", f"Cluster {c_id}")
            desc = c_meta.get("description", "No description generated.")
            cat = c_meta.get("document_category", "General")
            conf = c_meta.get("confidence", "medium")

            with st.container():
                st.markdown(f"""
                <div class="cluster-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0; color:#60a5fa;">Cluster {c_id}: {label}</h3>
                        <div>
                            <span class="badge-category">{cat}</span>
                            <span class="badge-good">{c_data['size']} Documents</span>
                        </div>
                    </div>
                    <p style="margin: 8px 0; color:#cbd5e1; font-size:14px;">{desc}</p>
                    <small style="color:#94a3b8;">Gemini Confidence: <b>{conf}</b> | Avg Centroid Distance: <b>{c_data['avg_distance']}</b></small>
                </div>
                """, unsafe_allow_html=True)

                # Documents belonging to this cluster
                doc_items = []
                for idx in c_data["indices"]:
                    d = res["documents"][idx]
                    dist = cr["distances"][idx]
                    ambig = cr["is_ambiguous"][idx]
                    flag = " ⚠️ [AMBIGUOUS / REVIEW]" if ambig else " ✅ Good"
                    doc_items.append({
                        "Document ID": d.doc_id,
                        "Document Type": getattr(d, "document_type", "General"),
                        "Category": getattr(d, "category", "General"),
                        "Source File": d.source_filename,
                        "Page Range": d.page_range,
                        "Distance to Centroid": dist,
                        "Status": flag
                    })
                st.table(pd.DataFrame(doc_items))

# -------------------------------------------------------------
# PAGE 4: DOCUMENT DETAIL INSPECTOR
# -------------------------------------------------------------
elif nav_choice == "📄 Document Detail Inspector":
    st.title("📄 Document Detail Inspector")

    if not st.session_state.results:
        st.warning("Please process a batch of documents in the 'Upload & Processing' tab first.")
    else:
        res = st.session_state.results
        docs = res["documents"]
        doc_options = [
            f"{i+1}. [{getattr(d, 'document_type', 'Doc')}] {d.source_filename} ({d.page_range})" 
            for i, d in enumerate(docs)
        ]
        selected_option = st.selectbox("Select document to inspect", doc_options)
        sel_idx = doc_options.index(selected_option)
        doc = docs[sel_idx]

        c_id = res["cluster_results"]["labels"][sel_idx]
        c_meta = res["gemini_labels"].get(c_id, {})
        dist = res["cluster_results"]["distances"][sel_idx]
        is_ambig = res["cluster_results"]["is_ambiguous"][sel_idx]
        doc_type = getattr(doc, "document_type", "General Document")
        category = getattr(doc, "category", "General")
        summary = getattr(doc, "type_summary", "")
        key_fields = getattr(doc, "key_fields", {})

        # Top Highlight Card
        st.markdown(f"""
        <div style="background:#1e293b; padding:18px; border-radius:12px; margin-bottom:20px; border-left: 6px solid #10b981;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="margin:0; color:#ffffff;">🏷️ {doc_type}</h2>
                <span class="badge-category">{category}</span>
            </div>
            <p style="margin:8px 0; color:#cbd5e1; font-size:15px;">💡 {summary or 'Identified by AI reasoning.'}</p>
        </div>
        """, unsafe_allow_html=True)

        col_img, col_info = st.columns([1, 1])

        with col_img:
            st.subheader("Document Preview")
            for p_img in doc.preview_image_paths:
                if Path(p_img).exists():
                    st.image(p_img, caption=Path(p_img).name, use_column_width=True)

        with col_info:
            st.subheader("Metadata & Classification")
            st.write(f"**Document ID:** `{doc.doc_id}`")
            st.write(f"**Source File:** `{doc.source_filename}`")
            st.write(f"**Page Range:** {doc.page_range} ({doc.page_count} pages)")
            st.write(f"**Assigned Semantic Cluster:** Cluster {c_id} ({c_meta.get('label')})")
            st.write(f"**Distance to Centroid:** `{dist:.4f}`")
            if is_ambig:
                st.warning("⚠️ Flagged for Human Review (borderline distance or competing cluster proximity).")
            else:
                st.success("✅ High confidence assignment.")

            if key_fields:
                st.markdown("##### 🔑 Key Extracted Fields")
                kf_data = []
                for k, v in key_fields.items():
                    if v:
                        label_name = k.replace('_', ' ').title()
                        kf_data.append({"Field": label_name, "Value": str(v)})
                if kf_data:
                    st.table(pd.DataFrame(kf_data))

            st.write(f"**Invoice ID:** `{doc.primary_invoice_id or 'None'}`")
            st.write(f"**Account ID:** `{doc.primary_account_id or 'None'}`")
            st.write(f"**Word Count:** {doc.word_count}")

            st.subheader("Extracted OCR Text")
            st.text_area("Full Text", doc.combined_text, height=280)

# -------------------------------------------------------------
# PAGE 5: EXPORT CENTER
# -------------------------------------------------------------
elif nav_choice == "💾 Export Center":
    st.title("💾 Export Center")
    st.write("Download processing outputs, metadata, cluster catalogs, and segmented PDFs.")

    if not st.session_state.results:
        st.warning("Please process a batch of documents in the 'Upload & Processing' tab first.")
    else:
        res = st.session_state.results
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("📊 CSV Summary")
            st.write("Tabular breakdown of documents, clusters, distances, and identifiers.")
            csv_str = ExportService.generate_csv(
                res["documents"], res["cluster_results"], res["gemini_labels"]
            )
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_str,
                file_name="neoflio_document_clusters.csv",
                mime="text/csv",
                use_container_width=True
            )

        with c2:
            st.subheader("🧬 Full JSON Tree")
            st.write("Complete structured JSON containing document boundaries, embeddings, and Gemini metadata.")
            json_str = ExportService.generate_json(
                res["documents"], res["cluster_results"], res["gemini_labels"], res["transitions_by_file"]
            )
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name="neoflio_classification_results.json",
                mime="application/json",
                use_container_width=True
            )

        with c3:
            st.subheader("📦 Segmented ZIP Bundle")
            st.write("ZIP archive containing the CSV, JSON, and each segmented document exported as its own PDF.")
            if st.button("Generate ZIP Bundle", use_container_width=True):
                with st.spinner("Extracting individual PDFs and packaging ZIP archive..."):
                    zip_bytes = ExportService.create_export_zip(
                        res["documents"], res["cluster_results"], res["gemini_labels"], res["transitions_by_file"]
                    )
                    st.download_button(
                        label="⬇️ Download Complete ZIP Package",
                        data=zip_bytes,
                        file_name="neoflio_segmented_documents_bundle.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
