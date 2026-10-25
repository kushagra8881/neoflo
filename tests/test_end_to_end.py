import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config import TEST_SAMPLES_DIR, PROCESSED_DIR
from services.export_service import ExportService

def run_end_to_end_test():
    print("=" * 80)
    print("END-TO-END PIPELINE BENCHMARK: Phases 1 through 6")
    print("Testing Scenarios A, B, C, D in a Single Unified Multi-Format Batch")
    print("=" * 80)

    # Gather test samples
    test_files = [
        TEST_SAMPLES_DIR / "scenario_a_multi_doc_on_single_page.png",
        TEST_SAMPLES_DIR / "scenario_b_multi_doc_compound.pdf",
        TEST_SAMPLES_DIR / "scenario_c_multipage_bank_statement.pdf",
        TEST_SAMPLES_DIR / "scenario_d_unseen_compliance_report.pdf",
    ]

    orchestrator = PipelineOrchestrator()

    def progress_logger(fraction, message):
        bar_len = 25
        filled = int(bar_len * fraction)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"[{bar}] {int(fraction * 100):3d}% - {message}")

    print("\n--- LAUNCHING BATCH PIPELINE ---")
    results = orchestrator.process_batch(
        file_paths=test_files,
        manual_k=None,  # Test automatic K estimation
        progress_cb=progress_logger
    )

    print("\n" + "=" * 80)
    print("BATCH PROCESSING RESULTS SUMMARY")
    print("=" * 80)
    print(f"• Input Files:              {results['total_files']}")
    print(f"• Total Pages Processed:    {results['total_pages']}")
    print(f"• Logical Documents Found:  {results['total_logical_documents']}")
    print(f"• Embedding Matrix Shape:   {results['embedding_matrix'].shape}")
    print(f"• Discovered Clusters (K):  {results['cluster_results']['k']}")
    print(f"• Cluster Quality Status:   {results['cluster_results']['quality']['status']} (Silhouette: {results['cluster_results']['quality']['silhouette_score']})")

    # Verify document count (3 from A + 4 from B + 1 from C + 1 from D = 9)
    assert results['total_logical_documents'] == 9, f"Expected 9 logical documents, got {results['total_logical_documents']}"
    assert results['embedding_matrix'].shape == (9, 384), f"Expected (9, 384) embedding matrix, got {results['embedding_matrix'].shape}"

    print("\n--- CLUSTERS DISCOVERED & GEMINI REASONING ---")
    for c_id, c_info in results['gemini_labels'].items():
        doc_count = results['cluster_results']['clusters'][c_id]['size']
        print(f"\n[Cluster {c_id}] '{c_info.get('label')}' ({doc_count} documents)")
        print(f"  Category:    {c_info.get('document_category')}")
        print(f"  Description: {c_info.get('description')}")
        print(f"  Confidence:  {c_info.get('confidence')}")

    print("\n--- DOCUMENT ASSIGNMENTS & METADATA ---")
    labels = results['cluster_results']['labels']
    distances = results['cluster_results']['distances']
    ambiguities = results['cluster_results']['is_ambiguous']

    for i, doc in enumerate(results['documents']):
        c_id = labels[i]
        c_label = results['gemini_labels'][c_id].get('label', f'Cluster {c_id}')
        ambig_str = " [FLAGGED FOR REVIEW]" if ambiguities[i] else ""
        doc_type = getattr(doc, "document_type", "General")
        category = getattr(doc, "category", "General")
        print(f"  Doc {i+1:02d} [{doc.doc_id}]:")
        print(f"       Type:      🏷️ {doc_type} ({category})")
        print(f"       File/Span: {doc.source_filename} | {doc.page_range}")
        print(f"       Cluster:   {c_label} (dist={distances[i]:.3f}){ambig_str}")
        if getattr(doc, "type_summary", ""):
            print(f"       Summary:   \"{doc.type_summary}\"")

    # Test Phase 6: Exports
    print("\n--- TESTING EXPORT SERVICES (CSV, JSON, ZIP) ---")
    csv_content = ExportService.generate_csv(
        results['documents'], 
        results['cluster_results'], 
        results['gemini_labels']
    )
    assert len(csv_content) > 200, "CSV content should not be empty"
    print(f"  ✓ CSV Export generated ({len(csv_content.splitlines())} rows)")

    json_content = ExportService.generate_json(
        results['documents'],
        results['cluster_results'],
        results['gemini_labels'],
        results['transitions_by_file']
    )
    assert len(json_content) > 500, "JSON content should not be empty"
    print(f"  ✓ JSON Export generated ({len(json_content)} bytes)")

    zip_bytes = ExportService.create_export_zip(
        results['documents'],
        results['cluster_results'],
        results['gemini_labels'],
        results['transitions_by_file']
    )
    assert len(zip_bytes) > 1000, "ZIP bundle should contain data"
    print(f"  ✓ Complete ZIP Bundle generated ({len(zip_bytes)} bytes)")

    print("\n" + "=" * 80)
    print("ALL PHASES (1 TO 6) VERIFIED AND PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_end_to_end_test()
