import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from pipeline.config import TEST_SAMPLES_DIR

def create_scenario_a_image() -> Path:
    """
    Scenario A: Multiple documents arranged on a single scanned page.
    Generates a 1400x1200 canvas with:
      - Document 1: Invoice Card (top-left)
      - Document 2: Store Receipt Card (top-right)
      - Document 3: Identification Card (bottom)
    """
    width, height = 1400, 1200
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 245  # Off-white scanner background

    # 1. Invoice Card (top-left: 60, 60, 680, 520)
    cv2.rectangle(canvas, (60, 60), (680, 520), (255, 255, 255), -1)
    cv2.rectangle(canvas, (60, 60), (680, 520), (180, 180, 180), 2)
    
    cv2.putText(canvas, "COMMERCIAL INVOICE", (90, 110), cv2.FONT_HERSHEY_DUPLEX, 0.85, (20, 20, 20), 2)
    cv2.putText(canvas, "Invoice Number: INV-88201", (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    cv2.putText(canvas, "Billed To: Horizon Logistics Ltd", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(canvas, "Date: 12-OCT-2026", (90, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.line(canvas, (90, 230), (650, 230), (200, 200, 200), 1)
    cv2.putText(canvas, "Item: Cloud Infrastructure Consulting", (90, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(canvas, "Qty: 40 hrs | Rate: $120.00 | Total: $4,800.00", (90, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.putText(canvas, "Item: Server Security Hardening", (90, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(canvas, "Qty: 10 hrs | Rate: $150.00 | Total: $1,500.00", (90, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.line(canvas, (90, 420), (650, 420), (200, 200, 200), 1)
    cv2.putText(canvas, "TOTAL AMOUNT DUE: $6,300.00 USD", (90, 460), cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 140), 2)
    cv2.putText(canvas, "Payment Due: Net 30 Days", (90, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)

    # 2. Receipt Card (top-right: 760, 60, 1340, 520)
    cv2.rectangle(canvas, (760, 60), (1340, 520), (255, 255, 255), -1)
    cv2.rectangle(canvas, (760, 60), (1340, 520), (180, 180, 180), 2)
    cv2.putText(canvas, "METRO COFFEE & BAKERY", (820, 110), cv2.FONT_HERSHEY_DUPLEX, 0.75, (20, 20, 20), 2)
    cv2.putText(canvas, "Receipt #: RCP-44910", (820, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    cv2.putText(canvas, "Cashier: Alex  | Register 03", (820, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(canvas, "Date: 12-OCT-2026 08:42 AM", (820, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.line(canvas, (800, 230), (1300, 230), (210, 210, 210), 1)
    cv2.putText(canvas, "1x Espresso Roast Blend       $16.50", (820, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(canvas, "2x Blueberry Danish           $9.00", (820, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(canvas, "1x Sparkling Mineral Water    $3.50", (820, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.line(canvas, (800, 380), (1300, 380), (210, 210, 210), 1)
    cv2.putText(canvas, "Subtotal: $29.00  Tax: $2.32", (820, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(canvas, "TOTAL PAID (CARD): $31.32", (820, 455), cv2.FONT_HERSHEY_DUPLEX, 0.65, (10, 10, 140), 2)
    cv2.putText(canvas, "THANK YOU FOR YOUR VISIT!", (850, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # 3. ID Card (bottom: 220, 590, 1180, 1120)
    cv2.rectangle(canvas, (220, 590), (1180, 1120), (255, 255, 255), -1)
    cv2.rectangle(canvas, (220, 590), (1180, 1120), (180, 180, 180), 2)
    # Header banner
    cv2.rectangle(canvas, (220, 590), (1180, 660), (45, 90, 140), -1)
    cv2.putText(canvas, "NATIONAL IDENTITY VERIFICATION CARD", (320, 638), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # Photo frame placeholder
    cv2.rectangle(canvas, (260, 710), (460, 970), (210, 210, 210), -1)
    cv2.rectangle(canvas, (260, 710), (460, 970), (150, 150, 150), 2)
    cv2.putText(canvas, "[PHOTO]", (320, 845), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)

    # Personal info
    cv2.putText(canvas, "Full Name: EMMA JANE WATSON", (500, 740), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(canvas, "ID Document No: ID-90812-4410", (500, 785), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.putText(canvas, "Date of Birth: 15-AUG-1994", (500, 825), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.putText(canvas, "Nationality: Citizen", (500, 865), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    cv2.putText(canvas, "Issue Date: 01-JAN-2022  |  Expiry Date: 01-JAN-2032", (500, 905), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(canvas, "Address: 742 Evergreen Terrace, Sector 4", (500, 945), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.line(canvas, (260, 1010), (1140, 1010), (200, 200, 200), 1)
    cv2.putText(canvas, "<<IDDOCEMMA<<JANE<<<<<<<<<<<<<<<<<<9081244109408154M3201015<<<<", (260, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

    out_path = TEST_SAMPLES_DIR / "scenario_a_multi_doc_on_single_page.png"
    cv2.imwrite(str(out_path), canvas)
    return out_path


def create_scenario_b_pdf() -> Path:
    """
    Scenario B: Compound PDF containing multiple distinct documents across pages:
      - Pages 1–2: Invoice A (INV-A-101)
      - Pages 3–4: Customer KYC Identity Verification Document
      - Pages 5–6: Invoice B (INV-B-505)
      - Pages 7–9: Bank Statement (Account # 4401-2290-88)
    Total: 9 pages.
    """
    doc = fitz.open()

    def add_header(page, title, subtitle):
        page.draw_rect(fitz.Rect(50, 40, 545, 95), fill=(0.95, 0.95, 0.97), color=(0.7, 0.7, 0.8))
        page.insert_text((70, 70), title, fontsize=16, fontname="helv", color=(0.1, 0.1, 0.2))
        page.insert_text((70, 87), subtitle, fontsize=10, fontname="helv", color=(0.3, 0.3, 0.4))

    # --- DOC 1: INVOICE A (Pages 1 & 2) ---
    p1 = doc.new_page()
    add_header(p1, "INVOICE - ACME INDUSTRIAL CORP", "Invoice Number: INV-A-101 | Date: October 01, 2026")
    p1.insert_text((70, 130), "Bill To: Nexa Systems Pvt Ltd", fontsize=11, fontname="helv")
    p1.insert_text((70, 150), "Client Account: NEX-8800", fontsize=10, fontname="helv")
    p1.insert_text((70, 180), "Line Items Description:\n1. Server Hardware Blades Type X  -  Qty: 4  -  Amount: $12,000.00\n2. Dual Redundant Power Supplies  -  Qty: 8  -  Amount: $3,200.00", fontsize=10, fontname="helv")
    p1.insert_text((70, 240), "[Continued on Page 2 for installation and assembly breakdown...]", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))
    p1.insert_text((250, 780), "Page 1 of 2", fontsize=10, fontname="helv")

    p2 = doc.new_page()
    add_header(p2, "INVOICE - ACME INDUSTRIAL CORP (CONT.)", "Invoice Number: INV-A-101 | Continued")
    p2.insert_text((70, 130), "3. On-site Rack Installation & Cabling  -  Qty: 1  -  Amount: $2,500.00\n4. 3-Year Enterprise SLA Maintenance  -  Qty: 1  -  Amount: $4,800.00", fontsize=10, fontname="helv")
    p2.insert_text((70, 200), "Subtotal: $22,500.00\nSales Tax (8%): $1,800.00\nTOTAL INVOICE AMOUNT DUE: $24,300.00 USD", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.6))
    p2.insert_text((70, 270), "Payment Terms: Net 30 days. Remit to Acme Corp Wells Fargo Wire.", fontsize=9, fontname="helv")
    p2.insert_text((250, 780), "Page 2 of 2", fontsize=10, fontname="helv")

    # --- DOC 2: CUSTOMER KYC DOCUMENT (Pages 3 & 4) ---
    p3 = doc.new_page()
    add_header(p3, "CUSTOMER KYC ONBOARDING FORM", "Compliance Record: KYC-2026-992")
    p3.insert_text((70, 130), "Applicant Full Name: Robert Michael Chang\nDate of Birth: 22-MAR-1988\nCitizenship: United States\nGovernment Identification: US Passport ending 4892", fontsize=10, fontname="helv")
    p3.insert_text((70, 210), "Residential Address: 1044 Beacon Street, Boston, MA\nEmployment Status: Full-Time Software Architect\nSource of Funds: Employment Income\nExpected Monthly Volume: $10,000 - $50,000", fontsize=10, fontname="helv")
    p3.insert_text((70, 280), "Please see page 4 for beneficial ownership declaration and signature.", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))
    p3.insert_text((250, 780), "Page 1 of 2", fontsize=10, fontname="helv")

    p4 = doc.new_page()
    add_header(p4, "CUSTOMER KYC - VERIFICATION & DECLARATION", "Compliance Record: KYC-2026-992")
    p4.insert_text((70, 130), "Risk Assessment Category: Low Risk Tier 1\nAML Automated Database Check: Passed (Zero Adverse Findings)\nPEP (Politically Exposed Person) Status: Negative", fontsize=10, fontname="helv")
    p4.insert_text((70, 200), "Declaration: I hereby confirm that all information provided in this KYC dossier is accurate and verifiable under penalties of perjury.\n\nSigned: Robert M. Chang\nAuthorized KYC Compliance Officer: Sarah Jenkins, CAMS", fontsize=10, fontname="helv")
    p4.insert_text((250, 780), "Page 2 of 2", fontsize=10, fontname="helv")

    # --- DOC 3: INVOICE B (Pages 5 & 6) ---
    p5 = doc.new_page()
    add_header(p5, "TAX INVOICE - APEX CLOUD NETWORKS", "Invoice No: INV-B-505 | Date: October 15, 2026")
    p5.insert_text((70, 130), "Customer: Nexa Systems Pvt Ltd\nBilling Ref: APEX-90112\nCurrency: USD", fontsize=10, fontname="helv")
    p5.insert_text((70, 180), "Services Rendered:\n1. Dedicated Fiber Internet 10Gbps  -  Period: 01-Oct to 31-Oct  -  $4,200.00\n2. DDoS Shield Enterprise Mitigation  -  Monthly Tier  -  $1,500.00", fontsize=10, fontname="helv")
    p5.insert_text((250, 780), "Page 1 of 2", fontsize=10, fontname="helv")

    p6 = doc.new_page()
    add_header(p6, "TAX INVOICE - APEX CLOUD NETWORKS (PAGE 2)", "Invoice No: INV-B-505 | Balance Summary")
    p6.insert_text((70, 130), "Total Services: $5,700.00\nApplicable Service Tax: $456.00\nGRAND TOTAL PAYABLE: $6,156.00", fontsize=12, fontname="helv", color=(0.1, 0.1, 0.6))
    p6.insert_text((70, 190), "Due Date: 30-OCT-2026. Electronic bank transfers to Apex Cloud Citibank.", fontsize=9, fontname="helv")
    p6.insert_text((250, 780), "Page 2 of 2", fontsize=10, fontname="helv")

    # --- DOC 4: BANK STATEMENT (Pages 7, 8 & 9) ---
    p7 = doc.new_page()
    add_header(p7, "FIRST NATIONAL COMMERCIAL BANK", "Monthly Statement of Account | Account Number: 4401-2290-88")
    p7.insert_text((70, 130), "Account Holder: Nexa Systems Pvt Ltd\nStatement Period: 01-SEP-2026 to 30-SEP-2026\nOpening Balance: $148,290.40", fontsize=10, fontname="helv")
    p7.insert_text((70, 190), "Transactions Ledger (Part 1):\n02-SEP: Deposit Wire from Horizon Logistics  + $25,000.00  Balance: $173,290.40\n05-SEP: Payroll Transfer Direct Debit       - $42,100.00  Balance: $131,190.40\n08-SEP: Vendor Payment Acme Corp            - $18,000.00  Balance: $113,190.40", fontsize=10, fontname="helv")
    p7.insert_text((250, 780), "Page 1 of 3", fontsize=10, fontname="helv")

    p8 = doc.new_page()
    add_header(p8, "FIRST NATIONAL COMMERCIAL BANK (CONT.)", "Account Number: 4401-2290-88 | Page 2")
    p8.insert_text((70, 130), "Transactions Ledger (Part 2):\n12-SEP: AWS Cloud Hosting Bill               - $6,450.00   Balance: $106,740.40\n18-SEP: Client Invoice Settlement Inbound    + $54,300.00  Balance: $161,040.40\n22-SEP: Office Lease Rent Payment            - $12,500.00  Balance: $148,540.40", fontsize=10, fontname="helv")
    p8.insert_text((250, 780), "Page 2 of 3", fontsize=10, fontname="helv")

    p9 = doc.new_page()
    add_header(p9, "FIRST NATIONAL COMMERCIAL BANK (SUMMARY)", "Account Number: 4401-2290-88 | Summary")
    p9.insert_text((70, 130), "Transactions Ledger (Part 3):\n28-SEP: Interest Earned                      + $210.15     Balance: $148,750.55\n30-SEP: Bank Service Maintenance Fee         - $45.00      Balance: $148,705.55", fontsize=10, fontname="helv")
    p9.insert_text((70, 200), "Total Deposits: $79,510.15 | Total Withdrawals: $79,095.00\nCLOSING LEDGER BALANCE: $148,705.55", fontsize=12, fontname="helv", color=(0.1, 0.4, 0.1))
    p9.insert_text((250, 780), "Page 3 of 3", fontsize=10, fontname="helv")

    out_path = TEST_SAMPLES_DIR / "scenario_b_multi_doc_compound.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def create_scenario_c_pdf() -> Path:
    """
    Scenario C: One multi-page document (4-page bank statement).
    The system should identify all 4 pages belong to the same logical document.
    """
    doc = fitz.open()

    for p_num in range(1, 5):
        page = doc.new_page()
        page.draw_rect(fitz.Rect(50, 40, 545, 95), fill=(0.93, 0.96, 0.95), color=(0.6, 0.75, 0.7))
        page.insert_text((70, 70), "GLOBAL PACIFIC TRUST BANK", fontsize=15, fontname="helv", color=(0.05, 0.25, 0.15))
        page.insert_text((70, 87), f"Quarterly Financial Statement | Account: ACCT-8899-0012 | Page {p_num} of 4", fontsize=10, fontname="helv")

        page.insert_text((70, 130), f"Customer Name: Pacific BioVentures Inc\nAccount Type: Commercial Money Market Ledger\nStatement Window: Q3 2026", fontsize=10, fontname="helv")

        # Ledger entries
        ledger_text = f"Activity Ledger for Section {p_num}:\n"
        for row in range(1, 8):
            ledger_text += f"Day {p_num*7 + row}: Routine clearing transaction item #{row*100}  |  Ref: TRN-880{p_num}{row}  |  Amount: ${row * 1250}.00\n"
        page.insert_text((70, 180), ledger_text, fontsize=9, fontname="helv")

        if p_num < 4:
            page.insert_text((70, 360), "[Account activity continues on the following statement page...]", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))
        else:
            page.insert_text((70, 360), "End of Statement. Verified by Global Pacific Trust Treasury Services.", fontsize=10, fontname="helv", color=(0.1, 0.3, 0.1))

        page.insert_text((250, 780), f"Page {p_num} of 4", fontsize=10, fontname="helv")

    out_path = TEST_SAMPLES_DIR / "scenario_c_multipage_bank_statement.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def create_scenario_d_pdf() -> Path:
    """
    Scenario D: Previously unseen document type (ISO Environmental & ESG Audit Report).
    Dataset-independent test to verify zero-shot unsupervised capability.
    """
    doc = fitz.open()

    p1 = doc.new_page()
    p1.draw_rect(fitz.Rect(50, 40, 545, 95), fill=(0.96, 0.94, 0.90), color=(0.8, 0.7, 0.5))
    p1.insert_text((70, 70), "ENVIRONMENTAL AUDIT & ESG COMPLIANCE REPORT", fontsize=14, fontname="helv", color=(0.3, 0.2, 0.05))
    p1.insert_text((70, 87), "Standard: ISO 14001:2015 Environmental Management Systems", fontsize=10, fontname="helv")

    p1.insert_text((70, 130), "Facility Inspected: Plant 04 Chemical Synthesis Complex\nAuditing Agency: EcoVeritas Global Standards\nAssessment Date: 20-OCT-2026", fontsize=10, fontname="helv")
    p1.insert_text((70, 190), "Executive Findings:\n1. Carbon Scope 1 Direct Emissions: 1,420 metric tons CO2e (Compliant with regional caps).\n2. Wastewater Effluent Treatment: pH normalized between 6.8 and 7.4 prior to municipal discharge.\n3. Hazardous Chemical Storage: Secondary containment dykes inspected and certified sound.\n4. Zero critical non-conformances identified during on-site surveillance audit.", fontsize=10, fontname="helv")
    p1.insert_text((70, 310), "Certification Status: Recommended for 3-Year ISO 14001 Recertification.\nLead Environmental Auditor: Dr. Marcus Vance, PhD, QEP", fontsize=10, fontname="helv")
    p1.insert_text((250, 780), "Page 1 of 1", fontsize=10, fontname="helv")

    out_path = TEST_SAMPLES_DIR / "scenario_d_unseen_compliance_report.pdf"
    doc.save(str(out_path))
    doc.close()
    return out_path


def generate_all_test_data():
    print("Generating comprehensive test dataset for Scenarios A, B, C, D...")
    p_a = create_scenario_a_image()
    print(f"  [✓] Scenario A (Multi-doc on 1 page image): {p_a}")
    p_b = create_scenario_b_pdf()
    print(f"  [✓] Scenario B (Compound PDF with 4 documents): {p_b}")
    p_c = create_scenario_c_pdf()
    print(f"  [✓] Scenario C (4-page single bank statement): {p_c}")
    p_d = create_scenario_d_pdf()
    print(f"  [✓] Scenario D (Unseen ISO Environmental Audit): {p_d}")
    print("All test samples successfully generated in data/test_samples/")

if __name__ == "__main__":
    generate_all_test_data()
