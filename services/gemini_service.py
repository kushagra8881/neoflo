import os
import json
import re
import hashlib
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from pipeline.config import GEMINI_API_KEY

class GeminiService:
    """
    Integrates Google Gemini API for high-level semantic reasoning:
    cluster naming, concise explanations, metadata extraction, and ambiguity resolution.
    Complies with project rules by keeping routine processing local and calling Gemini selectively.
    """
    _instance: Optional["GeminiService"] = None

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.client_available = False
        self._cache: Dict[str, Any] = {}

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Use gemini-3.8-flash for high accuracy structured responses
                self.model = genai.GenerativeModel("gemini-3.8-flash")
                self.client_available = True
            except Exception as e:
                print(f"[GeminiService] Warning: Could not initialize Gemini API: {e}")

    @classmethod
    def get_instance(cls) -> "GeminiService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def describe_cluster(
        self, 
        cluster_id: int, 
        sample_texts: List[str], 
        sample_signals: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generates a human-readable label and concise explanation for a discovered cluster
        from representative document texts.
        """
        # Create cache key
        combined_samples = " | ".join([s[:200] for s in sample_texts])
        cache_key = hashlib.md5(f"{cluster_id}_{combined_samples}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.client_available or not sample_texts:
            return self._heuristic_cluster_label(cluster_id, sample_texts, sample_signals)

        # Prepare limited, privacy-friendly text snippets
        truncated_samples = []
        for i, text in enumerate(sample_texts[:3]):
            clean = " ".join(text.split()[:120])
            truncated_samples.append(f"Sample {i+1}:\n\"{clean}\"")
        prompt_text = "\n\n".join(truncated_samples)

        prompt = f"""You are an expert AI document intelligence assistant.
Analyze these representative text excerpts from a newly discovered document cluster (Cluster {cluster_id}).
Identify what kind of business/legal documents these are.

Return ONLY a valid JSON object matching this exact schema:
{{
  "cluster_id": {cluster_id},
  "label": "Short Title (2 to 4 words, e.g., 'Commercial Invoices', 'Bank Statements', 'KYC Verification Forms', 'Environmental Compliance Reports')",
  "description": "A concise 1-2 sentence description explaining what unites these documents.",
  "document_category": "Financial | Legal | Identification | Operations | Technical | General",
  "confidence": "high"
}}

Document samples:
{prompt_text}
"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
                request_options={"timeout": 5.0}
            )
            parsed = json.loads(response.text.strip())
            self._cache[cache_key] = parsed
            return parsed
        except Exception as e:
            print(f"[GeminiService] API call failed for cluster {cluster_id}: {e}. Falling back to heuristic.")
            fallback = self._heuristic_cluster_label(cluster_id, sample_texts, sample_signals)
            self._cache[cache_key] = fallback
            return fallback

    def _heuristic_cluster_label(
        self, 
        cluster_id: int, 
        sample_texts: List[str],
        sample_signals: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Deterministic rule-based fallback if Gemini API is disabled or unreachable."""
        combined = " ".join(sample_texts).lower()

        if any(w in combined for w in ["invoice", "bill to", "tax invoice", "amount due", "subtotal"]):
            label = "Invoices & Billing"
            desc = "Commercial and tax invoices containing line items, billing terms, and payment totals."
            category = "Financial"
        elif any(w in combined for w in ["statement of account", "bank statement", "ledger", "deposit", "withdrawal"]):
            label = "Bank & Financial Statements"
            desc = "Banking statements summarizing running account balances, deposits, and transaction ledgers."
            category = "Financial"
        elif any(w in combined for w in ["kyc", "identity", "passport", "driving license", "national id", "dob"]):
            label = "Identity & KYC Documents"
            desc = "Identification forms, identity cards, and customer onboarding verification records."
            category = "Identification"
        elif any(w in combined for w in ["receipt", "cashier", "pos terminal", "card payment"]):
            label = "Retail & POS Receipts"
            desc = "Transaction receipts detailing cashier register purchases and point-of-sale totals."
            category = "Financial"
        elif any(w in combined for w in ["iso", "audit", "compliance", "environmental", "esg", "inspection"]):
            label = "Audit & Compliance Reports"
            desc = "Specialized audit reports evaluating regulatory standards and organizational compliance."
            category = "Operations"
        else:
            label = f"Document Group {cluster_id + 1}"
            desc = "Semantically cohesive documents identified by unsupervised embedding clustering."
            category = "General"

        return {
            "cluster_id": cluster_id,
            "label": label,
            "description": desc,
            "document_category": category,
            "confidence": "medium (heuristic)"
        }

    def classify_document(
        self, 
        doc_text: str, 
        signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classifies an individual document into a specific document type
        (e.g., 'Commercial Invoice', 'Passport', 'Driver License', 'National ID Card',
               'Bank Statement', 'Retail Receipt', 'Audit Report', 'Employment Contract').
        Fast local classification runs in < 5ms with optional fast Gemini enrichment.
        """
        # 1. Run instant local heuristic classification
        base_result = self._heuristic_document_type(doc_text, signals)

        # 2. If Gemini is available, attempt fast enrichment with strict 3-second timeout
        if not self.client_available:
            return base_result

        clean_snip = " ".join(doc_text.split()[:150])
        cache_key = hashlib.md5(f"doc_{clean_snip}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = f"""Identify the document type of this document excerpt (e.g., Commercial Invoice, Passport, Driver License, National ID Card, Bank Account Statement, Retail POS Receipt, Customer KYC Onboarding Form, Compliance Audit Report):

\"\"\"{clean_snip}\"\"\"

Return JSON:
{{
  "document_type": "...",
  "category": "Financial | Identification | Legal | Operations | General",
  "confidence": "high",
  "summary": "1 sentence summary"
}}
"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
                request_options={"timeout": 3.0}
            )
            parsed = json.loads(response.text.strip())
            # Preserve extracted key fields
            parsed["key_fields"] = base_result.get("key_fields", {})
            self._cache[cache_key] = parsed
            return parsed
        except Exception:
            # On any network timeout or delay, immediately return instant local classification
            self._cache[cache_key] = base_result
            return base_result

    def _heuristic_document_type(
        self, 
        doc_text: str, 
        signals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deterministic heuristic for document type classification."""
        text_lower = doc_text.lower()
        signals = signals or {}

        doc_type = "Business Document"
        category = "General"
        summary = "General business document."
        issuer = None
        ref_id = None
        date = None
        total = None

        if "passport" in text_lower or "<<iddoc" in text_lower or "<<mrz" in text_lower:
            doc_type = "Passport / Travel Document"
            category = "Identification"
            summary = "Official government travel passport or machine-readable travel document."
        elif "driving license" in text_lower or "driver license" in text_lower or "driver's license" in text_lower:
            doc_type = "Driver's License"
            category = "Identification"
            summary = "Government-issued motor vehicle driving license."
        elif "identity" in text_lower or "national id" in text_lower or "aadhaar" in text_lower or signals.get("has_id_cues"):
            doc_type = "National Identity Card"
            category = "Identification"
            summary = "National identity verification card containing personal demographic information."
        elif "receipt" in text_lower or "cashier" in text_lower or "pos terminal" in text_lower or signals.get("has_receipt_cues"):
            doc_type = "Retail POS Receipt"
            category = "Financial"
            summary = "Point-of-sale customer sales receipt with register timestamp and tender total."
        elif "statement of account" in text_lower or "bank statement" in text_lower or "ledger" in text_lower or signals.get("has_bank_cues"):
            doc_type = "Bank Account Statement"
            category = "Financial"
            summary = "Periodic bank account financial statement summarizing transactions and balance."
        elif "tax invoice" in text_lower or "commercial invoice" in text_lower or "invoice" in text_lower or signals.get("has_invoice_cues"):
            doc_type = "Commercial Invoice"
            category = "Financial"
            summary = "Itemized commercial sales invoice with payment terms and billing totals."
        elif "kyc" in text_lower or "customer onboarding" in text_lower:
            doc_type = "Customer KYC Onboarding Form"
            category = "Legal / Compliance"
            summary = "Customer KYC compliance dossier for account opening and identity verification."
        elif "audit" in text_lower or "compliance report" in text_lower or "iso" in text_lower:
            doc_type = "Regulatory Audit & Compliance Report"
            category = "Operations / Compliance"
            summary = "Formal audit report verifying compliance with industrial and environmental standards."

        if signals.get("invoice_id"):
            ref_id = signals["invoice_id"]
        elif signals.get("account_id"):
            ref_id = signals["account_id"]

        return {
            "document_type": doc_type,
            "category": category,
            "confidence": "medium (heuristic)",
            "summary": summary,
            "key_fields": {
                "issuer_or_institution": issuer,
                "reference_or_id_number": ref_id,
                "date": date,
                "total_amount": total
            }
        }

