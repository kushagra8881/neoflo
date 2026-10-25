import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Union
from pipeline.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS

class FileLoader:
    """Handles file validation, ingestion, and hashing for multi-format uploads."""

    @staticmethod
    def validate_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {"valid": False, "error": f"File does not exist: {path}"}
        
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "valid": False, 
                "error": f"Unsupported format '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
            }
        
        size = path.stat().st_size
        if size == 0:
            return {"valid": False, "error": "File is empty (0 bytes)."}

        # Calculate SHA256 checksum for deduplication and caching
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()[:16]

        return {
            "valid": True,
            "filename": path.name,
            "filepath": str(path.resolve()),
            "extension": ext,
            "size_bytes": size,
            "file_hash": file_hash,
            "is_pdf": ext == ".pdf"
        }

    @classmethod
    def save_uploaded_stream(cls, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """Save an in-memory stream (e.g. from Streamlit or API upload) to disk."""
        target_path = UPLOAD_DIR / filename
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        return cls.validate_file(target_path)

    @classmethod
    def load_directory_or_files(cls, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Validate and gather metadata for a list of input files."""
        results = []
        for fp in file_paths:
            path = Path(fp)
            if path.is_file():
                meta = cls.validate_file(path)
                if meta["valid"]:
                    results.append(meta)
            elif path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                        meta = cls.validate_file(child)
                        if meta["valid"]:
                            results.append(meta)
        return results
