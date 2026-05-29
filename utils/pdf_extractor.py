"""
utils/pdf_extractor.py
----------------------
Utility to extract text from uploaded PDF files.
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract raw text from PDF file bytes page-by-page.
    """
    try:
        import pypdf
        
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
                
        full_text = "\n".join(text_parts)
        logger.info(f"Successfully extracted {len(full_text)} characters from {len(reader.pages)} PDF pages.")
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to parse PDF file: {str(e)}")
