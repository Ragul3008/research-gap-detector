"""
utils/preprocessing.py
-----------------------
Text cleaning, normalisation, and feature engineering
for the research paper dataset.
"""

import re
import string
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Text cleaning helpers
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Lowercase, strip HTML tags, normalise whitespace,
    and remove non-printable characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)              # strip HTML
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"\s+", " ", text)                   # collapse spaces
    text = text.strip()
    return text


def remove_punctuation(text: str) -> str:
    """Remove punctuation (keep hyphens for compound words)."""
    translator = str.maketrans(
        string.punctuation.replace("-", ""),
        " " * (len(string.punctuation) - 1)
    )
    return text.translate(translator)


def build_combined_text(
    title: str,
    abstract: str,
    keywords: str,
    title_weight: int = 3,
    keyword_weight: int = 2,
) -> str:
    """
    Concatenate title (repeated for weight), abstract, and keywords
    into a single embedding-ready string.

    Repetition-based weighting is a simple but effective trick
    for sentence-transformer models.
    """
    title_part    = (clean_text(title) + " ") * title_weight
    keyword_part  = (clean_text(keywords) + " ") * keyword_weight
    abstract_part = clean_text(abstract)
    combined = f"{title_part}{keyword_part}{abstract_part}"
    return combined.strip()


# ─────────────────────────────────────────────
# DataFrame-level pipeline
# ─────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a papers DataFrame.

    Expected columns: title, abstract, keywords, year, domain
    Returns an augmented DataFrame with 'combined_text' column.
    """
    required = {"title", "abstract", "keywords"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    df = df.copy()

    # Fill NaN
    for col in ["title", "abstract", "keywords"]:
        df[col] = df[col].fillna("").astype(str)

    # Clean individual fields
    df["title_clean"]    = df["title"].apply(clean_text)
    df["abstract_clean"] = df["abstract"].apply(clean_text)
    df["keywords_clean"] = df["keywords"].apply(clean_text)

    # Build weighted combined text for embedding
    df["combined_text"] = df.apply(
        lambda r: build_combined_text(
            r["title"], r["abstract"], r["keywords"]
        ),
        axis=1,
    )

    # Year as int (handle missing)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    logger.info(f"Preprocessing complete. Total records: {len(df)}")
    return df


# ─────────────────────────────────────────────
# Query preprocessing
# ─────────────────────────────────────────────

def preprocess_query(
    title: str,
    abstract: Optional[str] = "",
    keywords: Optional[str] = "",
) -> str:
    """
    Preprocess a user query in the same way as the dataset.
    Returns a single combined text string ready for embedding.
    """
    abstract = abstract or ""
    keywords = keywords or ""
    return build_combined_text(title, abstract, keywords)
