"""String and semantic similarity comparison utilities for SilentAuditor."""

import logging
import re
import unicodedata
from typing import Optional

import numpy as np
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Common corporate suffixes to strip during vendor-name normalisation.
_CORP_SUFFIXES: list[str] = [
    "incorporated",
    "corporation",
    "company",
    "limited",
    "inc",
    "llc",
    "ltd",
    "corp",
    "co",
    "lp",
    "llp",
    "plc",
    "gmbh",
    "sa",
    "ag",
]

# Build a regex pattern that matches any suffix at the end of a string,
# optionally preceded/followed by a dot or comma.
_SUFFIX_PATTERN: re.Pattern = re.compile(
    r"[,.]?\s+\b("
    + "|".join(re.escape(s) for s in _CORP_SUFFIXES)
    + r")\.?\s*$",
    re.IGNORECASE,
)

# Address abbreviation mappings (lowercased key → canonical form).
_ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "rd": "road",
    "rd.": "road",
    "ste": "suite",
    "ste.": "suite",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "ct": "court",
    "ct.": "court",
    "pl": "place",
    "pl.": "place",
    "hwy": "highway",
    "hwy.": "highway",
    "pkwy": "parkway",
    "pkwy.": "parkway",
}


# ------------------------------------------------------------------
# Text normalisation
# ------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, normalise unicode.

    Returns an empty string for ``None`` or empty input.
    """
    if not text:
        return ""
    # Unicode NFKD decomposition then drop combining marks
    normalised = unicodedata.normalize("NFKD", text)
    normalised = "".join(
        ch for ch in normalised if not unicodedata.combining(ch)
    )
    normalised = normalised.lower()
    # Replace punctuation with spaces
    normalised = re.sub(r"[^\w\s]", " ", normalised)
    # Collapse whitespace
    normalised = re.sub(r"\s+", " ", normalised).strip()
    return normalised


def normalize_vendor_name(name: str) -> str:
    """Normalise a vendor name for comparison.

    Applies :func:`normalize_text` then strips common corporate suffixes
    (Inc, LLC, Ltd, Corp, Co, etc.).
    """
    if not name:
        return ""
    result = normalize_text(name)
    # Strip corporate suffixes iteratively (handles "Inc. Co." edge case)
    prev = None
    while prev != result:
        prev = result
        result = _SUFFIX_PATTERN.sub("", result).strip()
    return result


def normalize_address(address: str) -> str:
    """Normalise a street address for comparison.

    Lowercases, expands common abbreviations (St→Street, Ave→Avenue, etc.),
    and removes unit / suite numbers so that "123 Main St Ste 400" and
    "123 Main Street" compare as equivalent.
    """
    if not address:
        return ""
    result = address.lower().strip()
    # Remove unit/suite/apt numbers  (e.g. "Suite 400", "Apt 3B", "Unit 12")
    result = re.sub(
        r"\b(suite|ste\.?|apt\.?|unit|#)\s*\w+",
        "",
        result,
        flags=re.IGNORECASE,
    )
    # Expand abbreviations word-by-word
    tokens = result.split()
    expanded = [_ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    result = " ".join(expanded)
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result


def normalize_phone(phone: str) -> str:
    """Strip a phone number to digits only, removing country-code prefix.

    Returns only the local digits (assumes US 10-digit if 11 digits
    starting with ``'1'``).
    """
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)
    # Strip leading country code '1' for US numbers
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


# ------------------------------------------------------------------
# Fuzzy / exact matching
# ------------------------------------------------------------------

def fuzzy_match_score(text_a: str, text_b: str) -> float:
    """Return a 0.0–1.0 fuzzy similarity score using token-sort ratio.

    Uses ``rapidfuzz.fuzz.token_sort_ratio`` which is order-insensitive
    and handles minor typos well.
    """
    if not text_a or not text_b:
        return 0.0
    return fuzz.token_sort_ratio(text_a, text_b) / 100.0


def exact_match(text_a: str, text_b: str) -> bool:
    """Case-insensitive, whitespace-normalised exact comparison."""
    if text_a is None or text_b is None:
        return False
    return normalize_text(text_a) == normalize_text(text_b)


# ------------------------------------------------------------------
# Embedding / vector operations
# ------------------------------------------------------------------

def compute_embedding(text: str, model: object) -> np.ndarray:
    """Generate a sentence embedding using a ``sentence_transformers`` model.

    Args:
        text: The input text.
        model: A ``SentenceTransformer`` instance (or any object with an
               ``encode`` method returning an ndarray).

    Returns:
        A 1-D float32 numpy array.
    """
    vec = model.encode(text, convert_to_numpy=True)
    return vec.astype(np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def batch_cosine_similarity(
    query_vec: np.ndarray, corpus_vecs: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between a query vector and a corpus matrix.

    Args:
        query_vec: 1-D array of shape ``(d,)``.
        corpus_vecs: 2-D array of shape ``(n, d)``.

    Returns:
        1-D array of shape ``(n,)`` with similarity scores.
    """
    if corpus_vecs.ndim == 1:
        corpus_vecs = corpus_vecs.reshape(1, -1)

    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0.0:
        return np.zeros(corpus_vecs.shape[0], dtype=np.float32)

    corpus_norms = np.linalg.norm(corpus_vecs, axis=1)
    # Avoid division by zero
    corpus_norms = np.where(corpus_norms == 0.0, 1.0, corpus_norms)

    dots = corpus_vecs @ query_vec
    return (dots / (query_norm * corpus_norms)).astype(np.float32)
