import re
import unicodedata
from typing import Tuple, List, Set

# Regex patterns precompiled for performance
RE_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|org|net|in|co\.in|io|ai|biz|info|fr|co)$", re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r"\.(?:com|org|net|in|co\.in|io|ai|biz|info|fr|co)\b", re.IGNORECASE)
RE_AMP = re.compile(r"\s*&\s*")
RE_NON_ALPHANUM_UNICODE = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\d+")

# Legal suffixes to standardize or strip for core name matching
LEGAL_SUFFIXES = [
    (re.compile(r"\bprivate\s+limited\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\bpvt\.?\s*ltd\.?\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\bcorporation\b", re.IGNORECASE), "corp"),
    (re.compile(r"\bincorporated\b", re.IGNORECASE), "inc"),
    (re.compile(r"\bcompany\b", re.IGNORECASE), "co"),
    (re.compile(r"\blimited\b", re.IGNORECASE), "ltd"),
    (re.compile(r"\bl\.?l\.?c\.?\b", re.IGNORECASE), "llc"),
    (re.compile(r"\bl\.?l\.?p\.?\b", re.IGNORECASE), "llp"),
    (re.compile(r"\bs\.?a\.?r\.?l\.?\b", re.IGNORECASE), "sarl"),
    (re.compile(r"\bs\.?a\.?s\.?\b", re.IGNORECASE), "sas"),
    (re.compile(r"\bs\.?a\.?\b", re.IGNORECASE), "sa"),
]

# Patterns specifically to strip off the tail of a business name to obtain the core entity name
RE_CORE_SUFFIX_STRIP = re.compile(
    r"\s*(?:\b(?:pvt\s+ltd|private\s+limited|corp|corporation|inc|incorporated|co|company|ltd|limited|llc|llp|sarl|sas|sa|gmbh|enterprises|enterprise|group|holdings|services|solutions|consulting)\b\s*)+$",
    re.IGNORECASE
)

# Address abbreviations standardization
ADDR_ABBREVIATIONS = [
    (re.compile(r"\bstreet\b", re.IGNORECASE), "st"),
    (re.compile(r"\broad\b", re.IGNORECASE), "rd"),
    (re.compile(r"\bavenue\b", re.IGNORECASE), "ave"),
    (re.compile(r"\bboulevard\b", re.IGNORECASE), "blvd"),
    (re.compile(r"\bdrive\b", re.IGNORECASE), "dr"),
    (re.compile(r"\blane\b", re.IGNORECASE), "ln"),
    (re.compile(r"\bfloor\b", re.IGNORECASE), "fl"),
    (re.compile(r"\bapartment\b", re.IGNORECASE), "apt"),
    (re.compile(r"\bsuite\b", re.IGNORECASE), "ste"),
]

def normalize_text_unicode(text: str) -> str:
    """Standardizes Unicode representation via NFKC and casefolding."""
    if not text:
        return ""
    # NFKC normalizes compatibility characters while preserving scripts
    text = unicodedata.normalize("NFKC", text)
    return text.casefold()

def normalize_business_name(name: str) -> str:
    """
    Cleans business name:
      - NFKC + casefold
      - Handles domain names (e.g. 'company.com' -> 'company')
      - Replaces '&' with ' and '
      - Standardizes legal suffixes
      - Collapses whitespace and removes punctuation
    """
    if not name:
        return ""
    
    text = unicodedata.normalize("NFKC", name).casefold()
    
    # Check domain name pattern:
    m_domain = RE_DOMAIN.match(text)
    if m_domain:
        text = m_domain.group(1)
    else:
        text = RE_DOMAIN_SUFFIX.sub("", text)
        
    text = RE_AMP.sub(" and ", text)
    
    for pat, rep in LEGAL_SUFFIXES:
        text = pat.sub(rep, text)
        
    text = RE_NON_ALPHANUM_UNICODE.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text

def extract_core_business_name(clean_name: str) -> str:
    """
    Strips trailing legal and corporate entity markers from clean name.
    Example: 'payne enterprises llc' -> 'payne'
    """
    if not clean_name:
        return ""
    core = RE_CORE_SUFFIX_STRIP.sub("", clean_name).strip()
    return core if len(core) >= 2 else clean_name

def get_alphanumeric_skeleton(text: str) -> str:
    """Extracts only alphanumeric Unicode characters without whitespace."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in norm if c.isalnum())

def normalize_address(address: str) -> str:
    """
    Cleans and standardizes address text:
      - NFKC + casefold
      - Standardizes street, road, ave, blvd, etc.
      - Collapses whitespace and strips punctuation
    """
    if not address or address.strip().lower() in ("null", "none", "nan"):
        return ""
    
    text = unicodedata.normalize("NFKC", address).casefold()
    text = RE_AMP.sub(" and ", text)
    
    for pat, rep in ADDR_ABBREVIATIONS:
        text = pat.sub(rep, text)
        
    text = RE_NON_ALPHANUM_UNICODE.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text

def extract_all_digits(text: str) -> List[str]:
    """Extracts all continuous digit sequences (ZIP/PIN, house numbers)."""
    if not text:
        return []
    return RE_DIGITS.findall(text)
