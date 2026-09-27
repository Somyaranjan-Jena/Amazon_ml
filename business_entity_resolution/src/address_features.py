import re
from typing import Dict, List, Set, Optional, Any

RE_PIN_INDIA = re.compile(r"\b\d{6}\b")
RE_ZIP_US_FR = re.compile(r"\b\d{5}\b")
RE_NUMBERS = re.compile(r"\b\d+\b")

COMMON_ADDR_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "road", "street", "avenue",
    "floor", "fl", "suite", "ste", "apt", "near", "opp", "opposite",
    "behind", "beside", "lane", "colony", "block", "sector", "phase",
    "city", "district", "dist", "nagar", "state", "india", "us", "usa",
    "france", "rue", "boulevard", "cedex"
}

def extract_address_components(norm_addr: str) -> Dict[str, Any]:
    """
    Extracts structural address components:
      - postal_code: 5 or 6 digit sequence
      - house_number: first digit sequence in address
      - all_digits: set of all numeric strings
      - significant_tokens: distinctive words excluding common structural address terms
    """
    if not norm_addr:
        return {
            "postal_code": "",
            "house_number": "",
            "all_digits": set(),
            "significant_tokens": set()
        }

    # Extract 6-digit or 5-digit postal code
    pin_match = RE_PIN_INDIA.findall(norm_addr)
    postal_code = pin_match[-1] if pin_match else ""
    if not postal_code:
        zip_match = RE_ZIP_US_FR.findall(norm_addr)
        postal_code = zip_match[-1] if zip_match else ""

    all_nums = RE_NUMBERS.findall(norm_addr)
    # Strip leading zeros so that 0684 matches 684
    norm_nums = [str(int(n)) for n in all_nums]
    house_num = norm_nums[0] if norm_nums else ""

    tokens = norm_addr.split()
    sig_tokens = {t for t in tokens if len(t) >= 3 and t not in COMMON_ADDR_STOPWORDS and not t.isdigit()}

    return {
        "postal_code": postal_code,
        "house_number": house_num,
        "all_digits": set(norm_nums),
        "significant_tokens": sig_tokens
    }
