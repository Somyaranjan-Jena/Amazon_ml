import re
from typing import Dict, List, Any, Optional, Tuple, Union
from rapidfuzz import fuzz

RE_DIGITS = re.compile(r"\b\d+\b")

FEATURE_NAMES = [
    # Name features
    "name_lev_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "exact_name_match",
    "exact_core_name_match",
    "name_token_jaccard",
    "name_common_tokens",
    "name_len_diff",
    "name_len_ratio",
    # Address features
    "addr_lev_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "exact_addr_match",
    "addr_token_jaccard",
    "addr_common_tokens",
    "pin_match",
    "house_num_match",
    "digits_jaccard",
    "addr_len_diff",
    "target_addr_missing",
    # Interaction & source features
    "is_s2",
    "name_x_addr_score",
    "high_name_high_addr",
    "high_name_missing_addr"
]

def extract_pair_features(
    s1_rec: Union[Tuple, Dict[str, Any]],
    target_rec: Union[Tuple, Dict[str, Any]],
    is_s2: bool
) -> List[float]:
    """
    Extracts rich numerical similarity features between an S1 entity and a target (S2/S3) entity.
    Supports compact tuples: (norm_name, core_name, norm_addr, pin, house).
    """
    if isinstance(s1_rec, tuple):
        s1_name, s1_core, s1_addr, s1_pin, s1_h = s1_rec
    else:
        s1_name = s1_rec.get("norm_name", "")
        s1_core = s1_rec.get("core_name", "")
        s1_addr = s1_rec.get("norm_addr", "")
        s1_pin = s1_rec.get("postal_code", "")
        s1_h = s1_rec.get("house_number", "")

    if isinstance(target_rec, tuple):
        t_name, t_core, t_addr, t_pin, t_h = target_rec
    else:
        t_name = target_rec.get("norm_name", "")
        t_core = target_rec.get("core_name", "")
        t_addr = target_rec.get("norm_addr", "")
        t_pin = target_rec.get("postal_code", "")
        t_h = target_rec.get("house_number", "")

    # 1. Name features
    exact_name = 1.0 if (s1_name and s1_name == t_name) else 0.0
    exact_core = 1.0 if (s1_core and s1_core == t_core) else 0.0

    if s1_name and t_name:
        name_lev = fuzz.ratio(s1_name, t_name) / 100.0
        name_sort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
        name_set = fuzz.token_set_ratio(s1_name, t_name) / 100.0
        name_part = fuzz.partial_ratio(s1_name, t_name) / 100.0

        s1_toks = set(s1_name.split())
        t_toks = set(t_name.split())
        tok_inter = s1_toks.intersection(t_toks)
        tok_union = s1_toks.union(t_toks)
        name_jaccard = (len(tok_inter) / len(tok_union)) if tok_union else 0.0
        name_common = float(len(tok_inter))

        l1, l2 = len(s1_name), len(t_name)
        name_len_diff = float(abs(l1 - l2))
        name_len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0
    else:
        name_lev = name_sort = name_set = name_part = 0.0
        name_jaccard = name_common = name_len_diff = name_len_ratio = 0.0

    # 2. Address features
    target_addr_missing = 1.0 if not t_addr else 0.0
    exact_addr = 1.0 if (s1_addr and t_addr and s1_addr == t_addr) else 0.0

    if s1_addr and t_addr:
        addr_lev = fuzz.ratio(s1_addr, t_addr) / 100.0
        addr_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
        addr_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0

        s1_atok = set(s1_addr.split())
        t_atok = set(t_addr.split())
        atok_inter = s1_atok.intersection(t_atok)
        atok_union = s1_atok.union(t_atok)
        addr_jaccard = (len(atok_inter) / len(atok_union)) if atok_union else 0.0
        addr_common = float(len(atok_inter))

        pin_match = 1.0 if (s1_pin and t_pin and s1_pin == t_pin) else 0.0
        house_match = 1.0 if (s1_h and t_h and s1_h == t_h) else 0.0

        s1_dig = set(RE_DIGITS.findall(s1_addr))
        t_dig = set(RE_DIGITS.findall(t_addr))
        dig_inter = s1_dig.intersection(t_dig)
        dig_union = s1_dig.union(t_dig)
        digits_jaccard = (len(dig_inter) / len(dig_union)) if dig_union else 0.0

        addr_len_diff = float(abs(len(s1_addr) - len(t_addr)))
    else:
        addr_lev = addr_sort = addr_set = addr_jaccard = addr_common = 0.0
        pin_match = house_match = digits_jaccard = 0.0
        addr_len_diff = float(len(s1_addr)) if s1_addr else 0.0

    # 3. Cross-field interaction features
    source_flag = 1.0 if is_s2 else 0.0
    name_x_addr = name_lev * (addr_lev if not target_addr_missing else 0.5)
    high_name_high_addr = 1.0 if (name_lev >= 0.82 and addr_lev >= 0.70) else 0.0
    high_name_missing_addr = 1.0 if (name_lev >= 0.90 and target_addr_missing == 1.0) else 0.0

    return [
        name_lev,
        name_sort,
        name_set,
        name_part,
        exact_name,
        exact_core,
        name_jaccard,
        name_common,
        name_len_diff,
        name_len_ratio,
        addr_lev,
        addr_sort,
        addr_set,
        exact_addr,
        addr_jaccard,
        addr_common,
        pin_match,
        house_match,
        digits_jaccard,
        addr_len_diff,
        target_addr_missing,
        source_flag,
        name_x_addr,
        high_name_high_addr,
        high_name_missing_addr
    ]
