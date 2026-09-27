import os
import sys
from typing import Dict, List, Set, Tuple, Iterator, Any, Optional
from .normalization import (
    normalize_business_name,
    extract_core_business_name,
    normalize_address
)
from .address_features import extract_address_components

DELIM = "\t"

def stream_source_file(file_path: str) -> Iterator[Tuple[str, str, str, str]]:
    """Yields (entity_id, business_name, business_address, country) from a TSV source file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n\r").split(DELIM)
            if len(parts) < 4:
                parts = (parts + [""] * 4)[:4]
            yield parts[0], parts[1], parts[2], parts[3]

def load_compact_records(
    file_path: str,
    target_ids: Optional[Set[str]] = None,
    filter_country: Optional[str] = None,
    max_records: Optional[int] = None
) -> Dict[str, Tuple[str, str, str, str, str]]:
    """
    Loads source records as compact tuples:
      eid -> (norm_name, core_name, norm_addr, postal_code, house_number)
    Saves ~80% memory compared to full python dicts.
    """
    records = {}
    count = 0
    for eid, name, addr, ctry in stream_source_file(file_path):
        if target_ids is not None and eid not in target_ids:
            continue
        ctry_clean = ctry.strip()
        if filter_country is not None and ctry_clean != filter_country:
            continue

        norm_name = normalize_business_name(name)
        core_name = extract_core_business_name(norm_name)
        norm_addr = normalize_address(addr)
        addr_comp = extract_address_components(norm_addr)

        records[eid] = (
            norm_name,
            core_name,
            norm_addr,
            addr_comp["postal_code"],
            addr_comp["house_number"]
        )
        count += 1
        if max_records and count >= max_records:
            break
    return records

def load_ground_truth(file_path: str) -> Dict[str, Set[str]]:
    """Loads ground truth mapping {s1_id: set(matched_ids)}."""
    gt = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n\r").split(DELIM)
            s1_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if rest.strip():
                gt[s1_id] = {m.strip() for m in rest.split(",") if m.strip()}
            else:
                gt[s1_id] = set()
    return gt
