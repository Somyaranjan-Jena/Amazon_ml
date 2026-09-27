import os
from typing import Dict, List, Set

DELIM = "\t"

def write_matching_results_tsv(
    predictions: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str
):
    """
    Writes matching_results.tsv:
    source1_entity_id \t matched_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id{DELIM}matched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = predictions.get(s1_id, set())
            # Clean and ensure S2/S3 prefix only
            clean_matches = [m for m in sorted(matches) if m.startswith(("S2-", "S3-")) and not m.startswith("S1-")]
            # De-duplicate preserving order
            seen = set()
            unique_matches = []
            for m in clean_matches:
                if m not in seen:
                    seen.add(m)
                    unique_matches.append(m)
            m_str = ",".join(unique_matches)
            f.write(f"{s1_id}{DELIM}{m_str}\n")
    print(f"  Wrote {len(all_s1_ids):,} rows to {output_path}")

def write_candidate_pairs_tsv(
    candidates: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str
):
    """
    Writes candidate_pairs.tsv:
    source1_entity_id \t candidate_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, set())
            clean_cands = [c for c in sorted(cands) if c.startswith(("S2-", "S3-")) and not c.startswith("S1-")]
            seen = set()
            unique_cands = []
            for c in clean_cands:
                if c not in seen:
                    seen.add(c)
                    unique_cands.append(c)
            c_str = ",".join(unique_cands)
            f.write(f"{s1_id}{DELIM}{c_str}\n")
    print(f"  Wrote {len(all_s1_ids):,} rows to {output_path}")
