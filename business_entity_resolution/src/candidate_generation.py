import os
import time
import collections
from typing import Dict, List, Set, Any, Optional
from .blocking import MultiPassBlockingIndex

DELIM = "\t"

def generate_candidates_by_country(
    s1_records: Dict[str, Dict[str, Any]],
    target_records: Dict[str, Dict[str, Any]],
    max_candidates_per_s1: int = 50
) -> Dict[str, Set[str]]:
    """
    Partitions records by country, builds country-specific blocking indexes,
    and retrieves candidates for each S1 entity.
    """
    # Group S1 by country
    s1_by_country = collections.defaultdict(dict)
    for eid, r in s1_records.items():
        s1_by_country[r.get("country", "")][eid] = r

    # Group target records by country
    targets_by_country = collections.defaultdict(dict)
    for eid, r in target_records.items():
        targets_by_country[r.get("country", "")][eid] = r

    all_candidates = {}

    for ctry, s1_subset in s1_by_country.items():
        targets_subset = targets_by_country.get(ctry, {})
        print(f"\n--- Processing country [{ctry}]: {len(s1_subset):,} S1 queries, {len(targets_subset):,} candidate targets ---")
        
        if not targets_subset:
            print(f"  Warning: No targets found for country [{ctry}]. Singletons will be produced.")
            for eid in s1_subset:
                all_candidates[eid] = set()
            continue

        index = MultiPassBlockingIndex()
        t0 = time.time()
        index.build_index(targets_subset)
        t_index = time.time() - t0
        print(f"  Index built in {t_index:.2f}s.")

        t0 = time.time()
        for idx, (eid, s1_rec) in enumerate(s1_subset.items(), 1):
            cands = index.retrieve_candidates_for_query(s1_rec, max_candidates=max_candidates_per_s1)
            all_candidates[eid] = cands
            if idx % 100000 == 0 or idx == len(s1_subset):
                print(f"  Retrieved candidates for {idx:,}/{len(s1_subset):,} S1 queries...")
        t_query = time.time() - t0
        print(f"  Candidate retrieval completed in {t_query:.2f}s.")

    return all_candidates

def write_candidate_pairs_tsv(
    candidates: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str
):
    """
    Writes candidate_pairs.tsv in the exact required schema:
    source1_entity_id \t candidate_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, set())
            cand_str = ",".join(sorted(cands)) if cands else ""
            f.write(f"{s1_id}{DELIM}{cand_str}\n")
    print(f"Successfully wrote {len(all_s1_ids):,} rows to {output_path}")
