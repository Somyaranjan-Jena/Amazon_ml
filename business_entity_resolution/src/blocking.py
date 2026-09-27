import collections
from typing import Dict, List, Set, Tuple, Any, Union
from .address_features import COMMON_ADDR_STOPWORDS

# Compact record: (norm_name, core_name, norm_addr, postal_code, house_number)

class MultiPassBlockingIndex:
    """
    Priority-Ranked Multi-Pass Blocking Index:
      - Accumulates priority points for candidates across passes
      - Ranks candidates by match quality before truncation
      - Yields optimal candidate recall and precision
    """
    def __init__(
        self,
        max_name_df: int = 15000,
        max_addr_df: int = 8000,
        max_bucket_size: int = 250
    ):
        self.max_name_df = max_name_df
        self.max_addr_df = max_addr_df
        self.max_bucket_size = max_bucket_size
        
        # Inverted index tables
        self.idx_exact_name = collections.defaultdict(list)
        self.idx_exact_addr = collections.defaultdict(list)
        self.idx_name_skeleton = collections.defaultdict(list)
        self.idx_name_tokens = collections.defaultdict(list)
        self.idx_house_token = collections.defaultdict(list)
        
        # Document frequencies
        self.name_token_df = collections.Counter()
        self.addr_token_df = collections.Counter()

    def build_index(self, target_records: Dict[str, Union[Tuple, Dict[str, Any]]]):
        """Indexes target records (S2 and S3)."""
        print(f"  Building priority-ranked blocking index over {len(target_records):,} target records...")
        
        # Pass 1: Compute token document frequencies
        for eid, r in target_records.items():
            if isinstance(r, tuple):
                norm_name, core_name, norm_addr, pin, house = r
            else:
                core_name = r.get("core_name", "")
                norm_addr = r.get("norm_addr", "")
                house = r.get("house_number", "")

            if core_name:
                for t in set(core_name.split()):
                    if len(t) >= 3:
                        self.name_token_df[t] += 1
            if norm_addr:
                for at in set(norm_addr.split()):
                    if len(at) >= 3 and at not in COMMON_ADDR_STOPWORDS and not at.isdigit():
                        self.addr_token_df[at] += 1
                    
        # Pass 2: Populate inverted index tables
        for eid, r in target_records.items():
            if isinstance(r, tuple):
                norm_name, core_name, norm_addr, pin, house = r
            else:
                core_name = r.get("core_name", "")
                norm_addr = r.get("norm_addr", "")
                house = r.get("house_number", "")
            
            # Pass A: Exact Core Name
            if core_name and len(core_name) >= 3:
                if len(self.idx_exact_name[core_name]) < self.max_bucket_size:
                    self.idx_exact_name[core_name].append(eid)
                    
            # Pass B: Exact Address
            if norm_addr and len(norm_addr) >= 8:
                if len(self.idx_exact_addr[norm_addr]) < self.max_bucket_size:
                    self.idx_exact_addr[norm_addr].append(eid)
                    
            # Pass C: Alphanumeric Skeleton
            if core_name and len(core_name) >= 4:
                skel = "".join(c for c in core_name if c.isalnum())
                if len(skel) >= 4:
                    if len(self.idx_name_skeleton[skel]) < self.max_bucket_size:
                        self.idx_name_skeleton[skel].append(eid)

            # Pass D: Rare Name Tokens
            if core_name:
                for t in set(core_name.split()):
                    if len(t) >= 3 and self.name_token_df[t] <= self.max_name_df:
                        if len(self.idx_name_tokens[t]) < self.max_bucket_size:
                            self.idx_name_tokens[t].append(eid)
                                
            # Pass E: Compound House Number + Distinctive Address Token
            if house and norm_addr:
                for at in set(norm_addr.split()):
                    if len(at) >= 3 and at not in COMMON_ADDR_STOPWORDS and not at.isdigit():
                        if self.addr_token_df[at] <= self.max_addr_df:
                            k = f"{house}@{at}"
                            if len(self.idx_house_token[k]) < self.max_bucket_size:
                                self.idx_house_token[k].append(eid)

        print(f"  Enhanced index built: {len(self.idx_exact_name):,} names, {len(self.idx_name_tokens):,} rare name tokens, {len(self.idx_house_token):,} house-token keys.")

    def retrieve_candidates_for_query(
        self,
        query: Union[Tuple, Dict[str, Any]],
        max_candidates: int = 65
    ) -> Set[str]:
        """
        Retrieves candidate entity IDs for a single S1 query.
        Accumulates priority points and ranks candidates before truncation.
        """
        cand_scores = collections.defaultdict(float)

        if isinstance(query, tuple):
            norm_name, core_name, norm_addr, pin, house = query
        else:
            core_name = query.get("core_name", "")
            norm_addr = query.get("norm_addr", "")
            house = query.get("house_number", "")
        
        # 1. Exact core name (Highest priority: +10.0)
        if core_name in self.idx_exact_name:
            for cid in self.idx_exact_name[core_name]:
                cand_scores[cid] += 10.0
            
        # 2. Exact address (+10.0)
        if norm_addr and norm_addr in self.idx_exact_addr:
            for cid in self.idx_exact_addr[norm_addr]:
                cand_scores[cid] += 10.0
            
        # 3. Skeleton match (+6.0)
        if core_name and len(core_name) >= 4:
            skel = "".join(c for c in core_name if c.isalnum())
            if skel in self.idx_name_skeleton:
                for cid in self.idx_name_skeleton[skel]:
                    cand_scores[cid] += 6.0
                
        # 4. Rare name tokens (+3.0 per matching token)
        if core_name:
            tokens = [t for t in set(core_name.split()) if len(t) >= 3 and t in self.idx_name_tokens]
            tokens.sort(key=lambda t: self.name_token_df[t])
            for t in tokens[:4]:
                for cid in self.idx_name_tokens[t]:
                    cand_scores[cid] += 3.0

        # 5. House Number + Distinctive Address Token (+3.0 per matching token)
        if house and norm_addr:
            addr_tokens = [
                at for at in set(norm_addr.split())
                if len(at) >= 3 and at not in COMMON_ADDR_STOPWORDS and not at.isdigit() and f"{house}@{at}" in self.idx_house_token
            ]
            addr_tokens.sort(key=lambda at: self.addr_token_df[at])
            for at in addr_tokens[:4]:
                for cid in self.idx_house_token[f"{house}@{at}"]:
                    cand_scores[cid] += 3.0

        if not cand_scores:
            return set()

        if len(cand_scores) <= max_candidates:
            return set(cand_scores.keys())

        # Source-aware ranked selection
        s2_items = [(cid, sc) for cid, sc in cand_scores.items() if cid.startswith("S2-")]
        s3_items = [(cid, sc) for cid, sc in cand_scores.items() if cid.startswith("S3-")]

        s2_items.sort(key=lambda x: x[1], reverse=True)
        s3_items.sort(key=lambda x: x[1], reverse=True)

        half = max_candidates // 2
        selected = {cid for cid, _ in s2_items[:half]}
        selected.update({cid for cid, _ in s3_items[:half]})

        # Fill remaining slots with the highest remaining scores overall
        if len(selected) < max_candidates:
            remaining = [
                (cid, sc) for cid, sc in cand_scores.items() if cid not in selected
            ]
            remaining.sort(key=lambda x: x[1], reverse=True)
            needed = max_candidates - len(selected)
            selected.update({cid for cid, _ in remaining[:needed]})

        return selected
