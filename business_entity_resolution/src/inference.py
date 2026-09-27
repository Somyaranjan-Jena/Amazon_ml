import os
import sys
import time
import collections
import joblib
import numpy as np
import xgboost as xgb
from typing import Dict, List, Set, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .data_loader import stream_source_file, load_compact_records
from .blocking import MultiPassBlockingIndex
from .pair_features import extract_pair_features, FEATURE_NAMES
from .normalization import (
    normalize_business_name,
    extract_core_business_name,
    normalize_address
)
from .address_features import extract_address_components

DELIM = "\t"

# Larger batch size for GPU — amortizes kernel launch overhead
GPU_BATCH_SIZE = 200000
CPU_BATCH_SIZE = 50000


def _detect_gpu() -> bool:
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        try:
            dtmp = xgb.DMatrix(np.zeros((2, 2), dtype=np.float32))
            bst = xgb.train({"device": "cuda", "verbosity": 0}, dtmp, num_boost_round=1)
            return True
        except Exception:
            return False


GPU_AVAILABLE = _detect_gpu()


def partition_test_files_by_country(test_dir: str, split_dir: str) -> List[str]:
    """
    Single-pass fast splitter that divides test_source1, test_source2, test_source3
    into per-country files in split_dir. Returns the sorted list of discovered countries.
    """
    os.makedirs(split_dir, exist_ok=True)
    t0 = time.time()
    print(f"Partitioning test sources by country into {split_dir}...", flush=True)

    countries = set()
    open_handles = {}

    for src_name in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
        src_path = os.path.join(test_dir, src_name)
        prefix = src_name.split(".")[0]
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n\r").split(DELIM)
                if len(parts) >= 4:
                    ctry = parts[3].strip()
                    countries.add(ctry)
                    key = (prefix, ctry)
                    if key not in open_handles:
                        out_path = os.path.join(split_dir, f"{prefix}_{ctry}.tsv")
                        h = open(out_path, "w", encoding="utf-8")
                        h.write(header)
                        open_handles[key] = h
                    open_handles[key].write(line)

    for h in open_handles.values():
        h.close()

    sorted_countries = sorted(countries)
    print(f"Partitioning completed in {time.time() - t0:.2f}s. Countries discovered: {sorted_countries}", flush=True)
    return sorted_countries


def _gpu_batch_predict(model, X_batch: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated batch prediction using XGBoost native API.
    Falls back to sklearn-compatible predict_proba for legacy models.
    """
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X_batch, feature_names=FEATURE_NAMES)
        probs = model.predict(dmat)
        return probs
    elif hasattr(model, 'booster') and isinstance(model.booster, xgb.Booster):
        # XGBWrapper
        dmat = xgb.DMatrix(X_batch, feature_names=FEATURE_NAMES)
        probs = model.booster.predict(dmat)
        return probs
    else:
        # Legacy LightGBM or sklearn model
        probs = model.predict_proba(X_batch)[:, 1]
        return probs


def _normalize_record(name: str, addr: str) -> Tuple[str, str, str, str, str]:
    """Normalize a single S1 record into compact tuple. Used for thread-pool mapping."""
    norm_name = normalize_business_name(name)
    core_name = extract_core_business_name(norm_name)
    norm_addr = normalize_address(addr)
    addr_comp = extract_address_components(norm_addr)
    return (norm_name, core_name, norm_addr, addr_comp["postal_code"], addr_comp["house_number"])


def _extract_features_for_candidates(
    s1_rec: Tuple, cands: Set[str], target_records: Dict
) -> List[Tuple[str, List[float]]]:
    """Extract pair features for all candidates of a single S1 query."""
    results = []
    for mid in cands:
        t_rec = target_records.get(mid)
        if t_rec:
            feats = extract_pair_features(s1_rec, t_rec, is_s2=mid.startswith("S2-"))
            results.append((mid, feats))
    return results


def _process_batch_results(
    probs: np.ndarray,
    batch_pairs_meta: list,
    batch_s1_cands: dict,
    threshold: float,
    f_match,
    f_cand,
) -> Tuple[int, int]:
    """Processes scored batch and writes results. Returns (written, matches) counts."""
    total_written = 0
    total_matches = 0

    # Pre-compute feature indices for deterministic override check
    feat_name_idx = FEATURE_NAMES.index("exact_name_match")
    feat_addr_idx = FEATURE_NAMES.index("exact_addr_match")

    s1_matches = collections.defaultdict(list)

    for idx, (s1_eid, t_eid, f_vals) in enumerate(batch_pairs_meta):
        p_score = float(probs[idx])

        is_match = False
        # Deterministic override: exact name + exact address
        if f_vals[feat_name_idx] == 1.0 and f_vals[feat_addr_idx] == 1.0:
            is_match = True
        elif p_score >= threshold:
            is_match = True

        if is_match:
            s1_matches[s1_eid].append(t_eid)

    # Write completed S1 records
    for s1_eid, cands_set in batch_s1_cands.items():
        clean_c = sorted({c for c in cands_set if c.startswith(("S2-", "S3-"))})
        c_str = ",".join(clean_c)
        f_cand.write(f"{s1_eid}{DELIM}{c_str}\n")

        pred_list = s1_matches.get(s1_eid, [])
        clean_m = sorted({m for m in pred_list if m in cands_set and m.startswith(("S2-", "S3-"))})
        m_str = ",".join(clean_m)
        f_match.write(f"{s1_eid}{DELIM}{m_str}\n")

        total_written += 1
        total_matches += len(clean_m)

    return total_written, total_matches


def _process_s1_chunk(
    chunk: List[Tuple[str, str, str, str]],
    index: MultiPassBlockingIndex,
    target_records: Dict,
    max_candidates: int
) -> Tuple[Dict[str, Set[str]], List[List[float]], List[Tuple[str, str, List[float]]]]:
    """
    Process a chunk of S1 queries: normalize, block, extract features.
    Returns (s1_cands, feats_list, meta_list).
    """
    s1_cands = {}
    feats_list = []
    meta_list = []
    
    for eid, name, addr, _ in chunk:
        s1_rec = _normalize_record(name, addr)
        cands = index.retrieve_candidates_for_query(s1_rec, max_candidates=max_candidates)
        s1_cands[eid] = cands
        
        for mid in cands:
            t_rec = target_records.get(mid)
            if t_rec:
                feats = extract_pair_features(s1_rec, t_rec, is_s2=mid.startswith("S2-"))
                feats_list.append(feats)
                meta_list.append((eid, mid, feats))
    
    return s1_cands, feats_list, meta_list


def run_test_inference_streaming(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = "cache/lgb_model.joblib",
    threshold: float = 0.85,
    max_candidates_per_s1: int = 50,
    batch_size: int = None,
    n_workers: int = None
):
    """
    High-performance GPU-accelerated streaming test inference:
      - Divides test files by country in one quick pass
      - Uses thread pool for parallel normalization + feature extraction
      - Runs blocking + XGBoost GPU batch scoring per country
      - Writes candidate pairs and matching results directly to disk
      - Uses larger batch sizes for GPU to maximize throughput
    """
    if batch_size is None:
        batch_size = GPU_BATCH_SIZE if GPU_AVAILABLE else CPU_BATCH_SIZE
    
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    t_start = time.time()
    device_str = "CUDA GPU" if GPU_AVAILABLE else "CPU"
    print("\n==========================================", flush=True)
    print(f"GPU-ACCELERATED STREAMING TEST INFERENCE", flush=True)
    print(f"Device: {device_str} | Threshold: {threshold:.2f} | Batch: {batch_size:,} | Workers: {n_workers}", flush=True)
    print("==========================================", flush=True)

    split_dir = "cache/test_country_splits"
    countries = partition_test_files_by_country(test_dir, split_dir)

    os.makedirs(output_dir, exist_ok=True)
    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Load trained model
    print(f"Loading trained model from {model_path}...", flush=True)
    model = joblib.load(model_path)
    print(f"  Model type: {type(model).__name__}", flush=True)

    # Open output files
    f_match = open(match_path, "w", encoding="utf-8")
    f_cand = open(cand_path, "w", encoding="utf-8")

    f_match.write(f"source1_entity_id{DELIM}matched_entity_ids\n")
    f_cand.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")

    total_written = 0
    total_matches_predicted = 0

    # Order countries by file size (smallest first for fast early feedback)
    ordered_countries = sorted(
        countries,
        key=lambda c: os.path.getsize(os.path.join(split_dir, f"test_source1_{c}.tsv"))
    )

    for ctry in ordered_countries:
        s1_file = os.path.join(split_dir, f"test_source1_{ctry}.tsv")
        s2_file = os.path.join(split_dir, f"test_source2_{ctry}.tsv")
        s3_file = os.path.join(split_dir, f"test_source3_{ctry}.tsv")

        print(f"\n==========================================", flush=True)
        print(f"PROCESSING COUNTRY: {ctry}", flush=True)
        print(f"==========================================", flush=True)

        t_c0 = time.time()
        print(f"  Loading target records...", flush=True)
        s2_records = load_compact_records(s2_file)
        s3_records = load_compact_records(s3_file)
        target_records = {**s2_records, **s3_records}
        print(f"  Loaded {len(target_records):,} target records for {ctry}.", flush=True)

        # Build priority-ranked blocking index
        index = MultiPassBlockingIndex()
        index.build_index(target_records)

        print(f"  Streaming S1 queries with {n_workers} workers + GPU scoring for {ctry}...", flush=True)

        batch_pairs_feats = []
        batch_pairs_meta = []
        batch_s1_cands = {}

        ctry_queries = 0
        ctry_written = 0
        ctry_matches = 0

        # Pre-read S1 records into micro-chunks for threaded processing
        micro_chunk = []
        MICRO_CHUNK_SIZE = 5000

        for eid, name, addr, ctry_val in stream_source_file(s1_file):
            ctry_queries += 1
            
            # Normalize inline (this is CPU-bound but fast per-record)
            s1_rec = _normalize_record(name, addr)
            
            cands = index.retrieve_candidates_for_query(s1_rec, max_candidates=max_candidates_per_s1)
            batch_s1_cands[eid] = cands

            for mid in cands:
                t_rec = target_records.get(mid)
                if t_rec:
                    feats = extract_pair_features(s1_rec, t_rec, is_s2=mid.startswith("S2-"))
                    batch_pairs_feats.append(feats)
                    batch_pairs_meta.append((eid, mid, feats))

            # GPU batch scoring — use larger batches for better GPU utilization
            if len(batch_pairs_feats) >= batch_size:
                X_b = np.array(batch_pairs_feats, dtype=np.float32)
                probs = _gpu_batch_predict(model, X_b)

                w, m = _process_batch_results(
                    probs, batch_pairs_meta, batch_s1_cands,
                    threshold, f_match, f_cand
                )
                ctry_written += w
                ctry_matches += m

                f_match.flush()
                f_cand.flush()
                batch_pairs_feats.clear()
                batch_pairs_meta.clear()
                batch_s1_cands.clear()

                if ctry_queries % 50000 == 0:
                    elapsed = time.time() - t_c0
                    rate = ctry_queries / max(1.0, elapsed)
                    print(f"  [{ctry}] {ctry_queries:,} queries ({rate:.1f} q/s), {ctry_matches:,} matches...", flush=True)

        # Process final remaining queries for country
        if batch_s1_cands:
            if batch_pairs_feats:
                X_b = np.array(batch_pairs_feats, dtype=np.float32)
                probs = _gpu_batch_predict(model, X_b)

                w, m = _process_batch_results(
                    probs, batch_pairs_meta, batch_s1_cands,
                    threshold, f_match, f_cand
                )
                ctry_written += w
                ctry_matches += m
            else:
                # S1 entities with 0 candidates — all singletons
                for s1_eid, cands_set in batch_s1_cands.items():
                    f_cand.write(f"{s1_eid}{DELIM}\n")
                    f_match.write(f"{s1_eid}{DELIM}\n")
                    ctry_written += 1

            f_match.flush()
            f_cand.flush()
            batch_pairs_feats.clear()
            batch_pairs_meta.clear()
            batch_s1_cands.clear()

        total_written += ctry_written
        total_matches_predicted += ctry_matches

        del s2_records, s3_records, target_records, index
        elapsed_c = time.time() - t_c0
        print(f"  Country {ctry} finished: {ctry_queries:,} queries, {ctry_matches:,} matches in {elapsed_c:.2f}s ({ctry_queries/max(1,elapsed_c):.1f} q/s)", flush=True)

    f_match.close()
    f_cand.close()

    total_elapsed = time.time() - t_start
    print(f"\n==========================================", flush=True)
    print("FULL TEST INFERENCE COMPLETED SUCCESSFULLY", flush=True)
    print(f"Device: {device_str}", flush=True)
    print(f"Total runtime: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(f"Total S1 entities written: {total_written:,}", flush=True)
    print(f"Total match links predicted: {total_matches_predicted:,}", flush=True)
    print(f"Matching Results: {match_path}", flush=True)
    print(f"Candidate Pairs:  {cand_path}", flush=True)
    print("==========================================", flush=True)
    return True
