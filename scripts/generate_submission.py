"""
Amazon ML Challenge 2026: Generate Final Submission
=====================================================
Re-runs inference with the pre-trained XGBoost model using
precision-optimized thresholding for F0.5 maximization.

Key improvements over baseline:
  1. Higher decision threshold (precision-heavy for F0.5)
  2. Score-based match capping (GT max is ~11; truncate by score)
  3. Full ID validation against test S2/S3 files
  4. Deterministic exact-match overrides
  5. Country-partitioned processing (no cross-country matches)

Usage:
    python scripts/generate_submission.py [--threshold 0.80] [--max-matches 12]
"""

import os
import sys
import time
import argparse
import collections
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from business_entity_resolution.src.normalization import (
    normalize_business_name,
    extract_core_business_name,
    normalize_address
)
from business_entity_resolution.src.address_features import extract_address_components
from business_entity_resolution.src.pair_features import extract_pair_features, FEATURE_NAMES
from business_entity_resolution.src.blocking import MultiPassBlockingIndex

DELIM = "\t"


def load_valid_ids(test_dir: str):
    """Load all valid S2 and S3 entity IDs from test files."""
    valid_s2 = set()
    valid_s3 = set()
    for fname, target_set in [("test_source2.tsv", valid_s2), ("test_source3.tsv", valid_s3)]:
        fpath = os.path.join(test_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            f.readline()  # skip header
            for line in f:
                eid = line.split(DELIM, 1)[0].strip()
                if eid:
                    target_set.add(eid)
    print(f"  Valid S2 IDs: {len(valid_s2):,}, Valid S3 IDs: {len(valid_s3):,}")
    return valid_s2, valid_s3


def stream_source_file(file_path):
    """Yields (entity_id, business_name, business_address, country)."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n\r").split(DELIM)
            if len(parts) < 4:
                parts = (parts + [""] * 4)[:4]
            yield parts[0], parts[1], parts[2], parts[3]


def normalize_record(name, addr):
    """Normalize a record into compact tuple for blocking + features."""
    norm_name = normalize_business_name(name)
    core_name = extract_core_business_name(norm_name)
    norm_addr = normalize_address(addr)
    addr_comp = extract_address_components(norm_addr)
    return (norm_name, core_name, norm_addr, addr_comp["postal_code"], addr_comp["house_number"])


def load_compact_records(file_path):
    """Load and normalize all records from a source file."""
    records = {}
    for eid, name, addr, ctry in stream_source_file(file_path):
        records[eid] = normalize_record(name, addr)
    return records


def partition_test_by_country(test_dir, split_dir):
    """Partition test files by country. Returns sorted list of countries."""
    os.makedirs(split_dir, exist_ok=True)
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
    print(f"  Countries discovered: {sorted_countries}")
    return sorted_countries


def gpu_batch_predict(model, X_batch):
    """Predict with the XGBoost model (handles both Booster and wrapper)."""
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X_batch, feature_names=FEATURE_NAMES)
        return model.predict(dmat)
    elif hasattr(model, "booster") and isinstance(model.booster, xgb.Booster):
        dmat = xgb.DMatrix(X_batch, feature_names=FEATURE_NAMES)
        bst = model.booster
        return bst.predict(dmat, iteration_range=(0, bst.best_iteration + 1))
    else:
        return model.predict_proba(X_batch)[:, 1]


def run_optimized_inference(
    test_dir="dataset/test",
    output_dir="output",
    model_path="cache/lgb_model.joblib",
    threshold=0.80,
    max_candidates_per_s1=50,
    max_matches_per_s1=12,
    batch_size=100000
):
    """
    Precision-optimized inference pipeline for F0.5 maximization.
    
    Key design decisions:
      - Higher threshold (0.80 vs 0.65) to minimize false positives
      - Match capping: At most max_matches_per_s1 per entity (sorted by score)
      - Deterministic override: exact name + exact address always matches
      - Country-partitioned: enforces country invariant
      - Full ID validation: only emits IDs that exist in test S2/S3
    """
    t_start = time.time()

    print("=" * 60)
    print("OPTIMIZED SUBMISSION GENERATION")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Max matches/entity: {max_matches_per_s1}")
    print(f"  Max candidates/entity: {max_candidates_per_s1}")
    print(f"  Batch size: {batch_size:,}")
    print("=" * 60)

    # Step 1: Load valid IDs
    print("\n[1/5] Loading valid S2/S3 entity IDs...")
    valid_s2, valid_s3 = load_valid_ids(test_dir)
    valid_target_ids = valid_s2 | valid_s3

    # Step 2: Partition by country (reuse if exists)
    split_dir = "cache/test_country_splits"
    print(f"\n[2/5] Partitioning test data by country...")
    if os.path.exists(split_dir) and any(f.endswith(".tsv") for f in os.listdir(split_dir)):
        # Discover countries from existing splits
        countries = set()
        for fname in os.listdir(split_dir):
            if fname.startswith("test_source1_") and fname.endswith(".tsv"):
                ctry = fname.replace("test_source1_", "").replace(".tsv", "")
                countries.add(ctry)
        countries = sorted(countries)
        print(f"  Reusing existing splits. Countries: {countries}")
    else:
        countries = partition_test_by_country(test_dir, split_dir)

    # Step 3: Load model
    print(f"\n[3/5] Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    print(f"  Model type: {type(model).__name__}")

    # Step 4: Open output files and process per country
    os.makedirs(output_dir, exist_ok=True)
    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Feature indices for deterministic overrides
    feat_exact_name_idx = FEATURE_NAMES.index("exact_name_match")
    feat_exact_addr_idx = FEATURE_NAMES.index("exact_addr_match")

    # Collect all results in memory for final sorted write
    all_results = {}  # s1_id -> (matched_ids_str, candidate_ids_str)

    print(f"\n[4/5] Running inference per country...")

    for ctry in countries:
        s1_file = os.path.join(split_dir, f"test_source1_{ctry}.tsv")
        s2_file = os.path.join(split_dir, f"test_source2_{ctry}.tsv")
        s3_file = os.path.join(split_dir, f"test_source3_{ctry}.tsv")

        t_c = time.time()
        print(f"\n  --- Country: {ctry} ---")

        # Load target records for this country
        s2_records = load_compact_records(s2_file) if os.path.exists(s2_file) else {}
        s3_records = load_compact_records(s3_file) if os.path.exists(s3_file) else {}
        target_records = {**s2_records, **s3_records}
        print(f"  Loaded {len(target_records):,} target records")

        # Build blocking index
        index = MultiPassBlockingIndex()
        if target_records:
            index.build_index(target_records)

        # Process S1 entities
        # Accumulate in batches for GPU scoring
        batch_feats = []
        batch_meta = []    # (s1_eid, target_eid, feat_vector)
        s1_cands = {}      # s1_eid -> set of candidate ids
        ctry_s1_count = 0
        ctry_matches = 0

        for eid, name, addr, _ in stream_source_file(s1_file):
            ctry_s1_count += 1
            s1_rec = normalize_record(name, addr)

            if target_records:
                cands = index.retrieve_candidates_for_query(s1_rec, max_candidates=max_candidates_per_s1)
                # Filter to only valid IDs
                cands = {c for c in cands if c in valid_target_ids}
            else:
                cands = set()

            s1_cands[eid] = cands

            for mid in cands:
                t_rec = target_records.get(mid)
                if t_rec:
                    feats = extract_pair_features(s1_rec, t_rec, is_s2=mid.startswith("S2-"))
                    batch_feats.append(feats)
                    batch_meta.append((eid, mid, feats))

            # Score batch when large enough
            if len(batch_feats) >= batch_size:
                _score_and_collect(
                    model, batch_feats, batch_meta, s1_cands,
                    threshold, max_matches_per_s1,
                    feat_exact_name_idx, feat_exact_addr_idx,
                    valid_target_ids, all_results
                )
                batch_feats.clear()
                batch_meta.clear()
                s1_cands.clear()

            if ctry_s1_count % 100000 == 0:
                elapsed = time.time() - t_c
                print(f"    Processed {ctry_s1_count:,} S1 entities ({ctry_s1_count/max(1,elapsed):.0f} q/s)...")

        # Process remaining batch
        if s1_cands:
            if batch_feats:
                _score_and_collect(
                    model, batch_feats, batch_meta, s1_cands,
                    threshold, max_matches_per_s1,
                    feat_exact_name_idx, feat_exact_addr_idx,
                    valid_target_ids, all_results
                )
            else:
                # All singletons in this remaining batch
                for s1_eid in s1_cands:
                    all_results[s1_eid] = ("", "")
            batch_feats.clear()
            batch_meta.clear()
            s1_cands.clear()

        del target_records, index
        elapsed_c = time.time() - t_c
        print(f"  Country {ctry}: {ctry_s1_count:,} S1 entities in {elapsed_c:.1f}s")

    # Step 5: Write output files with all S1 entities from test_source1
    print(f"\n[5/5] Writing output files...")

    # Verify 100% coverage by reading all S1 IDs from original test file
    all_s1_ids = []
    with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            eid = line.split(DELIM, 1)[0].strip()
            if eid:
                all_s1_ids.append(eid)

    missing = [eid for eid in all_s1_ids if eid not in all_results]
    if missing:
        print(f"  WARNING: {len(missing)} S1 entities missing from inference results! Adding as singletons.")
        for eid in missing:
            all_results[eid] = ("", "")

    # Write matching_results.tsv
    with open(match_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id{DELIM}matched_entity_ids\n")
        for s1_id in all_s1_ids:
            match_str, _ = all_results.get(s1_id, ("", ""))
            f.write(f"{s1_id}{DELIM}{match_str}\n")

    # Write candidate_pairs.tsv
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id{DELIM}candidate_entity_ids\n")
        for s1_id in all_s1_ids:
            _, cand_str = all_results.get(s1_id, ("", ""))
            f.write(f"{s1_id}{DELIM}{cand_str}\n")

    # Summary statistics
    total_matches = 0
    total_singletons = 0
    match_count_dist = collections.Counter()
    for s1_id in all_s1_ids:
        match_str, _ = all_results.get(s1_id, ("", ""))
        if not match_str:
            total_singletons += 1
            match_count_dist[0] += 1
        else:
            n = len(match_str.split(","))
            total_matches += n
            match_count_dist[n] += 1

    total_elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Total S1 entities: {len(all_s1_ids):,}")
    print(f"  Singletons (no match): {total_singletons:,} ({total_singletons/len(all_s1_ids)*100:.1f}%)")
    print(f"  Total match links: {total_matches:,}")
    print(f"  Avg matches/non-singleton: {total_matches/max(1, len(all_s1_ids)-total_singletons):.2f}")
    print(f"  Match distribution:")
    for k in sorted(match_count_dist.keys()):
        if k <= 12 or match_count_dist[k] > 100:
            print(f"    {k:2d} matches: {match_count_dist[k]:>10,} entities")
    print(f"\n  Output files:")
    print(f"    {match_path}")
    print(f"    {cand_path}")
    print("=" * 60)

    return match_path, cand_path


def _score_and_collect(
    model, batch_feats, batch_meta, s1_cands,
    threshold, max_matches_per_s1,
    feat_exact_name_idx, feat_exact_addr_idx,
    valid_target_ids, all_results
):
    """Score a batch of pairs and collect results into all_results dict."""
    X = np.array(batch_feats, dtype=np.float32)
    probs = gpu_batch_predict(model, X)

    # Group scores by S1 entity
    s1_scored = collections.defaultdict(list)

    for idx, (s1_eid, t_eid, feat_vec) in enumerate(batch_meta):
        score = float(probs[idx])
        is_exact = (feat_vec[feat_exact_name_idx] == 1.0 and feat_vec[feat_exact_addr_idx] == 1.0)
        s1_scored[s1_eid].append((t_eid, score, is_exact))

    # Process each S1 entity in this batch
    for s1_eid, cands_set in s1_cands.items():
        # Candidates string (all blocking candidates)
        clean_cands = sorted({c for c in cands_set if c in valid_target_ids and c.startswith(("S2-", "S3-"))})
        cand_str = ",".join(clean_cands)

        scored_list = s1_scored.get(s1_eid, [])

        # Apply threshold + deterministic overrides
        matched_with_scores = []
        for t_eid, score, is_exact in scored_list:
            if t_eid not in valid_target_ids:
                continue
            if not t_eid.startswith(("S2-", "S3-")):
                continue

            if is_exact:
                # Deterministic override: always match exact name+addr
                matched_with_scores.append((t_eid, 1.0))
            elif score >= threshold:
                matched_with_scores.append((t_eid, score))

        # Sort by score descending, cap at max_matches_per_s1
        matched_with_scores.sort(key=lambda x: x[1], reverse=True)
        if len(matched_with_scores) > max_matches_per_s1:
            matched_with_scores = matched_with_scores[:max_matches_per_s1]

        # Deduplicate and format
        seen = set()
        final_matches = []
        for t_eid, _ in matched_with_scores:
            if t_eid not in seen:
                seen.add(t_eid)
                final_matches.append(t_eid)

        match_str = ",".join(sorted(final_matches))
        all_results[s1_eid] = (match_str, cand_str)


def validate_output(test_dir, output_dir):
    """Quick self-validation of the output files."""
    print("\n" + "=" * 60)
    print("SELF-VALIDATION")
    print("=" * 60)

    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Load all S1 IDs
    s1_ids = set()
    with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1_ids.add(line.split(DELIM, 1)[0].strip())

    issues = []

    # Check matching_results.tsv
    seen_s1 = set()
    with open(match_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n\r")
        expected_header = f"source1_entity_id{DELIM}matched_entity_ids"
        if header != expected_header:
            issues.append(f"matching_results.tsv header mismatch: got {repr(header)}")

        for line_no, line in enumerate(f, 2):
            parts = line.rstrip("\n\r").split(DELIM)
            s1_id = parts[0]

            if s1_id in seen_s1:
                issues.append(f"Duplicate S1 row: {s1_id} at line {line_no}")
            seen_s1.add(s1_id)

            if s1_id not in s1_ids:
                issues.append(f"Unknown S1 ID: {s1_id}")

            matched = parts[1] if len(parts) > 1 else ""
            if matched.strip():
                ids = matched.split(",")
                id_set = set()
                for mid in ids:
                    mid = mid.strip()
                    if not mid:
                        continue
                    if mid.startswith("S1-"):
                        issues.append(f"S1 ID in matches for {s1_id}: {mid}")
                    if not mid.startswith(("S2-", "S3-")):
                        issues.append(f"Invalid prefix in matches for {s1_id}: {mid}")
                    if mid in id_set:
                        issues.append(f"Duplicate match ID for {s1_id}: {mid}")
                    id_set.add(mid)

    missing_s1 = s1_ids - seen_s1
    if missing_s1:
        issues.append(f"Missing {len(missing_s1)} S1 entities from matching_results.tsv")

    extra_s1 = seen_s1 - s1_ids
    if extra_s1:
        issues.append(f"Extra {len(extra_s1)} unknown S1 entities in matching_results.tsv")

    if issues:
        print(f"  FAILED — {len(issues)} issues found:")
        for iss in issues[:20]:
            print(f"    - {iss}")
    else:
        print("  PASS — All validation checks passed!")
        print(f"  Total S1 entities: {len(seen_s1):,}")

    return len(issues) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optimized submission for Amazon ML Challenge 2026")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Decision threshold for matching (higher = more precision)")
    parser.add_argument("--max-matches", type=int, default=12,
                        help="Maximum matches per S1 entity (GT max is ~11)")
    parser.add_argument("--max-candidates", type=int, default=50,
                        help="Maximum blocking candidates per S1 entity")
    parser.add_argument("--batch-size", type=int, default=100000,
                        help="Batch size for GPU prediction")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--model-path", type=str, default="cache/lgb_model.joblib")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    match_path, cand_path = run_optimized_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        max_candidates_per_s1=args.max_candidates,
        max_matches_per_s1=args.max_matches,
        batch_size=args.batch_size
    )

    if not args.skip_validate:
        validate_output(args.test_dir, args.output_dir)
