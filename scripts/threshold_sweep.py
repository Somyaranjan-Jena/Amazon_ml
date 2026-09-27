"""
Quick threshold sweep on training data to find optimal F0.5 threshold.
Uses a random sample of training entities for speed.
"""
import os
import sys
import time
import random
import collections
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from business_entity_resolution.src.normalization import (
    normalize_business_name, extract_core_business_name, normalize_address
)
from business_entity_resolution.src.address_features import extract_address_components
from business_entity_resolution.src.pair_features import extract_pair_features, FEATURE_NAMES
from business_entity_resolution.src.blocking import MultiPassBlockingIndex
from business_entity_resolution.src.data_loader import load_ground_truth, stream_source_file

DELIM = "\t"
random.seed(42)


def normalize_record(name, addr):
    norm_name = normalize_business_name(name)
    core_name = extract_core_business_name(norm_name)
    norm_addr = normalize_address(addr)
    addr_comp = extract_address_components(norm_addr)
    return (norm_name, core_name, norm_addr, addr_comp["postal_code"], addr_comp["house_number"])


def compute_f05(true_set, pred_set):
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 and len(pred_set) > 0:
        return 0.0
    if len(true_set) > 0 and len(pred_set) == 0:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1.25 * p * r) / (0.25 * p + r)


def main():
    t0 = time.time()
    N_SAMPLE = 20000  # entities to evaluate

    print("Loading ground truth...")
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")

    # Sample S1 entities (stratified: include singletons and non-singletons)
    all_s1_ids = list(gt.keys())
    singletons = [k for k in all_s1_ids if len(gt[k]) == 0]
    non_singletons = [k for k in all_s1_ids if len(gt[k]) > 0]

    random.shuffle(singletons)
    random.shuffle(non_singletons)

    # Maintain realistic singleton ratio (~5.6%)
    n_singleton_sample = int(N_SAMPLE * 0.06)
    n_non_singleton_sample = N_SAMPLE - n_singleton_sample
    sample_ids = set(singletons[:n_singleton_sample] + non_singletons[:n_non_singleton_sample])

    print(f"Sampled {len(sample_ids):,} S1 entities ({n_singleton_sample} singletons, {n_non_singleton_sample} non-singletons)")

    # Find needed target IDs
    needed_targets = set()
    for s1_id in sample_ids:
        needed_targets.update(gt[s1_id])

    # Load S1 records with country info
    print("Loading S1 records...")
    s1_records = {}
    s1_countries = {}
    for eid, name, addr, ctry in stream_source_file("dataset/train/train_source1.tsv"):
        if eid in sample_ids:
            s1_records[eid] = normalize_record(name, addr)
            s1_countries[eid] = ctry.strip()
            if len(s1_records) >= len(sample_ids):
                break

    # Load target records (GT matches + background for blocking)
    print("Loading target records...")
    s2_records = {}
    s2_countries = {}
    for eid, name, addr, ctry in stream_source_file("dataset/train/train_source2.tsv"):
        if eid in needed_targets or len(s2_records) < 150000:
            s2_records[eid] = normalize_record(name, addr)
            s2_countries[eid] = ctry.strip()

    s3_records = {}
    s3_countries = {}
    for eid, name, addr, ctry in stream_source_file("dataset/train/train_source3.tsv"):
        if eid in needed_targets or len(s3_records) < 150000:
            s3_records[eid] = normalize_record(name, addr)
            s3_countries[eid] = ctry.strip()

    all_targets = {**s2_records, **s3_records}
    all_target_countries = {**s2_countries, **s3_countries}
    print(f"Loaded {len(all_targets):,} target records")

    # Load model
    print("Loading model...")
    model = joblib.load("cache/lgb_model.joblib")
    bst = model.booster

    # Build per-country blocking indexes and run inference
    print("Building blocking indexes and scoring pairs...")

    # Group by country
    country_groups = collections.defaultdict(list)
    for eid in s1_records:
        country_groups[s1_countries.get(eid, "US")].append(eid)

    # Score all pairs
    s1_pair_scores = collections.defaultdict(list)  # s1_id -> [(target_id, score, is_exact)]
    s1_cands_map = {}

    feat_exact_name = FEATURE_NAMES.index("exact_name_match")
    feat_exact_addr = FEATURE_NAMES.index("exact_addr_match")

    for ctry, eids in country_groups.items():
        # Build country-specific target set
        ctry_targets = {k: v for k, v in all_targets.items() if all_target_countries.get(k) == ctry}
        if not ctry_targets:
            for eid in eids:
                s1_cands_map[eid] = set()
            continue

        index = MultiPassBlockingIndex()
        index.build_index(ctry_targets)

        batch_feats = []
        batch_meta = []

        for eid in eids:
            s1_rec = s1_records[eid]
            cands = index.retrieve_candidates_for_query(s1_rec, max_candidates=50)
            s1_cands_map[eid] = cands

            for mid in cands:
                t_rec = ctry_targets.get(mid)
                if t_rec:
                    feats = extract_pair_features(s1_rec, t_rec, is_s2=mid.startswith("S2-"))
                    batch_feats.append(feats)
                    batch_meta.append((eid, mid, feats))

        if batch_feats:
            X = np.array(batch_feats, dtype=np.float32)
            dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
            probs = bst.predict(dmat, iteration_range=(0, bst.best_iteration + 1))

            for idx, (s1_eid, t_eid, feat_vec) in enumerate(batch_meta):
                score = float(probs[idx])
                is_exact = (feat_vec[feat_exact_name] == 1.0 and feat_vec[feat_exact_addr] == 1.0)
                s1_pair_scores[s1_eid].append((t_eid, score, is_exact))

    # Compute blocking recall first
    total_true = 0
    total_recalled = 0
    for s1_id in s1_records:
        true_m = gt.get(s1_id, set())
        cands = s1_cands_map.get(s1_id, set())
        for mid in true_m:
            total_true += 1
            if mid in cands:
                total_recalled += 1
    blocking_recall = total_recalled / max(1, total_true)
    print(f"\nBlocking recall: {blocking_recall*100:.2f}% ({total_recalled}/{total_true})")

    # Threshold sweep
    print("\n" + "=" * 70)
    print(f"{'Threshold':>10} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | {'Singleton Acc':>13} | {'Avg Matches':>11}")
    print("-" * 70)

    best_thresh = 0.0
    best_f05 = -1.0

    for thresh_int in range(40, 96, 5):
        thresh = thresh_int / 100.0

        f05_scores = []
        total_pred_matches = 0
        total_entities = 0
        singleton_correct = 0
        total_singletons = 0

        for s1_id in s1_records:
            true_m = gt.get(s1_id, set())
            scored = s1_pair_scores.get(s1_id, [])

            pred_set = set()
            for t_eid, score, is_exact in scored:
                if is_exact or score >= thresh:
                    pred_set.add(t_eid)

            f = compute_f05(true_m, pred_set)
            f05_scores.append(f)
            total_pred_matches += len(pred_set)
            total_entities += 1

            if len(true_m) == 0:
                total_singletons += 1
                if len(pred_set) == 0:
                    singleton_correct += 1

        macro_f05 = np.mean(f05_scores)
        avg_matches = total_pred_matches / max(1, total_entities)
        singleton_acc = singleton_correct / max(1, total_singletons)

        # Also compute macro precision and recall
        p_scores = []
        r_scores = []
        for s1_id in s1_records:
            true_m = gt.get(s1_id, set())
            scored = s1_pair_scores.get(s1_id, [])
            pred_set = set()
            for t_eid, score, is_exact in scored:
                if is_exact or score >= thresh:
                    pred_set.add(t_eid)
            if len(true_m) == 0 and len(pred_set) == 0:
                p_scores.append(1.0); r_scores.append(1.0)
            elif len(true_m) == 0:
                p_scores.append(0.0); r_scores.append(1.0)
            elif len(pred_set) == 0:
                p_scores.append(1.0); r_scores.append(0.0)
            else:
                tp = len(true_m & pred_set)
                p_scores.append(tp / len(pred_set))
                r_scores.append(tp / len(true_m))

        macro_p = np.mean(p_scores)
        macro_r = np.mean(r_scores)

        marker = " <-- BEST" if macro_f05 > best_f05 else ""
        print(f"    {thresh:.2f}   | {macro_f05:10.4f} | {macro_p:10.4f} | {macro_r:10.4f} | {singleton_acc:13.4f} | {avg_matches:11.2f}{marker}")

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh

    print("-" * 70)
    print(f"  OPTIMAL THRESHOLD: {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f})")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
