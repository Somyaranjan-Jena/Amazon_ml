import random
import collections
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from .metrics import evaluate_predictions, compute_blocking_recall
from .pair_features import extract_pair_features, FEATURE_NAMES
from .decision import optimize_decision_threshold, predict_matches_from_scores

def create_s1_validation_split(
    s1_records: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, Set[str]],
    val_size: int = 50000,
    random_seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Creates a stratified S1-level split preserving:
      - Country distribution
      - Singleton ratio
    Returns (train_s1_ids, val_s1_ids).
    """
    random.seed(random_seed)
    
    # Group by (country, is_singleton)
    groups = collections.defaultdict(list)
    for s1_id, r in s1_records.items():
        ctry = r.get("country", "")
        is_singleton = (len(ground_truth.get(s1_id, set())) == 0)
        groups[(ctry, is_singleton)].append(s1_id)

    total_s1 = len(s1_records)
    ratio = min(1.0, val_size / total_s1) if val_size < total_s1 else 0.25

    val_ids = set()
    train_ids = set()

    for grp_key, id_list in groups.items():
        random.shuffle(id_list)
        n_val = int(round(len(id_list) * ratio))
        val_ids.update(id_list[:n_val])
        train_ids.update(id_list[n_val:])

    print(f"Created S1-level split: {len(train_ids):,} train S1, {len(val_ids):,} validation S1")
    return train_ids, val_ids

def run_validation_pipeline(
    val_s1_records: Dict[str, Dict[str, Any]],
    target_records: Dict[str, Dict[str, Any]],
    val_candidates: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    model: Any
) -> Dict[str, Any]:
    """
    Simulates full inference on the validation set:
      1. Scores each (val S1, candidate) pair with model.
      2. Optimizes decision threshold on macro F0.5.
      3. Produces comprehensive evaluation report.
    """
    val_ids = set(val_s1_records.keys())
    val_gt = {k: ground_truth.get(k, set()) for k in val_ids}

    print(f"\n--- Running Full Validation Simulation ({len(val_s1_records):,} S1 queries) ---")

    # 1. Compute Candidate Recall on validation subset
    blocking_metrics = compute_blocking_recall(val_gt, val_candidates)
    print(f"Candidate Generation Metrics on Validation Set:")
    print(f"  Candidate Recall: {blocking_metrics['candidate_recall']*100:.2f}% (S2: {blocking_metrics['s2_recall']*100:.2f}%, S3: {blocking_metrics['s3_recall']*100:.2f}%)")
    print(f"  Avg candidates/S1: {blocking_metrics['avg_candidates_per_s1']:.1f}, Median: {blocking_metrics['median_candidates_per_s1']:.1f}, P95: {blocking_metrics['p95_candidates_per_s1']:.1f}")

    # 2. Extract features and predict probabilities
    candidate_scores = {}
    pair_features_batch = []
    pair_metadata = []

    for s1_id, s1_rec in val_s1_records.items():
        cands = val_candidates.get(s1_id, set())
        candidate_scores[s1_id] = []
        for mid in cands:
            t_rec = target_records.get(mid)
            if t_rec:
                feats = extract_pair_features(s1_rec, t_rec, is_s2=mid.startswith("S2-"))
                pair_features_batch.append(feats)
                pair_metadata.append((s1_id, mid, feats))

    if pair_features_batch:
        X_pairs = np.array(pair_features_batch, dtype=np.float32)
        print(f"  Scoring {X_pairs.shape[0]:,} validation candidate pairs with model...")
        probs = model.predict_proba(X_pairs)[:, 1]

        feat_dict_keys = FEATURE_NAMES
        for idx, (s1_id, mid, feats) in enumerate(pair_metadata):
            score = float(probs[idx])
            f_dict = dict(zip(feat_dict_keys, feats))
            candidate_scores[s1_id].append((mid, score, f_dict))

    # 3. Optimize Threshold
    best_thresh, best_f05, history = optimize_decision_threshold(candidate_scores, val_gt)

    # 4. Final Evaluation with best threshold
    final_preds = predict_matches_from_scores(candidate_scores, threshold=best_thresh)
    eval_results = evaluate_predictions(val_gt, final_preds, entity_metadata=val_s1_records)

    eval_results["blocking"] = blocking_metrics
    eval_results["best_threshold"] = best_thresh
    eval_results["threshold_history"] = history

    print("\n==========================================")
    print("FINAL VALIDATION PERFORMANCE REPORT")
    print("==========================================")
    print(f"Macro F0.5:           {eval_results['macro_f05']:.4f}")
    print(f"Macro Precision:      {eval_results['macro_precision']:.4f}")
    print(f"Macro Recall:         {eval_results['macro_recall']:.4f}")
    print(f"Singleton Accuracy:   {eval_results['singleton_accuracy']:.4f}")
    print(f"Non-Singleton F0.5:   {eval_results['non_singleton_f05']:.4f}")
    print(f"S2 Macro F0.5:        {eval_results['s2_f05']:.4f}")
    print(f"S3 Macro F0.5:        {eval_results['s3_f05']:.4f}")
    if "country_breakdown" in eval_results:
        print("Country Breakdown:")
        for ctry, c_data in eval_results["country_breakdown"].items():
            print(f"  {ctry:10s}: Macro F0.5 = {c_data['macro_f05']:.4f} ({c_data['count']:,} entities)")

    return eval_results
