import numpy as np
from typing import Dict, List, Set, Tuple, Any

def compute_f05_single(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single S1 entity.
    Singleton rules:
      - True empty, Pred empty: precision=1.0, recall=1.0, f05=1.0
      - True empty, Pred non-empty: precision=0.0, recall=1.0, f05=0.0
      - True non-empty, Pred empty: precision=1.0, recall=0.0, f05=0.0
    """
    len_true = len(true_set)
    len_pred = len(pred_set)

    if len_true == 0 and len_pred == 0:
        return 1.0, 1.0, 1.0
    if len_true == 0 and len_pred > 0:
        return 0.0, 0.0, 1.0
    if len_true > 0 and len_pred == 0:
        return 0.0, 1.0, 0.0

    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len_pred
    recall = tp / len_true
    
    # F0.5 formula: (1 + 0.5^2) * P * R / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
    return f05, precision, recall

def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    entity_metadata: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes macro F0.5 across all S1 entities, including:
      - overall macro F0.5
      - singleton accuracy
      - non-singleton macro F0.5
      - precision and recall macro
      - source-specific metrics (S2-only, S3-only)
      - country-wise metrics (if metadata provided)
    """
    all_s1 = sorted(ground_truth.keys())
    f05_scores = []
    precision_scores = []
    recall_scores = []

    singleton_scores = []
    non_singleton_scores = []

    s2_f05 = []
    s3_f05 = []

    country_scores = {}

    for s1_id in all_s1:
        true_m = ground_truth[s1_id]
        pred_m = predictions.get(s1_id, set())

        f, p, r = compute_f05_single(true_m, pred_m)
        f05_scores.append(f)
        precision_scores.append(p)
        recall_scores.append(r)

        if len(true_m) == 0:
            singleton_scores.append(f)
        else:
            non_singleton_scores.append(f)

        # S2 only check
        true_s2 = {x for x in true_m if x.startswith("S2-")}
        pred_s2 = {x for x in pred_m if x.startswith("S2-")}
        if true_s2 or pred_s2:
            f2, _, _ = compute_f05_single(true_s2, pred_s2)
            s2_f05.append(f2)

        # S3 only check
        true_s3 = {x for x in true_m if x.startswith("S3-")}
        pred_s3 = {x for x in pred_m if x.startswith("S3-")}
        if true_s3 or pred_s3:
            f3, _, _ = compute_f05_single(true_s3, pred_s3)
            s3_f05.append(f3)

        if entity_metadata and s1_id in entity_metadata:
            ctry = entity_metadata[s1_id].get("country", "Unknown")
            if ctry not in country_scores:
                country_scores[ctry] = []
            country_scores[ctry].append(f)

    results = {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precision_scores)),
        "macro_recall": float(np.mean(recall_scores)),
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "s2_f05": float(np.mean(s2_f05)) if s2_f05 else 0.0,
        "s3_f05": float(np.mean(s3_f05)) if s3_f05 else 0.0,
        "total_evaluated": len(all_s1),
        "total_singletons": len(singleton_scores),
        "total_non_singletons": len(non_singleton_scores)
    }

    if country_scores:
        results["country_breakdown"] = {
            c: {"macro_f05": float(np.mean(sc)), "count": len(sc)}
            for c, sc in country_scores.items()
        }

    return results

def compute_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    total_comparison_space: int = None
) -> Dict[str, Any]:
    """
    Computes candidate generation metrics:
      - candidate recall (true matches in candidates / total true matches)
      - S2 recall, S3 recall
      - candidate count distribution per S1 (mean, median, p95, max)
      - candidate reduction ratio
    """
    total_true_matches = 0
    recalled_matches = 0

    s2_true = 0
    s2_recalled = 0
    s3_true = 0
    s3_recalled = 0

    cand_counts = []
    total_candidates_generated = 0

    for s1_id, true_m in ground_truth.items():
        cands = candidates.get(s1_id, set())
        n_cands = len(cands)
        cand_counts.append(n_cands)
        total_candidates_generated += n_cands

        for mid in true_m:
            total_true_matches += 1
            is_s2 = mid.startswith("S2-")
            if is_s2:
                s2_true += 1
            else:
                s3_true += 1

            if mid in cands:
                recalled_matches += 1
                if is_s2:
                    s2_recalled += 1
                else:
                    s3_recalled += 1

    arr = np.array(cand_counts)
    recall = (recalled_matches / total_true_matches) if total_true_matches > 0 else 0.0
    s2_rec = (s2_recalled / s2_true) if s2_true > 0 else 0.0
    s3_rec = (s3_recalled / s3_true) if s3_true > 0 else 0.0

    res = {
        "candidate_recall": float(recall),
        "s2_recall": float(s2_rec),
        "s3_recall": float(s3_rec),
        "total_true_matches": total_true_matches,
        "recalled_matches": recalled_matches,
        "avg_candidates_per_s1": float(arr.mean()),
        "median_candidates_per_s1": float(np.median(arr)),
        "p95_candidates_per_s1": float(np.percentile(arr, 95)),
        "max_candidates_per_s1": int(arr.max()) if len(arr) > 0 else 0,
        "total_candidates": total_candidates_generated
    }

    if total_comparison_space and total_comparison_space > 0:
        res["reduction_ratio"] = 1.0 - (total_candidates_generated / total_comparison_space)

    return res
