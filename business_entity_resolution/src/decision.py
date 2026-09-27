import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from .metrics import evaluate_predictions

def predict_matches_from_scores(
    candidate_scores: Dict[str, List[Tuple[str, float, Dict[str, float]]]],
    threshold: float = 0.65,
    exact_match_override: bool = True
) -> Dict[str, Set[str]]:
    """
    Given {s1_id: [(candidate_id, score, feature_dict), ...]},
    produces {s1_id: set(matched_ids)}.
    Applies:
      1. Deterministic high-precision overrides (exact name + exact addr)
      2. Calibrated thresholding (score >= threshold)
      3. Singleton preservation (empty set if no candidates meet threshold)
    """
    predictions = {}
    for s1_id, scored_cands in candidate_scores.items():
        matched = set()
        for cand_id, score, feats in scored_cands:
            # Deterministic override for identical name + address
            if exact_match_override and feats.get("exact_name_match", 0.0) == 1.0 and feats.get("exact_addr_match", 0.0) == 1.0:
                matched.add(cand_id)
                continue
                
            if score >= threshold:
                matched.add(cand_id)

        predictions[s1_id] = matched
    return predictions

def optimize_decision_threshold(
    candidate_scores: Dict[str, List[Tuple[str, float, Dict[str, float]]]],
    ground_truth: Dict[str, Set[str]],
    grid_start: float = 0.35,
    grid_end: float = 0.85,
    grid_step: float = 0.05
) -> Tuple[float, float, Dict[float, float]]:
    """
    Searches threshold grid to maximize macro F0.5 on validation set.
    Returns (best_threshold, best_f05, all_scores).
    """
    best_thresh = 0.60
    best_f05 = -1.0
    history = {}

    thresholds = np.arange(grid_start, grid_end + 1e-5, grid_step)
    print(f"\n--- Optimizing Decision Threshold over [{grid_start:.2f}, {grid_end:.2f}] ---")

    for t in thresholds:
        t_val = round(float(t), 3)
        preds = predict_matches_from_scores(candidate_scores, threshold=t_val)
        res = evaluate_predictions(ground_truth, preds)
        macro_f05 = res["macro_f05"]
        history[t_val] = macro_f05
        print(f"  Threshold {t_val:.2f} -> Macro F0.5: {macro_f05:.4f} (Precision: {res['macro_precision']:.4f}, Recall: {res['macro_recall']:.4f}, Singletons: {res['singleton_accuracy']:.4f})")
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = t_val

    print(f"Optimal Threshold: {best_thresh:.2f} with Macro F0.5: {best_f05:.4f}")
    return best_thresh, best_f05, history
