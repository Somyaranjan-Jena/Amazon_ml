import os
import sys
import argparse
import time
import subprocess
import collections
import random
from .config import Config
from .data_loader import load_compact_records, load_ground_truth, stream_source_file
from .candidate_generation import generate_candidates_by_country
from .train import construct_training_dataset, train_lightgbm_model
from .validation import create_s1_validation_split, run_validation_pipeline
from .inference import run_test_inference_streaming

def mode_train_and_validate(config: Config, n_samples: int = 50000):
    t0 = time.time()
    print("==========================================")
    print(f"MODE: TRAIN & VALIDATE (n_samples={n_samples:,})")
    print("==========================================")

    data_cfg = config.data
    gt_path = data_cfg.get("train_ground_truth", "dataset/train/train_ground_truth.tsv")
    s1_path = data_cfg.get("train_source1", "dataset/train/train_source1.tsv")
    s2_path = data_cfg.get("train_source2", "dataset/train/train_source2.tsv")
    s3_path = data_cfg.get("train_source3", "dataset/train/train_source3.tsv")

    print(f"Loading ground truth from {gt_path}...")
    ground_truth = load_ground_truth(gt_path)
    print(f"Loaded {len(ground_truth):,} ground truth entries.")

    print(f"Selecting stratified sample of {n_samples:,} S1 entities...")
    target_s1_ids = set()
    s1_countries = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) >= 4:
                eid, ctry = parts[0], parts[3].strip()
                target_s1_ids.add(eid)
                s1_countries[eid] = ctry
                if len(target_s1_ids) >= n_samples:
                    break

    s1_records = load_compact_records(s1_path, target_ids=target_s1_ids)
    # inject country back into record representation for partitioning
    s1_rec_with_country = {}
    for eid, rec in s1_records.items():
        s1_rec_with_country[eid] = {
            "entity_id": eid,
            "norm_name": rec[0],
            "core_name": rec[1],
            "norm_addr": rec[2],
            "postal_code": rec[3],
            "house_number": rec[4],
            "country": s1_countries.get(eid, "US")
        }
    print(f"Loaded {len(s1_records):,} compact S1 records.")

    # Split into train and validation
    train_s1_ids, val_s1_ids = create_s1_validation_split(
        s1_rec_with_country, ground_truth, val_size=int(n_samples * 0.25)
    )

    # Collect needed target IDs from ground truth
    needed_target_ids = set()
    for s1_id in target_s1_ids:
        needed_target_ids.update(ground_truth.get(s1_id, set()))
    print(f"Known ground truth targets for sample: {len(needed_target_ids):,}")

    # Load targets: all known GT targets + background records for realistic candidate generation
    print("Loading target records (GT matches + background distractors)...")
    s2_targets = load_compact_records(s2_path, target_ids=needed_target_ids)
    s3_targets = load_compact_records(s3_path, target_ids=needed_target_ids)
    
    # Add background distractor records — more distractors = harder negatives = better model
    s2_bg = load_compact_records(s2_path, max_records=200000)
    s3_bg = load_compact_records(s3_path, max_records=200000)
    
    # Merge all targets
    all_target_records = {**s2_targets, **s3_targets, **s2_bg, **s3_bg}
    print(f"Total target pool loaded: {len(all_target_records):,} records.")

    # Scan countries for loaded target IDs
    target_countries = {}
    for p in (s2_path, s3_path):
        with open(p, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) >= 4 and parts[0] in all_target_records:
                    target_countries[parts[0]] = parts[3].strip()

    target_rec_with_country = {}
    for eid, rec in all_target_records.items():
        target_rec_with_country[eid] = {
            "entity_id": eid,
            "norm_name": rec[0],
            "core_name": rec[1],
            "norm_addr": rec[2],
            "postal_code": rec[3],
            "house_number": rec[4],
            "country": target_countries.get(eid, "US")
        }

    # Generate candidates for train S1
    print("\nGenerating candidates for training subset...")
    train_s1_dict = {k: s1_rec_with_country[k] for k in train_s1_ids}
    train_candidates = generate_candidates_by_country(train_s1_dict, target_rec_with_country, max_candidates_per_s1=40)

    # Train GPU-accelerated model
    X_train, y_train = construct_training_dataset(
        train_s1_dict, target_rec_with_country, train_candidates, ground_truth, negative_ratio=3.0
    )
    model = train_lightgbm_model(X_train, y_train, model_save_path="cache/lgb_model.joblib")

    # Generate candidates for validation S1
    print("\nGenerating candidates for validation subset (blind)...")
    val_s1_dict = {k: s1_rec_with_country[k] for k in val_s1_ids}
    val_candidates = generate_candidates_by_country(val_s1_dict, target_rec_with_country, max_candidates_per_s1=50)

    # Run full validation simulation and threshold tuning
    val_report = run_validation_pipeline(val_s1_dict, target_rec_with_country, val_candidates, ground_truth, model)

    print(f"\nTraining and Validation completed in {time.time() - t0:.2f}s.")
    return val_report

def mode_predict(config: Config, threshold: float = 0.85):
    print("==========================================")
    print(f"MODE: PREDICT (threshold={threshold:.2f})")
    print("==========================================")
    from .inference import run_test_inference_streaming
    run_test_inference_streaming(
        test_dir="dataset/test",
        output_dir="output",
        model_path="cache/lgb_model.joblib",
        threshold=threshold,
        max_candidates_per_s1=50
    )

def mode_validate_submission():
    print("\n==========================================")
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("==========================================")
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    if res.stderr:
        print("STDERR:")
        print(res.stderr)
    return res.returncode

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline (GPU-Accelerated)")
    parser.add_argument("--mode", type=str, choices=["train", "validate", "predict", "all", "check"], default="all")
    parser.add_argument("--config", type=str, default="business_entity_resolution/configs/default.yaml")
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    if args.mode in ("train", "validate"):
        mode_train_and_validate(cfg, n_samples=args.n_samples)
    elif args.mode == "predict":
        mode_predict(cfg, threshold=args.threshold)
    elif args.mode == "check":
        mode_validate_submission()
    elif args.mode == "all":
        val_report = mode_train_and_validate(cfg, n_samples=args.n_samples)
        best_t = val_report.get("best_threshold", 0.65)
        mode_predict(cfg, threshold=best_t)
        mode_validate_submission()

if __name__ == "__main__":
    main()
