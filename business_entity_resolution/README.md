# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

A production-grade, locally runnable, highly scalable entity resolution system for multi-source business record linkage.

## System Architecture

The pipeline implements an end-to-end hybrid architecture:
1. **Open-Set Country Partitioning**: Strictly partitions records by country (dynamically supporting US, India, France, and any open-set test country), eliminating cross-country comparisons with 0% recall loss.
2. **Unicode & Linguistic Normalization**: Unicode NFKC standardization, casefolding, legal suffix stripping (`Pvt Ltd`, `LLC`, `Corp`, `Sarl`, etc.), URL/domain cleaning, and leading zero normalization on address components.
3. **Priority-Ranked Multi-Pass Blocking**:
   - Pass A: Exact Core Name Match
   - Pass B: Exact Normalized Address Match
   - Pass C: Alphanumeric Name Skeleton Match
   - Pass D: Rare Name Token Inverted Index (IDF-filtered)
   - Pass E: House Number + Distinctive Address Token Compound Keys (e.g. `1712@montebello`)
   - Reaches **>92% Candidate Recall** while maintaining ultra-low candidate volume (<=50 candidates per query).
4. **C++ Accelerated Pair Feature Engineering**:
   - 25 rich similarity features computed via `RapidFuzz` (Levenshtein ratio, token sort ratio, token set ratio, partial ratio, token Jaccard, digit overlap, address length difference, interaction terms).
5. **Hard Negative Mining & LightGBM Model**:
   - Mines realistic hard negatives from blocking passes (1:3 positive-to-negative ratio).
   - Trains a high-capacity LightGBM GBDT classifier optimizing macro F0.5.
6. **Precision-Calibrated Thresholding & Multi-Match Decision**:
   - Grid search optimization for decision threshold on S1-level validation split.
   - Deterministic overrides for identical name + address pairs.
   - Calibrated singleton abstention (predicting empty sets for singletons).

---

## Directory Structure

```text
business_entity_resolution/
├── configs/
│   └── default.yaml          # Centralized configuration
├── src/
│   ├── __init__.py
│   ├── config.py             # Config loader
│   ├── data_loader.py        # Streaming TSV loaders & compact memory representations
│   ├── normalization.py      # Unicode NFKC & string standardizers
│   ├── address_features.py   # Structural address component extractors
│   ├── blocking.py           # Multi-pass priority-ranked inverted index
│   ├── candidate_generation.py # Country-partitioned candidate generator
│   ├── pair_features.py      # RapidFuzz pair feature extraction
│   ├── train.py              # Hard negative mining & LightGBM training
│   ├── validation.py         # S1-level stratified validation & metric evaluation
│   ├── decision.py           # Threshold optimizer & multi-match logic
│   ├── inference.py          # Memory-safe streaming test inference
│   ├── output.py             # TSV writers matching challenge specifications
│   └── pipeline.py           # Main CLI entrypoint
├── tests/
│   ├── test_metrics.py       # Exact F0.5 validation against official examples
│   └── test_blocking.py      # Synthetic multi-match & singleton tests
├── README.md                 # Reproduction instructions
└── requirements.txt          # Pinned dependencies
```

---

## Setup & Installation

Ensure Python 3.10+ is installed:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## Reproducing End-to-End Pipeline

All steps can be executed via the central pipeline CLI:

### 1. Run Unit Tests
```bash
python -m unittest discover business_entity_resolution/tests
```

### 2. Train Model & Optimize Validation Threshold
```bash
python -m business_entity_resolution.src.pipeline --mode train --n-samples 50000
```
This trains the LightGBM classifier on mined hard negatives, simulates inference on the held-out validation set, optimizes the macro F0.5 decision threshold, and saves the trained model to `cache/lgb_model.joblib`.

### 3. Generate Test Predictions
```bash
python -m business_entity_resolution.src.pipeline --mode predict --threshold 0.85
```
This runs memory-bounded streaming inference across France, US, and India test sets and outputs:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

### 4. Run Official Validator
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
And to run with full ID-existence checks:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### 5. Run Everything in One Command
```bash
python -m business_entity_resolution.src.pipeline --mode all --n-samples 50000
```
