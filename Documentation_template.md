# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity Engineering  
**Team Members:** Senior ML Systems & Research Engineering  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We designed and implemented a production-grade, GPU-accelerated, locally reproducible hybrid entity resolution architecture to resolve multi-source business records across Source 1 (deduplicated anchor), Source 2, and Source 3. Our system introduces three core technical innovations: (1) an **open-set dynamic country partitioning engine** grounded in our empirical proof that 0.000000% of true matches cross national borders, (2) a **priority-ranked multi-pass blocking inverted index** leveraging compound house-number and distinctive-token keys that achieves a **~89% candidate recall ceiling** with strict volume capping ($\le 50$ candidates per query), and (3) an **XGBoost GPU-accelerated gradient boosted decision tree classifier** (device='cuda', RTX 3060) trained on 1M+ hard-negative pairs with **C++ accelerated RapidFuzz pair features** and precision-calibrated thresholding. On the stratified held-out validation set (25,000 S1 entities), our solution achieves an official **Macro $F_{0.5}$ score of 0.9279** with **98.04% Precision**, **87.13% Recall**, and **93.73% Singleton Accuracy**.

---

## 2. Methodology

### 2.1 Problem Analysis & Empirical Audit
A comprehensive audit of the raw data (>26.4 million total records across 7 TSV files) and ground-truth pairings (7,638,365 true matches) revealed critical structural characteristics:
- **Absolute Country Invariant**: Verification over all 7.64 million ground truth links proved that **zero matches cross country boundaries**. Training data covers US (59.98%) and India (40.02%); the test set introduces an unseen country, France (14.98% of S1). Country is never missing.
- **Multilingual & Indic Transliteration**: S1 business names are 100% Latin text. In S2 and S3, **14.89% and 11.68% of business names are transliterated into non-Latin scripts** (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali). Despite script divergence in name fields, addresses in S2/S3 frequently preserve Latin municipal numbers, plot numbers, and street tokens.
- **Syntactic & Web Noise**: S3 records frequently manifest as web domain names (e.g., `maurewilliamscolombier.com`). Suffix variations (`Corporation` vs. `Corp`, `Private Limited` vs. `Pvt Ltd`, `LLC`, `SARL`, `SAS`), typos (`Enterpires`), and missing addresses (~3.3% in S2/S3) are widespread.
- **Multi-Match & Singletons**: Exactly 123,247 (5.58%) S1 entities in train are true singletons (0 matches). Non-singletons match an average of 3.67 records (up to 11). Over 80.48% of S1 entities possess concurrent matches in both S2 and S3.

### 2.2 Solution Strategy
We adopted a **Hybrid Multi-Pass Blocking + GPU Gradient Boosted Pair Classifier + Calibrated Decision Pipeline**:
- **Strict Open-Set Partitioning**: Dynamically groups records by the `country` string attribute. Fully supports US, India, France, and any unseen test label without hard-coded rules.
- **Unicode NFKC Normalization**: Standardizes Unicode compatibility, normalizes case, expands abbreviations, removes web domain extensions, strips trailing corporate suffixes, and normalizes address numeric strings (stripping leading zeros so `0684` matches `684`).
- **Priority-Ranked Blocking**: Combines exact core names, exact addresses, name skeletons, rare name tokens, and compound `(house_number @ distinctive_token)` keys. Scores candidates based on pass overlap and takes the top-ranked candidates.
- **XGBoost GPU Pair Classifier**: CUDA-accelerated gradient boosted tree booster with 25 fine-grained similarity and interaction features, trained on 1,029,588 hard-negative pairs.
- **Decision Calibration**: Maximizes macro $F_{0.5}$ through validation grid search with deterministic overrides for identical name + address pairs.

---

## 3. Candidate Generation (Blocking)

To avoid an impossible all-pairs comparison space of $1.73\text{M} \times 10\text{M} \approx 1.7 \times 10^{13}$ pairs, we developed an inverted index with 5 complementary blocking passes:

- **Pass A (Exact Core Name)**: Normalized name with trailing legal corporate suffixes removed (e.g., `payne enterprises llc` $\rightarrow$ `payne enterprises`).
- **Pass B (Exact Normalized Address)**: Standardized address strings with collapsed whitespace and normalized directional/street abbreviations.
- **Pass C (Alphanumeric Name Skeleton)**: Strips whitespace and non-alphanumeric characters, capturing spacing and hyphenation typos (`payne-enterprises` $\leftrightarrow$ `payneenterprises`).
- **Pass D (Rare Name Tokens Inverted Index)**: Tokenizes names and indexes informative words where document frequency $\le 15,000$. Prioritizes rare tokens for queries.
- **Pass E (Compound House Number + Distinctive Token Keys)**: Pairs building/plot numbers with distinctive street/locality tokens (e.g., `1712@montebello`, `16@sagar`). This pass successfully bridges transliterated Indian records where the business name is in Devanagari/Telugu but the physical address tokens match.
- **Candidate Volume & True-Match Preservation**:
  - Each pass accumulates priority points. Candidates appearing in multiple passes or matching exact keys are ranked highest.
  - Queries are capped at $K=50$ candidates (balanced across S2 and S3), reducing the comparison space by **99.999%** while achieving an **88.57% validation candidate recall** (S2: 89.34%, S3: 87.85%).

---

## 4. Matching Model

### Features Used (25 Total):
- **Name Features**: RapidFuzz Levenshtein ratio, Token Sort ratio, Token Set ratio, Partial Token ratio, Exact Name indicator, Exact Core Name indicator, Token Jaccard similarity, Common token count, Absolute length difference, Length ratio.
- **Address Features**: Levenshtein ratio, Token Sort ratio, Token Set ratio, Exact Address indicator, Token Jaccard similarity, Common token count, Postal code match indicator, House number match indicator, Digit sequence Jaccard similarity, Address length difference, Target address missing indicator.
- **Cross-Field & Source Features**: Source indicator (`is_s2`), Name $\times$ Address interaction score (`name_x_addr`), High-name + High-address indicator, High-name + Missing-address indicator.

### Model Architecture & Training:
- **Model Type**: XGBoost Booster (Apache-2.0 license, 500 rounds, max depth 8, eta 0.05, `device='cuda'`, `tree_method='hist'`).
- **GPU Acceleration**: Trained and predicted on NVIDIA GeForce RTX 3060 (6GB VRAM) via XGBoost CUDA backend.
- **Negative Sampling**: Mined hard negatives directly from blocking candidate output (1:2.97 positive-to-negative ratio on 1,029,588 training pairs from 100,000 S1 entities and 732,716 target records).
- **Top Features by Gain**: `addr_token_set_ratio` (4340.76), `addr_token_jaccard` (880.58), `name_x_addr_score` (382.54), `high_name_missing_addr` (221.73), `addr_token_sort_ratio` (189.36).
- **Threshold Selection**: Validation grid search over $[0.30, 0.85]$ in $0.05$ increments directly optimizing the official macro $F_{0.5}$ metric. The optimal threshold was empirically identified at **$\tau = 0.85$**, reflecting the metric's heavy precision penalty ($\beta = 0.5$).

---

## 5. Results & Error Analysis

### Official Metric Performance (Stratified Validation, 25,000 S1 entities):
- **Macro $F_{0.5}$ Score:** **0.9279**
- **Macro Precision:** **0.9804** (98.04%)
- **Macro Recall:** **0.8713** (87.13%)
- **Singleton Accuracy:** **0.9373** (93.73%)
- **Non-Singleton Macro $F_{0.5}$:** **0.9274**
- **S2 Macro $F_{0.5}$:** **0.8926**
- **S3 Macro $F_{0.5}$:** **0.8901**
- **US Macro $F_{0.5}$:** **0.9529**
- **India Macro $F_{0.5}$:** **0.8906**

### Error Analysis:
- **False Positives (Wrong Merges)**: Almost entirely eliminated by our high decision threshold ($\tau = 0.85$), achieving 98.04% precision. The few remaining false positives occur when two distinct small businesses share an identical generic name (e.g., "City Barbershop") and occupy adjacent suites in the same commercial plaza.
- **False Negatives (Missed Matches)**: Primarily caused by entities with extreme simultaneous corruption: a transliterated regional script name coupled with a completely missing or malformed address in S2/S3. India F0.5 (0.8906) is lower than US (0.9529) due to higher prevalence of script transliteration.

---

## 6. Conclusion
Our solution demonstrates that principled data-driven architecture—open-set country partitioning, priority-ranked multi-pass inverted indexing, and GPU-accelerated calibrated gradient boosted trees with hard negative mining—solves massive-scale business entity resolution with exceptional precision and scalability. The complete system processes 1.73 million S1 queries against 10 million target records locally within ~57 minutes on a single RTX 3060 GPU, adheres strictly to all challenge constraints, and achieves a validated Macro $F_{0.5}$ of **0.9279**.

---

## Appendix

### A. Code Artefacts & Reproduction
The self-contained codebase is organized under `business_entity_resolution/`:
- `src/pipeline.py`: Central entrypoint supporting `--mode train`, `--mode predict`, and `--mode all`.
- `src/blocking.py`: Multi-pass priority inverted index.
- `src/pair_features.py`: C++ RapidFuzz feature extraction (25 features).
- `src/train.py`: XGBoost GPU training with hard negative mining and sklearn-compatible wrapper.
- `src/inference.py`: GPU-accelerated streaming test inference engine with 200K batch scoring.
- `requirements.txt`: Pinned dependencies (XGBoost, RapidFuzz, CuPy, NumPy, SciPy, etc.).

To reproduce end-to-end:
```bash
python -m business_entity_resolution.src.pipeline --mode all --n-samples 100000
```

### B. Summary Performance Across Threshold Grid

| Threshold ($\tau$) | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 0.35 | 0.9029 | 0.9392 | 0.8884 | 0.8140 |
| 0.45 | 0.9101 | 0.9489 | 0.8871 | 0.8450 |
| 0.55 | 0.9151 | 0.9561 | 0.8856 | 0.8609 |
| 0.65 | 0.9195 | 0.9628 | 0.8837 | 0.8832 |
| 0.75 | 0.9237 | 0.9704 | 0.8791 | 0.9070 |
| **0.85 (Optimal)** | **0.9279** | **0.9804** | **0.8713** | **0.9373** |
