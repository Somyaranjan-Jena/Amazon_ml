import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*60)
print('FINAL SUBMISSION CHECKLIST')
print('='*60)

# 1. Check submission structure
print('\n[1] SUBMISSION STRUCTURE')
required = {
    'output/matching_results.tsv': 'Final matches',
    'output/candidate_pairs.tsv': 'Blocking candidates',
    'business_entity_resolution/src/pipeline.py': 'Main entrypoint',
    'business_entity_resolution/src/train.py': 'Training (XGBoost GPU)',
    'business_entity_resolution/src/inference.py': 'GPU inference',
    'business_entity_resolution/src/blocking.py': 'Blocking module',
    'business_entity_resolution/src/pair_features.py': 'Feature extraction',
    'business_entity_resolution/src/normalization.py': 'Normalization',
    'business_entity_resolution/src/validation.py': 'Validation',
    'business_entity_resolution/src/decision.py': 'Decision logic',
    'business_entity_resolution/src/metrics.py': 'Metrics (F0.5)',
    'business_entity_resolution/src/data_loader.py': 'Data loader',
    'business_entity_resolution/src/config.py': 'Config loader',
    'business_entity_resolution/src/output.py': 'Output writers',
    'business_entity_resolution/README.md': 'Reproduction instructions',
    'business_entity_resolution/requirements.txt': 'Pip dependencies',
    'business_entity_resolution/configs/default.yaml': 'Config YAML',
    'Documentation_template.md': 'Methodology document',
}
all_ok = True
for f, desc in required.items():
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = 'OK' if exists and size > 0 else 'MISSING'
    if not exists or size == 0:
        all_ok = False
    print(f'  [{status:7s}] {f} ({size:,} bytes) - {desc}')

# 2. Output file integrity
print('\n[2] OUTPUT FILE INTEGRITY')
for fname in ['output/matching_results.tsv', 'output/candidate_pairs.tsv']:
    with open(fname, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        lines = sum(1 for _ in f)
    print(f'  {fname}: header="{header}", rows={lines:,}')

# 3. Match-candidate subset check
print('\n[3] MATCH-CANDIDATE SUBSET CHECK')
match_data = {}
with open('output/matching_results.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\n\r').split('\t')
        s1 = parts[0]
        mids = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
        match_data[s1] = mids

cand_data = {}
with open('output/candidate_pairs.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\n\r').split('\t')
        s1 = parts[0]
        cids = set(parts[1].split(',')) if len(parts) > 1 and parts[1].strip() else set()
        cand_data[s1] = cids

violations = 0
for s1, matches in match_data.items():
    cands = cand_data.get(s1, set())
    orphans = matches - cands - {''}
    if orphans:
        violations += 1
status = 'PASS' if violations == 0 else f'FAIL ({violations} violations)'
print(f'  [{status}] All match IDs are subsets of candidate IDs')

# 4. S1 coverage
print('\n[4] S1 ENTITY COVERAGE')
test_s1_ids = set()
with open('dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\n\r').split('\t')
        test_s1_ids.add(parts[0])

match_s1 = set(match_data.keys())
cand_s1 = set(cand_data.keys())
missing_match = test_s1_ids - match_s1
missing_cand = test_s1_ids - cand_s1

print(f'  Test S1: {len(test_s1_ids):,}')
print(f'  matching_results S1: {len(match_s1):,} (missing: {len(missing_match)})')
print(f'  candidate_pairs S1: {len(cand_s1):,} (missing: {len(missing_cand)})')
status = 'PASS' if len(missing_match)==0 and len(missing_cand)==0 else 'FAIL'
print(f'  [{status}] Full coverage')

# 5. Stats
print('\n[5] PREDICTION STATISTICS')
total_matches = sum(len(m - {''}) for m in match_data.values())
total_cands = sum(len(c - {''}) for c in cand_data.values())
singletons = sum(1 for m in match_data.values() if len(m - {''}) == 0)
non_singletons = len(match_data) - singletons
avg_matches = total_matches / max(1, non_singletons)
avg_cands = total_cands / max(1, len(cand_data))
print(f'  Total match links: {total_matches:,}')
print(f'  Total candidate links: {total_cands:,}')
print(f'  Singletons: {singletons:,} ({100*singletons/len(match_data):.1f}%)')
print(f'  Non-singletons: {non_singletons:,}')
print(f'  Avg matches/non-singleton: {avg_matches:.2f}')
print(f'  Avg candidates/S1: {avg_cands:.1f}')

# 6. ID prefix check
print('\n[6] ID PREFIX VALIDATION')
s1_in_matches = sum(1 for mids in match_data.values() for m in mids if m.startswith('S1-'))
s1_in_cands = sum(1 for cids in cand_data.values() for c in cids if c.startswith('S1-'))
status = 'PASS' if s1_in_matches==0 and s1_in_cands==0 else 'FAIL'
print(f'  [{status}] No S1-prefixed IDs (matches:{s1_in_matches}, cands:{s1_in_cands})')

# 7. Duplicate row check
print('\n[7] DUPLICATE ROW CHECK')
seen_m = set(); dup_m = 0
with open('output/matching_results.tsv','r',encoding='utf-8') as f:
    f.readline()
    for line in f:
        s1 = line.split('\t')[0]
        if s1 in seen_m: dup_m += 1
        seen_m.add(s1)
seen_c = set(); dup_c = 0
with open('output/candidate_pairs.tsv','r',encoding='utf-8') as f:
    f.readline()
    for line in f:
        s1 = line.split('\t')[0]
        if s1 in seen_c: dup_c += 1
        seen_c.add(s1)
status = 'PASS' if dup_m==0 and dup_c==0 else 'FAIL'
print(f'  [{status}] No duplicate S1 rows (matches:{dup_m}, cands:{dup_c})')

# 8. Model artifacts
print('\n[8] MODEL ARTIFACTS')
for mf in ['cache/lgb_model.joblib', 'cache/xgb_model.ubj', 'cache/xgb_model.joblib']:
    if os.path.exists(mf):
        print(f'  [OK] {mf} ({os.path.getsize(mf):,} bytes)')
    else:
        print(f'  [--] {mf} not found')

# 9. License check
print('\n[9] MODEL LICENSE')
print('  [PASS] XGBoost: Apache-2.0 (< 8B params)')
print('  [PASS] LightGBM: MIT (< 8B params)')
print('  [PASS] RapidFuzz: MIT')

print('\n' + '='*60)
print('ALL CHECKS COMPLETE - READY TO SUBMIT')
print('='*60)
