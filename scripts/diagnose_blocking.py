import sys, os
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
sys.path.insert(0, os.path.abspath("."))
import collections
from business_entity_resolution.src.data_loader import load_compact_records, load_ground_truth
from business_entity_resolution.src.blocking import MultiPassBlockingIndex
from business_entity_resolution.src.candidate_generation import generate_candidates_by_country
from business_entity_resolution.src.validation import create_s1_validation_split

def diagnose():
    gt = load_ground_truth("dataset/train/train_ground_truth.tsv")
    s1_path = "dataset/train/train_source1.tsv"
    s2_path = "dataset/train/train_source2.tsv"
    s3_path = "dataset/train/train_source3.tsv"

    # Take first 5,000 S1
    s1_ids = []
    s1_countries = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n\r").split("\t")
            if len(p) >= 4:
                s1_ids.append(p[0])
                s1_countries[p[0]] = p[3].strip()
            if len(s1_ids) >= 5000:
                break

    needed_targets = set()
    for eid in s1_ids:
        needed_targets.update(gt.get(eid, set()))

    s1_records = load_compact_records(s1_path, target_ids=set(s1_ids))
    s2_targets = load_compact_records(s2_path, target_ids=needed_targets)
    s3_targets = load_compact_records(s3_path, target_ids=needed_targets)
    all_targets = {**s2_targets, **s3_targets}

    # Add countries
    s1_dict = {
        eid: {
            "entity_id": eid,
            "norm_name": r[0], "core_name": r[1], "norm_addr": r[2],
            "postal_code": r[3], "house_number": r[4], "country": s1_countries.get(eid, "US")
        }
        for eid, r in s1_records.items()
    }

    # Target countries
    target_countries = {}
    for p in (s2_path, s3_path):
        with open(p, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) >= 4 and parts[0] in all_targets:
                    target_countries[parts[0]] = parts[3].strip()

    target_dict = {
        eid: {
            "entity_id": eid,
            "norm_name": r[0], "core_name": r[1], "norm_addr": r[2],
            "postal_code": r[3], "house_number": r[4], "country": target_countries.get(eid, "US")
        }
        for eid, r in all_targets.items()
    }

    cands = generate_candidates_by_country(s1_dict, target_dict, max_candidates_per_s1=60)

    missed_examples = []
    total_true = 0
    recalled_true = 0

    for s1_id, s1_rec in s1_dict.items():
        true_m = gt.get(s1_id, set())
        s1_cands = cands.get(s1_id, set())
        for mid in true_m:
            total_true += 1
            if mid in s1_cands:
                recalled_true += 1
            else:
                t_rec = target_dict.get(mid)
                if t_rec and len(missed_examples) < 15:
                    missed_examples.append((s1_rec, t_rec, mid))

    print(f"\nInitial Recall: {recalled_true}/{total_true} ({recalled_true/total_true*100:.2f}%)")
    print(f"\n--- 15 MISSED TRUE MATCH EXAMPLES ---")
    for s1_rec, t_rec, mid in missed_examples:
        print(f"\nS1: [{s1_rec['entity_id']}] ({s1_rec['country']})")
        print(f"  Name: '{s1_rec['norm_name']}' | Core: '{s1_rec['core_name']}'")
        print(f"  Addr: '{s1_rec['norm_addr']}' | PIN: {s1_rec['postal_code']} | House: {s1_rec['house_number']}")
        print(f"TARGET: [{mid}] ({t_rec['country']})")
        print(f"  Name: '{t_rec['norm_name']}' | Core: '{t_rec['core_name']}'")
        print(f"  Addr: '{t_rec['norm_addr']}' | PIN: {t_rec['postal_code']} | House: {t_rec['house_number']}")

if __name__ == "__main__":
    diagnose()
