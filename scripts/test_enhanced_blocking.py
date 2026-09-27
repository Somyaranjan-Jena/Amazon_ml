import sys, os
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
sys.path.insert(0, os.path.abspath("."))
import collections
from business_entity_resolution.src.data_loader import load_compact_records, load_ground_truth
from business_entity_resolution.src.address_features import COMMON_ADDR_STOPWORDS

def test_enhanced_blocking():
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
    target_countries = {}
    for p in (s2_path, s3_path):
        with open(p, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) >= 4 and parts[0] in all_targets:
                    target_countries[parts[0]] = parts[3].strip()

    # Index by country
    for ctry in ["US", "India"]:
        s1_ctry = {eid: s1_records[eid] for eid in s1_ids if s1_countries.get(eid) == ctry}
        tgt_ctry = {eid: rec for eid, rec in all_targets.items() if target_countries.get(eid) == ctry}

        # Build enhanced index
        idx_exact_name = collections.defaultdict(list)
        idx_exact_addr = collections.defaultdict(list)
        idx_name_tokens = collections.defaultdict(list)
        idx_house_token = collections.defaultdict(list)
        idx_skel = collections.defaultdict(list)

        token_df = collections.Counter()
        addr_token_df = collections.Counter()

        for eid, r in tgt_ctry.items():
            norm_name, core_name, norm_addr, pin, house = r
            for t in set(core_name.split()):
                if len(t) >= 3:
                    token_df[t] += 1
            for at in set(norm_addr.split()):
                if len(at) >= 3 and at not in COMMON_ADDR_STOPWORDS and not at.isdigit():
                    addr_token_df[at] += 1

        for eid, r in tgt_ctry.items():
            norm_name, core_name, norm_addr, pin, house = r
            if core_name:
                idx_exact_name[core_name].append(eid)
                sk = "".join(c for c in core_name if c.isalnum())
                if len(sk) >= 4:
                    idx_skel[sk].append(eid)
            if norm_addr and len(norm_addr) >= 8:
                idx_exact_addr[norm_addr].append(eid)
            if core_name:
                for t in set(core_name.split()):
                    if len(t) >= 3 and token_df[t] <= 10000:
                        idx_name_tokens[t].append(eid)
            # Pass C2: House + Distinctive Addr Token
            if house:
                for at in set(norm_addr.split()):
                    if len(at) >= 3 and at not in COMMON_ADDR_STOPWORDS and not at.isdigit():
                        if addr_token_df[at] <= 5000:
                            idx_house_token[f"{house}@{at}"].append(eid)

        # Retrieve and evaluate
        recalled = 0
        total = 0
        for s1_id, r in s1_ctry.items():
            norm_name, core_name, norm_addr, pin, house = r
            true_m = gt.get(s1_id, set())
            cands = set()

            if core_name in idx_exact_name:
                cands.update(idx_exact_name[core_name])
            if norm_addr in idx_exact_addr:
                cands.update(idx_exact_addr[norm_addr])
            if core_name:
                sk = "".join(c for c in core_name if c.isalnum())
                if sk in idx_skel:
                    cands.update(idx_skel[sk])
                for t in set(core_name.split()):
                    if t in idx_name_tokens:
                        cands.update(idx_name_tokens[t])
            if house:
                for at in set(norm_addr.split()):
                    if len(at) >= 3 and at not in COMMON_ADDR_STOPWORDS and not at.isdigit():
                        k = f"{house}@{at}"
                        if k in idx_house_token:
                            cands.update(idx_house_token[k])

            for mid in true_m:
                total += 1
                if mid in cands:
                    recalled += 1

        print(f"Country [{ctry}] Recall with House+AddrToken Pass: {recalled}/{total} ({recalled/total*100:.2f}%)")

if __name__ == "__main__":
    test_enhanced_blocking()
