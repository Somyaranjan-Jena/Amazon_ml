import os
import sys
import collections
import unicodedata
import re
import pandas as pd
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def analyze_source_file(file_path, sample_limit=50000):
    print(f"\n==========================================")
    print(f"AUDITING: {file_path}")
    print(f"==========================================")
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    row_count = 0
    header = None
    countries = collections.Counter()
    missing_counts = collections.defaultdict(int)
    name_lengths = []
    addr_lengths = []
    
    # Unicode / script detection
    non_ascii_name_count = 0
    non_ascii_addr_count = 0
    script_counts = collections.Counter()
    
    # Duplicate entity_ids
    sample_rows = []
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\n\r")
        header = header_line.split("\t")
        print(f"Header ({len(header)} cols): {header}")
        
        for line in f:
            row_count += 1
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) != len(header):
                # malformed line or tab in field
                parts = (parts + [""] * len(header))[:len(header)]
            
            eid, bname, baddr, ctry = parts[0], parts[1], parts[2], parts[3]
            
            if not bname.strip():
                missing_counts["business_name"] += 1
            if not baddr.strip():
                missing_counts["business_address"] += 1
            if not ctry.strip():
                missing_counts["country"] += 1
            
            countries[ctry.strip()] += 1
            
            # length sampling
            if row_count <= sample_limit:
                name_lengths.append(len(bname))
                addr_lengths.append(len(baddr))
                
                # Check unicode
                if any(ord(c) >= 128 for c in bname):
                    non_ascii_name_count += 1
                if any(ord(c) >= 128 for c in baddr):
                    non_ascii_addr_count += 1
                    
                for c in bname[:20]:
                    if ord(c) >= 128:
                        try:
                            script_counts[unicodedata.name(c).split()[0]] += 1
                        except Exception:
                            script_counts["OTHER"] += 1
                            
            if row_count <= 5:
                sample_rows.append((eid, bname, baddr, ctry))
                
    print(f"Total rows: {row_count:,}")
    print("Missing counts:", dict(missing_counts))
    print("Country distribution:")
    for ctry, count in countries.most_common():
        pct = (count / row_count) * 100
        print(f"  {ctry if ctry else '[BLANK]'}: {count:,} ({pct:.2f}%)")
        
    if name_lengths:
        nl = np.array(name_lengths)
        al = np.array(addr_lengths)
        print("Name lengths (sample):")
        print(f"  min={nl.min()}, median={np.median(nl):.1f}, mean={nl.mean():.1f}, p95={np.percentile(nl, 95):.1f}, max={nl.max()}")
        print("Address lengths (sample):")
        print(f"  min={al.min()}, median={np.median(al):.1f}, mean={al.mean():.1f}, p95={np.percentile(al, 95):.1f}, max={al.max()}")
        
    print(f"Non-ASCII name rate (sample): {non_ascii_name_count / min(row_count, sample_limit):.4f}")
    print(f"Non-ASCII address rate (sample): {non_ascii_addr_count / min(row_count, sample_limit):.4f}")
    if script_counts:
        print("Top Unicode scripts:", script_counts.most_common(10))
        
    print("First 3 sample rows:")
    for r in sample_rows[:3]:
        print(" ", r)
        
    return {
        "file": file_path,
        "rows": row_count,
        "cols": header,
        "countries": dict(countries),
        "missing": dict(missing_counts)
    }

def analyze_ground_truth(file_path):
    print(f"\n==========================================")
    print(f"AUDITING GROUND TRUTH: {file_path}")
    print(f"==========================================")
    
    row_count = 0
    singletons = 0
    total_matches = 0
    s2_matches = 0
    s3_matches = 0
    both_matches = 0
    match_counts = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n\r").split("\t")
        print(f"Header: {header}")
        
        for line in f:
            row_count += 1
            parts = line.rstrip("\n\r").split("\t")
            s1_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            mids = [m.strip() for m in rest.split(",") if m.strip()] if rest.strip() else []
            k = len(mids)
            match_counts.append(k)
            
            if k == 0:
                singletons += 1
            else:
                total_matches += k
                has_s2 = any(m.startswith("S2-") for m in mids)
                has_s3 = any(m.startswith("S3-") for m in mids)
                s2_count = sum(1 for m in mids if m.startswith("S2-"))
                s3_count = sum(1 for m in mids if m.startswith("S3-"))
                s2_matches += s2_count
                s3_matches += s3_count
                if has_s2 and has_s3:
                    both_matches += 1
                    
    mc = np.array(match_counts)
    print(f"Total S1 entities in GT: {row_count:,}")
    print(f"Total True Matches: {total_matches:,}")
    print(f"Singletons (0 matches): {singletons:,} ({singletons/row_count*100:.2f}%)")
    print(f"Non-singletons: {row_count - singletons:,} ({(row_count - singletons)/row_count*100:.2f}%)")
    print(f"S2 matches: {s2_matches:,} ({s2_matches/total_matches*100:.2f}%)")
    print(f"S3 matches: {s3_matches:,} ({s3_matches/total_matches*100:.2f}%)")
    print(f"Entities with matches in BOTH S2 and S3: {both_matches:,} ({both_matches/row_count*100:.2f}%)")
    print(f"Match count distribution across ALL S1:")
    print(f"  Mean: {mc.mean():.3f}, Median: {np.median(mc):.1f}, P75: {np.percentile(mc, 75):.1f}, P90: {np.percentile(mc, 90):.1f}, P99: {np.percentile(mc, 99):.1f}, Max: {mc.max()}")
    
    non_sing = mc[mc > 0]
    print(f"Match count distribution across NON-SINGLETONS:")
    print(f"  Mean: {non_sing.mean():.3f}, Median: {np.median(non_sing):.1f}, P75: {np.percentile(non_sing, 75):.1f}, P90: {np.percentile(non_sing, 90):.1f}, Max: {non_sing.max()}")

def inspect_corruption_samples(n=10):
    print(f"\n==========================================")
    print("INSPECTING GROUND TRUTH TRUE MATCH CORRUPTIONS")
    print(f"==========================================")
    # Read ground truth first 100 rows to find non-singletons
    gt_pairs = []
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) > 1 and parts[1].strip():
                s1_id = parts[0]
                mids = parts[1].strip().split(",")
                gt_pairs.append((s1_id, mids))
                if len(gt_pairs) >= n:
                    break
                    
    target_s1 = {p[0] for p in gt_pairs}
    target_matches = {m for p in gt_pairs for m in p[1]}
    
    s1_dict = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if parts[0] in target_s1:
                s1_dict[parts[0]] = parts
                
    s2_dict = {}
    with open("dataset/train/train_source2.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if parts[0] in target_matches:
                s2_dict[parts[0]] = parts
                
    s3_dict = {}
    with open("dataset/train/train_source3.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if parts[0] in target_matches:
                s3_dict[parts[0]] = parts
                
    for s1_id, mids in gt_pairs[:n]:
        s1_row = s1_dict.get(s1_id)
        print(f"\n--- S1 [{s1_id}] ({s1_row[3] if s1_row else '?'}) ---")
        print(f"  NAME: {s1_row[1] if s1_row else '?'}")
        print(f"  ADDR: {s1_row[2] if s1_row else '?'}")
        for mid in mids:
            row = s2_dict.get(mid) if mid.startswith("S2-") else s3_dict.get(mid)
            if row:
                print(f"  -> Match [{mid}] ({row[3]}):")
                print(f"     NAME: {row[1]}")
                print(f"     ADDR: {row[2]}")
            else:
                print(f"  -> Match [{mid}]: NOT FOUND IN SAMPLE")

if __name__ == "__main__":
    sources = [
        "dataset/train/train_source1.tsv",
        "dataset/train/train_source2.tsv",
        "dataset/train/train_source3.tsv",
        "dataset/test/test_source1.tsv",
        "dataset/test/test_source2.tsv",
        "dataset/test/test_source3.tsv",
    ]
    for s in sources:
        analyze_source_file(s)
        
    analyze_ground_truth("dataset/train/train_ground_truth.tsv")
    inspect_corruption_samples(8)
