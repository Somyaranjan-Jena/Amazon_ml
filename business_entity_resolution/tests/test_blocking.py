import unittest
from business_entity_resolution.src.normalization import normalize_business_name, extract_core_business_name, normalize_address
from business_entity_resolution.src.address_features import extract_address_components
from business_entity_resolution.src.blocking import MultiPassBlockingIndex
from business_entity_resolution.src.candidate_generation import generate_candidates_by_country

class TestBlocking(unittest.TestCase):
    def test_synthetic_candidate_generation(self):
        # S1 queries
        s1_data = {
            "S1-A": {
                "raw_name": "Payne Enterprises LLC",
                "raw_addr": "3315 Fremont St, Peoria, IL 61602",
                "country": "US"
            },
            "S1-B": {
                "raw_name": "Unique Quantum Robotics Corp",
                "raw_addr": "999 Mars Way, Sector 4, Bangalore 560001",
                "country": "India"
            },
            "S1-C": {
                "raw_name": "Ss Food Private Limited",
                "raw_addr": "Af-684, Nandgram, Ghaziabad, UP 201001",
                "country": "India"
            }
        }
        
        # Targets (S2 and S3)
        targets = {
            # S2-A matches S1-A (typo in name)
            "S2-A": {
                "raw_name": "Payne Enterpires",
                "raw_addr": "3315 FREMONT ST, PEORIA, IL 61602",
                "country": "US"
            },
            # S3-A matches S1-A (exact address)
            "S3-A": {
                "raw_name": "Payne Enterprises",
                "raw_addr": "3315 Fremont Street, Peoria, IL 61602",
                "country": "US"
            },
            # S2-B matches S1-C (exact Hindi transliteration or address)
            "S2-B": {
                "raw_name": "एसएस फूड प्राइवेट लिमिटेड",
                "raw_addr": "AF-0684, Nandgram, Ghaziabad, UP 201001",
                "country": "India"
            },
            # S3-D matches S1-C (name match)
            "S3-D": {
                "raw_name": "Ss Food Pvt Ltd",
                "raw_addr": "Nandgram, Ghaziabad 201001",
                "country": "India"
            },
            # Irrelevant US entity
            "S2-X": {
                "raw_name": "Global Timber Co",
                "raw_addr": "100 Pine St, Seattle, WA 98101",
                "country": "US"
            }
        }
        
        # Pre-normalize records
        for d in (s1_data, targets):
            for eid, r in d.items():
                norm_name = normalize_business_name(r["raw_name"])
                r["norm_name"] = norm_name
                r["core_name"] = extract_core_business_name(norm_name)
                r["norm_addr"] = normalize_address(r["raw_addr"])
                ac = extract_address_components(r["norm_addr"])
                r["postal_code"] = ac["postal_code"]
                r["house_number"] = ac["house_number"]
                r["all_digits"] = ac["all_digits"]
                r["significant_tokens"] = ac["significant_tokens"]

        cands = generate_candidates_by_country(s1_data, targets, max_candidates_per_s1=10)
        
        # S1-A should retrieve S2-A (via pin+house or address) and S3-A (exact name)
        self.assertIn("S3-A", cands["S1-A"])
        self.assertIn("S2-A", cands["S1-A"])
        
        # S1-B should have no targets (singleton)
        self.assertEqual(len(cands["S1-B"]), 0)
        
        # S1-C should retrieve S2-B (compound PIN/house match) and S3-D (exact core name)
        self.assertIn("S3-D", cands["S1-C"])
        self.assertIn("S2-B", cands["S1-C"])

if __name__ == "__main__":
    unittest.main()
