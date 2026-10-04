# check_species_identity.py
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pathlib import Path
from collections import Counter

analyzer = Analyzer()

for folder in ["Grey-throated_Babbler", "Humes_Bar-tailed_Scimitar_Babbler"]:
    print(f"\n=== {folder} ===")
    files = list(Path(f"ibc53/{folder}").glob("*"))[:10]  # sample 10 files
    votes = Counter()
    for f in files:
        try:
            rec = Recording(analyzer, str(f), min_conf=0.1)
            rec.analyze()
            for d in rec.detections:
                votes[d["common_name"]] += 1
        except Exception as e:
            print(f"  ! {f.name}: {e}")
    for species, count in votes.most_common(5):
        print(f"  {species:35s} {count}")
