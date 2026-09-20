"""
01_download_weak_species.py
============================
Pull additional recordings from Xeno-canto for the species that are
underrepresented in the iBC53 dataset (the 15 "moderate" + 6 "weak" tier
species from your README, i.e. everything under ~50 estimated 5s chunks).

Xeno-canto API v3 requires a free API key as of Oct 2025:
  1. Register / sign in at https://xeno-canto.org
  2. Verify your email
  3. Grab your key from https://xeno-canto.org/account
  4. export XC_API_KEY="your-key-here"   (or pass --api_key)

Usage:
  python 01_download_weak_species.py --out_dir ./xc_downloads --api_key $XC_API_KEY
  python 01_download_weak_species.py --preview            # just show counts, no download

Notes:
  - Searches India first. If a species returns fewer than --min_per_species
    recordings from India alone, it automatically broadens the search to
    neighboring countries (Bangladesh, Nepal, Sri Lanka, Myanmar, Bhutan,
    Pakistan) since the vocalization itself doesn't change at the border —
    more usable audio for the same species/call type still helps the model.
  - Skips files you've already downloaded (checked by Xeno-canto recording id).
  - Writes a metadata.csv per species folder so you can trace provenance,
    license, and recording quality later.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import requests

XC_API_URL = "https://xeno-canto.org/api/3/recordings"
FALLBACK_COUNTRIES = ["bangladesh", "nepal", "sri lanka", "myanmar", "bhutan", "pakistan"]

# Common names as listed in the iBC53 README, moderate (🟡) + weak (🔴) tiers.
TARGET_SPECIES = [
    "Yellow-browed Warbler", "Yellow-throated Leaf Warbler", "Asian Palm Swift",
    "Streaked Spiderhunter", "Baikal Bush Warbler", "Scarlet-backed Flowerpecker",
    "Chinspot Wren-Babbler", "Long-tailed Shrike", "Spot-breasted Parrotbill",
    "Oriental Dollarbird", "Mrs Gould's Sunbird", "Plain Flowerpecker",
    "Black-bellied Plover", "Ruddy Kingfisher", "Cachar Bulbul",
    "White-tailed Flycatcher", "Yellow-vented Flowerpecker", "Grey-throated Martin",
    "Cinnamon Bittern", "Tickell's Leaf Warbler", "Blue-winged Leafbird",
]


def slugify(name: str) -> str:
    return name.replace("'", "").replace(" ", "_").replace("-", "-")


def query_xc(species_en: str, country: str, api_key: str, page: int = 1) -> dict:
    q = f'en:"{species_en}" cnt:"{country}"'
    resp = requests.get(XC_API_URL, params={"query": q, "key": api_key, "page": page}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_recordings(species_en: str, country: str, api_key: str) -> list:
    first = query_xc(species_en, country, api_key, page=1)
    recordings = list(first.get("recordings", []))
    num_pages = int(first.get("numPages", 1))
    for p in range(2, num_pages + 1):
        time.sleep(1)  # be polite to the API
        page_data = query_xc(species_en, country, api_key, page=p)
        recordings.extend(page_data.get("recordings", []))
    return recordings


def download_recording(rec: dict, dest_dir: Path) -> bool:
    file_url = rec.get("file") or ""
    if file_url.startswith("//"):
        file_url = "https:" + file_url
    if not file_url:
        return False
    dest_path = dest_dir / f"xc_{rec['id']}.mp3"
    if dest_path.exists():
        return False  # already have it
    try:
        r = requests.get(file_url, timeout=60)
        r.raise_for_status()
        dest_path.write_bytes(r.content)
        return True
    except requests.RequestException as e:
        print(f"    ! failed to download xc_{rec['id']}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./xc_downloads")
    ap.add_argument("--api_key", default=os.environ.get("XC_API_KEY", ""))
    ap.add_argument("--min_per_species", type=int, default=15,
                     help="if India-only results are below this, broaden to neighboring countries")
    ap.add_argument("--preview", action="store_true", help="only print counts, download nothing")
    args = ap.parse_args()

    if not args.preview and not args.api_key:
        raise SystemExit("Need an API key: pass --api_key or set XC_API_KEY. "
                          "Get one free at https://xeno-canto.org/account")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for species in TARGET_SPECIES:
        print(f"\n=== {species} ===")
        recordings = fetch_all_recordings(species, "india", args.api_key) if not args.preview else []
        countries_used = ["india"]

        if args.preview:
            # Cheap preview: just report the India count from page 1's numRecordings.
            data = query_xc(species, "india", args.api_key) if args.api_key else {}
            n = int(data.get("numRecordings", 0)) if data else "?"
            print(f"  India recordings available: {n}")
            summary.append((species, n, "india"))
            time.sleep(1)
            continue

        if len(recordings) < args.min_per_species:
            print(f"  Only {len(recordings)} from India, broadening search...")
            for country in FALLBACK_COUNTRIES:
                time.sleep(1)
                more = fetch_all_recordings(species, country, args.api_key)
                if more:
                    recordings.extend(more)
                    countries_used.append(country)
                if len(recordings) >= args.min_per_species:
                    break

        species_dir = out_root / slugify(species)
        species_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        with open(species_dir / "metadata.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "country", "quality", "length", "recordist", "license", "url"])
            for rec in recordings:
                ok = download_recording(rec, species_dir)
                if ok:
                    downloaded += 1
                writer.writerow([rec.get("id"), rec.get("cnt"), rec.get("q"),
                                  rec.get("length"), rec.get("rec"), rec.get("lic"), rec.get("url")])

        print(f"  Found {len(recordings)} recordings across {countries_used} -> downloaded {downloaded} new files")
        summary.append((species, len(recordings), "+".join(countries_used)))
        time.sleep(1)

    print("\n=== Summary ===")
    for species, n, countries in summary:
        print(f"  {species:40s} {n:>4} recordings  ({countries})")


if __name__ == "__main__":
    main()
