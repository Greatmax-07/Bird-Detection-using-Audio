"""
01_download_weak_species.py
============================
Pull recordings from Xeno-canto for Maharashtra-region bird species.

Xeno-canto API v3 requires a free API key:
  1. Register / sign in at https://xeno-canto.org
  2. Verify your email
  3. Grab your key from https://xeno-canto.org/account
  4. export XC_API_KEY="your-key-here"   (or pass --api_key)

Usage:
  python 01_download_weak_species.py --out_dir ./xc_maharashtra --api_key $XC_API_KEY
  python 01_download_weak_species.py --preview

Notes:
  - Searches India first. If a species returns fewer than --min_per_species
    recordings from India alone, it automatically broadens to neighboring
    countries (Bangladesh, Nepal, Sri Lanka, Myanmar, Bhutan, Pakistan).
  - Skips files you've already downloaded, checked by Xeno-canto recording ID.
  - Writes a metadata.csv per species folder for provenance tracking.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import requests


XC_API_URL = "https://xeno-canto.org/api/3/recordings"

FALLBACK_COUNTRIES = [
    "bangladesh",
    "nepal",
    "sri lanka",
    "myanmar",
    "bhutan",
    "pakistan",
]


# ---------------------------------------------------------------------------
# Maharashtra target species
# ---------------------------------------------------------------------------

TARGET_SPECIES = [
    # --- Urban / commensal ---
    "Rock Pigeon",
    "Spotted Dove",
    "Eurasian Collared Dove",
    "House Sparrow",
    "House Crow",
    "Large-billed Crow",
    "Common Myna",
    "Bank Myna",
    "Rose-ringed Parakeet",
    "Asian Koel",
    "Red-vented Bulbul",
    "Red-whiskered Bulbul",
    "Oriental Magpie-Robin",
    "Purple Sunbird",
    "Barn Swallow",

    # --- Forest / open country ---
    "Indian Peafowl",
    "Indian Roller",
    "White-throated Kingfisher",
    "Pied Kingfisher",
    "Stork-billed Kingfisher",
    "Coppersmith Barbet",
    "Common Hawk-Cuckoo",
    "Greater Coucal",
    "Indian Grey Hornbill",
    "Black Drongo",
    "Ashy Drongo",
    "White-browed Fantail",
    "Indian Paradise Flycatcher",
    "Tickell's Blue Flycatcher",
    "Indian Robin",
    "Jungle Babbler",
    "Yellow-billed Babbler",
    "Common Tailorbird",
    "Zitting Cisticola",
    "Baya Weaver",
    "Scaly-breasted Munia",
    "Indian Silverbill",
    "Spotted Owlet",
    "Indian Nightjar",
    "Common Iora",

    # --- Western Ghats / Sahyadri ---
    "Malabar Whistling Thrush",
    "Orange-headed Thrush",
    "Indian Scimitar Babbler",
    "Tawny-bellied Babbler",
    "Verditer Flycatcher",
    "Asian Brown Flycatcher",
    "Malabar Trogon",

    # --- Raptors ---
    "Shikra",
    "Black Kite",
    "Brahminy Kite",
    "White-eyed Buzzard",
    "Crested Serpent Eagle",
    "Changeable Hawk-Eagle",

    # --- Wetland / waterside ---
    "Indian Pond Heron",
    "Little Egret",
    "Green Bee-eater",
    "Blue-tailed Bee-eater",

    # --- Additional common Maharashtra species ---
    "Jungle Owlet",
    "Indian Cuckoo",
    "Black-rumped Flameback",
    "Lesser Goldenback",
    "White-rumped Munia",
    "Indian Wren-Babbler",
    "Ashy Prinia",
    "Plain Prinia",
    "Jungle Prinia",
    "Paddyfield Pipit",
    "Richard's Pipit",
    "Small Minivet",
    "Scarlet Minivet",
    "Common Kingfisher",
    "Brown-headed Barbet",
]

# Remove duplicates while preserving order.
TARGET_SPECIES = list(dict.fromkeys(TARGET_SPECIES))


def slugify(name: str) -> str:
    return name.replace("'", "").replace(" ", "_")


def query_xc(
    species_en: str,
    country: str,
    api_key: str,
    page: int = 1,
) -> dict:
    q = f'en:"{species_en}" cnt:"{country}"'

    resp = requests.get(
        XC_API_URL,
        params={
            "query": q,
            "key": api_key,
            "page": page,
        },
        timeout=30,
    )

    resp.raise_for_status()
    return resp.json()


def fetch_all_recordings(
    species_en: str,
    country: str,
    api_key: str,
) -> list:
    first = query_xc(
        species_en,
        country,
        api_key,
        page=1,
    )

    recordings = list(first.get("recordings", []))
    num_pages = int(first.get("numPages", 1))

    for page in range(2, num_pages + 1):
        time.sleep(1)

        page_data = query_xc(
            species_en,
            country,
            api_key,
            page=page,
        )

        recordings.extend(
            page_data.get("recordings", [])
        )

    return recordings


def download_recording(
    rec: dict,
    dest_dir: Path,
) -> bool:
    file_url = rec.get("file") or ""

    if file_url.startswith("//"):
        file_url = "https:" + file_url

    if not file_url:
        return False

    dest_path = dest_dir / f"xc_{rec['id']}.mp3"

    if dest_path.exists():
        return False

    try:
        response = requests.get(
            file_url,
            timeout=60,
        )

        response.raise_for_status()

        dest_path.write_bytes(
            response.content
        )

        return True

    except requests.RequestException as e:
        print(
            f"    ! failed to download "
            f"xc_{rec['id']}: {e}"
        )

        return False


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out_dir",
        default="./xc_maharashtra",
    )

    ap.add_argument(
        "--api_key",
        default=os.environ.get(
            "XC_API_KEY",
            "",
        ),
    )

    ap.add_argument(
        "--min_per_species",
        type=int,
        default=15,
    )

    ap.add_argument(
        "--preview",
        action="store_true",
    )

    args = ap.parse_args()

    if not args.preview and not args.api_key:
        raise SystemExit(
            "Need an API key: pass --api_key "
            "or set XC_API_KEY.\n"
            "Get one free at "
            "https://xeno-canto.org/account"
        )

    out_root = Path(args.out_dir)

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    print(
        f"Target species: {len(TARGET_SPECIES)}"
    )

    for species in TARGET_SPECIES:
        print(f"\n=== {species} ===")

        if args.preview:
            data = (
                query_xc(
                    species,
                    "india",
                    args.api_key,
                )
                if args.api_key
                else {}
            )

            n = (
                int(data.get("numRecordings", 0))
                if data
                else "?"
            )

            print(
                f"  India recordings available: {n}"
            )

            summary.append(
                (species, n, "india")
            )

            time.sleep(1)
            continue

        recordings = fetch_all_recordings(
            species,
            "india",
            args.api_key,
        )

        countries_used = ["india"]

        if len(recordings) < args.min_per_species:
            print(
                f"  Only {len(recordings)} from India, "
                "broadening search..."
            )

            for country in FALLBACK_COUNTRIES:
                time.sleep(1)

                more = fetch_all_recordings(
                    species,
                    country,
                    args.api_key,
                )

                if more:
                    recordings.extend(more)
                    countries_used.append(country)

                if len(recordings) >= args.min_per_species:
                    break

        # Deduplicate by Xeno-canto recording ID.
        unique = {
            rec.get("id"): rec
            for rec in recordings
            if rec.get("id")
        }

        recordings = list(unique.values())

        species_dir = (
            out_root / slugify(species)
        )

        species_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        downloaded = 0

        with open(
            species_dir / "metadata.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "id",
                "country",
                "quality",
                "length",
                "recordist",
                "license",
                "url",
            ])

            for rec in recordings:
                ok = download_recording(
                    rec,
                    species_dir,
                )

                if ok:
                    downloaded += 1

                writer.writerow([
                    rec.get("id"),
                    rec.get("cnt"),
                    rec.get("q"),
                    rec.get("length"),
                    rec.get("rec"),
                    rec.get("lic"),
                    rec.get("url"),
                ])

        print(
            f"  Found {len(recordings)} recordings "
            f"across {countries_used} -> "
            f"downloaded {downloaded} new files"
        )

        summary.append(
            (
                species,
                len(recordings),
                "+".join(countries_used),
            )
        )

        time.sleep(1)

    print("\n=== Summary ===")

    for species, n, countries in summary:
        print(
            f"  {species:40s} "
            f"{n:>4} recordings  "
            f"({countries})"
        )


if __name__ == "__main__":
    main()