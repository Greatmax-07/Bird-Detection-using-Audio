"""
01_download_weak_species.py
============================
Pull additional recordings from Xeno-canto for the species that are
underrepresented in the iBC53 dataset, plus a set of common urban species.

Xeno-canto API v3 requires a free API key:
  1. Register / sign in at https://xeno-canto.org
  2. Verify your email
  3. Grab your key from https://xeno-canto.org/account
  4. export XC_API_KEY="your-key-here"   (or pass --api_key)

Usage:
  python 01_download_weak_species.py --out_dir ./xc_downloads --api_key $XC_API_KEY
  python 01_download_weak_species.py --preview

Notes:
  - Searches India first. If a species returns fewer than --min_per_species
    recordings from India alone, it automatically broadens the search to
    neighboring countries (Bangladesh, Nepal, Sri Lanka, Myanmar, Bhutan,
    Pakistan).
  - Skips files you've already downloaded, checked by Xeno-canto recording ID.
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

FALLBACK_COUNTRIES = [
    "bangladesh",
    "nepal",
    "sri lanka",
    "myanmar",
    "bhutan",
    "pakistan",
]


# ---------------------------------------------------------------------------
# iBC53 weak / moderate species
# ---------------------------------------------------------------------------

TARGET_SPECIES = [
    "Yellow-browed Warbler",
    "Yellow-throated Leaf Warbler",
    "Asian Palm Swift",
    "Streaked Spiderhunter",
    "Baikal Bush Warbler",
    "Scarlet-backed Flowerpecker",
    "Chinspot Wren-Babbler",
    "Long-tailed Shrike",
    "Spot-breasted Parrotbill",
    "Oriental Dollarbird",
    "Mrs Gould's Sunbird",
    "Plain Flowerpecker",
    "Black-bellied Plover",
    "Ruddy Kingfisher",
    "Cachar Bulbul",
    "White-tailed Flycatcher",
    "Yellow-vented Flowerpecker",
    "Grey-throated Martin",
    "Cinnamon Bittern",
    "Tickell's Leaf Warbler",
    "Blue-winged Leafbird",
]


# ---------------------------------------------------------------------------
# Common urban species
# ---------------------------------------------------------------------------

COMMON_URBAN_SPECIES = [
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
]


# Combine both groups while preserving order and avoiding duplicates.
TARGET_SPECIES = list(dict.fromkeys(TARGET_SPECIES + COMMON_URBAN_SPECIES))


def slugify(name: str) -> str:
    """Convert a species name into a filesystem-friendly folder name."""
    return name.replace("'", "").replace(" ", "_").replace("-", "-")


def query_xc(
    species_en: str,
    country: str,
    api_key: str,
    page: int = 1,
) -> dict:
    """Query Xeno-canto for recordings of one species in one country."""
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
    """Fetch all available recordings for a species/country combination."""

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

        recordings.extend(page_data.get("recordings", []))

    return recordings


def download_recording(
    rec: dict,
    dest_dir: Path,
) -> bool:
    """Download one Xeno-canto recording."""

    file_url = rec.get("file") or ""

    if file_url.startswith("//"):
        file_url = "https:" + file_url

    if not file_url:
        return False

    dest_path = dest_dir / f"xc_{rec['id']}.mp3"

    # Skip recordings that already exist.
    if dest_path.exists():
        return False

    try:
        response = requests.get(
            file_url,
            timeout=60,
        )

        response.raise_for_status()

        dest_path.write_bytes(response.content)

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
        default="./xc_downloads",
        help="directory where downloaded recordings will be stored",
    )

    ap.add_argument(
        "--api_key",
        default=os.environ.get("XC_API_KEY", ""),
        help="Xeno-canto API key",
    )

    ap.add_argument(
        "--min_per_species",
        type=int,
        default=15,
        help=(
            "if India-only results are below this, "
            "broaden to neighboring countries"
        ),
    )

    ap.add_argument(
        "--preview",
        action="store_true",
        help="only print counts, download nothing",
    )

    args = ap.parse_args()

    if not args.preview and not args.api_key:
        raise SystemExit(
            "Need an API key: pass --api_key or set XC_API_KEY. "
            "Get one free at https://xeno-canto.org/account"
        )

    out_root = Path(args.out_dir)

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    for species in TARGET_SPECIES:

        print(f"\n=== {species} ===")

        # ---------------------------------------------------------------
        # Preview mode
        # ---------------------------------------------------------------

        if args.preview:

            data = query_xc(
                species,
                "india",
                args.api_key,
            ) if args.api_key else {}

            n = (
                int(data.get("numRecordings", 0))
                if data
                else "?"
            )

            print(
                f"  India recordings available: {n}"
            )

            summary.append(
                (
                    species,
                    n,
                    "india",
                )
            )

            time.sleep(1)

            continue

        # ---------------------------------------------------------------
        # Normal download mode
        # ---------------------------------------------------------------

        recordings = fetch_all_recordings(
            species,
            "india",
            args.api_key,
        )

        countries_used = ["india"]

        # ---------------------------------------------------------------
        # Broaden search if India has too few recordings
        # ---------------------------------------------------------------

        if len(recordings) < args.min_per_species:

            print(
                f"  Only {len(recordings)} from India, "
                f"broadening search..."
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

        # ---------------------------------------------------------------
        # Remove duplicate Xeno-canto recordings
        #
        # This can happen if the same recording appears through
        # overlapping search results or country searches.
        # ---------------------------------------------------------------

        unique_recordings = {}
        for rec in recordings:
            rec_id = rec.get("id")

            if rec_id:
                unique_recordings[rec_id] = rec

        recordings = list(unique_recordings.values())

        # ---------------------------------------------------------------
        # Create species directory
        # ---------------------------------------------------------------

        species_dir = out_root / slugify(species)

        species_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        downloaded = 0

        metadata_path = species_dir / "metadata.csv"

        with open(
            metadata_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "id",
                    "country",
                    "quality",
                    "length",
                    "recordist",
                    "license",
                    "url",
                ]
            )

            for rec in recordings:

                ok = download_recording(
                    rec,
                    species_dir,
                )

                if ok:
                    downloaded += 1

                writer.writerow(
                    [
                        rec.get("id"),
                        rec.get("cnt"),
                        rec.get("q"),
                        rec.get("length"),
                        rec.get("rec"),
                        rec.get("lic"),
                        rec.get("url"),
                    ]
                )

        print(
            f"  Found {len(recordings)} recordings "
            f"across {countries_used} "
            f"-> downloaded {downloaded} new files"
        )

        summary.append(
            (
                species,
                len(recordings),
                "+".join(countries_used),
            )
        )

        time.sleep(1)

    # -------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------

    print("\n=== Summary ===")

    for species, n, countries in summary:

        print(
            f"  {species:40s} "
            f"{n:>4} recordings  "
            f"({countries})"
        )


if __name__ == "__main__":
    main()
