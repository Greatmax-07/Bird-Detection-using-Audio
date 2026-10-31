"""
01_download_weak_species.py
===========================
Download training audio from Xeno-canto for the species in
config/species_maharashtra.txt.

Needs a free API key: https://xeno-canto.org/account  ->  export XC_API_KEY=...

    python scripts/01_download_weak_species.py --preview              # counts only
    python scripts/01_download_weak_species.py                        # download
    python scripts/01_download_weak_species.py --species "House Crow" "Shikra"

Per species: India first; if fewer than --min_per_species recordings, the
neighbouring countries are added; if still short, the quality filter is relaxed
to --relax_quality. Species-file lines may list alternative xeno-canto spellings
('Rock Dove | Rock Pigeon'); each is tried until one yields recordings. Only
recordings whose English name matches one of them (ignoring case, hyphens and
spaces) are kept, since the API does substring matching. The best
--max_per_species recordings (quality A first) are downloaded to
<out_dir>/<Species_Name>/xc_<id>.mp3 with a metadata.csv alongside.

NOTE: query syntax (q:">D", len:"5-60", grp:birds) follows the Xeno-canto API v3
docs; if the API rejects a filter, use --quality "" / --length "" to drop it.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from birdear.core import DEFAULT_DATA_DIR, load_species_entries, name_key, slugify  # noqa: E402

XC_API_URL = "https://xeno-canto.org/api/3/recordings"
FALLBACK_COUNTRIES = ["bangladesh", "nepal", "sri lanka", "myanmar", "bhutan", "pakistan"]
QUALITY_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def api_get(params, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(XC_API_URL, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.RequestException(f"HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise SystemExit(f"xeno-canto API error: {data.get('error')} - {data.get('message')}")
            return data
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    ! {e}; retrying in {wait}s")
            time.sleep(wait)


def build_query(species, country, quality, length):
    q = f'en:"{species}" cnt:"{country}" grp:birds'
    if quality:
        q += f' q:"{quality}"'
    if length:
        q += f' len:"{length}"'
    return q


def fetch_recordings(species, country, key, quality, length):
    params = {"query": build_query(species, country, quality, length), "key": key, "page": 1}
    first = api_get(params)
    recs = list(first.get("recordings", []))
    for page in range(2, int(first.get("numPages", 1)) + 1):
        time.sleep(1)
        params["page"] = page
        recs.extend(api_get(params).get("recordings", []))
    return recs


def download(rec, dest_dir):
    url = rec.get("file") or ""
    if url.startswith("//"):
        url = "https:" + url
    if not url:
        return False
    dest = dest_dir / f"xc_{rec['id']}.mp3"
    if dest.exists():
        return False
    tmp = dest.with_suffix(".part")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        tmp.write_bytes(r.content)
        tmp.rename(dest)
        return True
    except requests.RequestException as e:
        print(f"    ! failed xc_{rec['id']}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--api_key", default=os.environ.get("XC_API_KEY", ""))
    ap.add_argument("--species", nargs="*", help="subset of names (default: whole species file)")
    ap.add_argument("--min_per_species", type=int, default=15,
                    help="below this many recordings in India, add neighbouring countries")
    ap.add_argument("--max_per_species", type=int, default=40)
    ap.add_argument("--quality", default=">D", help='xeno-canto quality filter, ">D" = A,B,C ("" = any)')
    ap.add_argument("--relax_quality", default=">E",
                    help='if still below --min_per_species, retry with this filter (">E" = A-D; "" = off)')
    ap.add_argument("--length", default="5-60", help='recording length filter in seconds ("" = any)')
    ap.add_argument("--preview", action="store_true", help="only print how many recordings match")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Need an API key: --api_key or XC_API_KEY (https://xeno-canto.org/account)")

    entries = load_species_entries()
    if args.species:
        wanted = {name_key(n) for n in args.species}
        entries = [e for e in entries if any(name_key(n) in wanted for n in e)]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(entries)} species -> {out_root}")

    def gather(names, quality):
        """India, then neighbours until min_per_species; exact-name matches only."""
        found, used = {}, []
        for country in ["india"] + FALLBACK_COUNTRIES:
            if country != "india" and len(found) >= args.min_per_species:
                break
            before = len(found)
            for name in names:
                recs = fetch_recordings(name, country, args.api_key, quality, args.length)
                keys = {name_key(n) for n in names}
                for r in recs:
                    if r.get("id") and name_key(r.get("en", "")) in keys:
                        found[r["id"]] = r
                if recs:
                    time.sleep(1)
            if len(found) > before:
                used.append(country)
            time.sleep(1)
        return found, used

    summary = []
    for names in entries:
        species = names[0]
        print(f"\n=== {species} ===" + (f"  (also: {', '.join(names[1:])})" if names[1:] else ""))
        exact, used = gather(names, args.quality)
        if len(exact) < args.min_per_species and args.relax_quality and args.relax_quality != args.quality:
            more, used2 = gather(names, args.relax_quality)
            if len(more) > len(exact):
                print(f"  relaxed quality to {args.relax_quality!r}: {len(exact)} -> {len(more)}")
                exact, used = more, used2
        ranked = sorted(exact.values(), key=lambda r: QUALITY_RANK.get(r.get("q"), 9))
        ranked = ranked[: args.max_per_species]
        print(f"  {len(ranked)} usable recordings ({'+'.join(used) or 'none'})")
        summary.append((species, len(ranked)))

        if not args.preview and ranked:
            sdir = out_root / slugify(species)
            sdir.mkdir(parents=True, exist_ok=True)
            new = 0
            with open(sdir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "country", "quality", "length", "recordist", "license", "url"])
                for rec in ranked:
                    new += download(rec, sdir)
                    w.writerow([rec.get("id"), rec.get("cnt"), rec.get("q"), rec.get("length"),
                                rec.get("rec"), rec.get("lic"), rec.get("url")])
            print(f"  downloaded {new} new files")

    print("\n=== Summary (species with fewer than 10 usable recordings) ===")
    for species, n in summary:
        if n < 10:
            print(f"  {species:35s} {n}")
    print(f"total recordings: {sum(n for _, n in summary)}")


if __name__ == "__main__":
    main()
