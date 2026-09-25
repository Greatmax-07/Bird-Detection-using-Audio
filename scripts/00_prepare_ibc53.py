#!/usr/bin/env python3
"""
00_prepare_ibc53.py

Convert raw iBC53 class directories from scientific names to the
common-name_with_underscores convention used by this project.

Also removes the 3 classes excluded by the project:
    Mystery
    Black-browed_Reed_Warbler
    Asian_Emerald_Cuckoo

Usage:
    python scripts/00_prepare_ibc53.py --data_dir ./ibc53 --dry-run
    python scripts/00_prepare_ibc53.py --data_dir ./ibc53

The script expects the raw iBC53 directory to contain one directory per class.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# Final common-name class list from the project README.
# These are the classes we want to retain.
KEEP_CLASSES = {
    "Puff-throated_Babbler",
    "Indian_Cuckoo",
    "Hill_Partridge",
    "Streak-breasted_Scimitar_Babbler",
    "Asian_Barred_Owlet",
    "Pale_Blue_Flycatcher",
    "Humes_Bar-tailed_Scimitar_Babbler",
    "Pygmy_Cupwing",
    "Pale-chinned_Blue_Flycatcher",
    "Red-faced_Liocichla",
    "Leafbird",
    "Long-billed_Wren-Babbler",
    "Vernal_Hanging_Parrot",
    "Grey-cheeked_Tit",
    "Lineated_Barbet",
    "Alexandrine_Parakeet",
    "Chestnut-capped_Babbler",
    "Andaman_Drongo",
    "Yellow-bellied_Fantail",
    "Jerdons_Leafbird",
    "Andaman_Coucal",
    "Collared_Kingfisher",
    "Chestnut-tailed_Starling",
    "Slender-billed_Babbler",
    "Jungle_Myna",
    "Large-billed_Blue_Flycatcher",
    "Citrine_Wagtail",
    "Grey_Peacock-Pheasant",
    "Yellow-browed_Warbler",
    "Yellow-throated_Leaf_Warbler",
    "Asian_Palm_Swift",
    "Streaked_Spiderhunter",
    "Baikal_Bush_Warbler",
    "Scarlet-backed_Flowerpecker",
    "Chinspot_Wren-Babbler",
    "Long-tailed_Shrike",
    "Spot-breasted_Parrotbill",
    "Oriental_Dollarbird",
    "Mrs_Goulds_Sunbird",
    "Plain_Flowerpecker",
    "Black-bellied_Plover",
    "Ruddy_Kingfisher",
    "Cachar_Bulbul",
    "White-tailed_Flycatcher",
    "Yellow-vented_Flowerpecker",
    "Grey-throated_Martin",
    "Cinnamon_Bittern",
    "Tickells_Leaf_Warbler",
    "Blue-winged_Leafbird",
}

DROP_CLASSES = {
    "Mystery",
    "Black-browed_Reed_Warbler",
    "Asian_Emerald_Cuckoo",
}


def normalize_common_name(name: str) -> str:
    """Convert BirdNET common name to this project's directory convention."""
    return (
        name.strip()
        .replace("'", "")
        .replace(" ", "_")
    )


def load_birdnet_labels() -> dict[str, str]:
    """
    Return mapping:

        scientific_name -> project_common_directory_name

    Pulls the label list straight from the already-installed birdnetlib
    package (same file used to build your embeddings) instead of fetching
    from GitHub — no network call, and guaranteed to match your installed
    model version.
    """
    print("Loading BirdNET labels from installed birdnetlib package...")

    from birdnetlib.analyzer import Analyzer

    analyzer = Analyzer()
    labels = getattr(analyzer, "labels", None)

    if not labels:
        raise SystemExit(
            "Could not read labels from birdnetlib's Analyzer object. "
            "Run this to inspect what's available:\n"
            "  python -c \"from birdnetlib.analyzer import Analyzer; "
            "a = Analyzer(); print(vars(a).keys())\""
        )

    mapping: dict[str, str] = {}

    for raw_line in labels:
        line = raw_line.strip()

        if not line or "_" not in line:
            continue

        scientific, common = line.split("_", 1)
        scientific = scientific.strip()
        common = common.strip()

        if not scientific or not common:
            continue

        common_dir = normalize_common_name(common)

        if common_dir in KEEP_CLASSES:
            mapping[scientific] = common_dir

    return mapping


def rename_two_phase(rename_pairs: list[tuple[Path, Path]]) -> None:
    """
    Rename through temporary names first so swapping/conflicting names
    cannot collide.
    """
    temporary_pairs: list[tuple[Path, Path]] = []

    for src, dst in rename_pairs:
        tmp = src.parent / f".__tmp_rename__{src.name}"
        counter = 1

        while tmp.exists():
            tmp = src.parent / f".__tmp_rename__{src.name}_{counter}"
            counter += 1

        src.rename(tmp)
        temporary_pairs.append((tmp, dst))

    for tmp, dst in temporary_pairs:
        if dst.exists():
            raise RuntimeError(
                f"Destination already exists, refusing to overwrite: {dst}"
            )
        tmp.rename(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Raw iBC53 directory containing one subdirectory per class",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without modifying anything",
    )
    parser.add_argument(
        "--delete-dropped",
        action="store_true",
        help="Delete excluded classes instead of leaving them untouched",
    )

    args = parser.parse_args()

    root = args.data_dir.resolve()

    if not root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {root}")

    print(f"\nDataset: {root}")

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    if not class_dirs:
        raise SystemExit("No class directories found.")

    print(f"Found {len(class_dirs)} class directories.\n")

    labels = load_birdnet_labels()

    rename_pairs: list[tuple[Path, Path]] = []
    already_common: list[Path] = []
    unresolved: list[Path] = []
    dropped: list[Path] = []

    for src in class_dirs:
        name = src.name

        # Already in target naming convention.
        if name in KEEP_CLASSES:
            already_common.append(src)
            continue

        # Explicitly dropped classes.
        if name in DROP_CLASSES:
            dropped.append(src)
            continue

        # Direct scientific-name match.
        target_name = labels.get(name)

        if target_name is None:
            unresolved.append(src)
            continue

        dst = root / target_name

        if src != dst:
            rename_pairs.append((src, dst))

    print("=== PLAN ===")

    if already_common:
        print(f"\nAlready common-name folders: {len(already_common)}")

    print(f"Folders to rename:           {len(rename_pairs)}")
    print(f"Folders to drop:             {len(dropped)}")
    print(f"Unresolved folders:           {len(unresolved)}")

    if rename_pairs:
        print("\nRenames:")
        for src, dst in rename_pairs:
            print(f"  {src.name}  ->  {dst.name}")

    if dropped:
        print("\nDropped classes:")
        for p in dropped:
            print(f"  {p.name}")

    if unresolved:
        print("\nUNRESOLVED folders:")
        for p in unresolved:
            print(f"  {p.name}")

    if unresolved:
        print(
            "\nNothing will be renamed while unresolved folders exist."
        )
        print(
            "This prevents silently assigning the wrong species label."
        )
        raise SystemExit(2)

    if args.dry_run:
        print("\nDry run complete. No changes made.")
        return

    print("\nApplying changes...")

    if rename_pairs:
        rename_two_phase(rename_pairs)

    if args.delete_dropped:
        for p in dropped:
            print(f"  Removing: {p.name}")
            shutil.rmtree(p)
    else:
        print(
            "\nExcluded classes were NOT deleted."
            "\nRun again with --delete-dropped after checking the plan."
        )

    print("\nPreparation complete.")

    final_dirs = sorted(
        p.name for p in root.iterdir()
        if p.is_dir()
    )

    print(f"Final class directories: {len(final_dirs)}")


if __name__ == "__main__":
    main()

