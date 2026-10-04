# Bird call imbalance pipeline

Three scripts, run in order:

1. **`01_download_weak_species.py`** — pulls more Xeno-canto recordings
   for your 15 moderate-tier + 6 weak-tier species (the ones under ~50
   estimated 5s chunks). Needs a free API key from
   https://xeno-canto.org/account (API v3 requires one since Oct 2025).

   ```
   python 01_download_weak_species.py --out_dir ./xc_downloads --api_key $XC_API_KEY
   ```

2. **`02_extract_birdnet_embeddings.py`** — instead of training a CNN from
   scratch on your ~5,100 chunks, this extracts 1024-dim embeddings from
   BirdNET (pretrained on millions of global bird recordings) for every
   file in both your existing dataset and the new downloads. Point it at
   both folders:

   ```
   python 02_extract_birdnet_embeddings.py \
       --data_dirs ./ibc53_processed ./xc_downloads \
       --out_dir ./features
   ```

3. **`03_train_classifier_head.py`** — trains a small MLP on top of the
   frozen embeddings, using focal loss + class weighting so the weak
   classes actually get learned instead of ignored. Prints a per-class
   F1/precision/recall report so you can see whether the weak-tier
   species actually improved (accuracy alone will hide this).

   ```
   python 03_train_classifier_head.py --features_dir ./features
   ```

## Notes

- Adjust `TARGET_SPECIES` in script 1 if your final class list changes.
- Script 1 automatically expands to neighboring countries (Bangladesh,
  Nepal, Sri Lanka, Myanmar, Bhutan, Pakistan) if India alone doesn't
  return enough recordings for a species — same vocalizations, more data.
- `birdnetlib` needs `ffmpeg` installed on your system PATH.
- If a class is still starved for data after step 1, consider merging it
  into a coarser label (e.g. genus-level) rather than force-fitting a
  50-way classifier — the per-class report from step 3 will tell you
  which ones, if any, still need this.
