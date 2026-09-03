# iBC53 — Indian Bird Call Dataset (Processed)

> **Project:** Bird identification from audio, India-specific species  
> **Base dataset:** [iBC53 on Kaggle](https://www.kaggle.com/datasets/arghyasahoo/ibc53-indian-bird-call-dataset)  
> **Directories renamed** to common names with underscores (e.g. `Puff-throated_Babbler`)
 
---
 
## Dataset Decisions
 
| Class | Reason |
|---|---|
| `Mystery` | **Dropped** — "unidentified" recordings, not a real species label |
| `Black-browed_Reed_Warbler` | **Dropped** — only 5.25 s of audio total (< 1 usable chunk) |
| `Asian_Emerald_Cuckoo` | **Dropped** — only 29.5 s of audio (~5 chunks), insufficient to learn from |
 
**Final class count: 50 species**
 
---
 
## Species Summary (sorted by total audio duration)
 
| # | Species | Recordings | Total Duration (s) | Est. 5s Chunks | Tier |
|---|---|---|---|---|---|
| 1 | Puff-throated_Babbler | 102 | 3973.39 | ~794 | 🟢 Strong |
| 2 | Indian_Cuckoo | 66 | 1979.51 | ~395 | 🟢 Strong |
| 3 | Hill_Partridge | 33 | 1712.63 | ~342 | 🟢 Strong |
| 4 | Streak-breasted_Scimitar_Babbler | 41 | 1652.54 | ~330 | 🟢 Strong |
| 5 | Asian_Barred_Owlet | 41 | 1376.27 | ~275 | 🟢 Strong |
| 6 | Pale_Blue_Flycatcher | 28 | 1359.92 | ~271 | 🟢 Strong |
| 7 | Hume's_Bar-tailed_Scimitar_Babbler | 43 | 1227.38 | ~245 | 🟢 Strong |
| 8 | Pygmy_Cupwing | 29 | 1043.50 | ~208 | 🟢 Strong |
| 9 | Pale-chinned_Blue_Flycatcher | 29 | 1019.68 | ~203 | 🟢 Strong |
| 10 | Red-faced_Liocichla | 23 | 990.22 | ~198 | 🟢 Strong |
| 11 | Leafbird | 21 | 971.50 | ~194 | 🟢 Strong |
| 12 | Long-billed_Wren-Babbler | 20 | 891.51 | ~178 | 🟢 Strong |
| 13 | Vernal_Hanging_Parrot | 34 | 825.06 | ~165 | 🟢 Strong |
| 14 | Grey-cheeked_Tit | 24 | 803.44 | ~160 | 🟢 Strong |
| 15 | Lineated_Barbet | 22 | 786.13 | ~157 | 🟢 Strong |
| 16 | Alexandrine_Parakeet | 28 | 707.09 | ~141 | 🟢 Strong |
| 17 | Chestnut-capped_Babbler | 24 | 695.83 | ~139 | 🟢 Strong |
| 18 | Andaman_Drongo | 20 | 682.48 | ~136 | 🟢 Strong |
| 19 | Yellow-bellied_Fantail | 16 | 660.55 | ~132 | 🟢 Strong |
| 20 | Jerdon's_Leafbird | 13 | 586.33 | ~117 | 🟢 Strong |
| 21 | Andaman_Coucal | 16 | 567.85 | ~113 | 🟢 Strong |
| 22 | Collared_Kingfisher | 23 | 479.75 | ~95 | 🟢 Strong |
| 23 | Chestnut-tailed_Starling | 18 | 415.34 | ~83 | 🟢 Strong |
| 24 | Slender-billed_Babbler | 11 | 406.04 | ~81 | 🟢 Strong |
| 25 | Jungle_Myna | 17 | 398.86 | ~79 | 🟢 Strong |
| 26 | Large-billed_Blue_Flycatcher | 8 | 377.36 | ~75 | 🟢 Strong |
| 27 | Grey-throated_Babbler | 15 | 376.24 | ~75 | 🟢 Strong |
| 28 | Citrine_Wagtail | 19 | 320.63 | ~64 | 🟢 Strong |
| 29 | Grey_Peacock-Pheasant | 11 | 310.10 | ~62 | 🟢 Strong |
| 30 | Yellow-browed_Warbler | 11 | 268.58 | ~53 | 🟡 Moderate |
| 31 | Yellow-throated_Leaf_Warbler | 7 | 258.30 | ~51 | 🟡 Moderate |
| 32 | Asian_Palm_Swift | 9 | 250.10 | ~50 | 🟡 Moderate |
| 33 | Streaked_Spiderhunter | 11 | 237.11 | ~47 | 🟡 Moderate |
| 34 | Baikal_Bush_Warbler | 8 | 216.69 | ~43 | 🟡 Moderate |
| 35 | Scarlet-backed_Flowerpecker | 8 | 188.80 | ~37 | 🟡 Moderate |
| 36 | Chinspot_Wren-Babbler | 5 | 185.39 | ~37 | 🟡 Moderate |
| 37 | Long-tailed_Shrike | 5 | 166.50 | ~33 | 🟡 Moderate |
| 38 | Spot-breasted_Parrotbill | 4 | 151.96 | ~30 | 🟡 Moderate |
| 39 | Oriental_Dollarbird | 5 | 137.19 | ~27 | 🟡 Moderate |
| 40 | Mrs_Gould's_Sunbird | 6 | 133.80 | ~26 | 🟡 Moderate |
| 41 | Plain_Flowerpecker | 7 | 128.59 | ~25 | 🟡 Moderate |
| 42 | Black-bellied_Plover | 7 | 122.92 | ~24 | 🟡 Moderate |
| 43 | Ruddy_Kingfisher | 8 | 110.57 | ~22 | 🟡 Moderate |
| 44 | Cachar_Bulbul | 6 | 105.94 | ~21 | 🟡 Moderate |
| 45 | White-tailed_Flycatcher | 3 | 92.67 | ~18 | 🔴 Weak |
| 46 | Yellow-vented_Flowerpecker | 3 | 83.03 | ~16 | 🔴 Weak |
| 47 | Grey-throated_Martin | 7 | 74.40 | ~14 | 🔴 Weak |
| 48 | Cinnamon_Bittern | 4 | 58.58 | ~11 | 🔴 Weak |
| 49 | Tickell's_Leaf_Warbler | 2 | 58.14 | ~11 | 🔴 Weak |
| 50 | Blue-winged_Leafbird | 1 | 47.08 | ~9 | 🔴 Weak |
 
**Totals (excl. dropped classes):** ~614 recordings · ~25,588 s of audio · ~5,117 estimated 5s chunks
 
---
 
## Tier Definitions
 
| Tier | Chunk count | Strategy |
|---|---|---|
| 🟢 Strong (≥50 chunks) | 29 classes | Standard training + mild augmentation |
| 🟡 Moderate (20–49 chunks) | 15 classes | SpecAugment + weighted loss |
| 🔴 Weak (<20 chunks) | 6 classes | Aggressive augmentation + high class weight; monitor closely |
 
---
 

