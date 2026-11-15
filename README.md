# Driver Drowsiness Detection: Eye-State Classification (MRL Eye Dataset)

**Course project, Phase 1 (data) and Phase 2 (modeling).**
Team: _add names_ | Deadline Phase 1: 10 Nov | Phase 2: 28 Nov

## 1. Short description
A camera-based system that decides whether a driver's eye is **open or closed**. Repeated or long eye closures are an early sign of drowsiness, so a reliable eye-state classifier is the core building block of a drowsiness alarm. This repository contains the data work (download, quality checks, cleaning, EDA, preprocessing) and a baseline CNN.

## 2. Problem statement
Given a single grayscale (infrared) image of an eye, predict whether the eye is closed (**1**) or open (**0**). Because a missed closed eye is the costly mistake, models are judged by recall / F1 of the closed class and AUC, not accuracy alone. The model must work across lighting, glasses, reflections and camera sensors, and on people it has never seen.

## 3. Dataset
| | |
|---|---|
| Name | MRL Eye Dataset (`mrlEyes_2018_01`) |
| Source | http://mrl.cs.vsb.cz/eyedataset (direct ZIP: http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip) |
| Size | 84,898 infrared eye images from 37 people (33 men, 4 women) per the dataset authors |
| Features | Image pixels + metadata encoded in the filename: subject, image ID, gender, glasses, reflections, lighting, sensor |
| Target | **`closed`**: 1 = closed, 0 = open (MRL codes eye state the other way round: 0 = closed, 1 = open; we flip it) |

The dataset is not stored in this repo (too large). `notebooks/00_data_download_and_quality_checks.ipynb` downloads it, or place the images in `data/raw/mrl/`.

Filename format: `s0001_00001_0_0_0_0_0_01.png` = `subject_imageID_gender_glasses_eyeState_reflections_lighting_sensor`.

## 4. Cleaning summary (`01_cleaning.ipynb`)
| Issue | How it was handled |
|---|---|
| Missing / malformed labels | Filenames must have 8 fields; rows with missing or invalid values are dropped |
| Inconsistent formats | Subject IDs lower-cased, coded fields converted to integers; image size and colour mode standardised later (grayscale, 64x64) |
| Corrupt images | Files that cannot be decoded are dropped |
| Duplicates | Identical pixel content removed (first copy kept); duplicate groups with conflicting labels removed entirely |
| Outliers | Blank / tiny images removed; extreme brightness flagged but kept (real lighting conditions) |

Exact counts per step: `data/processed/cleaning_log.csv`. Raw-data audit: `data/processed/quality_report.json`.

## 5. Key findings
Generated from the data by `02_eda.ipynb` into `images/key_findings.md` - **paste that text here after running the notebooks on the real data.**

Figures (each notebook explains what it shows) are in `images/`: `quality_*` (audit), `clean_*` (cleaning), `eda_*` (EDA), `prep_*` (preprocessing), `train_*` (baseline model).

## 6. Preprocessing (`03_preprocessing.ipynb`)
Grayscale, 64x64, pixels scaled to [0, 1], subject-wise 70/15/15 split (no person in two splits, checked by assertion), class weights from the training set. Output: `data/processed/eye_dataset_64.npz`, `splits.csv`, `normalization_stats.json`.

## 7. How to run
```bash
git clone <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Run the notebooks in `notebooks/` in this order (Run All in each):

| # | Notebook | Purpose |
|---|---|---|
| 00 | `00_data_download_and_quality_checks.ipynb` | Download/upload the data, check labels, corrupt files, duplicates |
| 01 | `01_cleaning.ipynb` | Clean the data, write `metadata_clean.csv` |
| 02 | `02_eda.ipynb` | EDA, figures, key findings |
| 03 | `03_preprocessing.ipynb` | Normalise, split, save arrays |
| 04 | `04_training_baseline_cnn.ipynb` | Baseline CNN (optional for Phase 1) |

Set `SOURCE` in notebook 00 (`'official'` to download automatically). To point the notebooks at another image folder: `export MRL_DIR=/path/to/images` before starting Jupyter. On Colab, upload the repo, set `SOURCE='upload'` and run the same notebooks.

## 8. Repository structure
```
data/raw/          images (not committed)
data/processed/    metadata_clean.csv, cleaning_log.csv, quality_report.json, splits.csv, normalization_stats.json
images/            all figures + key_findings.md
models/            trained model + metrics (Phase 2)
notebooks/         00-04
drowsiness_detection_system.ipynb   full demo: training, video test, live webcam
requirements.txt
```
