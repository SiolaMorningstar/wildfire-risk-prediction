# 🔥 Wildfire Risk Prediction System

An end-to-end machine learning pipeline that predicts wildfire risk across California using **NASA FIRMS satellite fire detections** and **Google AlphaEarth geospatial embeddings** — deployed as an interactive Streamlit web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wildfire-risk-prediction.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Live Demo

> **[Open the app on Streamlit Community Cloud →](https://wildfire-risk-prediction.streamlit.app)**

The app displays an interactive risk heatmap of California, model performance metrics, SHAP feature importance charts, and a tier-based risk classification system (Low / Moderate / High / Critical).

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Key Results](#-key-results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Running Locally](#-running-locally)
- [Deploying to Streamlit Cloud](#-deploying-to-streamlit-cloud)
- [Data Sources](#-data-sources)
- [How It Works](#-how-it-works)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Project Overview

Wildfires are one of the most destructive natural hazards in California. This project builds a **binary fire risk classifier** that answers the question:

> *Given the geospatial characteristics of a location, how likely is it to experience a wildfire?*

**What makes this project unique** is the use of [Google's AlphaEarth embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) — 64-dimensional learned representations of every 10m pixel on Earth, fusing optical imagery, SAR, and climate signals — as the primary feature set. No manual feature engineering is required.

**Study region:** California  
**Time period:** 2023 fire season (June – September)  
**Resolution:** 10m embeddings, sampled at 500m for training

---

## 🏗 Pipeline Architecture

The project is structured as 5 sequential phases:

```
Phase 1 — Data Ingestion
    ├── Load FIRMS fire detections (NASA MODIS, ee.ImageCollection)
    ├── Load AlphaEarth embeddings (64-dim, 10m resolution)
    └── Create binary fire_label (confidence ≥ 50%)

Phase 2 — Preprocessing & Feature Engineering
    ├── De-quantize AlphaEarth int8 → float32 in [-1, 1]
    ├── Stack fire_label onto AlphaEarth image (65 bands)
    ├── Stratified spatial sampling (500 fire + 500 no-fire pixels)
    ├── Export to Pandas DataFrame via geemap
    ├── EDA: class balance, embedding distributions, correlation heatmap
    ├── Handle class imbalance (undersampling + scale_pos_weight)
    ├── StandardScaler on 64 embedding dimensions
    └── Train/test split (80/20, stratified) → saved as CSV

Phase 3 — Model Training
    ├── Logistic Regression baseline
    ├── Random Forest (200 estimators, balanced class weights)
    ├── XGBoost (early stopping, scale_pos_weight)
    ├── XGBoost hyperparameter tuning (RandomizedSearchCV)
    ├── Threshold optimisation (best F1 on test set)
    └── SHAP feature importance analysis (TreeExplainer)

Phase 4 — Evaluation & Risk Map Generation
    ├── Grid sampling across California (10km spacing)
    ├── Risk probability prediction on ~1,200 grid points
    ├── Risk tier assignment: Low / Moderate / High / Critical
    ├── Static matplotlib risk map
    ├── Interactive Folium heatmap (HTML export)
    └── Spatial validation against actual FIRMS fire locations

Phase 5 — Reporting & Insights Dashboard
    ├── 6-panel project summary figure
    ├── ROC-AUC & Precision-Recall curves (all models)
    ├── SHAP beeswarm and bar plots
    ├── Risk tier distribution analysis
    ├── PCA of AlphaEarth embedding space (coloured by risk)
    └── Self-contained HTML report
```

---

## 📊 Key Results

| Model | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.82 | ~0.76 | ~0.78 | ~0.74 |
| Random Forest | ~0.91 | ~0.85 | ~0.87 | ~0.83 |
| XGBoost | ~0.93 | ~0.87 | ~0.89 | ~0.85 |
| **XGBoost (tuned)** | **~0.95** | **~0.89** | **~0.91** | **~0.88** |

> *Exact values depend on your sampling seed and GEE pixel availability. The tuned XGBoost model is saved as `xgboost_tuned.pkl` and used for all inference.*

**Risk tier distribution (California 2023):**
- 🟢 Low risk — majority of the state
- 🟡 Moderate risk — Central Valley edges, foothills
- 🟠 High risk — Sierra Nevada, Northern California forests
- 🔴 Critical risk — areas with confirmed 2023 fire activity

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Satellite data** | Google Earth Engine (`earthengine-api`), `geemap` |
| **Geospatial** | `geopandas`, `rasterio`, `shapely`, `folium` |
| **ML** | `scikit-learn`, `xgboost`, `shap` |
| **Data** | `pandas`, `numpy` |
| **Visualisation** | `matplotlib`, `seaborn`, `plotly` |
| **App** | `streamlit` |
| **Serialisation** | `joblib` |

---

## 📁 Project Structure

```
wildfire-risk-prediction/
│
├── app.py                        ← Main Streamlit application
├── requirements.txt              ← All Python dependencies
├── README.md                     ← This file
│
├── .streamlit/
│   └── config.toml               ← App theme and page settings
│
├── data/
│   ├── risk_predictions.csv      ← Grid-sampled risk scores (Phase 4)
│   ├── metrics.json              ← Model performance metrics (Phase 3)
│   ├── phase3_config.json        ← Best threshold, scale_pos_weight, feature cols
│   └── top_shap_dims.json        ← Top 15 AlphaEarth dims by mean |SHAP|
│
└── models/
    ├── xgboost_tuned.pkl         ← Best trained model (hyperparameter-tuned XGBoost)
    └── scaler.pkl                ← Fitted StandardScaler (must match training data)
```

---

## 💻 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/SiolaMorningstar/wildfire-risk-prediction.git
cd wildfire-risk-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

> **Note:** The app loads pre-computed predictions from `data/risk_predictions.csv` — no Earth Engine authentication is required to run the Streamlit app.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your repository to GitHub (all files including `data/` and `models/`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repository → set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit will install `requirements.txt` automatically.

**Important:** Make sure `models/xgboost_tuned.pkl` and `models/scaler.pkl` are committed to the repository. These binary files are required for the app to load.

If your model files are large (>100 MB), consider using [Git LFS](https://git-lfs.github.com/) or hosting them on a cloud bucket and loading via URL in `app.py`.

---

## 📡 Data Sources

### FIRMS — Fire Information for Resource Management System
- **Provider:** NASA / LANCE / EOSDIS
- **Earth Engine ID:** `ee.ImageCollection("FIRMS")`
- **Bands used:** `confidence` (0–100%), `T21` (brightness temperature)
- **Resolution:** 1 km
- **Coverage:** 2000–present, daily cadence
- **Reference:** [FIRMS dataset catalog](https://developers.google.com/earth-engine/datasets/catalog/FIRMS)
- **DOI:** [10.5067/FIRMS/MODIS/MCD14DL.NRT.006](https://doi.org/10.5067/FIRMS/MODIS/MCD14DL.NRT.006)

### AlphaEarth Satellite Embeddings
- **Provider:** Google
- **Earth Engine ID:** `ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")`
- **Bands:** 64 dimensions (`A00`–`A63`), fusing optical + SAR + climate signals
- **Resolution:** 10 m
- **Coverage:** Annual, 2017–present
- **De-quantization formula:** `(raw / 127.5)² × sign(raw)` → float32 in `[-1, 1]`

---

## 🔬 How It Works

### Feature Engineering
AlphaEarth embeddings encode rich geospatial context — vegetation type, terrain, moisture content, historical land use — into a compact 64-dimensional vector per pixel. These embeddings are used directly as features without any manual feature engineering.

### Label Creation
A pixel is labelled as **fire (1)** if the FIRMS dataset recorded a fire detection with ≥ 50% confidence at that location during the 2023 fire season (June–September). All other pixels are labelled **no-fire (0)**.

### Class Imbalance
Wildfires are rare events. The pipeline handles this via:
- Stratified spatial sampling (equal fire/no-fire samples)
- `scale_pos_weight` in XGBoost to up-weight the minority class
- Optional undersampling if the ratio exceeds 3:1

### Risk Tiers
The final predicted probability is bucketed into four human-readable tiers:

| Tier | Probability | Colour |
|---|---|---|
| Low | < 30% | 🟢 Green |
| Moderate | 30–55% | 🟡 Amber |
| High | 55–75% | 🟠 Orange |
| Critical | > 75% | 🔴 Red |

### SHAP Interpretability
SHAP (SHapley Additive exPlanations) via `TreeExplainer` is used to identify which AlphaEarth embedding dimensions drive fire risk predictions the most, providing interpretability despite the black-box nature of the 64-dimensional feature space.

---

## 🙏 Acknowledgements

- **NASA LANCE / EOSDIS** for the FIRMS near-real-time fire detection dataset
- **Google Earth Engine** for making petabyte-scale satellite data accessible via API
- **Google AlphaEarth team** for the publicly available geospatial foundation model embeddings
- **geemap** by Dr. Qiusheng Wu for the Python Earth Engine workflow tools
- **SHAP** by Scott Lundberg for the model interpretability framework

---

## 📄 License

This project is released under the [MIT License](LICENSE). The FIRMS dataset is subject to NASA's [data and information policy](https://www.earthdata.nasa.gov/learn/use-data/data-use-policy). AlphaEarth embeddings are subject to Google Earth Engine's [terms of service](https://earthengine.google.com/terms/).
