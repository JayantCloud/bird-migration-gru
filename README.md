# Bird Migration Trajectory Prediction using GRU

## Project Overview

This project analyzes and predicts bird migration patterns using GPS tracking data from 3 White Storks (Eric, Nico, Sanne) tracked from August 2013 to April 2014. The system combines deep learning (GRU neural network), unsupervised clustering (HDBSCAN), real-time climate data integration (Open-Meteo API), and AI-powered insights (Google Gemini) to model and predict migration trajectories.

**Developed as part of ISI Kolkata IDEAS Advanced Data Science Internship Programme, 2026.**

---

## Team

| Member | Contribution |
|--------|-------------|
| Jayant Pandey | Data ingestion, cleaning, feature engineering, climate data integration, MinMaxScaler preprocessing, GRU sequence creation, cross-pipeline debugging, inverse transform post-processing (40%) |
| B. Jasvanth | HDBSCAN clustering, Folium visualisation, GRU model architecture and training, model evaluation, Gemini AI climate insights, final integration and submission (60%) |

---

## Dataset

- **Source:** GPS tracking data — White Stork migration
- **Records:** 61,920 GPS readings across 3 birds
- **Period:** August 2013 – April 2014
- **Features:** latitude, longitude, altitude, speed, direction, timestamp
- **Climate data:** Fetched via Open-Meteo Archive API (temperature, wind speed, precipitation per GPS coordinate)

---

## Project Architecture

```
Raw GPS Data
     │
     ▼
Data Cleaning & Feature Engineering (Jayant)
     │  ├── Datetime parsing, null handling, chronological sorting
     │  ├── Temporal features: unix_timestamp, day_of_year, hour
     │  └── Climate fetch: Open-Meteo API (61,920 coordinates)
     │
     ▼
MinMaxScaler Preprocessing (Jayant)
     │  ├── Per-bird scaling (latitude, longitude, unix_timestamp, day_of_year, hour)
     │  └── Scalers saved for inverse transform
     │
     ▼
Sequence Creation (Jayant)
     │  └── Sliding window: 10 timesteps × 5 features → predict next lat/lon
     │
     ▼
GRU Model Training (Jasvanth)
     │  ├── Input shape: (n, 10, 5) → Output: (n, 2)
     │  └── Per-bird models for Eric, Nico, Sanne
     │
     ▼
HDBSCAN Clustering + Folium Visualisation (Jasvanth)
     │  └── 11 migration clusters identified
     │
     ▼
Gemini AI Climate Insights (Jasvanth)
     │  └── Climate patterns correlated with migration behaviour
     │
     ▼
Inverse Transform → Real-world Coordinates (Jayant)
```

---

## Jayant's Contribution — Technical Detail

**Data Pipeline & Preprocessing**
- Loaded and cleaned 61,920 GPS records — fixed datetime parsing (UTC), forward filled 443 nulls in `direction` and `speed_2d`, sorted chronologically per bird
- Engineered 3 temporal features for GRU: `unix_timestamp` (chronological ordering), `day_of_year` (seasonal patterns), `hour` (intra-day behaviour)
- Applied MinMaxScaler independently per bird — each bird has a different geographic range requiring separate scaling

**Climate Data Integration**
- Fetched historical weather for all 61,920 GPS coordinates via Open-Meteo Archive API using parallel requests (`ThreadPoolExecutor`, 5 workers)
- Implemented checkpoint saving every 500 rows for fault tolerance — recovered mid-run from internet outages without data loss
- Identified and resolved a genuine data gap: 10,388 GPS points over West Africa (lat 12–16°N, lon 16–17°W) had no hourly data in the Open-Meteo archive — forward fill applied within each bird group as standard time-series imputation

**Cross-pipeline Debugging**
- Debugged GRU input specifications iteratively with teammate — confirmed feature set (5 columns), sequence length (10), column ordering, and date_time encoding strategy (3 separate temporal features vs single timestamp)
- Identified mismatch between initial sequence spec (seq_length=30, 8 features) and final confirmed spec (seq_length=10, 5 features) — prevented downstream model errors
- Will debug inverse transform output once GRU predictions are received — real-world coordinate reconstruction using saved scalers

**Sequence Creation**
- Built sliding window sequences: input shape `(n, 10, 5)`, target shape `(n, 2)` — 80/20 train/test split, no shuffle (time-series order preserved)
- Output: 12 `.npy` files + `scalers.pkl` handed off to teammate for GRU training

---

## Repository Structure

```
bird-migration-gru/
│
├── jayant_bird_migration_github.ipynb   ← Jayant's preprocessing pipeline
├── bird_migration.csv                   ← Raw GPS dataset
├── bird_migration_with_climate.csv      ← Cleaned dataset with climate columns
├── scalers.pkl                          ← Fitted MinMaxScaler per bird
├── X_train_Eric.npy                     ← GRU input sequences
├── X_test_Eric.npy
├── y_train_Eric.npy
├── y_test_Eric.npy
├── X_train_Nico.npy
├── X_test_Nico.npy
├── y_train_Nico.npy
├── y_test_Nico.npy
├── X_train_Sanne.npy
├── X_test_Sanne.npy
├── y_train_Sanne.npy
├── y_test_Sanne.npy
└── README.md
```

---

## Sequence Specifications

| Parameter | Value |
|-----------|-------|
| GRU features | latitude, longitude, unix_timestamp, day_of_year, hour |
| Sequence length | 10 timesteps |
| Target | Next latitude, longitude |
| Input shape | (n, 10, 5) |
| Output shape | (n, 2) |
| Train/test split | 80/20, no shuffle |

---

## Climate Data Notes

Historical weather fetched per GPS coordinate using the Open-Meteo Archive API. 10,388 GPS points over West Africa (lat 12–16°N, lon 16–17°W) had no hourly data available in the archive. Forward fill applied within each bird group as standard time-series imputation.

---

## Tech Stack

- Python 3.10
- pandas, numpy
- scikit-learn (MinMaxScaler)
- TensorFlow/Keras (GRU)
- HDBSCAN
- Folium
- Open-Meteo Archive API
- Google Gemini API

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/<your-username>/bird-migration-gru.git
cd bird-migration-gru
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn requests folium hdbscan tensorflow
```

3. Run Jayant's preprocessing notebook
```
jayant_bird_migration_github.ipynb
```
Note: Climate fetch section is already executed. `bird_migration_with_climate.csv` is pre-generated.

4. Run Jasvanth's modeling notebook *(coming soon)*
