# 🚦 Traffic Volume Prediction System

An end-to-end Machine Learning web application that predicts hourly urban **traffic volume (vehicles per hour)** based on historical traffic logs, calendar factors, and atmospheric weather metrics.

---

## 👨‍💻 Developer Information

* **Developer Name:** Swapna V
* **Role:** ML Engineer
* **Company:** IPEC Solutions
* **Project Name:** Traffic Volume Prediction System
* **Primary Objective:** Real-time urban traffic modeling, congestion forecasting, and smart city traffic management optimization.

---

## 📌 Project Overview

Urban traffic congestion creates severe economic loss, increased fuel consumption, and higher carbon emissions. The **Traffic Volume Prediction System** uses supervised Machine Learning regression models to forecast traffic volume hours in advance based on time-of-day, day-of-week, seasonal variations, temperature, rainfall, snow, and weather severity.

### Key Capabilities
* **Dataset Profiling & EDA**: Interactive visual breakdown of hourly rush-hour curves, weekday/weekend traffic differentials, and precipitation impacts.
* **Multi-Model Comparison**: Evaluates Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor side-by-side.
* **Automated Model Selection**: Automatically identifies the best model based on R² Score and Root Mean Squared Error (RMSE).
* **Interactive Web Dashboard**: Streamlit-based web application with customizable prediction forms, road capacity utilization gauges, and 4-level traffic classification.

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.10
* **Data Processing & Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly Express, Plotly Graph Objects, Matplotlib
* **Machine Learning & Pipeline:** Scikit-Learn
* **Model Serialization:** Joblib
* **Web Application Framework:** Streamlit

---

## 📁 Project Directory Structure

```text
traffic-volume-prediction/
│
├── app.py                      # Main Streamlit web application
├── train_model.py              # ML training & model selection script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & deployment guide
├── .gitignore                  # Git ignore rules
├── .streamlit/
│   └── config.toml             # Streamlit server theme configuration
│
├── data/
│   ├── generate_dataset.py     # Dataset generator script
│   └── traffic_data.csv        # Hourly traffic dataset (8,760 records)
│
├── models/
│   ├── traffic_model.pkl       # Serialized best ML model (Gradient Boosting)
│   ├── preprocessing_pipeline.pkl # Preprocessing ColumnTransformer
│   └── model_metrics.json      # Evaluation metrics & feature importances
│
├── notebooks/
│   └── traffic_analysis.ipynb  # Jupyter notebook for exploratory analysis
│
└── images/
    └── dashboard.png           # Dashboard preview artifact
```

---

## 📊 Dataset Description

The dataset contains 8,760 hourly records (1 full year) with the following features:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `date` | Datetime | Timestamp of observation (`YYYY-MM-DD HH:MM:SS`) |
| `hour` | Integer | Hour of the day (0 – 23) |
| `holiday` | Categorical | National holiday indicator (e.g., Thanksgiving, Christmas, No Holiday) |
| `temperature` | Float | Temperature in degrees Celsius (°C) |
| `rain_1h` | Float | Rain amount in mm during the past hour |
| `snow_1h` | Float | Snow amount in mm during the past hour |
| `clouds_all` | Integer | Cloud cover percentage (0% to 100%) |
| `weather_main` | Categorical | Weather condition category (Clear, Clouds, Rain, Snow, Mist, Fog, Drizzle, Thunderstorm) |
| `weather_description` | Categorical | Detailed weather text description |
| **`traffic_volume`** | **Integer** | **Target Variable: Number of vehicles per hour** |

> *Note: A realistic 8,760-row hourly dataset is included in `data/traffic_data.csv` for development and reproducible training.*

---

## 🔄 Machine Learning Workflow

```text
Dataset (8,760 records)
        ↓
Data Cleaning & Missing Value Imputation
        ↓
Feature Engineering (Extract Year, Month, Day, Hour, Weekend Flag)
        ↓
Chronological Train/Test Split (80% Train, 20% Test)
        ↓
Preprocessing Pipeline (StandardScaler for Numeric, OneHotEncoder for Categoricals)
        ↓
Model Training (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
        ↓
Model Evaluation & Metric Leaderboard (MAE, MSE, RMSE, R²)
        ↓
Model Serialization (Joblib Artifacts)
        ↓
Streamlit Dashboard Inference & Traffic Level Classification
```

---

## 📈 Model Performance & Results

Models evaluated on the out-of-sample 20% test dataset (1,752 hourly records):

| Model | MAE (veh/h) | RMSE (veh/h) | R² Score | Performance |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** 🏆 | **162.52** | **217.79** | **0.9821** | **Best Accuracy (98.2%)** |
| Random Forest Regressor | 167.67 | 231.47 | 0.9797 | High Performance |
| Tuned Random Forest | 169.88 | 233.94 | 0.9793 | High Performance |
| Decision Tree Regressor | 204.65 | 284.92 | 0.9693 | Good Performance |
| Linear Regression | 1492.21 | 1870.80 | -0.3231 | Poor (Non-linear relationships) |

**Winner:** `Gradient Boosting Regressor` achieved the lowest Root Mean Squared Error (**217.79 vehicles/hour**) and an **R² Score of 0.9821**.

---

## 🚦 Traffic Volume Classification

Predictions are automatically classified into four operational traffic levels:

* 🟢 **Low Traffic** (`< 1,500 vehicles/hour`): Smooth traffic flow. Minimal delays expected.
* 🟡 **Moderate Traffic** (`1,500 – 3,500 vehicles/hour`): Normal urban traffic density.
* 🟠 **High Traffic** (`3,500 – 5,200 vehicles/hour`): Heavy congestion. Expect moderate travel delays.
* 🔴 **Very High Traffic** (`> 5,200 vehicles/hour`): Severe congestion / Peak rush hour. High risk of delays.

---

## ⚡ Quick Start & Setup Guide

### 1. Clone the Repository
```bash
git clone <repository-url>
cd traffic-volume-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Models
```bash
python data/generate_dataset.py
python train_model.py
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` or access the cloud preview.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

To deploy this project for free on **Streamlit Community Cloud**:

1. **GitHub Upload**: Push this repository to GitHub.
2. **Streamlit Account**: Log in to [share.streamlit.io](https://share.streamlit.io).
3. **New App**: Click **"New app"** and select your GitHub repository.
4. **Branch & File**: Select branch `main` and set `Main file path` to `app.py`.
5. **Deploy**: Click **"Deploy!"**.

**Live Demo URL:** `[Add deployed Streamlit URL here]`

---

## 🚀 Future Enhancements

* **Real-time Traffic API Integration**: Ingest live TomTom or Google Maps Traffic API feeds.
* **Deep Learning Time-Series Forecasting**: Implement LSTM / GRU neural network architectures for multi-step ahead forecasting.
* **GPS & Route Guidance Integration**: Recommend alternate green routes based on predicted congestion bottlenecks.

---

## 👨‍💻 Author

* **Name:** Swapna V
* **Role:** ML Engineer
* **Company:** IPEC Solutions
* **Project:** Traffic Volume Prediction System
