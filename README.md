# 📦 LogistiX AI — Enterprise Warehouse Management & Demand Intelligence System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Interactive Plots](https://img.shields.io/badge/Plotly-5.15%2B-purple.svg)](https://plotly.com/)

An end-to-end, enterprise-grade **Warehouse Management System (WMS)** and **Demand Forecasting Intelligence Platform** built with **Python**, **Scikit-Learn**, and **Streamlit**. Powered by high-velocity machine learning algorithms, it predicts warehouse demand, determines optimal Safety Stock and Reorder Points (ROP), conducts Pareto (80/20) ABC inventory classification, and audits stockout risks in real time.

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [Dataset Architecture & Key Stats](#-dataset-architecture--key-stats)
- [Machine Learning Models & Performance](#-machine-learning-models--performance)
- [Mathematical Inventory Optimization Formulas](#-mathematical-inventory-optimization-formulas)
- [Application Features](#-application-features)
- [Project Directory Structure](#-project-directory-structure)
- [Streamlit Cloud Deployment Guide](#-streamlit-cloud-deployment-guide)
- [Local Setup & Execution](#-local-setup--execution)

---

## 🚀 Executive Overview

Traditional warehouse and supply chain management relies on static min/max levels that fail to adapt to seasonality, supplier volatility, and retail store transfer spikes. **LogistiX AI** transforms warehouse operations into a data-driven, proactive supply chain by combining:
1. **Gradient-Boosted SKU Demand Forecasting** to anticipate monthly outbound warehouse demand.
2. **Dynamic Safety Stock & Reorder Point (ROP) Calculation** adapted to supplier lead times and desired service levels (90% to 99%).
3. **Automated Inventory Velocity Classification (ABC/FSN Analysis)** to distinguish high-velocity Fast-Movers (Class A) from dead stock (Class C).
4. **Batch Stockout Auditing** allowing warehouse managers to upload raw CSV files and generate instant replenishment orders.

---

## 📊 Dataset Architecture & Key Stats

Trained and validated on Montgomery County Department of Liquor Control's official transactional warehouse records:

| Metric | Benchmark Value |
| :--- | :--- |
| **Total Transaction Records** | **307,645** |
| **Active Catalog SKUs** | **34,056 items** |
| **Monitored Suppliers & Distributors** | **397 companies** |
| **Total Warehouse Sales Throughput** | **7,925,509 cases/units** |
| **Total Retail Direct Sales** | **2,160,943 cases/units** |
| **Total Retail Transfers Dispatched** | **2,134,722 cases/units** |
| **Time Period Coverage** | **2017 – 2020** |
| **Product Categories** | Wine, Liquor, Beer, Kegs, Non-Alcohol, Supplies, Ref |

---

## 🤖 Machine Learning Models & Performance

Two supervised models were engineered, cross-validated, and deployed:

### 1. Demand Forecaster (`HistGradientBoostingRegressor`)
Predicts monthly warehouse dispatch volume for any item using log-transformed demand with cyclical calendar features, supplier throughput statistics, and transfer ratios.

- **Log-Scale $R^2$ Score**: **0.7093** (explains >70% of variance in log volume)
- **Log-Scale Mean Absolute Error (MAE)**: **0.5373**
- **Original-Scale Median Absolute Error (MedAE)**: **1.05 units**
- **Original-Scale Mean Absolute Error (MAE)**: **17.35 units**
- **Training Samples**: 246,116 | **Test Samples**: 61,529 (80/20 stratified split)

### 2. Stock Velocity Classifier (`HistGradientBoostingClassifier`)
Categorizes inventory into ABC velocity tiers:
- **Class A (Fast-Moving)**: $\ge 20$ units/month (High priority replenishment)
- **Class B (Moderate)**: $2 - 20$ units/month (Standard inventory cycle)
- **Class C (Slow/Tail)**: $< 2$ units/month (Low turnover / rationalization candidate)

#### Benchmark Classification Metrics:
- **Overall Multiclass Accuracy**: **71.14%**
- **Weighted Precision**: **0.7159**
- **Weighted Recall**: **0.7114**
- **Weighted F1-Score**: **0.7125**
- **Class A (Fast-Moving) Precision**: **84.34%** (precision on top revenue drivers)

#### Confusion Matrix (Test Split $N=61,529$):
```
Actual \ Predicted       Class A (Fast)    Class B (Moderate)   Class C (Slow)
-------------------------------------------------------------------------------
Class A (Fast)               4,399                 1,784               355
Class B (Moderate)             754                14,236             7,360
Class C (Slow)                  63                 7,441            25,137
```

---

## 🧮 Mathematical Inventory Optimization Formulas

LogistiX AI dynamically calculates warehouse replenishment thresholds:

### 1. Safety Stock ($SS$)
Buffers against demand surges and supplier delivery variability:
$$SS = Z \times \sigma_{\text{demand}} \times \sqrt{\frac{L}{30}}$$
- $Z$: Service factor ($Z=1.65$ for 95%, $Z=2.33$ for 99%)
- $\sigma_{\text{demand}}$: Estimated standard deviation of monthly demand
- $L$: Supplier lead time in days

### 2. Reorder Point ($ROP$)
The inventory level that triggers an automated purchase order:
$$ROP = \left(D_{\text{monthly}} \times \frac{L}{30}\right) + SS$$

### 3. Economic Order Quantity ($EOQ$)
The optimal batch order quantity minimizing holding and ordering costs:
$$EOQ = \sqrt{\frac{2 \times D_{\text{annual}} \times S}{H}}$$
- $D_{\text{annual}}$: Projected annual demand ($12 \times D_{\text{monthly}}$)
- $S$: Purchase order administration cost (default: \$50)
- $H$: Annual holding/carrying cost per unit (default: \$5)

---

## 🖥️ Application Features

1. **📊 Executive Analytics Dashboard**: Real-time KPI summary, monthly demand/transfer trends, category distributions, and top supplier rankings.
2. **🤖 AI Demand & Replenishment Engine**: Interactive single-SKU forecasting simulator with real-time Safety Stock, ROP, EOQ, and visual stock gauges.
3. **📁 Batch Stockout Audit**: One-click scoring of entire inventory CSV spreadsheets with downloadable audit reports.
4. **📦 ABC Pareto Velocity Matrix**: Visual 80/20 Pareto curve and searchable catalog classified into Class A, B, and C items.
5. **🚚 Supplier Assortment Analytics**: Distributor throughput comparison, SKU concentration, and volume share.
6. **📈 Model Stats & Governance**: Transparent view of $R^2$, MAE, Confusion Matrix Heatmap, and hyperparameters.
7. **🚀 Cloud Deployment Guide**: In-app instructions for instant GitHub and Streamlit Cloud hosting.

---

## 📂 Project Directory Structure

```
warehouse/
├── .streamlit/
│   └── config.toml               # Modern dark theme and server configurations
├── models/
│   ├── demand_forecaster.pkl     # Trained Regressor Pipeline
│   ├── velocity_classifier.pkl   # Trained Classifier Pipeline
│   ├── metrics.json              # Full benchmark metrics (R2, MAE, Accuracy, Confusion Matrix)
│   └── metadata.json             # Feature names, supplier lookup statistics, categories
├── data/
│   ├── Warehouse_and_Retail_Sales.csv   # Raw historical warehouse dataset (307k rows)
│   └── sample_inventory.csv      # Sample batch upload template (25 SKUs)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Cached memory-optimized data loading & aggregation
│   ├── ml_pipeline.py            # Inference engine, ROP, Safety Stock, EOQ calculations
│   └── visualizations.py         # Plotly charts (Timelines, Donut, Heatmap, Gauges)
├── train_model.py                # Standalone model training and evaluation script
├── app.py                        # Streamlit web application
├── requirements.txt              # Production Python package requirements
├── .gitignore                    # Git ignore file for temp files and large archives
└── README.md                     # Documentation and deployment guide
```

---

## ☁️ Streamlit Cloud Deployment Guide

Deploying this app to **Streamlit Community Cloud** is 100% free and takes under 2 minutes:

### Step 1: Push Repository to GitHub
```bash
git init
git add .
git commit -m "feat: complete warehouse management system with ML models and Streamlit app"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/warehouse-management-system.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Navigate to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository: `<YOUR_USERNAME>/warehouse-management-system`.
4. Set Main file path: `app.py`.
5. Click **"Deploy!"**.

---

## 💻 Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/warehouse-management-system.git
cd warehouse-management-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain Models
```bash
python train_model.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
