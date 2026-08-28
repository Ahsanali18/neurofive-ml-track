# 🇵🇰 GariQeemat: Smart Used Car Price Estimator & Market Valuation Tool

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](gariqeemat.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GariQeemat** is an advanced machine learning-powered web application designed to predict and analyze used car prices in Pakistan. Built for the **Neurofive ML Track**, it scrapes real-world automotive market data (from PakWheels), engineers robust pricing features, and evaluates vehicle asking prices against predicted fair market values with localized insights.

---

## 🚀 Live Demo
Experience the live application here: **[GariQeemat on Streamlit Cloud](YOUR_STREAMLIT_APP_URL_HERE)**

---

## ✨ Key Features

* **🤖 Advanced ML Pricing Engine:** Powered by an optimized XGBoost regression model trained on Pakistani automotive listings.
* **📊 Interactive Analytics & Gauge Charts:** Real-time visual comparison showing your target car's **Asking Price** vs. **Market Median** vs. **ML Fair Price**, complete with a Deal Quality Gauge.
* **💡 Localized Market Insights:** Generates context-aware smart alerts tailored to the Pakistani market (e.g., brand liquidity, Karachi vs. Punjab market pricing variance, and high mileage impact).
* **💰 Automated PKR Formatting:** Clean financial readouts automatically converting large numbers into traditional Pakistani **Lac** and **Crore** notations.
* **🎨 Modern Responsive UI:** Built with custom CSS styling, metric flex-cards, and dynamic verdict banners for an intuitive user experience.

---

## 🛠️ Project Architecture & Tech Stack

* **Frontend & UI:** [Streamlit](https://streamlit.io/)
* **Data Visualization:** [Plotly](https://plotly.com/)
* **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
* **Data Processing & Serialization:** Pandas, NumPy, Joblib
* **Data Source:** PakWheels (Scraped & Preprocessed)

---

## 📂 Project Directory Structure

```text
neurofive-ml-track/
│
├── .gitignore
├── requirements.txt
└── GariQeemat/
    ├── dashboard/
    │   └── app.py                  # Main Streamlit web application
    ├── data/
    │   └── processed/
    │       └── fair_price_lookup.csv # Market baseline lookup data
    ├── models/
    │   ├── best_model.pkl          # Trained XGBoost regression model
    │   ├── scaler.pkl              # Numerical feature standard scaler
    │   ├── feature_columns.json    # Saved one-hot encoded columns
    │   ├── rare_brands.json        # Brand grouping classifications
    │   └── model_freq_lookup.json  # Model frequency encoding mapping
    ├── notebooks/                  # EDA & Model evaluation notebooks
    └── src/                        # Feature engineering & training scripts
