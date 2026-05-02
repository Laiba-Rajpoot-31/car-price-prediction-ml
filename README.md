# 🚗 Car Price Prediction — Machine Learning Web App

> **CodeAlpha Data Science Internship — Task 3**

A fully interactive web application that predicts the **resale price of used cars**
using a Gradient Boosting ML model trained on real Indian market data.

---

## 🎯 Project Overview

This project covers the complete data science workflow — from raw data to a
live, deployable web application built with Streamlit.

---

## ✨ Features

- 🔮 **Instant Price Prediction** with ±8% confidence range
- 📊 **Model Analysis Dashboard** — Predicted vs Actual, Residuals, Feature Importance
- 🧠 **How It Works Tab** — full pipeline explanation inside the app
- 📉 **Depreciation Gauge** — shows how much value the car has lost
- 🏷️ **Brand Goodwill Engine** — auto-fetches average market price per brand

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data wrangling |
| Scikit-learn | ML models & evaluation |
| Matplotlib & Seaborn | Visualizations |
| Streamlit | Web app deployment |

---

## 🧪 Model Performance

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.8878 | ₹1.07L | ₹1.61L |
| Random Forest | 0.9825 | ₹0.39L | ₹0.64L |
| **Gradient Boosting** ✅ | **0.9850** | **₹0.31L** | **₹0.59L** |

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `Car_Age` | Years since manufacture (2024 − Year) |
| `Brand_Goodwill` | Average market price per brand |
| `KM_per_Year` | Annual mileage — usage intensity proxy |
| `Price_Drop_Pct` | Depreciation as % of present price |
| `Price_Drop` | Absolute depreciation in Lakhs |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/car-price-prediction-ml.git
cd car-price-prediction-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📂 Project Structure
