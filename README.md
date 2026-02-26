# Price Optimizer and Recommender

This repository contains a proof-of-concept **Price Optimization Dashboard** for laboratory equipment. The goal is to demonstrate how machine learning (or, in this demo, a simulated model) can be used to recommend optimal selling prices that maximize profit based on product, customer segment, market conditions, competitive dynamics, and seasonality.

> 💡 This is a demo app that uses mock models and synthetic data. The underlying file `app.py` contains both the Streamlit frontend and simulated feature engineering / prediction logic. The real implementation would replace the mock functions with a trained ML model and preprocessor pipelines.

---

## 🚀 Features

- **Interactive Streamlit app** (`app.py`) with three modes:
  - **Single Product Optimization** – compute optimal price for a given scenario, visualize price-profit curve, and get actionable recommendations.
  - **Batch Optimization** – run price optimization over multiple products simultaneously and download results as CSV.
  - **Scenario Comparison** – compare optimal pricing outcomes across predefined market scenarios.
- **Mocked prediction engine** that simulates profit using heuristics, allowing the dashboard to run without serialized model files.
- **Feature engineering utilities** that mirror training-time preprocessing (features such as price ratios, season flags, encoded segments, etc.).
- Utility notebooks for exploratory data analysis (`EDA.ipynb`), feature engineering experiments (`Feature-Engineering.ipynb`), and model development (`Modelling.ipynb`).
- A simple `Data-Generator.py` script to fabricate synthetic lab equipment pricing datasets.

---

## 🧠 Getting Started

### Prerequisites

- Python 3.10+ (the dev container is running Ubuntu 24.04 with Python 3.12).
- Install dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the App

1. Start the Streamlit server from the workspace root:

   ```bash
   streamlit run app.py
   ```

2. Open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

3. Explore the three app modes from the sidebar.

### Developing & Re-training

- Use `Data-Generator.py` to produce synthetic datasets if you want to experiment with new features or models.
- Open the Jupyter notebooks (`EDA.ipynb`, `Feature-Engineering.ipynb`, `Modelling.ipynb`) for step-by-step analysis and model training workflows.
- When a real model is ready, export it as a `.pkl` along with the scaler and update `app.py` to load them instead of using the mocks.

---

## 🗂️ Repository Structure

```
app.py                     # Streamlit dashboard (entry point)
Data-Generator.py          # Script to generate synthetic lab-equipment pricing data
EDA.ipynb                  # Exploratory Data Analysis notebook
Feature-Engineering.ipynb  # Feature engineering experiments
Modelling.ipynb            # Model development and evaluation notebook
requirements.txt           # Python dependencies
model_metadata.json        # Metadata template used in the app
feature_metadata.json      # Feature order template used in the app
lab_equipment_pricing.csv  # Sample dataset
lab_equipment_pricing_features.csv  # Feature-engineered version of sample data
price_optimization_model.pkl  # (placeholder) trained model file
feature_scaler*.pkl        # (placeholders) scaling objects
```

---

## 📝 Notes

- The dashboard currently runs in **simulated mode**, warning users when model files are absent. Replace the mocks with a real model pipeline for production use.
- The feature list and product configuration are hard-coded for demonstration; real applications should load these from a configuration store or database.
- Profit optimization uses a simple grid search over a price range; more advanced techniques (convex optimization, gradient-based methods) can be integrated as needed.

---

## 📄 License

This project is provided for educational/demonstration purposes. No license is specified.

---

For questions or enhancements, feel free to open issues or pull requests in the repository.
