# Price Optimizer & Recommender

A proof-of-concept **Price Optimization Dashboard** for laboratory equipment, demonstrating how machine learning can recommend optimal selling prices to maximize profit — factoring in product type, customer segment, market conditions, competitive dynamics, and seasonality.

> **Note:** This is a demo built on mock models and synthetic data. The simulated prediction logic lives alongside the Streamlit frontend in `app.py`. Replacing the mock functions with a trained ML model and preprocessor pipeline is all that's needed to move to production.

---

## Dashboard Preview

![Single Product Optimization](D1.png)
![Batch Optimization & Scenario Comparison](D2.png)

---

## Features

**Three optimization modes** (accessible from the sidebar):

- **Single Product Optimization** — compute the optimal price for a given scenario, visualize the price-profit curve, and receive actionable recommendations.
- **Batch Optimization** — run price optimization across multiple products simultaneously and export results as a CSV.
- **Scenario Comparison** — compare optimal pricing outcomes across predefined market scenarios side by side.

**Under the hood:**
- A mocked prediction engine that simulates profit using heuristics, so the dashboard runs without any serialized model files.
- Feature engineering utilities mirroring training-time preprocessing (price ratios, season flags, encoded segments, etc.).
- Jupyter notebooks for EDA, feature engineering experiments, and model development.
- A synthetic data generator script to fabricate lab equipment pricing datasets.

---

## Getting Started

### Prerequisites

- Python 3.10+ (the dev container runs Ubuntu 24.04 with Python 3.12)
```bash
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) and explore the three modes from the sidebar.

### Developing & Re-training

1. Run `Data-Generator.py` to produce synthetic datasets for experimenting with new features or models.
2. Use the Jupyter notebooks for step-by-step analysis and training workflows:
   - `EDA.ipynb` — exploratory data analysis
   - `Feature-Engineering.ipynb` — feature engineering experiments
   - `Modelling.ipynb` — model development and evaluation
3. When a real model is ready, export it as a `.pkl` (along with the scaler), then update `app.py` to load them in place of the mocks.

---

## Repository Structure
```
app.py                              # Streamlit dashboard (entry point)
Data-Generator.py                   # Synthetic lab-equipment pricing data generator
EDA.ipynb                           # Exploratory Data Analysis
Feature-Engineering.ipynb           # Feature engineering experiments
Modelling.ipynb                     # Model development and evaluation
requirements.txt                    # Python dependencies
model_metadata.json                 # Metadata template used by the app
feature_metadata.json               # Feature order template used by the app
lab_equipment_pricing.csv           # Sample dataset
lab_equipment_pricing_features.csv  # Feature-engineered sample data
price_optimization_model.pkl        # (placeholder) trained model
feature_scaler*.pkl                 # (placeholders) scaling objects
```

---

For questions or improvements, feel free to open an issue or pull request.
