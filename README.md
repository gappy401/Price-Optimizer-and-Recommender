# Price Optimization Dashboard — Laboratory Equipment

An end-to-end pricing analytics application that models optimal sell-price recommendations for laboratory equipment across diverse customer segments, market conditions, and competitive landscapes. The project covers the full workflow: raw data generation, exploratory analysis, feature engineering, model development, and an interactive decision-support dashboard, simulating data-driven pricing infrastructure used to build and maintain pricing guidelines at scale.

> **Note:** Built on mock models and synthetic data. The simulated prediction logic lives alongside the Streamlit frontend in `app.py`. Replacing the mock functions with a trained ML model and preprocessor pipeline is all that's needed to move to production.

---

## Dashboard Preview

![Single Product Optimization](D1.jpg)

---

## What It Does

Pricing decisions in scientific equipment markets are rarely straightforward. List price, customer segment, competitive position, and seasonal demand all interact in ways that simple margin targets miss. This dashboard operationalizes those dynamics:

- **Demand & margin modeling** — quantifies how price elasticity, segment sensitivity, and 
 competitive index jointly drive profitability, providing a data-driven basis for pricing 
  decisions rather than intuition.
- **Scenario analysis** — enables side-by-side comparison of pricing outcomes across market 
  conditions, supporting the kind of ad-hoc analysis that informs commercial and 
  cross-functional pricing discussions.
- **Batch optimization** — runs recommended price points across a product portfolio 
  simultaneously, producing outputs ready for review or export to downstream reporting.
- **Transparent feature logic** — every input variable (price ratio, seasonality flag, 
  segment encoding, competitive positioning) is documented and reproducible, consistent with 
  how pricing models are handed off to commercial teams.

---

## Features

**Three optimization modes** (accessible from the sidebar):

- **Single Product Optimization** — compute the optimal price for a given scenario, 
  visualize the price-profit curve, and receive actionable recommendations.
- **Batch Optimization** — run price optimization across multiple products simultaneously 
  and export results as CSV.
- **Scenario Comparison** — compare optimal pricing outcomes across predefined market 
  scenarios side by side.

**Under the hood:**

- A mocked prediction engine that simulates profit using heuristics, so the dashboard runs 
  without any serialized model files.
- Feature engineering utilities mirroring training-time preprocessing (price ratios, season 
  flags, encoded segments, etc.).
- Jupyter notebooks covering EDA, feature engineering experiments, and model development.
- A synthetic data generator to fabricate realistic lab equipment pricing datasets.

![Batch Optimization & Scenario Comparison](D2.jpg)

---

## Getting Started

### Prerequisites

- Python 3.10+ 
```bash
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```

Then open the URL shown in the terminal.

### Developing & Re-training
1. Use the Jupyter notebooks for step-by-step analysis and training workflows:
   - `EDA.ipynb` — exploratory data analysis
   - `Feature-Engineering.ipynb` — feature engineering experiments
   - `Modelling.ipynb` — model development and evaluation
2. When a real model is ready, export it as a `.pkl` (along with the scaler), then update `app.py` to load them in place of the mocks.

---

## Repository Structure
```
app.py                               # Streamlit dashboard (entry point)
requirements.txt                     # Python dependencies

# raw and processed data samples
lab_equipment_pricing.csv            # sample pricing dataset used in notebooks

# metadata files used by the app and preprocessing
Model/model_metadata.json            # metadata template consumed by `app.py`
"Feature Engineering"/feature_metadata.json  # feature order template for preprocessing

# project subfolders with supporting notebooks and artifacts
"Feature Engineering"/              # feature engineering experiments & outputs
   EDA.ipynb                        # exploratory data analysis notebook
   Feature-Engineering.ipynb        # feature creation and transformation steps
   lab_equipment_pricing_features.csv  # feature‑engineered sample data

Model/                                # model development and evaluation
   Modelling.ipynb                   # model training notebook
   (placeholders)                   # trained model (`.pkl`) and scaler files live here in practice
```

---

For questions or improvements, feel free to open an issue or pull request.
