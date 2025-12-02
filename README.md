# MatrixLab AI Studio

> **MatrixLab AI Studio** — A full-featured, Streamlit-based Machine Learning & Data Science playground for education, prototyping and light production use. Built to be modular, user-friendly and feature-rich: data upload, EDA, visualization, model training, explainability, predictions, model management and deployment.

Live demo: [https://matrixlab-ai.streamlit.app](https://matrixlab-ai.streamlit.app)

---

## Table of contents

1. [About](#about)
2. [Key features](#key-features)
3. [Tech stack](#tech-stack)
4. [Project structure](#project-structure)
5. [Installation & quick start](#installation--quick-start)
6. [Usage guide](#usage-guide)
7. [Advanced features](#advanced-features)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [Roadmap](#roadmap)
11. [License](#license)

---

## About

MatrixLab AI Studio is an educational and prototyping platform that brings together data ingestion, cleaning, visualization, model training, explainability and prediction in a single Streamlit app. It’s designed for students and practitioners who want a clean, interactive environment for building and evaluating machine learning models quickly.

The app focuses on usability and includes components inspired by Orange Data Mining and features found in ML platforms like WEKA, but built with modern Python tooling and a Streamlit UI.

---

## Key features

* **Data upload & preview** — CSV / Excel / Parquet support, quick preview and schema inspection.
* **Exploratory Data Analysis (EDA)** — Summary statistics, missing-value reports, correlation heatmaps, distribution plots and interactive filters.
* **Visualization** — Built-in plotting for numeric/categorical variables, pairplots, and customizable charts.
* **Model training & comparison** — Train multiple models (Logistic Regression, Decision Trees, Random Forest, XGBoost, LightGBM, SVM, KNN, etc.), compare using cross-validation and holdout metrics.
* **Explainability** — SHAP explainability support to inspect feature importance and per-prediction attributions.
* **Real-time prediction UI** — Upload new datasets or enter single records and get predictions.
* **Model management** — Save, load, version and delete trained models from disk; basic metadata and performance logging.
* **Authentication & multi-user (prototype)** — Lightweight auth flow for controlled access (can be extended for production).
* **Dark mode & responsive UI** — Toggle-friendly interface adaptable to different screen sizes.
* **MatrixLab brand** — Custom logo and theme for a polished presentation.

---

## Tech stack

* **Frontend / app**: Streamlit
* **Core libraries**: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matplotlib, plotly
* **Optional**: Streamlit-Authenticator or custom auth, joblib (model persistence)
* **Deployment**: Streamlit Community Cloud, Replit (for quick demo), or Docker for containerized deployments

---

## Project structure (example)

```
matrix-lab-ai/
├─ .github/
├─ assets/
│  ├─ logo.svg
│  └─ screenshots/
├─ data/
├─ models/
├─ pages/
│  ├─ Homepage.py
│  ├─ EDA.py
│  ├─ Train.py
│  ├─ Predict.py
│  └─ Manage.py
├─ src/
│  ├─ utils.py
│  ├─ data_processing.py
│  ├─ modeling.py
│  └─ explainability.py
├─ requirements.txt
├─ README.md
└─ .streamlit/
   └─ config.toml
```

---

## Installation & quick start

**Prerequisites**

* Python 3.9+ recommended
* Git

**Clone repository**

```bash
git clone https://github.com/MShriK17/matrix-lab-ai.git
cd matrix-lab-ai
```

**Create virtual environment & install**

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

**Run locally**

```bash
streamlit run Homepage.py
```

Open `http://localhost:8501` (Streamlit will print the exact URL).

---

## Usage guide

### Navigation

The app uses separate pages/tabs for common workflows:

* **Homepage** — Project overview & sample datasets
* **EDA** — Data cleaning, visualization and interactive exploration
* **Train** — Select target, features, choose model(s), train and evaluate
* **Predict** — Upload new data or type records for prediction
* **Manage** — Save/load models and view model metadata

### Typical workflow

1. Upload dataset (CSV / Excel).
2. Inspect summary statistics and plots in EDA.
3. Preprocess using built-in transformers (impute, encode, scale).
4. Train one or more models in Train page; compare metrics and confusion matrices.
5. Use SHAP to examine feature importance and local explanations.
6. Save a trained model and go to Predict to run new data through it.

---

## Advanced features

* **SHAP explainability**: Per-model SHAP plots and per-sample force plots for transparent model decisions.
* **Real-time emotion detection (research prototype)**: Modules for combining text, video and audio streams into emotion predictions (research code in `src/emotion_detection/`).
* **Model metadata & simple audit**: Each saved model stores training date, dataset hash, hyperparameters and key metrics for reproducibility.
* **Export**: Download predictions and model reports as CSV or PDF summaries.

---

## Deployment

### Streamlit Community Cloud

1. Push repo to GitHub.
2. Sign in to Streamlit Cloud and connect your GitHub repository.
3. Set the main file to `Homepage.py` (or your app entrypoint) and deploy.

### Replit / Colab + ngrok (for quick demos)

* A Colab notebook that starts the app using `ngrok` can be included in `/deploy` for quick temporary exposure.
* Not recommended for production due to ephemeral tunnels.

### Docker (recommended for stable deployments)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "Homepage.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

---

## Configuration & environment variables

Use `.streamlit/config.toml` for basic Streamlit settings. For secrets and credentials use Streamlit secrets or environment variables:

```
STREAMLIT_AUTHENTICATOR_SECRET=...
MODEL_STORE_PATH=./models
```

---

## Contributing

Contributions welcome! Suggested workflow:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Add tests or demo notebooks if relevant
4. Open a PR with a clear description of changes

Please follow the code style in the project and update `requirements.txt` for new dependencies.

---

## Roadmap

Planned improvements:

* Multi-user multi-tenant model store with per-user model isolation
* More model types (transformers for NLP, neural nets for vision)
* End-to-end CI/CD with automated testing & deployment
* Improved dataset versioning and lineage tracking

---

## Credits & inspiration

Inspired by Orange Data Mining, WEKA, and modern ML platforms. Built by the MatrixLab team.

---

## License

This project is released under the **MIT License** — see `LICENSE` for details.

---

