# 🏥 Hospital Readmission Analytics and Prediction

**Course:** Data Engineering and MLOps  
**Project:** Individual Assignment – Part 1 (Data Pipeline) + Part 2 (MLOps)  
**Dataset:** UCI Diabetes 130-US Hospitals Dataset  
**Disclaimer:** *This is an academic project. Predictions must NOT be used for clinical decision-making.*

---

## 1. Project Objective

Build a privacy-conscious healthcare data pipeline using the publicly available UCI Diabetes 130-US Hospitals dataset. The system:

- Ingests, cleans, validates, and transforms encounter-level hospital data
- Stores curated data in a PostgreSQL analytical data mart
- Orchestrates the ETL pipeline using Apache Airflow
- Provides interactive analytics via a Streamlit dashboard
- Trains and evaluates ML models for readmission risk prediction (Part 2)
- Tracks experiments with MLflow and serves predictions via FastAPI (Part 2)

---

## 2. Architecture

```
Public Dataset (UCI)
        │
        ▼
data/raw/            ← Raw CSV preserved unchanged
        │
        ▼
Ingestion (src/ingestion/ingest.py)
        │
        ▼
data/staging/        ← Lightly typed staging copy
        │
        ▼
Validation (src/validation/data_quality.py)
        │  ├── Rejected records → data/rejected/
        │  └── Validation log   → data/logs/
        ▼
Cleaning + Feature Engineering
(src/preprocessing/clean_data.py)
(src/transformation/feature_engineering.py)
        │
        ▼
data/curated/        ← Clean, analytical-ready CSV
        │
        ▼
PostgreSQL Healthcare Data Mart
  Schemas: raw | staging | curated | analytics
        │
        ├── analytics.dim_admission
        ├── analytics.dim_diagnosis
        ├── analytics.fact_patient_summary
        └── analytics.fact_readmission
        │
        ▼
Apache Airflow DAG (dags/hospital_readmission_etl.py)
  [extract → load_raw → validate_raw → transform
   → validate_cleaned → load_postgres
   → create_analytics_tables → pipeline_summary]
        │
        ▼
Streamlit Dashboard (dashboard/app.py)
  5+ Views: Readmission rates, Age groups, Diagnoses,
            Length-of-stay, High-risk cohorts
        │
        ▼  (Part 2)
ML Pipeline (ml/)
  Models: Logistic Regression | Random Forest | XGBoost
        │
        ▼
MLflow Experiment Tracking + Model Registry
        │
        ▼
FastAPI Prediction API (api/main.py)
  POST /predict  →  risk score + model version
  GET  /health   →  service status
        │
        ▼
Docker Container (Dockerfile.api + docker-compose.yml)
        │
        ▼
Monitoring + Retraining Design (ml/monitoring.py)
```

---

## 3. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data processing | Pandas, NumPy |
| Database | PostgreSQL 16 |
| ORM | SQLAlchemy 2.0 |
| Orchestration | Apache Airflow 2.9 |
| Dashboard | Streamlit + Plotly |
| ML | Scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |
| Monitoring | Evidently |
| Testing | pytest |

---

## 4. Dataset Information

| Field | Value |
|-------|-------|
| Name | UCI Diabetes 130-US Hospitals Dataset |
| Source | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 |
| Kaggle mirror | https://www.kaggle.com/datasets/brandao/diabetes |
| Rows (approx.) | 101,766 |
| Columns | 50 |
| Readmission label | `readmitted` (`<30`, `>30`, `NO`) |
| Privacy | De-identified — no PII |

### How to Download

1. Go to the Kaggle page: https://www.kaggle.com/datasets/brandao/diabetes
2. Click **Download** (requires free Kaggle account)
3. Extract the ZIP
4. Place `diabetic_data.csv` and `IDS_mapping.csv` into:

```
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

> **Important:** The ingestion script will stop with a clear error message if the file is missing. Do NOT place any other data in this folder.

---

## 5. Folder Structure

```
hospital-readmission-mlops/
│
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── .env.example                 ← Environment variable template
├── .gitignore
├── docker-compose.yml           ← All services
├── Dockerfile.api               ← FastAPI container
├── Dockerfile.dashboard         ← Streamlit container
│
├── config/
│   └── config.yaml              ← Central configuration
│
├── data/
│   ├── raw/                     ← Original CSV (never modified)
│   ├── staging/                 ← Lightly parsed copy
│   ├── processed/               ← Intermediate cleaned data
│   ├── curated/                 ← Final analytical dataset
│   ├── rejected/                ← Records that failed validation
│   └── logs/                    ← Ingestion and validation logs
│
├── dags/
│   └── hospital_readmission_etl.py  ← Airflow DAG
│
├── src/
│   ├── ingestion/ingest.py      ← CSV ingestion + metadata
│   ├── preprocessing/clean_data.py  ← Cleaning pipeline
│   ├── validation/data_quality.py   ← Quality checks
│   ├── transformation/feature_engineering.py
│   ├── database/
│   │   ├── connection.py        ← SQLAlchemy engine
│   │   └── schema.sql           ← DB schema DDL
│   ├── analytics/analytical_queries.sql
│   └── utils/logger.py          ← Shared logging
│
├── ml/
│   ├── preprocessing.py         ← ML feature pipeline
│   ├── train.py                 ← Model training
│   ├── evaluate.py              ← Metrics + charts
│   ├── predict.py               ← Inference helpers
│   ├── explainability.py        ← SHAP + feature importance
│   ├── model_registry.py        ← MLflow registration
│   └── monitoring.py            ← Drift + performance monitoring
│
├── api/
│   └── main.py                  ← FastAPI application
│
├── dashboard/
│   └── app.py                   ← Streamlit dashboard
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_validation.py
│   ├── test_transformation.py
│   └── test_api.py
│
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── validation_rules.md
│   └── model_card.md
│
└── reports/
    └── project_report.md
```

---

## 6. Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 16 (local or Docker)
- Docker Desktop (for full containerised setup)
- Git

### Step-by-step Local Setup

```powershell
# 1. Clone / navigate to project
cd "C:\Users\Vinod Meti\Hospital_readmission"

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
Copy-Item .env.example .env
# Then open .env and fill in your PostgreSQL password
```

---

## 7. Environment Variables

Copy `.env.example` to `.env` and set:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hospital_readmission
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<your_password>

MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=hospital_readmission_prediction
```

> Never commit `.env` to version control. It is excluded in `.gitignore`.

---

## 8. PostgreSQL Setup

### Option A – Local PostgreSQL

```sql
-- Connect to PostgreSQL as superuser and run:
CREATE DATABASE hospital_readmission;
```

Then run the schema script:

```powershell
psql -U postgres -d hospital_readmission -f src/database/schema.sql
```

### Option B – Docker (recommended)

```powershell
docker-compose up postgres -d
```

The schema is automatically applied via the init script mount.

---

## 9. Airflow Setup

```powershell
# Install Airflow with constraints (avoids dependency conflicts)
pip install "apache-airflow==2.9.2" `
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-3.11.txt"

# Set AIRFLOW_HOME
$env:AIRFLOW_HOME = "$PWD\airflow"

# Initialise the database
airflow db init

# Create admin user
airflow users create --username admin --password admin `
  --firstname Admin --lastname User --role Admin `
  --email admin@local.dev

# Copy DAG to Airflow dags folder (or set dags_folder in airflow.cfg)
# The dags/ directory in this project IS the dags folder when AIRFLOW_HOME is set
```

---

## 10. ETL Execution

```powershell
# 1. Make sure data/raw/diabetic_data.csv exists

# 2. Run ingestion manually (for testing)
python src/ingestion/ingest.py

# 3. Run cleaning manually
python src/preprocessing/clean_data.py

# 4. Run feature engineering manually
python src/transformation/feature_engineering.py

# 5. Run data quality checks
python src/validation/data_quality.py

# 6. Run full pipeline via Airflow (start scheduler + webserver)
airflow webserver --port 8080
airflow scheduler

# Then visit http://localhost:8080
# Enable and trigger the DAG: hospital_readmission_etl
```

---

## 11. Dashboard Execution

```powershell
# Make sure PostgreSQL is running and analytical tables are populated
streamlit run dashboard/app.py
# Visit http://localhost:8501
```

---

## 12. ML Training (Part 2)

```powershell
# Ensure curated data is in PostgreSQL first (run ETL pipeline)
python ml/train.py
```

---

## 13. MLflow Execution (Part 2)

```powershell
# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 --port 5000

# Visit http://localhost:5000
```

---

## 14. FastAPI Execution (Part 2)

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# API docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

---

## 15. Docker Execution

```powershell
# Copy and configure .env first
Copy-Item .env.example .env

# Build and start all services
docker-compose up --build

# Services available:
# PostgreSQL : localhost:5432
# Airflow UI : http://localhost:8080  (admin/admin)
# MLflow UI  : http://localhost:5000
# FastAPI    : http://localhost:8000/docs
# Streamlit  : http://localhost:8501
```

---

## 16. Testing

```powershell
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 17. Demonstration Checklist

- [ ] Show `data/raw/diabetic_data.csv` exists (unmodified)
- [ ] Run ingestion script → check `data/staging/` output
- [ ] Show ingestion metadata logged (timestamp, row count)
- [ ] Run cleaning → show `data/processed/` output
- [ ] Show `data/rejected/` contains rejected records with reasons
- [ ] Run Airflow DAG → show all tasks green
- [ ] Show PostgreSQL tables populated (`\dt analytics.*`)
- [ ] Open Streamlit dashboard → demonstrate 5 views
- [ ] Run ML training → show MLflow experiment
- [ ] Open MLflow UI → show runs and registered model
- [ ] Call FastAPI `/predict` endpoint
- [ ] Show Docker containers running
- [ ] Run pytest → all tests pass

---

## 18. Expected Outputs

| Output | Location |
|--------|----------|
| Raw data (preserved) | `data/raw/diabetic_data.csv` |
| Staging CSV | `data/staging/diabetic_staging.csv` |
| Curated CSV | `data/curated/diabetic_curated.csv` |
| Rejected records | `data/rejected/rejected_records_<date>.csv` |
| Ingestion log | `data/logs/ingestion_metadata_<date>.json` |
| Validation report | `data/logs/validation_results_<date>.json` |
| Pipeline logs | `logs/pipeline_<date>.log` |
| ML models | `models/` |
| MLflow experiments | `mlruns/` |

---

## 19. Troubleshooting

| Problem | Solution |
|---------|---------|
| `FileNotFoundError: diabetic_data.csv` | Place dataset in `data/raw/` |
| `psycopg2.OperationalError` | Check PostgreSQL is running + `.env` credentials |
| `ModuleNotFoundError` | Activate virtual environment + `pip install -r requirements.txt` |
| Airflow `ImportError` | Install Airflow with constraints file (see Section 9) |
| Docker port conflict | Change port mappings in `docker-compose.yml` |
| MLflow connection error | Start MLflow server first (`mlflow server ...`) |

---

## 20. Privacy Considerations

- The UCI Diabetes dataset is fully de-identified and publicly available
- No patient names, addresses, or direct identifiers are present
- Patient numbers in the dataset are anonymized encounter IDs
- This project creates additional surrogate keys to avoid exposing even those IDs
- No data is uploaded to external services
- All processing is local

---

## 21. Academic Disclaimer

> This project is submitted as an individual academic assignment for the course **Data Engineering and MLOps**.  
> All predictions and model outputs are for **academic demonstration only**.  
> They must **NOT** be used for clinical diagnosis, patient triage, or any medical decision-making.  
> The dataset is publicly available and fully de-identified.
