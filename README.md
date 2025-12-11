# 🚀 Machine Telemetry ETL & ML Pipeline  
### Airflow • PostgreSQL • Python • Machine Learning Models

This project implements a fully automated **ETL + ML pipeline** orchestrated by Apache Airflow, using PostgreSQL as the database and Python for feature engineering, ingestion, and machine learning utilities.

The pipeline performs:

- Database initialization (tables, indexes)
- CSV ingestion into PostgreSQL (high-performance batch insert)
- Automatic ML model feature-name fixes
- Scaler validation using real feature samples
- Daily or manual Airflow execution

---

## 📁 Project Structure

```
airflow/
│ docker-compose.yaml
│ .env
│
├── dags/
│   └── db_pipeline_dag.py
│
├── project/
│   ├── db.py
│   ├── database_and_model_tools.py
│   ├── data/
│   │   └── machine_data_cleaned.csv
│   ├── models/
│   │   ├── best_regressor_v18.pkl
│   │   ├── regression_scaler_v18.pkl
│   │   ├── classifier_fault_idle_v18.pkl
│   │   ├── classifier_fault_idle_scaler_v18.pkl
│   │   ├── classifier_active_maint_v18.pkl
│   │   ├── classifier_active_maint_scaler_v18.pkl
│   │   ├── best_anomaly_detector_v18.pkl
│   │   └── anomaly_scaler_v18.pkl
│   └── __init__.py
│
└── logs/
```

---

## 🐳 Running Airflow with Docker Compose

### 1️⃣ Download Airflow Compose template

```bash
curl -LfO "https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml"
```

### 2️⃣ Start the entire Airflow stack

```bash
docker compose up -d
```

Airflow Web UI:  
👉 http://localhost:8080  
Login: `airflow`  
Password: `airflow`

---

## ⚙️ Environment Variables (.env)

Place this file inside:

```
project/.env
```

```env
DB_HOST=postgres
DB_NAME=airflow
DB_USER=airflow
DB_PASSWORD=airflow
DB_PORT=5432
```

---

## 📊 Airflow DAG — ETL + ML Pipeline

DAG file: `dags/db_pipeline_dag.py`  
Pipeline ID: **machine_db_pipeline**

### ✔ Task 1 — init_database
- Creates required tables  
- Creates database indexes  
- Loads CSV telemetry dataset into PostgreSQL  

### ✔ Task 2 — fix_model_features
- Normalizes feature names  
- Updates stored models & scalers  
- Ensures compatibility  

### ✔ Task 3 — test_scaler_output
- Loads regression scaler  
- Evaluates transformation  

---

## 📅 Schedule (Daily Execution)

Set daily execution:

```python
schedule_interval="@daily"
```

---

## 🗄 Inspecting PostgreSQL Database

```bash
docker compose exec postgres psql -U airflow -d airflow
```

Useful commands:

```
\dt
\d telemetry
SELECT COUNT(*) FROM telemetry;
```

---

## 🎉 Summary

This repo provides:

✔ End-to-end ETL pipeline  
✔ Clean PostgreSQL schema  
✔ Automated ML model fixes  
✔ Scaler validation  
✔ Airflow orchestration  
✔ Full Docker environment  

