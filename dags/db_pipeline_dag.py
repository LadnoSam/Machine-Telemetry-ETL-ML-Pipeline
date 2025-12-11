from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

PROJECT_PATH = "/opt/airflow/project"

if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

print("PYTHONPATH:", sys.path)

from database_and_model_tools import (
    DatabaseInitializer,
    DataIngestor,
    FeatureNamesFixer,
    ScalerTester
)

CSV_FILE = "/opt/airflow/project/data/machine_data_cleaned.csv"
SCALER_PATH = "/opt/airflow/project/models/regression_scaler_v18.pkl"


def task_init_database():
    initializer = DatabaseInitializer()
    initializer.setup_complete_database(CSV_FILE)


def task_fix_models():
    fixer = FeatureNamesFixer()
    fixer.fix_all_models()


def task_test_scaler():
    tester = ScalerTester()
    test_data = {
        'fuelconsumption': 10.5, 'vibrationlevel': 4.0, 'humidity': 68.0,
        'pressure': 1000.0, 'poweroutput': 185.0, 'operatinghours': 120.0,
        'timestamp_epoch': 1756684800, 'hour': 12, 'dayofweek': 2, 'month': 9
    }
    tester.test_scaler_with_real_data(SCALER_PATH, test_data)


default_args = {
    'owner': 'I',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id="machine_db_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule='@daily',
    catchup=False,
    max_active_runs=1,
    tags=["ml", "db", "etl"],
) as dag:

    init_db = PythonOperator(
        task_id="init_database",
        python_callable=task_init_database,
    )

    fix_models = PythonOperator(
        task_id="fix_model_features",
        python_callable=task_fix_models,
    )

    test_scaler = PythonOperator(
        task_id="test_scaler_output",
        python_callable=task_test_scaler,
    )

    init_db >> fix_models >> test_scaler
