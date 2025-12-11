from psycopg2.extras import execute_values
import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from db import get_db  # using existing Database class from project

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestor:
    """Load CSV data into the database telemetry table."""
    def __init__(self):
        self.db = get_db()
        self.batch_size = 1000
        self.required_columns = [
            'machineid', 'type', 'location', 'timestamp', 'enginetemperature',
            'fuelconsumption', 'vibrationlevel', 'humidity', 'pressure',
            'poweroutput', 'operatinghours', 'status', 'status_encoded',
            'timestamp_epoch', 'hour', 'dayofweek', 'month'
        ]
        self.column_mapping = {
            'MachineID': 'machineid', 'Type': 'type', 'Location': 'location',
            'Timestamp': 'timestamp', 'EngineTemperature': 'enginetemperature',
            'FuelConsumption': 'fuelconsumption', 'VibrationLevel': 'vibrationlevel',
            'Humidity': 'humidity', 'Pressure': 'pressure', 'PowerOutput': 'poweroutput',
            'OperatingHours': 'operatinghours', 'Status': 'status',
            'Status_encoded': 'status_encoded', 'Timestamp_epoch': 'timestamp_epoch',
            'hour': 'hour', 'dayofweek': 'dayofweek', 'month': 'month'
        }

    def ingest_csv(self, file_path: str) -> int:
        try:
            logger.info(f"📖 Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"📊 Loaded {len(df)} rows from CSV")

            df = self._convert_column_names(df)
            df = self._clean_dataframe(df)

            count = self._insert_rows(df)
            logger.info(f"🎉 Successfully ingested {count} rows")
            return count
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            raise

    def _convert_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        mapping = {col: self.column_mapping.get(col, col.lower()) for col in df.columns}
        df_copy.rename(columns=mapping, inplace=True)
        return df_copy

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.fillna({
            'enginetemperature': 75.0,
            'fuelconsumption': 10.0,
            'vibrationlevel': 3.0,
            'humidity': 65.0,
            'pressure': 950.0,
            'poweroutput': 200.0,
            'operatinghours': 0.0,
            'status': 'Unknown'
        })
        df_clean.columns = [c.lower() for c in df_clean.columns]
        return df_clean



    def _insert_rows(self, df: pd.DataFrame) -> int:
        """FAST batch insert using psycopg2.execute_values"""
        rows = []
        for _, row in df.iterrows():
            values = tuple(row.get(c, None) for c in self.required_columns)
            rows.append(values)

        query = f"""
    INSERT INTO telemetry (
        {", ".join(self.required_columns)}
    ) VALUES %s
    """

        try:
            with self.db.conn.cursor() as cursor:
                execute_values(cursor, query, rows, page_size=1000)
            self.db.conn.commit()
            logger.info(f"⚡ Batch insert complete: {len(rows)} rows")
            return len(rows)
        except Exception as e:
            logger.error(f"❌ Batch insert failed: {e}")
            self.db.conn.rollback()
            raise

class DatabaseInitializer:
    """Initialize and verify database tables, and optionally ingest data."""
    def __init__(self):
        self.db = get_db()

    def setup_complete_database(self, csv_file_path: str = None):
        try:
            logger.info("🚀 Starting complete database setup...")
            self.db.init_db()
            if csv_file_path and os.path.exists(csv_file_path):
                ingestor = DataIngestor()
                count = ingestor.ingest_csv(csv_file_path)
                logger.info(f"📊 {count} rows inserted into telemetry.")
            self.verify_database_setup()
            logger.info("🎊 Database setup completed successfully!")
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")

    def verify_database_setup(self):
        tables = ['telemetry', 'user_query_log', 'predictions']
        for t in tables:
            try:
                res = self.db.execute_query(f"SELECT COUNT(*) as count FROM {t}")
                count = res[0]['count'] if res else 0
                logger.info(f"📋 {t}: {count} rows")
            except Exception as e:
                logger.warning(f"⚠️ Could not check {t}: {e}")

class FeatureNamesFixer:
    """Fix and normalize feature names for models and scalers."""
    def __init__(self):
        self.feature_mapping = {
            'FuelConsumption': 'fuelconsumption', 'VibrationLevel': 'vibrationlevel',
            'Humidity': 'humidity', 'Pressure': 'pressure', 'PowerOutput': 'poweroutput',
            'OperatingHours': 'operatinghours', 'Timestamp_epoch': 'timestamp_epoch',
            'Hour': 'hour', 'DayOfWeek': 'dayofweek', 'Month': 'month'
        }
        self.expected_features = [
            'fuelconsumption', 'vibrationlevel', 'humidity', 'pressure',
            'poweroutput', 'operatinghours', 'timestamp_epoch', 'hour',
            'dayofweek', 'month'
        ]
        self.model_paths = {
    "regression": {
        "model": os.path.join(MODEL_DIR, "best_regressor_v18.pkl"),
        "scaler": os.path.join(MODEL_DIR, "regression_scaler_v18.pkl"),
    },
    "classification_fault_idle": {
        "model": os.path.join(MODEL_DIR, "classifier_fault_idle_v18.pkl"),
        "scaler": os.path.join(MODEL_DIR, "classifier_fault_idle_scaler_v18.pkl"),
    },
    "classification_active_maint": {
        "model": os.path.join(MODEL_DIR, "classifier_active_maint_v18.pkl"),
        "scaler": os.path.join(MODEL_DIR, "classifier_active_maint_scaler_v18.pkl"),
    },
    "anomaly": {
        "model": os.path.join(MODEL_DIR, "best_anomaly_detector_v18.pkl"),
        "scaler": os.path.join(MODEL_DIR, "anomaly_scaler_v18.pkl"),
    },
}
    def fix_all_models(self) -> Dict[str, bool]:
        results = {}
        for intent, paths in self.model_paths.items():
            results[intent + "_model"] = self.fix_model_features(paths['model'], intent)
            results[intent + "_scaler"] = self.fix_scaler_features(paths['scaler'], intent)
        return results

    def fix_model_features(self, model_path: str, model_type: str) -> bool:
        try:
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return False

            model = joblib.load(model_path)
            if hasattr(model, 'feature_names_in_'):
                try:
                    original = list(model.feature_names_in_)
                    new_features = [self.feature_mapping.get(f, f.lower()) for f in original]
                    try:
                        model.feature_names_in_ = new_features
                        logger.info(f"✅ Updated feature names for {model_type}: {new_features}")
                    except AttributeError:
                        logger.warning(f"⚠️ Cannot modify feature_names_in_ for {model_type} (read-only). Skipped update.")
                    joblib.dump(model, model_path)
                    logger.info(f"💾 Saved model file: {model_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not fix features for {model_type}: {e}")
            else:
                logger.info(f"ℹ️ Model {model_type} has no feature_names_in_ attribute, skipped.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to fix model features ({model_type}): {e}")
            return False

    def fix_scaler_features(self, scaler_path: str, scaler_type: str) -> bool:
        try:
            if not os.path.exists(scaler_path):
                logger.error(f"❌ Scaler file not found: {scaler_path}")
                return False

            scaler = joblib.load(scaler_path)
            if hasattr(scaler, 'feature_names_in_'):
                original = list(scaler.feature_names_in_)
                new_features = [self.feature_mapping.get(f, f.lower()) for f in original]
                scaler.feature_names_in_ = new_features
                joblib.dump(scaler, scaler_path)
                logger.info(f"✅ Fixed scaler features for {scaler_type}: {new_features}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to fix scaler features ({scaler_type}): {e}")
            return False

class ScalerTester:
    """Test and validate scalers using real feature data."""
    def __init__(self):
        self.feature_order = [
            'fuelconsumption', 'vibrationlevel', 'humidity', 'pressure',
            'poweroutput', 'operatinghours', 'timestamp_epoch', 'hour',
            'dayofweek', 'month'
        ]

    def test_scaler_with_real_data(self, scaler_path: str, features_dict: Dict[str, float]):
        try:
            if not os.path.exists(scaler_path):
                logger.error(f"❌ Scaler not found: {scaler_path}")
                return
            scaler = joblib.load(scaler_path)
            feature_vector = np.array([[features_dict.get(f, 0.0) for f in self.feature_order]])
            scaled = scaler.transform(feature_vector)
            logger.info(f"✅ Scaled output for {scaler_path}: {scaled[0]}")
        except Exception as e:
            logger.error(f"❌ Scaler test failed: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("🧩 Unified Database + Model Tools (Safe Version with LightGBM fix)")
    print("=" * 80)

    initializer = DatabaseInitializer()
    initializer.setup_complete_database("data/machine_data_cleaned.csv")

    fixer = FeatureNamesFixer()
    fixer.fix_all_models()

    tester = ScalerTester()
    test_data = {
        'fuelconsumption': 10.5, 'vibrationlevel': 4.0, 'humidity': 68.0,
        'pressure': 1000.0, 'poweroutput': 185.0, 'operatinghours': 120.0,
        'timestamp_epoch': 1756684800, 'hour': 12, 'dayofweek': 2, 'month': 9
    }
    tester.test_scaler_with_real_data("models/regression_scaler_v18.pkl", test_data)

    print("=" * 80)
    print("🎉 All setup, ingestion, and model fix operations completed successfully!")
    print("=" * 80)
