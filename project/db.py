import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
import json
import numpy as np
import time
from pathlib import Path

_db_instance = None

_db_instance = None

def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

print("🔍 Loading .env from:", ENV_PATH)

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print("⚠️ WARNING: .env NOT FOUND at:", ENV_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.conn_params = {
            "host": os.getenv("DB_HOST"),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": os.getenv("DB_PORT")
        }
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            logger.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def execute_query(self, query, params=None):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                
                normalized_query = query.strip().upper()
                is_select_query = (
                    normalized_query.startswith('SELECT') or 
                    normalized_query.startswith('WITH') or
                    'SELECT' in normalized_query.split()[0:3]  
                )
                
                if is_select_query:
                    return cursor.fetchall()
                else:
                    self.conn.commit()
                    return cursor.rowcount
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            self.conn.rollback()
            raise

    def init_db(self):
        """Initialize database tables with optimized schema"""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS telemetry (
                id SERIAL PRIMARY KEY,
                machineid VARCHAR(50),
                type VARCHAR(50),
                location VARCHAR(100),
                timestamp TIMESTAMP,
                enginetemperature FLOAT,
                fuelconsumption FLOAT,
                vibrationlevel FLOAT,
                humidity FLOAT,
                pressure FLOAT,
                poweroutput FLOAT,
                operatinghours FLOAT,
                status VARCHAR(50),
                status_encoded INTEGER,
                timestamp_epoch BIGINT,
                hour INTEGER,
                dayofweek INTEGER,
                month INTEGER,
                ts_utc TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                ts_epoch BIGINT DEFAULT EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_query_log (
                id SERIAL PRIMARY KEY,
                role VARCHAR(100),
                query TEXT,
                intent VARCHAR(50),
                confidence FLOAT,
                machine_id VARCHAR(50),
                target_time_epoch BIGINT,
                ts_epoch BIGINT DEFAULT EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                machine_id VARCHAR(50),
                intent VARCHAR(50),
                numerical_answer FLOAT,
                features JSONB,
                ts_epoch BIGINT DEFAULT EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)
            )
            """
        ]
        
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_telemetry_machine_id ON telemetry(machineid)",
            "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp_epoch ON telemetry(timestamp_epoch)",
            "CREATE INDEX IF NOT EXISTS idx_telemetry_ts_epoch ON telemetry(ts_epoch)",
            "CREATE INDEX IF NOT EXISTS idx_query_log_epoch ON user_query_log(ts_epoch)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_epoch ON predictions(ts_epoch)",
            "CREATE INDEX IF NOT EXISTS idx_telemetry_composite ON telemetry(machineid, timestamp_epoch)"
        ]
        
        for query in queries:
            try:
                self.execute_query(query)
                logger.info(f"✅ Executed table creation query")
            except Exception as e:
                logger.warning(f"⚠️ Could not execute query: {e}")
                continue
        
        for query in index_queries:
            try:
                self.execute_query(query)
                logger.info(f"✅ Created index")
            except Exception as e:
                logger.warning(f"⚠️ Could not create index: {e}")
                continue

    def insert_telemetry(self, data):
        query = """
        INSERT INTO telemetry (
            machineid, type, location, timestamp, enginetemperature, fuelconsumption,
            vibrationlevel, humidity, pressure, poweroutput, operatinghours, status,
            status_encoded, timestamp_epoch, hour, dayofweek, month
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        return self.execute_query(query, data)

    def log_user_query(self, role, query, intent, confidence, machine_id=None, target_time_epoch=None):
        query = """
        INSERT INTO user_query_log (role, query, intent, confidence, machine_id, target_time_epoch)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        confidence = float(confidence) if confidence is not None else 0.0
        return self.execute_query(query, (role, query, intent, confidence, machine_id, target_time_epoch))

    def log_prediction(self, machine_id, intent, numerical_answer, features):
        query = """
        INSERT INTO predictions (machine_id, intent, numerical_answer, features)
        VALUES (%s, %s, %s, %s)
        """
        numerical_answer = float(numerical_answer) if numerical_answer is not None else 0.0
        
        features_serializable = {}
        for key, value in features.items():
            if hasattr(value, 'item'):
                features_serializable[key] = value.item()
            else:
                features_serializable[key] = float(value) if isinstance(value, (int, float)) else value
        
        return self.execute_query(query, (machine_id, intent, numerical_answer, json.dumps(features_serializable)))

    def get_latest_telemetry(self, machine_id, limit=1):
        query = """
        SELECT * FROM telemetry 
        WHERE machineid = %s 
        ORDER BY timestamp_epoch DESC 
        LIMIT %s
        """
        return self.execute_query(query, (machine_id, limit))

    def get_telemetry_range(self, machine_id, time_from_epoch, time_to_epoch):
        query = """
        SELECT * FROM telemetry 
        WHERE machineid = %s AND timestamp_epoch BETWEEN %s AND %s
        ORDER BY timestamp_epoch
        """
        return self.execute_query(query, (machine_id, time_from_epoch, time_to_epoch))
    
    def get_machine_list(self):
        """Get list of all available machines"""
        query = "SELECT DISTINCT machineid FROM telemetry ORDER BY machineid"
        return self.execute_query(query)
    
    def get_telemetry_stats(self, machine_id=None):
        """Get basic statistics for telemetry data"""
        if machine_id:
            query = """
            SELECT 
                COUNT(*) as record_count,
                MIN(timestamp_epoch) as earliest_time,
                MAX(timestamp_epoch) as latest_time,
                AVG(enginetemperature) as avg_temperature,
                AVG(fuelconsumption) as avg_fuel,
                AVG(vibrationlevel) as avg_vibration
            FROM telemetry 
            WHERE machineid = %s
            """
            return self.execute_query(query, (machine_id,))
        else:
            query = """
            SELECT 
                COUNT(*) as record_count,
                MIN(timestamp_epoch) as earliest_time,
                MAX(timestamp_epoch) as latest_time,
                COUNT(DISTINCT machineid) as machine_count
            FROM telemetry
            """
            return self.execute_query(query)

    def get_machines_with_highest_temperature(self, limit=5):
        """Get machines with highest current temperature"""
        query = """
        SELECT DISTINCT ON (machineid) 
            machineid,
            enginetemperature as temperature,
            timestamp_epoch,
            timestamp
        FROM telemetry 
        WHERE enginetemperature IS NOT NULL
        ORDER BY machineid, timestamp_epoch DESC
        """
        all_machines = self.execute_query(query)
        
        sorted_machines = sorted(all_machines, key=lambda x: x['temperature'], reverse=True)
        return sorted_machines[:limit]

    def get_machines_with_highest_humidity(self, limit=5):
        """Get machines with highest current humidity"""
        query = """
        SELECT DISTINCT ON (machineid) 
            machineid,
            humidity,
            timestamp_epoch,
            timestamp
        FROM telemetry 
        WHERE humidity IS NOT NULL
        ORDER BY machineid, timestamp_epoch DESC
        """
        all_machines = self.execute_query(query)
        
        sorted_machines = sorted(all_machines, key=lambda x: x['humidity'], reverse=True)
        return sorted_machines[:limit]

    def get_machines_with_highest_vibration(self, limit=5):
        """Get machines with highest current vibration level"""
        query = """
        SELECT DISTINCT ON (machineid) 
            machineid,
            vibrationlevel as vibration,
            timestamp_epoch,
            timestamp
        FROM telemetry 
        WHERE vibrationlevel IS NOT NULL
        ORDER BY machineid, timestamp_epoch DESC
        """
        all_machines = self.execute_query(query)
        
        sorted_machines = sorted(all_machines, key=lambda x: x['vibration'], reverse=True)
        return sorted_machines[:limit]

    def get_machines_with_highest_fuel_consumption(self, limit=5):
        """Get machines with highest current fuel consumption"""
        query = """
        SELECT DISTINCT ON (machineid) 
            machineid,
            fuelconsumption as fuel,
            timestamp_epoch,
            timestamp
        FROM telemetry 
        WHERE fuelconsumption IS NOT NULL
        ORDER BY machineid, timestamp_epoch DESC
        """
        all_machines = self.execute_query(query)
        
        sorted_machines = sorted(all_machines, key=lambda x: x['fuel'], reverse=True)
        return sorted_machines[:limit]

    def get_machines_by_status(self, status_filter=None):
        """Get machines filtered by status - FIXED ACCURATE DATA"""
        try:
            if status_filter:
                query = """
                SELECT DISTINCT ON (machineid) 
                    machineid,
                    status,
                    enginetemperature,
                    fuelconsumption,
                    vibrationlevel,
                    humidity,
                    timestamp_epoch,
                    timestamp
                FROM telemetry 
                WHERE status ILIKE %s AND machineid IS NOT NULL
                ORDER BY machineid, timestamp_epoch DESC
                """
                result = self.execute_query(query, (f'%{status_filter}%',))
            else:
                query = """
                SELECT DISTINCT ON (machineid) 
                    machineid,
                    status,
                    enginetemperature,
                    fuelconsumption,
                    vibrationlevel,
                    humidity,
                    timestamp_epoch,
                    timestamp
                FROM telemetry 
                WHERE machineid IS NOT NULL
                ORDER BY machineid, timestamp_epoch DESC
                """
                result = self.execute_query(query)
            
            if isinstance(result, int):
                logger.warning(f"⚠️ Query returned integer instead of list: {result}")
                return []
            
            if not result:
                logger.info(f"📊 No machines found with status filter: {status_filter}")
                return []
            
            valid_machines = []
            for machine in result:
                if (isinstance(machine, dict) and 
                    machine.get('machineid') and 
                    machine.get('status')):
                    
                    if status_filter:
                        actual_status = machine.get('status', '').lower()
                        filter_status = status_filter.lower()
                        if filter_status in actual_status:
                            valid_machines.append(machine)
                    else:
                        valid_machines.append(machine)
            
            logger.info(f"📊 Found {len(valid_machines)} ACCURATE machines with status filter: {status_filter}")
            return valid_machines
            
        except Exception as e:
            logger.error(f"❌ Error getting machines by status: {e}")
            return []

    def get_machine_comparison_stats(self):
        """Get comparative statistics for all machines"""
        query = """
        SELECT 
            machineid,
            COUNT(*) as record_count,
            AVG(enginetemperature) as avg_temperature,
            MAX(enginetemperature) as max_temperature,
            AVG(humidity) as avg_humidity,
            MAX(humidity) as max_humidity,
            AVG(vibrationlevel) as avg_vibration,
            MAX(vibrationlevel) as max_vibration,
            AVG(fuelconsumption) as avg_fuel,
            MAX(fuelconsumption) as max_fuel,
            MAX(timestamp_epoch) as last_update
        FROM telemetry 
        GROUP BY machineid
        ORDER BY avg_temperature DESC
        """
        return self.execute_query(query)
    

    def get_machines_with_lowest_temperature(self, limit=5):
        """Get machines with lowest current temperature"""
        query = """
        SELECT DISTINCT ON (machineid) 
            machineid,
            enginetemperature as temperature,
            timestamp_epoch,
            timestamp
        FROM telemetry 
        WHERE enginetemperature IS NOT NULL
        ORDER BY machineid, timestamp_epoch DESC
        """
        all_machines = self.execute_query(query)
        
        sorted_machines = sorted(all_machines, key=lambda x: x['temperature'])
        return sorted_machines[:limit]

    def get_machines_with_lowest_humidity(self, limit=5):
        """Get machines with lowest current humidity - FIXED VERSION"""
        try:
            query = """
            SELECT DISTINCT ON (machineid) 
                machineid,
                humidity,
                timestamp_epoch,
                timestamp
            FROM telemetry 
            WHERE humidity IS NOT NULL 
            AND humidity > 0 
            AND humidity <= 100
            AND machineid IS NOT NULL
            ORDER BY machineid, timestamp_epoch DESC
            """
            
            all_machines = self.execute_query(query)
            
            if not all_machines or isinstance(all_machines, int):
                logger.warning("⚠️ No valid humidity data found or query returned integer")
                return []
            
            sorted_machines = sorted(all_machines, key=lambda x: x['humidity'] if x['humidity'] is not None else float('inf'))
            
            logger.info(f"🔍 Database query returned {len(sorted_machines) if isinstance(sorted_machines, list) else 'non-list'} results")
            for i, machine in enumerate(sorted_machines[:limit]):
                logger.info(f"   {i+1}. {machine['machineid']}: {machine['humidity']}%")
            
            return sorted_machines[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error in get_machines_with_lowest_humidity: {e}")
            return []


    def get_machines_with_lowest_vibration(self, limit=5):
        """Get machines with lowest current vibration level"""
        query = """
        SELECT DISTINCT ON (machineid) 
            machineid,
            vibrationlevel as vibration,
            timestamp_epoch,
            timestamp
        FROM telemetry 
        WHERE vibrationlevel IS NOT NULL
        ORDER BY machineid, timestamp_epoch DESC
        """
        all_machines = self.execute_query(query)
        
        sorted_machines = sorted(all_machines, key=lambda x: x['vibration'])
        return sorted_machines[:limit]

    def get_machines_with_lowest_fuel_consumption(self, limit=5):
        """Get machines with lowest current fuel consumption"""
        query = """
        SELECT DISTINCT ON (machineid) 
            machineid,
            fuelconsumption as fuel,
            timestamp_epoch,
            timestamp
        FROM telemetry 
        WHERE fuelconsumption IS NOT NULL
        ORDER BY machineid, timestamp_epoch DESC
        """
        all_machines = self.execute_query(query)
        
        sorted_machines = sorted(all_machines, key=lambda x: x['fuel'])
        return sorted_machines[:limit]



