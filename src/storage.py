import sqlite3
import json
import logging
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageManager:
    """
    Layer 3: Manages persistence of classification results, feature vectors,
    and metadata using SQLite.
    """

    def __init__(self, config: Dict[str, Any]):
        self.db_path = config['storage']['database_path']
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        """Creates the necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Image Metadata Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_metadata (
            image_id TEXT PRIMARY KEY,
            file_name TEXT,
            capture_datetime TEXT,
            scale_um_per_pixel REAL,
            latitude REAL,
            longitude REAL,
            depth_m REAL,
            temperature_c REAL,
            salinity_ppt REAL
        )
        ''')

        # Particle Features Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS particle_features (
            particle_id TEXT,
            image_id TEXT,
            size_um REAL,
            area_um2 REAL,
            circularity REAL,
            aspect_ratio REAL,
            ri_estimate REAL,
            intensity_mean REAL,
            entropy REAL,
            contrast REAL,
            PRIMARY KEY (particle_id, image_id),
            FOREIGN KEY(image_id) REFERENCES image_metadata(image_id)
        )
        ''')

        # Classifications Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            particle_id TEXT,
            image_id TEXT,
            classification TEXT,
            confidence_microplastic REAL,
            confidence_biological REAL,
            polymer_type TEXT,
            organism_type TEXT,
            recommendation TEXT,
            reasoning TEXT,
            evidence_json TEXT,
            timestamp TEXT,
            FOREIGN KEY(particle_id, image_id) REFERENCES particle_features(particle_id, image_id)
        )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def save_classification(self,
                            image_meta: Dict[str, Any],
                            particle_data: Dict[str, Any],
                            classification_result: Dict[str, Any]):
        """Saves a complete record of a classified particle."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 1. Save Image Metadata (Ignore if exists)
            cursor.execute('''
            INSERT OR IGNORE INTO image_metadata 
            (image_id, file_name, capture_datetime, scale_um_per_pixel, latitude, longitude, depth_m, temperature_c, salinity_ppt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_meta['image_id'],
                image_meta['file_name'],
                image_meta['capture_datetime'],
                image_meta['scale_um_per_pixel'],
                image_meta['location']['latitude'],
                image_meta['location']['longitude'],
                image_meta['location']['depth_m'],
                image_meta['water_properties']['temperature_c'],
                image_meta['water_properties']['salinity_ppt']
            ))

            # 2. Save Features
            feats = particle_data['features']
            cursor.execute('''
            INSERT OR REPLACE INTO particle_features
            (particle_id, image_id, size_um, area_um2, circularity, aspect_ratio, ri_estimate, intensity_mean, entropy, contrast)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                particle_data['particle_id'],
                image_meta['image_id'],
                feats['size_um'],
                feats['area_um2'],
                feats['circularity'],
                feats['aspect_ratio'],
                feats['refractive_index_estimate'],
                feats['intensity_mean'],
                feats['entropy'],
                feats['contrast']
            ))

            # 3. Save Classification
            cursor.execute('''
            INSERT INTO classifications
            (particle_id, image_id, classification, confidence_microplastic, confidence_biological, 
            polymer_type, organism_type, recommendation, reasoning, evidence_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                particle_data['particle_id'],
                image_meta['image_id'],
                classification_result['classification'],
                classification_result['confidence_microplastic'],
                classification_result['confidence_biological'],
                classification_result.get('polymer_type'),
                classification_result.get('organism_type'),
                classification_result['recommendation'],
                classification_result['reasoning'],
                json.dumps(classification_result.get('evidence', {})),
                datetime.now().isoformat()
            ))

            conn.commit()
        except Exception as e:
            logger.error(f"DB Save Error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Retrieves aggregate statistics for reporting."""
        conn = sqlite3.connect(self.db_path)

        stats = {}
        try:
            df = pd.read_sql_query("SELECT * FROM classifications", conn)
            if not df.empty:
                stats['total_particles'] = len(df)
                stats['classification_counts'] = df['classification'].value_counts().to_dict()
                stats['polymer_counts'] = df['polymer_type'].value_counts().to_dict()
                stats['avg_confidence'] = df['confidence_microplastic'].mean()
            else:
                stats = {'total_particles': 0}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
        finally:
            conn.close()

        return stats

    def export_csv(self, output_path: str):
        """Exports joined data to CSV."""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT c.*, f.size_um, f.circularity, f.ri_estimate, m.file_name, m.depth_m 
        FROM classifications c
        JOIN particle_features f ON c.particle_id = f.particle_id AND c.image_id = f.image_id
        JOIN image_metadata m ON c.image_id = m.image_id
        """
        try:
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_path, index=False)
            logger.info(f"Data exported to {output_path}")
        finally:
            conn.close()