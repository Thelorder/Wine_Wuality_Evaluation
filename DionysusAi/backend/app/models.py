# backend/app/models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sqlite3
import json

class WineSample(BaseModel):
    id: Optional[int] = None
    type: int  # 1 for white, 0 for red
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    quality_prediction: str
    quality_score: int
    confidence: float
    probabilities: dict
    created_at: Optional[str] = None

class DatabaseManager:
    def __init__(self, db_path: str = "wine_samples.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with samples table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wine_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type INTEGER NOT NULL,
                fixed_acidity REAL NOT NULL,
                volatile_acidity REAL NOT NULL,
                citric_acid REAL NOT NULL,
                residual_sugar REAL NOT NULL,
                chlorides REAL NOT NULL,
                free_sulfur_dioxide REAL NOT NULL,
                density REAL NOT NULL,
                pH REAL NOT NULL,
                sulphates REAL NOT NULL,
                alcohol REAL NOT NULL,
                quality_prediction TEXT NOT NULL,
                quality_score INTEGER NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def save_sample(self, sample: WineSample) -> int:
        """Save a wine sample to database and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO wine_samples (
                type, fixed_acidity, volatile_acidity, citric_acid, 
                residual_sugar, chlorides, free_sulfur_dioxide, density,
                pH, sulphates, alcohol, quality_prediction, quality_score,
                confidence, probabilities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample.type, sample.fixed_acidity, sample.volatile_acidity,
            sample.citric_acid, sample.residual_sugar, sample.chlorides,
            sample.free_sulfur_dioxide, sample.density, sample.pH,
            sample.sulphates, sample.alcohol, sample.quality_prediction,
            sample.quality_score, sample.confidence, json.dumps(sample.probabilities)
        ))
        
        sample_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ Sample saved with ID: {sample_id}")
        return sample_id
    
    def get_all_samples(self) -> List[WineSample]:
        """Retrieve all wine samples from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM wine_samples ORDER BY created_at DESC
        ''')
        
        samples = []
        for row in cursor.fetchall():
            samples.append(WineSample(
                id=row[0],
                type=row[1],
                fixed_acidity=row[2],
                volatile_acidity=row[3],
                citric_acid=row[4],
                residual_sugar=row[5],
                chlorides=row[6],
                free_sulfur_dioxide=row[7],
                density=row[8],
                pH=row[9],
                sulphates=row[10],
                alcohol=row[11],
                quality_prediction=row[12],
                quality_score=row[13],
                confidence=row[14],
                probabilities=json.loads(row[15]),
                created_at=row[16]
            ))
        
        conn.close()
        return samples
    
    def get_sample(self, sample_id: int) -> Optional[WineSample]:
        """Retrieve a specific wine sample by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM wine_samples WHERE id = ?', (sample_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return WineSample(
                id=row[0],
                type=row[1],
                fixed_acidity=row[2],
                volatile_acidity=row[3],
                citric_acid=row[4],
                residual_sugar=row[5],
                chlorides=row[6],
                free_sulfur_dioxide=row[7],
                density=row[8],
                pH=row[9],
                sulphates=row[10],
                alcohol=row[11],
                quality_prediction=row[12],
                quality_score=row[13],
                confidence=row[14],
                probabilities=json.loads(row[15]),
                created_at=row[16]
            )
        return None