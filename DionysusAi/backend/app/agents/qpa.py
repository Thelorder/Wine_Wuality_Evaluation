# backend/app/agents/qpa.py
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
import traceback

class QualityPredictionAgent:
    def __init__(self):
        self.model = None
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def load_model(self, model_path: str, imputer_path: str, scaler_path: str, features_path: str):
        """Load pre-trained model and preprocessing objects"""
        try:
            print(f"🔍 Attempting to load model from: {model_path}")
            
            # Check if files exist first
            for path in [model_path, imputer_path, scaler_path, features_path]:
                if not os.path.exists(path):
                    print(f"❌ File not found: {path}")
                    return False
                else:
                    print(f"✅ File exists: {path}")
            
            print("📥 Loading model file...")
            self.model = joblib.load(model_path)
            print("✅ Model loaded successfully")
            
            print("📥 Loading imputer...")
            self.imputer = joblib.load(imputer_path)
            print("✅ Imputer loaded successfully")
            
            print("📥 Loading scaler...")
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully")
            
            print("📥 Loading feature names...")
            self.feature_names = joblib.load(features_path)
            print(f"✅ Feature names loaded: {self.feature_names}")
            
            self.is_trained = True
            print("🎉 QPA model loaded successfully - All systems go!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load QPA model: {e}")
            print("Full error traceback:")
            traceback.print_exc()
            return False
    
    def predict_quality(self, wine_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict wine quality from input features"""
        if not self.is_trained:
            return {
                "quality_prediction": "Model not loaded",
                "quality_score": -1,
                "confidence": 0.0,
                "probabilities": {"best_quality": 0.0, "standard_quality": 0.0},
                "success": False,
                "error": "QPA agent not ready"
            }
        
        try:
            print(f"🎯 Making prediction with features: {wine_data}")
            
            # Map API feature names (with underscores) to model feature names (with spaces)
            feature_mapping = {
                'fixed_acidity': 'fixed acidity',
                'volatile_acidity': 'volatile acidity',
                'citric_acid': 'citric acid', 
                'residual_sugar': 'residual sugar',
                'free_sulfur_dioxide': 'free sulfur dioxide'
                # Note: 'type', 'chlorides', 'density', 'pH', 'sulphates', 'alcohol' don't need mapping
            }
            
            # Create mapped data dictionary
            mapped_data = {}
            for api_name, value in wine_data.items():
                model_name = feature_mapping.get(api_name, api_name)  # Use mapping or original name
                mapped_data[model_name] = value
            
            # DEBUG: Check if we have all expected features
            print(f"🔤 Expected feature names: {self.feature_names}")
            print(f"🗂️  Mapped data keys: {list(mapped_data.keys())}")
            
            # Check for missing features
            missing_features = []
            input_values = []
            for feature in self.feature_names:
                if feature in mapped_data:
                    input_values.append(mapped_data[feature])
                else:
                    missing_features.append(feature)
                    input_values.append(None)  # This will cause issues!
            
            if missing_features:
                print(f"❌ MISSING FEATURES: {missing_features}")
                return {
                    "quality_prediction": "Error",
                    "quality_score": -1,
                    "confidence": 0.0,
                    "probabilities": {"best_quality": 0.0, "standard_quality": 0.0},
                    "success": False,
                    "error": f"Missing features: {missing_features}"
                }
            
            print(f"📊 Input values: {input_values}")
            print(f"🔤 Feature names: {self.feature_names}")
            
            input_df = pd.DataFrame([input_values], columns=self.feature_names)
            print(f"📋 Input DataFrame shape: {input_df.shape}")
            
            # Apply preprocessing
            print("🔄 Applying imputer...")
            input_imputed = self.imputer.transform(input_df)
            print(f"✅ Imputed data shape: {input_imputed.shape}")
            
            print("🔄 Applying scaler...")
            input_processed = self.scaler.transform(input_imputed)
            print(f"✅ Scaled data shape: {input_processed.shape}")
            
            # Make prediction
            print("🤖 Making prediction...")    
            prediction = self.model.predict(input_processed)[0]  # 0 or 1
            probabilities = self.model.predict_proba(input_processed)[0]
            
            print(f"🎉 Raw prediction: {prediction}")
            print(f"📈 Probabilities: {probabilities}")
            
            # INTERPRET THE PREDICTION PROPERLY
            # Based on your training: 0 = Standard (scores 1-5), 1 = Best (scores 6-10)
            
            if prediction == 1:
                # Best quality (scores 6-10)
                quality_prediction = "High Quality "
                quality_score = 1  # Binary score
                estimated_numeric_score = 7  # Middle of 6-10 range
                quality_range = "6-10"
                quality_category = "High Quality"
            else:
                # Standard quality (scores 1-5)
                quality_prediction = "Standard Quality"
                quality_score = 0  # Binary score
                estimated_numeric_score = 3  # Middle of 1-5 range
                quality_range = "1-5"
                quality_category = "Standard Quality"
            
            confidence = float(max(probabilities))
            
            result = {
                "quality_prediction": quality_prediction,
                "quality_score": int(prediction),  # Binary: 0 or 1
                "estimated_numeric_score": estimated_numeric_score,  # Estimated 1-10 score
                "quality_range": quality_range,  # Range interpretation
                "quality_category": quality_category,  # Category
                "confidence": confidence,
                "probabilities": {
                    "below_average": float(probabilities[0]),  # Probability of scores 1-5
                    "above_average": float(probabilities[1])   # Probability of scores 6-10
                },
                "success": True,
                "note": "Binary classification: 0 = scores 1-5, 1 = scores 6-10"
            }
            
            print(f"📤 Returning result: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            return {
                "success": False, 
                "error": str(e),
                "quality_prediction": "Error",
                "quality_score": -1,
                "confidence": 0.0,
                "probabilities": {"below_average": 0.0, "above_average": 0.0}
            }