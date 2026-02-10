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

    def load_model(self, model_path: str, imputer_path: str, scaler_path: str, features_path: str = None):
        """Load pre-trained model and preprocessing objects"""
        try:
            print(f"🔍 Attempting to load model from: {model_path}")

            for path in [model_path, imputer_path, scaler_path]:
                if not os.path.exists(path):
                    print(f"❌ File not found: {path}")
                    return False
                else:
                    print(f"✅ File exists: {path}")

            if features_path and os.path.exists(features_path):
                print(f"✅ File exists: {features_path}")
                self.feature_names = joblib.load(features_path)
                print(f"✅ Feature names loaded: {self.feature_names}")
            else:
                print("📌 Features path not provided or not found - using default features for regression")  
                self.feature_names = [
                    'type', 'fixed acidity', 'volatile acidity', 'citric acid',
                    'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'density', 'pH', 'sulphates', 'alcohol'
                ]  

            print("📥 Loading model file...")
            self.model = joblib.load(model_path)
            print("✅ Model loaded successfully")

            print("📥 Loading imputer...")
            self.imputer = joblib.load(imputer_path)
            print("✅ Imputer loaded successfully")

            print("📥 Loading scaler...")
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully")

            self.is_trained = True
            print("🎉 QPA model loaded successfully - All systems go!")
            return True

        except Exception as e:
            print(f"❌ Failed to load QPA model: {e}")
            print("Full error traceback:")
            traceback.print_exc()
            return False

    def quality_label(self, score: float) -> str:  
        if score < 5.5:
            return "Low"
        elif score < 6.5:
            return "Standard"
        elif score < 7.5:
            return "High"
        else:
            return "Excellent"

    def predict_quality(self, wine_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict wine quality from input features using regression"""  
        if not self.is_trained:
            return {
                "success": False,
                "error": "QPA agent not ready",
                "quality_score": None,
                "quality_label": None  
            }

        try:
            print(f"🎯 Making prediction with features: {wine_data}")

            feature_mapping = {
                'fixed_acidity': 'fixed acidity',
                'volatile_acidity': 'volatile acidity',
                'citric_acid': 'citric acid',
                'residual_sugar': 'residual sugar',
                'free_sulfur_dioxide': 'free sulfur dioxide'
            }

            mapped_data = {}
            for api_name, value in wine_data.items():
                model_name = feature_mapping.get(api_name, api_name) 
                mapped_data[model_name] = value

            print(f"🔤 Expected feature names: {self.feature_names}")
            print(f"🗂️  Mapped data keys: {list(mapped_data.keys())}")

            missing_features = [f for f in self.feature_names if f not in mapped_data]
            if missing_features:
                print(f"❌ MISSING FEATURES: {missing_features}")
                return {
                    "success": False,
                    "error": f"Missing features: {missing_features}",
                    "quality_score": None,
                    "quality_label": None
                }

            input_values = [mapped_data[f] for f in self.feature_names]
            print(f"📊 Input values: {input_values}")

            input_df = pd.DataFrame([input_values], columns=self.feature_names)
            print(f"📋 Input DataFrame shape: {input_df.shape}")

            print("🔄 Applying imputer...")
            input_imputed = self.imputer.transform(input_df)
            print(f"✅ Imputed data shape: {input_imputed.shape}")

            print("🔄 Applying scaler...")
            input_processed = self.scaler.transform(input_imputed)
            print(f"✅ Scaled data shape: {input_processed.shape}")

            print("🤖 Making prediction...")
            score = self.model.predict(input_processed)[0] 
            label = self.quality_label(score)

            print(f"🎉 Predicted score: {score}")
            print(f"🏷️ Label: {label}")

            result = {
                "success": True,
                "quality_score": round(float(score), 2),  
                "quality_label": label 
            } 

            print(f"📤 Returning result: {result}")
            return result

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "quality_score": None,
                "quality_label": None
            }