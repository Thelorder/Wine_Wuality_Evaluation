# backend/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, List,Dict
import uvicorn
import os
import pandas as pd 
import numpy as np
import urllib.parse

from app.agents.qpa import QualityPredictionAgent
from app.models import DatabaseManager, WineSample
from app.agents.wma import OllamaWineMentor
from app.agents.recommender import LLMEnhancedRecommender

# Initialize QPA agent
qpa_agent = QualityPredictionAgent()
db_manager = DatabaseManager()
wma_agent = OllamaWineMentor()
llm_recommender = LLMEnhancedRecommender(llm_agent=wma_agent)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting DionysusAI...")
    base_dir = r"C:\Users\User\Desktop\Python Code\DionysusAi\ml\models"
    
    print(f"🔍 Looking for models in: {base_dir}")
    
    model_loaded = qpa_agent.load_model(
        model_path=os.path.join(base_dir, "random_forest_model.pkl"),
        imputer_path=os.path.join(base_dir, "imputer.pkl"), 
        scaler_path=os.path.join(base_dir, "scaler.pkl"),
        features_path=os.path.join(base_dir, "feature_names.pkl")
    )
    
    if model_loaded:
        print("🎉 QPA agent ready for predictions!")
    else:
        print("⚠️  Running without ML model - some features disabled")
    
    print(f"📚 Recommender loaded: {len(llm_recommender.wine_df) if llm_recommender.wine_df is not None else 0} wines")
    yield  # The application runs here
    
    # Shutdown (if needed)
    print("🛑 Shutting down DionysusAI...")

app = FastAPI(
    title="DionysusAI Wine Quality API",
    description="Intelligent Wine Quality Assessment System",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for API
class WineFeatures(BaseModel):
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

class PredictionResponse(BaseModel):
    quality_prediction: str
    quality_score: int
    confidence: float
    probabilities: dict
    success: bool

class WinePreferences(BaseModel):
    wine_type: Optional[str] = None
    country: Optional[str] = None
    max_price: Optional[float] = None
    grape: Optional[str] = None
    style: Optional[str] = None

class RecommendationRequest(BaseModel):
    query: Optional[str] = None
    preferences: Optional[WinePreferences] = None
    limit: int = 5

@app.get("/")
async def root():
    return {
        "message": "DionysusAI Wine Quality Assessment System",
        "status": "operational",
        "qpa_loaded": qpa_agent.is_trained
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "qpa_agent_ready": qpa_agent.is_trained,
        "endpoints_available": ["/api/predict", "/api/health", "/api/samples"]
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_wine_quality(wine: WineFeatures):
    """Endpoint for wine quality prediction without saving to database"""
    if not qpa_agent.is_trained:
        raise HTTPException(status_code=503, detail="QPA agent not ready")
    
    try:
        # Convert to dict for the agent
        wine_dict = wine.model_dump()
        result = qpa_agent.predict_quality(wine_dict)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/samples", response_model=dict)
async def create_sample(wine: WineFeatures):
    """Analyze a wine sample and save it to database"""
    if not qpa_agent.is_trained:
        raise HTTPException(status_code=503, detail="QPA agent not ready")
    
    try:
        # Get prediction from QPA agent
        wine_dict = wine.dict()
        prediction_result = qpa_agent.predict_quality(wine_dict)
        
        if not prediction_result["success"]:
            raise HTTPException(status_code=400, detail=prediction_result.get("error", "Prediction failed"))
        
        # Create sample object
        sample = WineSample(
            **wine_dict,
            quality_prediction=prediction_result["quality_prediction"],
            quality_score=prediction_result["quality_score"],
            confidence=prediction_result["confidence"],
            probabilities=prediction_result["probabilities"]
        )
        
        # Save to database
        sample_id = db_manager.save_sample(sample)
        
        return {
            "sample_id": sample_id,
            "message": "Sample analyzed and saved successfully",
            "prediction": prediction_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample processing failed: {str(e)}")

@app.get("/api/samples", response_model=dict)
async def get_all_samples():
    """Get all analyzed wine samples"""
    try:
        samples = db_manager.get_all_samples()
        return {
            "count": len(samples),
            "samples": [sample.model_dump() for sample in samples]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve samples: {str(e)}")

@app.get("/api/samples/{sample_id}", response_model=WineSample)
async def get_sample(sample_id: int):
    """Get a specific wine sample by ID"""
    sample = db_manager.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample

@app.get("/api/analysis")
async def get_analysis():
    """Get analysis statistics"""
    try:
        samples = db_manager.get_all_samples()
        
        if not samples:
            return {"message": "No samples available for analysis"}
        
        # Basic statistics
        total_samples = len(samples)
        high_quality_count = sum(1 for s in samples if s.quality_score == 1)
        avg_confidence = sum(s.confidence for s in samples) / total_samples
        
        # Wine type distribution
        white_wines = sum(1 for s in samples if s.type == 1)
        red_wines = total_samples - white_wines
        
        return {
            "total_samples": total_samples,
            "high_quality_samples": high_quality_count,
            "standard_quality_samples": total_samples - high_quality_count,
            "high_quality_percentage": (high_quality_count / total_samples) * 100,
            "average_confidence": avg_confidence,
            "white_wines": white_wines,
            "red_wines": red_wines,
            "recent_samples": [s.model_dump() for s in samples[:5]]  
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
@app.post("/api/pairings/recommend")
async def get_pairing_recommendations(wine: WineFeatures):
    """Get food pairing recommendations based on wine characteristics"""
    try:
        wine_dict = wine.model_dump()
        result = wma_agent.get_pairing_recommendations(wine_dict)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Pairing generation failed"))
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pairing recommendation failed: {str(e)}")

@app.get("/api/education/{topic}")
async def get_education(topic: str, level: str = "beginner"):
    """Get educational content about wine topics"""
    try:
        result = wma_agent.get_educational_content(topic, level)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail="Educational content not available")
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Education retrieval failed: {str(e)}")

@app.get("/api/education/topics")
async def get_education_topics():
    """Get available educational topics"""
    return {
        "topics": [
            "wine_tasting_basics",
            "food_pairing_fundamentals", 
            "understanding_wine_labels",
            "wine_regions_overview",
            "serving_temperature_guide",
            "wine_storage_tips"
        ],
        "levels": ["beginner", "intermediate", "advanced"]
    }
    
@app.post("/api/mentor/chat")
async def mentor_chat(message: dict):
    """Conversational interface with the Wine Mentor"""
    try:
        user_message = message.get("message", "")
        user_id = message.get("user_id", "default")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        result = wma_agent.conversational_response(user_message, user_id)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail="Failed to generate response")
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/api/pairings/llm")
async def get_llm_pairings(wine: WineFeatures, preferences: str = ""):
    """Get LLM-generated pairing recommendations"""
    try:
        wine_dict = wine.model_dump()
        result = wma_agent.get_pairing_recommendations(wine_dict)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail="Pairing generation failed")
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM pairing failed: {str(e)}")

@app.get("/api/mentor/status")
async def get_mentor_status():
    """Check if LLM is available"""
    return {
        "ollama_available": wma_agent.ollama_working,
        "model": wma_agent.model_name,
        "features": ["conversational_chat", "llm_pairings", "educational_content"]
    }   
    
@app.post("/api/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Get wine recommendations based on preferences or natural language query"""
    try:
        preferences_dict = request.preferences.model_dump() if request.preferences else {}
        
        if request.query:
          
            result = llm_recommender.recommend_with_explanation(
                user_query=request.query,
                preferences=preferences_dict
            )
        else:
            wines = llm_recommender.search_by_preferences(preferences_dict)
            result = {
                "success": True,
                "preferences": preferences_dict,
                "wines_found": len(wines),
                "wines": wines[:request.limit]
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/api/wines/search")
async def search_wines(q: str, limit: int = 10):
    """Search wines by name, grape, or region"""
    try:
        results = []
        if llm_recommender.wine_df is not None and not llm_recommender.wine_df.empty:
            search_df = llm_recommender.wine_df
            
            # Search in relevant columns
            search_columns = ['title', 'grape', 'region', 'country', 'style']
            available_cols = [c for c in search_columns if c in search_df.columns]
            
            # Combine all matches
            all_matches = pd.DataFrame()
            for col in available_cols:
                # Handle NaN values before searching
                col_data = search_df[col].fillna('')
                matches = search_df[col_data.astype(str).str.contains(
                    q, case=False, na=False
                )]
                all_matches = pd.concat([all_matches, matches])
            
            # Remove duplicates
            all_matches = all_matches.drop_duplicates(subset=['title'] if 'title' in all_matches.columns else None)
            
            # Convert to clean dicts
            for _, row in all_matches.iterrows():
                clean_wine = {}
                for col_name, value in row.items():
                    if not pd.isna(value):
                        # Convert numpy/pandas types to Python types
                        if isinstance(value, (np.integer, np.int64)):
                            clean_wine[col_name] = int(value)
                        elif isinstance(value, (np.floating, np.float64)):
                            clean_wine[col_name] = float(value)
                        else:
                            clean_wine[col_name] = str(value)
                results.append(clean_wine)
        
        return {
            "query": q,
            "results": results[:limit],
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/wines/{wine_name:path}")
async def get_wine_details(wine_name: str):
    """Get details about a specific wine - using :path to capture everything including slashes"""
    try:
        # Decode URL-encoded wine name
        decoded_name = urllib.parse.unquote(wine_name)
        print(f"🔍 API endpoint looking up wine: '{decoded_name}'")
        
        wine = llm_recommender.get_wine_by_name(decoded_name)
        
        if not wine:
            raise HTTPException(status_code=404, detail=f"Wine '{decoded_name}' not found")
        
        # The wine dict is already cleaned by get_wine_by_name
        return wine
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in get_wine_details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get wine details: {str(e)}")
    
@app.post("/api/pairing/explain")
async def explain_pairing(wine_name: str, meal: str):
    """Explain why a wine pairs with a meal"""
    try:
        # Decode URL parameters
        decoded_wine_name = urllib.parse.unquote(wine_name)
        decoded_meal = urllib.parse.unquote(meal)
        
        result = llm_recommender.explain_wine_pairing(decoded_wine_name, decoded_meal)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        # FIX: Clean NaN values from the result
        def clean_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            else:
                return obj
        
        cleaned_result = clean_nan(result)
        return cleaned_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to explain pairing: {str(e)}")

@app.get("/api/recommender/status")
async def get_recommender_status():
    """Check recommender system status"""
    return {
        "database_loaded": llm_recommender.wine_df is not None and not llm_recommender.wine_df.empty,
        "wine_count": len(llm_recommender.wine_df) if llm_recommender.wine_df is not None else 0,
        "llm_connected": llm_recommender.llm is not None and llm_recommender.llm.ollama_working,
        "features": ["natural_language_recommendations", "preference_search", "wine_lookup", "pairing_explanations"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)