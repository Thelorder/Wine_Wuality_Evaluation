# app/agents/recommender.py

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import re
import requests
from typing import List, Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "WineDataset.csv")

class LLMEnhancedRecommender:
    def __init__(self, llm_agent=None, dataset_path: str = None):
        self.dataset_path = dataset_path or data_path
        self.llm_agent = llm_agent 
        self.wine_df = pd.read_csv(self.dataset_path)
        self.wine_df.columns = [col.strip().lower() for col in self.wine_df.columns]

        self.client = chromadb.EphemeralClient()
        
        self.embedding_func = embedding_functions.OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url="http://localhost:11434/api/embeddings",
            timeout = 300
        )

        self.collection = self.client.get_or_create_collection(
            name="wine_knowledge",
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0: 
            self._build_database()

    def _clean_price(self, price_str):
        if pd.isna(price_str):
            return 99.0
        match = re.search(r'(\d+\.?\d*)', str(price_str))
        return float(match.group(1)) if match else 99.0

    def _build_database(self):
        print("🍷 Building rich wine knowledge base with nomic embeddings...")
        documents, metadatas, ids = [], [], []

        for idx, row in self.wine_df.iterrows():
            title = row.get('title', 'Unknown Wine')
            desc = row.get('description', '')
            grape = row.get('grape', 'Unknown')
            region = row.get('region', 'Unknown')
            country = row.get('country', 'Unknown')
            style = row.get('style', '')

    
            text = f"{title}. {desc} Grape: {grape}. Region: {region}, {country}. Style: {style}."
            documents.append(text.strip())

            metadatas.append({
                "title": str(title),
                "grape": str(grape),
                "region": str(region),
                "country": str(country),
                "style": str(style),
                "price": self._clean_price(row.get('price')),
                "description": str(desc)
            })
            ids.append(str(idx))

        BATCH_SIZE = 50
        total = len(documents)
        for i in range(0, total, BATCH_SIZE):
            end = min(i + BATCH_SIZE, total)
            self.collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
            print(f"   Embedded {end}/{total} wines...")

        print(f"✅ Wine RAG database ready with {total} wines!")

    def _raw_llm_call(self, prompt: str) -> str:
        """Fallback if no llm_agent provided"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False},
                timeout=120
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    def recommend_with_explanation(
        self,
        user_query: str,
        preferences: Optional[Dict] = None,
        limit: int = 3
    ) -> Dict:
        """Main method used by /api/recommend"""

        where = {}
        if preferences:
            if preferences.get("country"):
                where["country"] = {"$eq": preferences["country"]}
            if preferences.get("grape"):
                where["grape"] = {"$eq": preferences["grape"]}
            if preferences.get("max_price"):
                where["price"] = {"$lte": float(preferences["max_price"])}

        results = self.collection.query(
            query_texts=[user_query],
            n_results=limit + 1,  
            where=where if where else None
        )

        wines_found = results['documents'][0]
        if not wines_found:
            return {"success": False, "error": "No wines matched your request."}

        wine_list = "\n".join([f"{i+1}. {wine}" for i, wine in enumerate(wines_found[:limit])])

        prompt = f"""You are a world-class sommelier with deep wine knowledge.

The customer said: "{user_query}"

Here are the best matching wines from your cellar:
{wine_list}

Recommend the top choices, explain why they fit perfectly, and give one tasting tip for the #1 recommendation.
Be warm, engaging, and concise (around 2 paragraphs max)."""

        if self.llm_agent and hasattr(self.llm_agent, 'get_response'):
            llm_response = self.llm_agent.get_response(prompt)
            explanation = llm_response.get("response", llm_response.get("text", "Great choices!"))
        else:
            explanation = self._raw_llm_call(prompt)

        recommendations = []
        for meta, doc in zip(results['metadatas'][0][:limit], wines_found[:limit]):
            recommendations.append({
                "title": meta.get("title", "Unknown"),
                "grape": meta.get("grape"),
                "region": meta.get("region"),
                "country": meta.get("country"),
                "price": meta.get("price"),
                "description": meta.get("description", "")[:300] + "..." if meta.get("description") else "",
                "full_context": doc  
            })

        return {
            "success": True,
            "query": user_query,
            "explanation": explanation.strip(),
            "recommendations": recommendations,
            "count": len(recommendations)
        }

    def get_wine_by_name(self, name: str):
        matches = self.wine_df[self.wine_df['title'].str.contains(name, case=False, na=False)]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return {k: v for k, v in row.items() if pd.notna(v)}