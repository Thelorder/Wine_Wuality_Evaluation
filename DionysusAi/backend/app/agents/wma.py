import ollama
import json
import requests
from typing import Dict, List, Optional

class OllamaWineMentor:
    def __init__(self):
        self.model_name = "llama3.2:latest"
        self.host = 'http://127.0.0.1:11434'
        self.ollama_working = False
        self.client = ollama.Client(host=self.host)

        self._initialize_connection()

    def _initialize_connection(self):
        print("\n" + "="*50)
        print("🔍 DEBUG: Starting Ollama Connection Check...")
        try:
            print(f"📡 DEBUG: Pinging Ollama Server at {self.host}...")
            ping = requests.get(self.host)
            print(f"✅ DEBUG: Server status code: {ping.status_code}")

            print("🔄 DEBUG: Fetching model list from Ollama...")
            response = self.client.list()
            
            available_models = [m['model'] for m in response.get('models', [])]
            print(f"📦 DEBUG: Found Models: {available_models}")

            if not available_models:
                print("❌ DEBUG: NO MODELS FOUND. You need to run 'ollama pull llama3.2'")
                return

            if self.model_name not in available_models:
                print(f"⚠️ DEBUG: '{self.model_name}' missing. Defaulting to '{available_models[0]}'")
                self.model_name = available_models[0]

            print(f"🧪 DEBUG: Testing chat with {self.model_name}...")
            test = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'hi'}],
                options={'num_predict': 5}
            )
            print(f"🎉 DEBUG: Ollama is alive! Response: {test['message']['content']}")
            self.ollama_working = True
            
        except Exception as e:
            print(f"❌ DEBUG ERROR during initialization: {str(e)}")
            self.ollama_working = False
        print("="*50 + "\n")

    def _analyze_chemistry(self, wine: Dict) -> Dict:
        alc = wine.get("alcohol", 11.0)
        body = "Light-bodied" if alc < 12.5 else "Medium-bodied" if alc < 13.5 else "Full-bodied"
        
        ph = wine.get("pH", 3.2)
        acidity = "High Acidity (Crisp)" if ph < 3.3 else "Medium Acidity" if ph < 3.6 else "Low Acidity (Smooth)"
        
        sulph = wine.get("sulphates", 0.6)
        finish = "Clean Finish" if sulph < 0.8 else "Sharp/Mineral Finish"
        
        return {"body": body, "acidity": acidity, "finish": finish}

    def get_pairing_recommendations(self, wine_info: Dict) -> Dict:
        print("🍳 DEBUG: get_pairing_recommendations CALLED")
        if not self.ollama_working:
            return {"success": False, "error": "Ollama is not connected."}

        profile = self._analyze_chemistry(wine_info)
        wine_type = "White" if wine_info.get("type") == 1 else "Red"

        user_prompt = f"Suggest 3 food pairings for a {wine_type} wine with {profile['body']} and {profile['acidity']}."
        
        try:
            print(f"📤 DEBUG: Sending pairing request to Ollama ({self.model_name})...")
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': "You are a Michelin Star Sommelier. Return JSON only with keys: analysis, pairings (list), serving_temp."},
                    {'role': 'user', 'content': user_prompt}
                ],
                format="json"
            )
            print("📥 DEBUG: Received response from Ollama")
            result = json.loads(response['message']['content'])
            return {"success": True, **profile, **result}
        except Exception as e:
            print(f"❌ DEBUG: Pairing failed: {e}")
            return {"success": False, "error": str(e)}

    def conversational_response(self, user_message: str, user_id: str = "default") -> Dict:
        print(f"💬 DEBUG: conversational_response CALLED with message: {user_message}")
        if not self.ollama_working:
            return {"success": False, "error": "Ollama is not connected."}

        try:
            print(f"📤 DEBUG: Sending chat to Ollama...")
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': "You are Dionysus. Short, witty wine advice only."},
                    {'role': 'user', 'content': user_message}
                ]
            )
            print("📥 DEBUG: Chat response received")
            return {"success": True, "response": response['message']['content']}
        except Exception as e:
            print(f"❌ DEBUG: Chat failed: {e}")
            return {"success": False, "error": str(e)}

    def get_educational_content(self, topic, level):
        return self.conversational_response(f"Explain {topic} to a {level}.")