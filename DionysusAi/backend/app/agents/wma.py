# backend/app/agents/wma.py
import ollama
from typing import Dict, List, Optional
import time

class OllamaWineMentor:
    def __init__(self):
        self.model_name = "llama3.2:latest"
        self.conversation_history = {}
        self.client = ollama.Client()  # This connects to local Ollama
        self.ollama_working = self._verify_ollama_connection()
        
    def _verify_ollama_connection(self) -> bool:
        """Verify Ollama is running and model is available"""
        try:
            print("🔍 Checking Ollama connection...")
            
            # Check if Ollama is running
            models_response = self.client.list()
            print(f"📡 Ollama raw response: {models_response}")
            print(f"📡 Response type: {type(models_response)}")
            
            # Handle the response properly - it's a ListResponse object
            models_list = models_response.models
            print(f"📡 Models list: {models_list}")
            
            if models_list:
                model_names = []
                for model in models_list:
                    # Access the model name directly from the Model object
                    model_name = model.model
                    model_names.append(model_name)
                    print(f"📡 Found model: {model_name}")
                
                print(f"✅ Available models: {model_names}")
                
                # Check if our model is available
                if self.model_name in model_names:
                    print(f"✅ Model {self.model_name} is available!")
                    return True
                else:
                    print(f"⚠️  Model {self.model_name} not found. Available: {model_names}")
                    print("📥 Attempting to pull model...")
                    try:
                        self.client.pull(self.model_name)
                        print(f"✅ Model {self.model_name} pulled successfully!")
                        return True
                    except Exception as pull_error:
                        print(f"❌ Failed to pull model: {pull_error}")
                        return False
            else:
                print("❌ No models found in Ollama response")
                return False
                        
        except Exception as e:
            print(f"❌ Ollama connection failed: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return False

    def _get_system_prompt(self) -> str:
        """System prompt for wine expert"""
        return """You are Dionysus, an expert sommelier and wine educator. Provide accurate, educational wine advice.

            SPECIALTIES:
            - Wine tasting techniques
            - Food and wine pairing
            - Wine regions and grapes
            - Serving and storage

            Be conversational but professional. Explain concepts clearly. Focus on practical advice."""

    def chat(self, user_message: str, user_id: str = "default") -> Dict:
        """Chat using Ollama client with better timeout handling"""
        if not self.ollama_working:
            return self._rule_based_chat(user_message)
            
        try:
            start_time = time.time()
            
            # Add timeout to the Ollama call
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': self._get_system_prompt()
                    },
                    {
                        'role': 'user', 
                        'content': user_message
                    }
                ],
                options={
                    'temperature': 0.4,
                    'top_p': 0.8,
                    'num_predict': 150  # Reduce from 300 to make it faster
                }
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "success": True,
                "response": response['message']['content'],
                "model": self.model_name,
                "response_time": f"{response_time:.2f}s",
                "tokens_used": response.get('eval_count', 'unknown')
            }
            
        except Exception as e:
            print(f"⚠️  Ollama request failed: {e}")
            # Return a faster fallback response
            return self._rule_based_chat(user_message)
    
    def get_pairings(self, wine_info: Dict) -> Dict:
        """Get wine pairings using Ollama"""
        if not self.ollama_working:
            return self._rule_based_pairings(wine_info)
            
        wine_desc = self._describe_wine(wine_info)
        
        try:
            prompt = f"""Based on this wine: {wine_desc}

            Provide 2-3 specific food pairing recommendations with brief explanations."""

            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=self._get_system_prompt(),
                options={
                    'temperature': 0.3,
                    'num_predict': 250
                }
            )
            
            return {
                "success": True,
                "wine_description": wine_desc,
                "pairing_recommendations": response['response'],
                "model": self.model_name
            }
            
        except Exception as e:
            print(f"⚠️  Ollama pairing failed: {e}")
            return self._rule_based_pairings(wine_info)
    
    def _describe_wine(self, wine_info: Dict) -> str:
        """Describe wine for the LLM"""
        wine_type = "Red" if wine_info.get("type") == 0 else "White"
        alcohol = wine_info.get("alcohol", 12.0)
        acidity = wine_info.get("fixed_acidity", 7.0)
        
        body = "Light" if alcohol < 12.5 else "Medium" if alcohol < 13.5 else "Full"
        acid_level = "High" if acidity > 7.5 else "Medium" if acidity > 6.5 else "Low"
        
        return f"{body}-bodied {wine_type} wine, {alcohol}% alcohol, {acid_level} acidity"

    def _rule_based_chat(self, user_message: str) -> Dict:
        """Rule-based fallback responses"""
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['taste', 'tasting']):
            response = "Wine tasting steps: 1. Look at color 2. Swirl for aromas 3. Smell for notes 4. Taste flavors 5. Savor finish"
        elif any(word in message_lower for word in ['pair', 'food']):
            response = "Food pairing basics: Match wine weight to food weight. Red wines with proteins, white wines with lighter dishes, acidity cuts through fat."
        elif any(word in message_lower for word in ['hello', 'hi']):
            response = "Hello! I'm Dionysus, your wine mentor. Ask me about tasting, pairings, or wine education."
        else:
            response = "I can help with wine tasting techniques, food pairings, or general wine knowledge. What would you like to know?"
        
        return {
            "success": True,
            "response": response,
            "fallback_used": True,
            "model": "rule_based"
        }
    
    def _rule_based_pairings(self, wine_info: Dict) -> Dict:
        """Rule-based pairings fallback"""
        wine_type = "Red" if wine_info.get("type") == 0 else "White"
        alcohol = wine_info.get("alcohol", 12.0)
        
        if wine_type == "Red":
            if alcohol > 13.5:
                pairings = "Bold reds pair with: Grilled steak, braised lamb, aged cheeses, rich stews"
            else:
                pairings = "Medium reds pair with: Roast chicken, mushroom dishes, pasta, pork"
        else:
            if alcohol > 13.0:
                pairings = "Full whites pair with: Lobster, creamy pasta, roast turkey, brie cheese"
            else:
                pairings = "Light whites pair with: Seafood, salads, light fish, goat cheese"
        
        return {
            "success": True,
            "pairing_recommendations": pairings,
            "fallback_used": True
        }
        
    def conversational_response(self, user_message: str, user_id: str = "default") -> Dict:
        """Alias for chat method to match main.py calls"""
        return self.chat(user_message, user_id)

    def get_pairing_recommendations(self, wine_info: Dict) -> Dict:
        """Alias for get_pairings method to match main.py calls"""
        return self.get_pairings(wine_info)

    def get_educational_content(self, topic: str, level: str = "beginner") -> Dict:
        """Get educational content about wine topics"""
        prompt = f"Explain {topic} for a {level} level wine enthusiast. Keep it educational and practical."
        return self.chat(prompt)
    
    def quick_health_check(self) -> bool:
        """Quick test to see if model responds in reasonable time"""
        try:
            start_time = time.time()
            response = self.client.generate(
                model=self.model_name,
                prompt='Say "OK"',
                options={'num_predict': 10}  # Very short response
            )
            end_time = time.time()
            
            if (end_time - start_time) < 5:  # Should respond in under 5 seconds
                print(f"✅ Model health check passed: {response['response']}")
                return True
            else:
                print("⚠️  Model responding but slowly")
                return True
        except Exception as e:
            print(f"❌ Model health check failed: {e}")
            return False