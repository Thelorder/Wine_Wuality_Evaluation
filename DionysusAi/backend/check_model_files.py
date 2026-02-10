from app.agents.wma import OllamaWineMentor

print("🧪 STEP 1: Initializing the Mentor Class...")
try:
    mentor = OllamaWineMentor()
    
    if mentor.ollama_working:
        print("\n🧪 STEP 2: Testing Chat Response...")
        chat = mentor.conversational_response("What is a Merlot?")
        print(f"Result: {chat}")
        
        print("\n🧪 STEP 3: Testing Food Pairing...")
        fake_wine = {"type": 0, "alcohol": 14.5, "pH": 3.2, "sulphates": 0.5}
        pairing = mentor.get_pairing_recommendations(fake_wine)
        print(f"Result: {pairing}")
    else:
        print("\n❌ CRITICAL: Mentor initialized but 'ollama_working' is False. Check the debug prints above.")

except Exception as e:
    print(f"\n❌ CRITICAL CRASH: {e}")