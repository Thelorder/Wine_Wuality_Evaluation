def test_quick_chat():
    print("🧪 Testing quick chat...")
    
    chat_data = {
        "message": "Say hello",
        "user_id": "quick_test"
    }
    
    try:
        response = requests.post("http://localhost:8000/api/mentor/chat", json=chat_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Quick chat worked! Response: {result['response'][:100]}...")
        else:
            print(f"❌ Quick chat failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Quick chat error: {e}")

# Call this before your main test
test_quick_chat()