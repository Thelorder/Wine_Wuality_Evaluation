# test_dionysus.py - Quick test script for your full DionysusAI system

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("🩺 Testing health check...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(json.dumps(response.json(), indent=2))
    print()

def test_recommender_simple():
    print("🍷 Testing basic recommendation...")
    payload = {
        "query": "I want a crisp, refreshing white wine for summer evenings",
        "limit": 3
    }
    response = requests.post(f"{BASE_URL}/api/recommend", json=payload)
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Query: {result['query']}")
        print("\n🤵 Sommelier says:")
        print(result['explanation'])
        print("\n🍾 Top recommendations:")
        for wine in result['recommendations']:
            print(f"- {wine['title']} ({wine['grape']} from {wine['region']}) - ${wine.get('price', '??')}")
            print(f"  {wine['description'][:150]}...\n")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_recommender_with_filters():
    print("🔍 Testing with price & grape filter...")
    payload = {
        "query": "Something bold and fruity for barbecue",
        "preferences": {
            "max_price": 40,
            "grape": "Cabernet Sauvignon"
        },
        "limit": 2
    }
    response = requests.post(f"{BASE_URL}/api/recommend", json=payload)
    print(json.dumps(response.json(), indent=2))

def test_search():
    print("\n🔎 Testing wine search...")
    response = requests.get(f"{BASE_URL}/api/wines/search?q=Chardonnay&limit=3")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("🚀 DionysusAI Full System Test\n")
    test_health()
    test_recommender_simple()
    # test_recommender_with_filters()  # Uncomment to test filters
    # test_search()                     # Uncomment to test search