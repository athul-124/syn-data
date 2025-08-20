import requests

# Check available endpoints
try:
    response = requests.get("http://localhost:8000/docs")
    print("✅ FastAPI docs available at http://localhost:8000/docs")
except:
    print("❌ Cannot reach API docs")

# Test health endpoint
try:
    response = requests.get("http://localhost:8000/health")
    print(f"Health check: {response.json()}")
except Exception as e:
    print(f"Health check failed: {e}")

# Check if async generation endpoint exists
try:
    response = requests.get("http://localhost:8000/generate")
    print(f"Generate endpoint status: {response.status_code}")
except Exception as e:
    print(f"Generate endpoint test failed: {e}")