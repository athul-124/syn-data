import requests
import time

def check_backend_health():
    """Check if backend is running and responsive"""
    try:
        print("🔍 Checking backend health...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ Backend is running! Status: {response.status_code}")
        print(f"📋 Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Backend not running! Start with: uvicorn main:app --reload --port 8000")
        return False
    except requests.exceptions.Timeout:
        print("⏰ Backend is slow to respond")
        return False
    except Exception as e:
        print(f"❌ Backend check failed: {str(e)}")
        return False

if __name__ == "__main__":
    check_backend_health()