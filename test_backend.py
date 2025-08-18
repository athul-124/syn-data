import requests

try:
    response = requests.get('http://localhost:8000/health')
    print(f"Backend status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Backend not reachable: {e}")