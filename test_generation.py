import requests
import json

# Test preview endpoint
def test_preview():
    try:
        response = requests.post("http://localhost:8000/preview", 
            json={
                "file_id": "3faa9015",  # Use your uploaded file ID
                "n_rows": 10
            }
        )
        print(f"Preview Status: {response.status_code}")
        print(f"Preview Response: {response.json()}")
    except Exception as e:
        print(f"Preview Error: {e}")

# Test async generation
def test_generation():
    try:
        data = {
            "file_id": "3faa9015",
            "n_rows": "100",
            "target_column": ""
        }
        response = requests.post("http://localhost:8000/generate-async", data=data)
        print(f"Generation Status: {response.status_code}")
        result = response.json()
        print(f"Generation Response: {result}")
        
        if result.get("task_id"):
            # Check task status
            task_id = result["task_id"]
            status_response = requests.get(f"http://localhost:8000/tasks/{task_id}/status")
            print(f"Task Status: {status_response.json()}")
            
    except Exception as e:
        print(f"Generation Error: {e}")

# Add this function to test task status
def test_task_status(task_id):
    try:
        response = requests.get(f"http://localhost:8000/tasks/{task_id}/status")
        print(f"Task Status Response: {response.json()}")
    except Exception as e:
        print(f"Task Status Error: {e}")

if __name__ == "__main__":
    print("Testing generation functionality...")
    test_preview()
    print("\n" + "="*50 + "\n")
    test_generation()
    # Use the task ID from your generation
    test_task_status("0ed31b06-02ec-469d-b61e-6c13c070afd0")
