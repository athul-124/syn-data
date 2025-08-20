import requests

with open('financial_test_data.csv', 'rb') as f:
    files = {'file': f}
    data = {'rows': 100}  # Changed from 'n_rows' to 'rows'
    
    response = requests.post("http://localhost:8000/generate-synthetic", files=files, data=data)
    
    if response.status_code == 200:
        # This endpoint returns the CSV directly, not a JSON with download_url
        with open('downloaded_synthetic.csv', 'wb') as output_file:
            output_file.write(response.content)
        print("✅ File downloaded successfully as 'downloaded_synthetic.csv'")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

