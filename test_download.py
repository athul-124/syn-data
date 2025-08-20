import requests

# Replace with actual filename from your generation response
filename = "synthetic_data_12345.csv"  # Use the actual filename
response = requests.get(f"http://localhost:8000/download/{filename}")

print(f"Download status: {response.status_code}")
if response.status_code == 200:
    with open(f"downloaded_{filename}", "wb") as f:
        f.write(response.content)
    print(f"File saved as downloaded_{filename}")
else:
    print(f"Error: {response.text}")
