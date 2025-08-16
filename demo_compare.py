# demo/demo_compare.py
# run locally to compare model performance quickly
import requests

GEN_URL = "http://localhost:8000/generate"
DEMO_URL = "http://localhost:8000/demo-train"

# Use curl or requests to POST a CSV and get back summary / train demo.
# Example usage with requests:
files = {'file': open('samples/your_dataset.csv','rb')}
data = {'n_rows': 1000}
r = requests.post(GEN_URL, files=files, data=data)
print(r.json())
