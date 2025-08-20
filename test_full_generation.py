import requests
import json

# Test the async generation endpoint that provides quality reports
with open('financial_test_data.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'n_rows': 100,
        'target_column': 'target'  # Adjust based on your data
    }
    
    # Use the async generation endpoint
    response = requests.post("http://localhost:8000/generate", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Generation started successfully!")
        print(f"Task ID: {result.get('task_id')}")
        
        # Check task status
        task_id = result.get('task_id')
        if task_id:
            status_response = requests.get(f"http://localhost:8000/status/{task_id}")
            status_data = status_response.json()
            
            print(f"Status: {status_data.get('status')}")
            print(f"Progress: {status_data.get('progress', 0)}%")
            
            # If completed, get the quality report
            if status_data.get('status') == 'completed':
                print("\n📊 Quality Report:")
                quality_report = status_data.get('quality_report', {})
                
                if 'overall_score' in quality_report:
                    overall = quality_report['overall_score']
                    print(f"Overall Score: {overall.get('overall_quality_score', 'N/A')}")
                    print(f"Grade: {overall.get('grade', 'N/A')}")
                
                # Save quality report
                with open('quality_report.json', 'w') as f:
                    json.dump(quality_report, f, indent=2)
                print("✅ Quality report saved to quality_report.json")
                
                # Download the file if available
                if 'output_file' in status_data:
                    download_url = f"/download/{status_data['output_file'].split('/')[-1]}"
                    download_response = requests.get(f"http://localhost:8000{download_url}")
                    
                    if download_response.status_code == 200:
                        with open('synthetic_with_report.csv', 'wb') as f:
                            f.write(download_response.content)
                        print("✅ Synthetic data downloaded as synthetic_with_report.csv")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")