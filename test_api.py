#!/usr/bin/env python3
"""Test SynData API with realistic datasets"""

import requests
import pandas as pd
import os

def test_with_financial_data():
    """Test API with financial dataset"""
    print("🏦 Testing with Financial Dataset...")
    
    # Create the dataset if it doesn't exist
    if not os.path.exists('financial_test_data.csv'):
        print("Creating financial test dataset...")
        exec(open('test_financial_dataset.py').read())
    
    try:
        with open('financial_test_data.csv', 'rb') as f:
            files = {'file': f}
            data = {
                'n_rows': 500,
                'target_column': 'loan_approved'  # Specify target for utility
            }
            
            response = requests.post("http://localhost:8000/generate", files=files, data=data)
            result = response.json()
            
            if result.get('success'):
                print("✅ Financial data generation successful!")
                
                # Print detailed quality metrics
                if 'quality_report' in result:
                    report = result['quality_report']
                    
                    if 'overall_score' in report:
                        overall = report['overall_score']['overall_quality_score']
                        grade = report['overall_score']['grade']
                        print(f"📊 Overall Score: {overall:.1%} ({grade})")
                    
                    if 'fidelity_metrics' in report:
                        fidelity = report['fidelity_metrics']['summary_scores']['overall_fidelity']
                        print(f"🎯 Fidelity: {fidelity:.1%}")
                    
                    if 'utility_metrics' in report and 'utility_score' in report['utility_metrics']:
                        utility = report['utility_metrics']['utility_score']
                        task_type = report['utility_metrics'].get('task_type', 'unknown')
                        print(f"🤖 Utility: {utility:.1%} ({task_type})")
                    else:
                        print("❌ Utility: N/A")
                    
                    if 'privacy_metrics' in report:
                        privacy = report['privacy_metrics']['privacy_score']
                        print(f"🔒 Privacy: {privacy:.1%}")
                
                print(f"📁 Download URL: {result['download_url']}")
            else:
                print(f"❌ Generation failed: {result}")
                
    except Exception as e:
        print(f"❌ Error testing financial data: {e}")

def test_with_customer_data():
    """Test API with customer churn dataset"""
    print("\n👥 Testing with Customer Dataset...")
    
    try:
        with open('customer_test_data.csv', 'rb') as f:
            files = {'file': f}
            data = {
                'n_rows': 400,
                'target_column': 'churned'  # Specify target for utility
            }
            
            response = requests.post("http://localhost:8000/generate", files=files, data=data)
            result = response.json()
            
            if result.get('success'):
                print("✅ Customer data generation successful!")
                
                if 'quality_report' in result:
                    report = result['quality_report']
                    
                    if 'overall_score' in report:
                        overall = report['overall_score']['overall_quality_score']
                        print(f"📊 Overall Score: {overall:.1%}")
                    
                    if 'utility_metrics' in report and 'utility_score' in report['utility_metrics']:
                        utility = report['utility_metrics']['utility_score']
                        print(f"🤖 Utility: {utility:.1%}")
                    else:
                        print("❌ Utility: N/A")
                
            else:
                print(f"❌ Generation failed: {result}")
                
    except Exception as e:
        print(f"❌ Error testing customer data: {e}")

def test_api():
    print("🧪 Testing SynData API with Realistic Datasets...")
    
    # Health check
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Health check: {response.json()}")
    except:
        print("❌ API server not running. Start with: uvicorn main:app --reload --port 8000")
        return
    
    # Test with different datasets
    test_with_financial_data()
    test_with_customer_data()

if __name__ == "__main__":
    test_api()
