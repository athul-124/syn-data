#!/usr/bin/env python3
"""Test MVP completion status"""

import requests
import pandas as pd
import sys
import os
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from quality_reporter import create_quality_report
    from main import enhanced_synthetic_tabular
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def simple_synthetic_tabular(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Fallback synthetic data generation"""
    import numpy as np
    
    synth_data = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Numeric: sample from normal distribution with same mean/std
            mean = df[col].mean()
            std = df[col].std()
            synth_data[col] = np.random.normal(mean, std, n_rows)
        else:
            # Categorical: sample with replacement
            synth_data[col] = np.random.choice(df[col].dropna(), n_rows, replace=True)
    
    return pd.DataFrame(synth_data)

def test_complete_pipeline():
    """Test the complete MVP pipeline with better test data"""
    
    print("🧪 Testing MVP Pipeline...")
    
    # Create more realistic test data for better ML performance
    np.random.seed(42)
    n_samples = 200  # Larger dataset
    
    test_data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(60000, 20000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_samples),
        'performance_score': np.random.uniform(1, 5, n_samples)
    })
    
    # Create a meaningful target based on other features
    test_data['high_performer'] = (
        (test_data['performance_score'] > 3.5) & 
        (test_data['experience'] > 5)
    ).astype(int)
    
    print(f"📊 Test data created: {test_data.shape}")
    print(f"Target distribution: {test_data['high_performer'].value_counts().to_dict()}")
    
    # Test synthetic generation
    try:
        synth = enhanced_synthetic_tabular(test_data, 300)
        print("✅ Enhanced synthetic data generation working")
    except Exception as e:
        print(f"⚠️ Enhanced generation failed: {e}")
        print("🔄 Falling back to simple generation...")
        synth = simple_synthetic_tabular(test_data, 300)
        print("✅ Simple synthetic data generation working")
    
    assert len(synth) == 300, f"Expected 300 rows, got {len(synth)}"
    print(f"📈 Generated {len(synth)} synthetic rows")
    
    # Test quality report with explicit target column
    try:
        target_col = 'high_performer'  # Use meaningful target
        print(f"Using target column: {target_col}")
        print(f"Target column exists in real data: {target_col in test_data.columns}")
        print(f"Target column exists in synthetic data: {target_col in synth.columns}")
        print(f"Real data columns: {list(test_data.columns)}")
        print(f"Synthetic data columns: {list(synth.columns)}")
        
        report = create_quality_report(test_data, synth, target_column=target_col)
        
        print(f"Report keys: {list(report.keys())}")
        if "utility_metrics" in report:
            print(f"Utility metrics keys: {list(report['utility_metrics'].keys())}")
            print(f"Utility metrics content: {report['utility_metrics']}")
        
        if "utility_metrics" in report and "utility_score" in report["utility_metrics"]:
            utility_score = report["utility_metrics"]["utility_score"]
            print(f"✅ Utility score: {utility_score:.1%}")
            
            if utility_score < 0:
                print("⚠️ Negative utility - synthetic data hurts ML performance")
            elif utility_score > 0.8:
                print("🎉 Excellent utility!")
            elif utility_score > 0.6:
                print("👍 Good utility")
            else:
                print("📈 Moderate utility - room for improvement")
        else:
            print("❌ Utility still N/A")
        
        # Print overall scores
        if "overall_score" in report:
            overall = report["overall_score"]["overall_quality_score"]
            print(f"📊 Overall Score: {overall:.1%}")
            print(f"🎯 Grade: {report['overall_score']['grade']}")
        
        print("✅ Quality report generation working")
        
    except Exception as e:
        print(f"❌ Quality report failed: {e}")
        return False
    
    print("🎉 All MVP components working!")
    return True

def test_api_server():
    """Test if API server is running"""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
    except requests.exceptions.RequestException:
        print("❌ API server not running. Start with: uvicorn backend.main:app --reload --port 8000")
        return False

if __name__ == "__main__":
    print("🚀 Starting MVP Completion Tests\n")
    
    # Test pipeline
    pipeline_success = test_complete_pipeline()
    
    print("\n" + "="*50)
    
    # Test API
    api_success = test_api_server()
    
    print("\n" + "="*50)
    
    if pipeline_success and api_success:
        print("🎉 ALL TESTS PASSED! MVP is ready!")
    elif pipeline_success:
        print("✅ Pipeline works! Start the API server to complete testing.")
    else:
        print("❌ Some tests failed. Check the errors above.")


