"""
Test script for enhanced quality reporting
"""
import pandas as pd
import numpy as np
from backend.quality_reporter import SynDataQualityReporter

def test_enhanced_quality_reporter():
    """Test the enhanced quality reporter with sample data"""
    
    print("🧪 Testing Enhanced Quality Reporter...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    real_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Create synthetic data (slightly different)
    synthetic_data = pd.DataFrame({
        'age': np.random.normal(36, 9, n_samples),
        'income': np.random.normal(51000, 14000, n_samples),
        'score': np.random.uniform(5, 95, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.4, 0.4, 0.2]),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    
    print(f"📊 Real data shape: {real_data.shape}")
    print(f"📊 Synthetic data shape: {synthetic_data.shape}")
    
    # Test the enhanced quality reporter
    try:
        reporter = SynDataQualityReporter()
        
        print("🔍 Generating comprehensive report...")
        report = reporter.generate_comprehensive_report(
            real_data=real_data,
            synthetic_data=synthetic_data,
            target_column='target',
            generate_visuals=False  # Skip visuals for testing
        )
        
        print("✅ Report generated successfully!")
        print(f"📋 Report keys: {list(report.keys())}")
        
        # Print key metrics
        if "overall_score" in report:
            overall = report["overall_score"]
            print(f"🎯 Overall Score: {overall['overall_quality_score']:.3f} ({overall['grade']})")
        
        if "fidelity_metrics" in report and "summary_scores" in report["fidelity_metrics"]:
            fidelity = report["fidelity_metrics"]["summary_scores"]
            print(f"📊 Fidelity Score: {fidelity['overall_fidelity']:.3f}")
            print(f"📊 Wasserstein Score: {fidelity.get('wasserstein_fidelity', 'N/A')}")
            print(f"📊 KS Score: {fidelity.get('kolmogorov_smirnov_fidelity', 'N/A')}")
        
        if "utility_metrics" in report:
            utility = report["utility_metrics"]
            print(f"🤖 Utility Score: {utility.get('utility_score', 'N/A')}")
        
        if "privacy_metrics" in report:
            privacy = report["privacy_metrics"]
            print(f"🔒 Privacy Score: {privacy.get('privacy_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_quality_reporter()
    if success:
        print("🎉 Enhanced Quality Reporter test PASSED!")
    else:
        print("💥 Enhanced Quality Reporter test FAILED!")