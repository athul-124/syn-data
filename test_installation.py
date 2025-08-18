#!/usr/bin/env python3
"""
Test script to verify SynData Plus installation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from syndata_plus.generator.main import DataGenerator
        from syndata_plus.generator.schemas import CreditScoringSchema, FraudDetectionSchema
        
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_generator():
    """Test basic generator functionality"""
    try:
        from syndata_plus.generator.main import DataGenerator
        from syndata_plus.generator.schemas import CreditScoringSchema
        
        generator = DataGenerator()
        schema = CreditScoringSchema.get_schema()
        
        # Generate small test dataset
        data = generator.create_synthetic_data(
            schema=schema,
            num_rows=10,
            scenario="credit_scoring"
        )
        
        print(f"✅ Generated {len(data)} rows successfully")
        print(f"✅ Columns: {list(data.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Generator test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing SynData Plus installation...")
    
    tests = [
        ("Import Tests", test_imports),
        ("Generator Tests", test_generator)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        results.append(test_func())
    
    if all(results):
        print("\n🎉 All tests passed! SynData Plus is ready to use.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
