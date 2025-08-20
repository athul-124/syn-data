import os
import pandas as pd
import numpy as np
from backend.main import enhanced_synthetic_tabular, create_quality_report

def run_new_quality_test():
    """
    Tests the enhanced synthetic data generation with a more realistic dataset.
    """
    print("🔬 Starting new data quality test...")

    # Create a more realistic sample DataFrame
    data = {
        'age': np.random.randint(18, 70, size=100),
        'income': np.random.randint(20000, 150000, size=100),
        'loan_amount': np.random.randint(1000, 50000, size=100),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], size=100),
        'credit_score': np.random.randint(300, 850, size=100)
    }
    original_df = pd.DataFrame(data)

    # Generate synthetic data
    print("🔄 Generating synthetic data...")
    synthetic_df = enhanced_synthetic_tabular(original_df, n_rows=200)

    # Create quality report
    print("📊 Creating quality report...")
    quality_report = create_quality_report(original_df, synthetic_df)

    # --- Assertions ---
    print("✅ Running assertions...")

    # 1. Check for negative values in columns where they are not expected
    assert (synthetic_df['age'] >= 0).all(), "Negative values found in 'age'"
    assert (synthetic_df['income'] >= 0).all(), "Negative values found in 'income'"
    assert (synthetic_df['loan_amount'] >= 0).all(), "Negative values found in 'loan_amount'"
    assert (synthetic_df['credit_score'] >= 0).all(), "Negative values found in 'credit_score'"

    # 2. Check if integer columns remain integers
    assert pd.api.types.is_integer_dtype(synthetic_df['age']), "'age' column is not integer"
    assert pd.api.types.is_integer_dtype(synthetic_df['income']), "'income' column is not integer"
    assert pd.api.types.is_integer_dtype(synthetic_df['credit_score']), "'credit_score' column is not integer"

    # 3. Check if the quality score has improved
    overall_score = quality_report.get('overall_score', {}).get('overall_quality_score', 0)
    print(f"📈 Overall Quality Score: {overall_score}")
    assert overall_score > 0.7, f"Quality score {overall_score} is below the acceptable threshold of 0.7"

    print("🎉 Test passed successfully!")

if __name__ == "__main__":
    run_new_quality_test()
