#!/usr/bin/env python3
"""Create realistic financial dataset for testing SynData"""

import pandas as pd
import numpy as np
import json

def create_financial_dataset():
    """Create a realistic financial dataset with clear patterns for ML"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated features that make sense for credit scoring
    data = {}
    
    # Demographics
    data['age'] = np.random.normal(40, 12, n_samples).clip(18, 80).astype(int)
    data['income'] = np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 500000).astype(int)
    
    # Credit history (correlated with age and income)
    data['credit_history_years'] = (data['age'] - 18 + np.random.normal(0, 3, n_samples)).clip(0, 50).astype(int)
    data['existing_loans'] = np.random.poisson(2, n_samples).clip(0, 10)
    
    # Employment
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
    employment_probs = [0.6, 0.2, 0.15, 0.05]
    data['employment_type'] = np.random.choice(employment_types, n_samples, p=employment_probs)
    
    # Loan details
    data['loan_amount'] = np.random.normal(50000, 25000, n_samples).clip(5000, 200000).astype(int)
    data['loan_term_months'] = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    
    # Debt-to-income ratio (key feature)
    monthly_income = data['income'] / 12
    monthly_payment = data['loan_amount'] / data['loan_term_months']
    data['debt_to_income_ratio'] = (monthly_payment / monthly_income).clip(0, 2)
    
    # Credit score (correlated with other features)
    base_score = 600 + (data['income'] / 1000) + (data['credit_history_years'] * 5)
    base_score -= (data['existing_loans'] * 20) + (data['debt_to_income_ratio'] * 100)
    data['credit_score'] = (base_score + np.random.normal(0, 50, n_samples)).clip(300, 850).astype(int)
    
    # Target: Loan approval (based on logical rules)
    approval_prob = (
        (data['credit_score'] > 650) * 0.4 +
        (data['debt_to_income_ratio'] < 0.4) * 0.3 +
        (data['income'] > 50000) * 0.2 +
        (data['employment_type'] == 'Full-time') * 0.1
    )
    
    data['loan_approved'] = (np.random.random(n_samples) < approval_prob).astype(int)
    
    df = pd.DataFrame(data)
    
    print(f"📊 Created financial dataset: {df.shape}")
    print(f"🎯 Loan approval rate: {df['loan_approved'].mean():.1%}")
    print(f"📈 Feature correlations with target:")
    
    # Only calculate correlations for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['loan_approved'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'loan_approved':
            print(f"   {feature}: {corr:.3f}")
    
    return df

def create_customer_dataset():
    """Create customer churn dataset"""
    
    np.random.seed(123)
    n_samples = 800
    
    data = {}
    
    # Customer demographics
    data['customer_age'] = np.random.normal(45, 15, n_samples).clip(18, 80).astype(int)
    data['monthly_charges'] = np.random.normal(70, 25, n_samples).clip(20, 200)
    data['total_charges'] = data['monthly_charges'] * np.random.uniform(1, 60, n_samples)
    
    # Service details
    data['tenure_months'] = np.random.exponential(24, n_samples).clip(1, 72).astype(int)
    data['contract_type'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                           n_samples, p=[0.5, 0.3, 0.2])
    data['payment_method'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                            n_samples, p=[0.4, 0.2, 0.2, 0.2])
    
    # Service usage
    data['internet_service'] = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])
    data['online_security'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    data['tech_support'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    
    # Support tickets (higher = more likely to churn)
    data['support_tickets'] = np.random.poisson(2, n_samples)
    
    # Target: Customer churn (based on logical patterns)
    churn_prob = (
        (data['contract_type'] == 'Month-to-month') * 0.3 +
        (data['monthly_charges'] > 80) * 0.2 +
        (data['tenure_months'] < 12) * 0.2 +
        (data['support_tickets'] > 3) * 0.2 +
        (data['payment_method'] == 'Electronic check') * 0.1
    )
    
    data['churned'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame(data)
    
    print(f"📊 Created customer dataset: {df.shape}")
    print(f"🎯 Churn rate: {df['churned'].mean():.1%}")
    
    return df

if __name__ == "__main__":
    # Create financial dataset
    financial_df = create_financial_dataset()
    financial_df.to_csv('financial_test_data.csv', index=False)
    print("✅ Saved financial_test_data.csv")
    
    # Create customer dataset  
    customer_df = create_customer_dataset()
    customer_df.to_csv('customer_test_data.csv', index=False)
    print("✅ Saved customer_test_data.csv")
    
    # Create JSON version of financial data
    financial_json = {
        "dataset_info": {
            "name": "Financial Loan Dataset",
            "description": "Synthetic financial data for loan approval prediction",
            "target_column": "loan_approved",
            "task_type": "classification",
            "features": len(financial_df.columns) - 1,
            "samples": len(financial_df)
        },
        "data": financial_df.to_dict('records')
    }
    
    with open('financial_test_data.json', 'w') as f:
        json.dump(financial_json, f, indent=2)
    print("✅ Saved financial_test_data.json")
    
    print("\n🧪 Test these datasets with:")
    print("python test_api.py")
    print("# Or upload financial_test_data.csv to the web interface")
