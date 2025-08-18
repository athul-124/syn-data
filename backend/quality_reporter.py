"""
SynData Quality Reporter
========================

Comprehensive quality assessment for synthetic data generation.
Provides fidelity, utility, and privacy metrics to build trust in synthetic data.

Key Features:
- Fidelity: KS tests, correlation preservation, distribution analysis
- Utility: Model performance comparison (classification/regression)
- Privacy: Basic membership inference resistance checks
- Visual reports: Statistical comparisons and model performance charts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class SynDataQualityReporter:
    """
    Comprehensive quality assessment for synthetic tabular data.
    
    Generates detailed reports covering:
    - Statistical fidelity (distributions, correlations)
    - Machine learning utility (model performance)
    - Basic privacy assessment
    """
    
    def __init__(self):
        self.report = {}
        
    def generate_comprehensive_report(
        self, 
        real_data: pd.DataFrame, 
        synthetic_data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report comparing real vs synthetic data.
        
        Args:
            real_data: Original dataset
            synthetic_data: Generated synthetic dataset  
            target_column: Target variable for utility assessment (optional)
            
        Returns:
            Comprehensive quality report dictionary
        """
        
        self.report = {
            "metadata": self._generate_metadata(real_data, synthetic_data),
            "fidelity_metrics": self._assess_fidelity(real_data, synthetic_data),
            "utility_metrics": self._assess_utility(real_data, synthetic_data, target_column),
            "privacy_metrics": self._assess_privacy(real_data, synthetic_data),
            "overall_score": None,
            "recommendations": []
        }
        
        # Calculate overall quality score
        self.report["overall_score"] = self._calculate_overall_score()
        
        # Generate recommendations
        self.report["recommendations"] = self._generate_recommendations()
        
        return self.report
    
    def _generate_metadata(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic metadata about the datasets."""
        return {
            "real_data_shape": real_data.shape,
            "synthetic_data_shape": synthetic_data.shape,
            "columns": list(real_data.columns),
            "numeric_columns": list(real_data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(real_data.select_dtypes(exclude=[np.number]).columns),
            "missing_values_real": real_data.isnull().sum().to_dict(),
            "missing_values_synthetic": synthetic_data.isnull().sum().to_dict()
        }
    
    def _assess_fidelity(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess statistical fidelity of synthetic data.
        
        Includes:
        - Kolmogorov-Smirnov tests for numeric columns
        - Jensen-Shannon divergence for categorical columns
        - Correlation matrix comparison
        - Total Variation Distance (TVD)
        """
        
        fidelity = {
            "column_wise_analysis": {},
            "correlation_analysis": {},
            "overall_tvd": None,
            "summary_scores": {}
        }
        
        # Column-wise analysis
        for col in real_data.columns:
            if col in synthetic_data.columns:
                fidelity["column_wise_analysis"][col] = self._analyze_column_fidelity(
                    real_data[col], synthetic_data[col]
                )
        
        # Correlation analysis
        fidelity["correlation_analysis"] = self._analyze_correlations(real_data, synthetic_data)
        
        # Overall TVD
        fidelity["overall_tvd"] = self._calculate_tvd(real_data, synthetic_data)
        
        # Summary scores
        fidelity["summary_scores"] = self._calculate_fidelity_scores(fidelity)
        
        return fidelity
    
    def _analyze_column_fidelity(self, real_col: pd.Series, synthetic_col: pd.Series) -> Dict[str, Any]:
        """Analyze fidelity for a single column."""
        
        analysis = {
            "data_type": str(real_col.dtype),
            "basic_stats": {},
            "distribution_test": {},
            "quality_score": 0.0
        }
        
        # Remove NaN values for analysis
        real_clean = real_col.dropna()
        synthetic_clean = synthetic_col.dropna()
        
        if pd.api.types.is_numeric_dtype(real_col):
            # Numeric column analysis
            analysis["basic_stats"] = {
                "real_mean": float(real_clean.mean()),
                "synthetic_mean": float(synthetic_clean.mean()),
                "real_std": float(real_clean.std()),
                "synthetic_std": float(synthetic_clean.std()),
                "real_min": float(real_clean.min()),
                "synthetic_min": float(synthetic_clean.min()),
                "real_max": float(real_clean.max()),
                "synthetic_max": float(synthetic_clean.max())
            }
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pvalue = stats.ks_2samp(real_clean, synthetic_clean)
                analysis["distribution_test"] = {
                    "test_type": "Kolmogorov-Smirnov",
                    "statistic": float(ks_stat),
                    "p_value": float(ks_pvalue),
                    "interpretation": "Similar distributions" if ks_pvalue > 0.05 else "Different distributions"
                }
                # Quality score based on KS test (lower KS stat = better)
                analysis["quality_score"] = max(0, 1 - ks_stat)
            except Exception as e:
                analysis["distribution_test"] = {"error": str(e)}
                analysis["quality_score"] = 0.0
                
        else:
            # Categorical column analysis
            real_freq = real_clean.value_counts(normalize=True)
            synthetic_freq = synthetic_clean.value_counts(normalize=True)
            
            # Align frequencies
            all_categories = set(real_freq.index) | set(synthetic_freq.index)
            real_aligned = [real_freq.get(cat, 0) for cat in all_categories]
            synthetic_aligned = [synthetic_freq.get(cat, 0) for cat in all_categories]
            
            analysis["basic_stats"] = {
                "real_unique_values": len(real_freq),
                "synthetic_unique_values": len(synthetic_freq),
                "common_categories": len(set(real_freq.index) & set(synthetic_freq.index)),
                "real_top_categories": real_freq.head(5).to_dict(),
                "synthetic_top_categories": synthetic_freq.head(5).to_dict()
            }
            
            # Jensen-Shannon divergence
            try:
                js_distance = jensenshannon(real_aligned, synthetic_aligned)
                analysis["distribution_test"] = {
                    "test_type": "Jensen-Shannon Divergence",
                    "distance": float(js_distance),
                    "interpretation": "Similar distributions" if js_distance < 0.1 else "Different distributions"
                }
                # Quality score based on JS distance (lower = better)
                analysis["quality_score"] = max(0, 1 - js_distance)
            except Exception as e:
                analysis["distribution_test"] = {"error": str(e)}
                analysis["quality_score"] = 0.0
        
        return analysis
    
    def _analyze_correlations(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict:
        """Analyze correlation preservation"""
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"correlation_score": 1.0, "message": "Insufficient numeric columns"}
        
        real_corr = real_data[numeric_cols].corr()
        synth_corr = synthetic_data[numeric_cols].corr()
        
        # Calculate correlation difference
        corr_diff = np.abs(real_corr - synth_corr).mean().mean()
        correlation_score = max(0.0, 1.0 - corr_diff)
        
        return {
            "correlation_score": float(correlation_score),
            "correlation_difference": float(corr_diff),
            "numeric_columns_analyzed": len(numeric_cols)
        }
    
    def _calculate_tvd(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Calculate Total Variation Distance between datasets."""
        try:
            # For simplicity, calculate TVD on first numeric column
            numeric_cols = real_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return 0.0
            
            col = numeric_cols[0]
            real_vals = real_data[col].dropna()
            synthetic_vals = synthetic_data[col].dropna()
            
            # Create histograms
            min_val = min(real_vals.min(), synthetic_vals.min())
            max_val = max(real_vals.max(), synthetic_vals.max())
            bins = np.linspace(min_val, max_val, 50)
            
            real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
            synthetic_hist, _ = np.histogram(synthetic_vals, bins=bins, density=True)
            
            # Normalize
            real_hist = real_hist / np.sum(real_hist)
            synthetic_hist = synthetic_hist / np.sum(synthetic_hist)
            
            # Calculate TVD
            tvd = 0.5 * np.sum(np.abs(real_hist - synthetic_hist))
            return float(tvd)
            
        except Exception:
            return 0.0
    
    def _calculate_fidelity_scores(self, fidelity: Dict) -> Dict:
        """Calculate comprehensive fidelity scores"""
        column_scores = []
        for col_analysis in fidelity["column_wise_analysis"].values():
            if "quality_score" in col_analysis:
                column_scores.append(col_analysis["quality_score"])
        
        avg_column_fidelity = np.mean(column_scores) if column_scores else 0.0
        correlation_score = fidelity["correlation_analysis"].get("correlation_score", 0.0)
        tvd_score = 1.0 - min(fidelity["overall_tvd"] or 1.0, 1.0)
        
        return {
            "column_fidelity": float(avg_column_fidelity),
            "correlation_fidelity": float(correlation_score),
            "distribution_fidelity": float(tvd_score),
            "overall_fidelity": float(np.mean([avg_column_fidelity, correlation_score, tvd_score]))
        }
    
    def _assess_utility(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """
        Enhanced utility assessment with better error handling and debugging
        """
        print(f"🔍 Utility assessment - Target column: {target_column}")
        print(f"🔍 Real data columns: {list(real_data.columns)}")
        print(f"🔍 Synthetic data columns: {list(synthetic_data.columns)}")
        
        if not target_column:
            return {"error": "No target column specified", "utility_score": 0.0}
        
        if target_column not in real_data.columns:
            return {"error": f"Target column '{target_column}' not found in real data", "utility_score": 0.0}
        
        if target_column not in synthetic_data.columns:
            return {"error": f"Target column '{target_column}' not found in synthetic data", "utility_score": 0.0}
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
            import numpy as np
            
            print("🔍 Starting utility calculation...")
            
            # Prepare data
            X_real = real_data.drop(columns=[target_column]).copy()
            y_real = real_data[target_column].copy()
            
            X_synth = synthetic_data.drop(columns=[target_column]).copy()
            y_synth = synthetic_data[target_column].copy()
            
            print(f"🔍 Real features shape: {X_real.shape}, target shape: {y_real.shape}")
            print(f"🔍 Synthetic features shape: {X_synth.shape}, target shape: {y_synth.shape}")
            
            # Handle missing values
            for col in X_real.columns:
                if X_real[col].dtype in ['object']:
                    mode_val = X_real[col].mode().iloc[0] if not X_real[col].mode().empty else 'Unknown'
                    X_real[col] = X_real[col].fillna(mode_val)
                    X_synth[col] = X_synth[col].fillna(mode_val)
                else:
                    mean_val = X_real[col].mean()
                    X_real[col] = X_real[col].fillna(mean_val)
                    X_synth[col] = X_synth[col].fillna(mean_val)
            
            # Encode categorical variables
            categorical_cols = X_real.select_dtypes(include=['object']).columns
            print(f"🔍 Categorical columns: {list(categorical_cols)}")
            
            for col in categorical_cols:
                le = LabelEncoder()
                # Combine all unique values from both datasets
                all_values = list(set(X_real[col].astype(str)) | set(X_synth[col].astype(str)))
                le.fit(all_values)
                
                X_real[col] = le.transform(X_real[col].astype(str))
                X_synth[col] = le.transform(X_synth[col].astype(str))
            
            # Determine if classification or regression
            is_classification = (y_real.dtype == 'object' or 
                               (y_real.dtype in ['int64', 'int32'] and y_real.nunique() <= 10))
            
            print(f"🔍 Task type: {'Classification' if is_classification else 'Regression'}")
            print(f"🔍 Target unique values: {y_real.nunique()}")
            
            # Encode target if needed
            if is_classification and y_real.dtype == 'object':
                le_target = LabelEncoder()
                y_real_encoded = le_target.fit_transform(y_real.astype(str))
                y_synth_encoded = le_target.transform(y_synth.astype(str))
            else:
                y_real_encoded = y_real.values
                y_synth_encoded = y_synth.values
            
            # Split real data for testing
            X_train_real, X_test, y_train_real, y_test = train_test_split(
                X_real, y_real_encoded, test_size=0.3, random_state=42,
                stratify=y_real_encoded if is_classification else None
            )
            
            print(f"🔍 Train/test split - Train: {X_train_real.shape}, Test: {X_test.shape}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_real_scaled = scaler.fit_transform(X_train_real)
            X_test_scaled = scaler.transform(X_test)
            X_synth_scaled = scaler.transform(X_synth)
            
            if is_classification:
                print("🔍 Training classification models...")
                
                # Classification models
                model_real = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                model_synth = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                
                # Train model on real data
                model_real.fit(X_train_real_scaled, y_train_real)
                pred_real = model_real.predict(X_test_scaled)
                
                # Train model on synthetic data, test on real test set
                model_synth.fit(X_synth_scaled, y_synth_encoded)
                pred_synth = model_synth.predict(X_test_scaled)
                
                # Calculate metrics
                acc_real = accuracy_score(y_test, pred_real)
                acc_synth = accuracy_score(y_test, pred_synth)
                
                print(f"🔍 Real model accuracy: {acc_real:.3f}")
                print(f"🔍 Synthetic model accuracy: {acc_synth:.3f}")
                
                # Utility score (ratio of synthetic to real performance)
                utility_score = acc_synth / acc_real if acc_real > 0 else 0.0
                utility_score = max(0.0, min(1.0, utility_score))  # Clamp between 0 and 1
                
                print(f"🔍 Calculated utility score: {utility_score:.3f}")
                
                return {
                    "task_type": "classification",
                    "real_model_accuracy": float(acc_real),
                    "synthetic_model_accuracy": float(acc_synth),
                    "utility_score": float(utility_score),
                    "performance_ratio": float(acc_synth / acc_real) if acc_real > 0 else 0.0,
                    "interpretation": self._interpret_utility_score(utility_score)
                }
                
            else:
                print("🔍 Training regression models...")
                
                # Regression models
                model_real = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                model_synth = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                
                # Train model on real data
                model_real.fit(X_train_real_scaled, y_train_real)
                pred_real = model_real.predict(X_test_scaled)
                
                # Train model on synthetic data, test on real test set
                model_synth.fit(X_synth_scaled, y_synth_encoded)
                pred_synth = model_synth.predict(X_test_scaled)
                
                # Calculate metrics
                r2_real = r2_score(y_test, pred_real)
                r2_synth = r2_score(y_test, pred_synth)
                
                print(f"🔍 Real model R²: {r2_real:.3f}")
                print(f"🔍 Synthetic model R²: {r2_synth:.3f}")
                
                # For regression, utility based on R² ratio
                utility_score = r2_synth / r2_real if r2_real > 0 else 0.0
                utility_score = max(0.0, min(1.0, utility_score))
                
                print(f"🔍 Calculated utility score: {utility_score:.3f}")
                
                return {
                    "task_type": "regression",
                    "real_model_r2": float(r2_real),
                    "synthetic_model_r2": float(r2_synth),
                    "utility_score": float(utility_score),
                    "performance_ratio": float(r2_synth / r2_real) if r2_real > 0 else 0.0,
                    "interpretation": self._interpret_utility_score(utility_score)
                }
                
        except Exception as e:
            print(f"❌ Utility assessment error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Utility assessment failed: {str(e)}",
                "utility_score": 0.0
            }
    
    def _interpret_utility_score(self, utility_score: float) -> str:
        """Interpret utility score."""
        if utility_score >= 0.9:
            return "Excellent utility - synthetic data performs nearly as well as real data"
        elif utility_score >= 0.8:
            return "Good utility - synthetic data is quite useful for ML tasks"
        elif utility_score >= 0.6:
            return "Moderate utility - synthetic data has some value but with limitations"
        else:
            return "Poor utility - synthetic data may not be suitable for ML tasks"
    
    def _assess_privacy(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced privacy assessment with membership inference attack simulation
        """
        privacy = {
            "duplicate_records": self._check_duplicate_records(real_data, synthetic_data),
            "value_overlap": self._check_value_overlap(real_data, synthetic_data),
            "membership_inference": self._membership_inference_test(real_data, synthetic_data),
            "privacy_score": 0.0,
            "recommendations": []
        }
        
        # Calculate comprehensive privacy score
        duplicate_penalty = min(1.0, privacy["duplicate_records"]["duplicate_percentage"] / 10.0)
        overlap_penalty = min(1.0, privacy["value_overlap"]["high_overlap_columns"] / len(real_data.columns))
        membership_penalty = 1.0 - privacy["membership_inference"]["privacy_score"]
        
        privacy["privacy_score"] = max(0.0, 1.0 - (duplicate_penalty + overlap_penalty + membership_penalty) / 3)
        
        # Generate recommendations
        if privacy["duplicate_records"]["duplicate_percentage"] > 5:
            privacy["recommendations"].append("High percentage of duplicate records - add more noise or use differential privacy")
        
        if privacy["membership_inference"]["privacy_score"] < 0.7:
            privacy["recommendations"].append("Vulnerable to membership inference attacks - consider differential privacy training")
        
        if privacy["privacy_score"] < 0.6:
            privacy["recommendations"].append("Overall privacy is low - implement stronger privacy-preserving techniques")
        
        return privacy
    
    def _check_duplicate_records(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for exact duplicate records between real and synthetic data."""
        
        # Convert to string representation for comparison
        real_strings = set(real_data.astype(str).apply(lambda x: '|'.join(x), axis=1))
        synthetic_strings = set(synthetic_data.astype(str).apply(lambda x: '|'.join(x), axis=1))
        
        duplicates = real_strings & synthetic_strings
        duplicate_percentage = (len(duplicates) / len(synthetic_data)) * 100
        
        return {
            "duplicate_count": len(duplicates),
            "duplicate_percentage": float(duplicate_percentage),
            "interpretation": "High privacy risk" if duplicate_percentage > 5 else "Low privacy risk"
        }
    
    def _check_value_overlap(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for high value overlap in individual columns."""
        
        high_overlap_columns = 0
        column_overlaps = {}
        
        for col in real_data.columns:
            if col in synthetic_data.columns:
                real_values = set(real_data[col].astype(str))
                synthetic_values = set(synthetic_data[col].astype(str))
                
                overlap = len(real_values & synthetic_values)
                overlap_percentage = (overlap / len(real_values)) * 100 if len(real_values) > 0 else 0
                
                column_overlaps[col] = float(overlap_percentage)
                
                if overlap_percentage > 80:  # High overlap threshold
                    high_overlap_columns += 1
        
        return {
            "high_overlap_columns": high_overlap_columns,
            "column_overlap_percentages": column_overlaps,
            "average_overlap": float(np.mean(list(column_overlaps.values()))) if column_overlaps else 0.0
        }
    
    def _membership_inference_test(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate membership inference attack to test privacy
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import accuracy_score
            import numpy as np
            
            # Prepare data for membership inference
            # Label: 1 = real data, 0 = synthetic data
            real_labeled = real_data.copy()
            real_labeled['is_real'] = 1
            
            synth_labeled = synthetic_data.copy()
            synth_labeled['is_real'] = 0
            
            # Combine datasets
            combined = pd.concat([real_labeled, synth_labeled], ignore_index=True)
            
            # Prepare features (exclude the label)
            X = combined.drop(columns=['is_real'])
            y = combined['is_real']
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train membership inference classifier
            mi_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            mi_classifier.fit(X_train, y_train)
            
            # Test accuracy
            y_pred = mi_classifier.predict(X_test)
            mi_accuracy = accuracy_score(y_test, y_pred)
            
            # Privacy score: lower accuracy = better privacy
            # Random guessing = 0.5 accuracy, perfect privacy = 0.5 accuracy
            privacy_score = max(0.0, 1.0 - (mi_accuracy - 0.5) * 2)
            
            return {
                "membership_inference_accuracy": mi_accuracy,
                "privacy_score": privacy_score,
                "interpretation": "Lower accuracy = better privacy (random guessing = 50%)"
            }
            
        except Exception as e:
            return {
                "error": f"Membership inference test failed: {str(e)}",
                "privacy_score": 0.5  # Neutral score
            }
    
    def _calculate_overall_score(self) -> Dict[str, float]:
        """Calculate overall quality score with proper handling of negative utility"""
        
        fidelity_score = self.report["fidelity_metrics"]["summary_scores"]["overall_fidelity"]
        
        # Handle utility score properly
        utility_score = 0.0
        if "utility_score" in self.report["utility_metrics"]:
            raw_utility = self.report["utility_metrics"]["utility_score"]
            # Ensure utility is between 0 and 1 (negative scores become 0)
            utility_score = max(0.0, min(1.0, raw_utility))
        
        privacy_score = self.report["privacy_metrics"]["privacy_score"]
        
        # Weighted average with emphasis on fidelity and utility
        overall_score = (0.4 * fidelity_score + 0.4 * utility_score + 0.2 * privacy_score)
        
        return {
            "overall_quality_score": float(overall_score),
            "fidelity_component": float(fidelity_score),
            "utility_component": float(utility_score),
            "privacy_component": float(privacy_score),
            "grade": self._score_to_grade(overall_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        
        recommendations = []
        
        # Fidelity recommendations
        fidelity_score = self.report["fidelity_metrics"]["summary_scores"]["overall_fidelity"]
        if fidelity_score < 0.7:
            recommendations.append("Consider using more sophisticated generation methods (CTGAN, SDV) to improve statistical fidelity")
        
        # Utility recommendations
        if "utility_score" in self.report["utility_metrics"]:
            utility_score = self.report["utility_metrics"]["utility_score"]
            if utility_score < 0.8:
                recommendations.append("Synthetic data shows reduced ML utility - consider increasing dataset size or improving generation quality")
        
        # Privacy recommendations
        privacy_score = self.report["privacy_metrics"]["privacy_score"]
        if privacy_score < 0.8:
            recommendations.append("Consider implementing differential privacy or adding more noise to improve privacy protection")
        
        # Add privacy-specific recommendations
        recommendations.extend(self.report["privacy_metrics"]["recommendations"])
        
        return recommendations


def create_quality_report(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to create a comprehensive quality report.
    
    Args:
        real_data: Original dataset
        synthetic_data: Generated synthetic dataset
        target_column: Target variable for utility assessment (optional)
        
    Returns:
        Comprehensive quality report
    """
    reporter = SynDataQualityReporter()
    return reporter.generate_comprehensive_report(real_data, synthetic_data, target_column)
