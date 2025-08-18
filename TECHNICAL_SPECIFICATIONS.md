# SynData MVP - Technical Specifications

## 🔧 System Architecture

### **Backend Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Synthetic      │    │  Quality        │
│   Web Server    │───▶│  Data Engine    │───▶│  Assessment     │
│                 │    │                 │    │  Engine         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Storage  │    │   ML Models     │    │   Report        │
│   & Management  │    │   (CTGAN, etc.) │    │   Generator     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Data Flow Architecture**
```
CSV Upload → Validation → Preprocessing → Model Training → 
Generation → Quality Assessment → Report Generation → Download
```

### **File Structure**
```
syn-data/
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── COMPREHENSIVE_DOCUMENTATION.md
├── TECHNICAL_SPECIFICATIONS.md
│
├── backend/                    # Core backend modules
│   ├── __init__.py
│   ├── synthetic_generator.py  # Data generation engine
│   ├── quality_reporter.py     # Quality assessment
│   ├── file_manager.py         # File handling utilities
│   └── config.py              # Configuration settings
│
├── mvp_tools/                  # MVP development toolkit
│   ├── __init__.py
│   ├── feature_prioritization.py
│   ├── user_research.py
│   ├── progress_tracker.py
│   ├── feedback_system.py
│   └── mvp_dashboard.py
│
├── frontend/                   # Web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── tests/                      # Test files
│   ├── test_api.py
│   ├── test_financial_dataset.py
│   └── demo_mvp_tools.py
│
├── data/                       # Sample datasets
│   ├── financial_test_data.csv
│   ├── customer_test_data.csv
│   └── financial_test_data.json
│
└── generated/                  # Generated files storage
    ├── synthetic_*.csv
    └── reports/
```

## 🧠 Machine Learning Pipeline

### **Model Selection Logic**
```python
def select_optimal_model(dataset_characteristics):
    """
    Intelligent model selection based on dataset properties
    """
    if dataset_characteristics['rows'] > 10000:
        if dataset_characteristics['categorical_ratio'] > 0.5:
            return 'CTGAN'  # Best for large, mixed datasets
        else:
            return 'GaussianCopula'  # Faster for numeric data
    elif dataset_characteristics['complexity_score'] > 0.7:
        return 'CTGAN'  # Complex relationships need GAN
    else:
        return 'GaussianCopula'  # Simple datasets
```

### **Training Pipeline**
```python
class SyntheticDataPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model_selector = ModelSelector()
        self.quality_assessor = QualityAssessor()
    
    def generate(self, data, n_samples, target_column=None):
        # 1. Data validation and preprocessing
        processed_data = self.preprocessor.fit_transform(data)
        
        # 2. Model selection and training
        model = self.model_selector.select_model(processed_data)
        model.fit(processed_data)
        
        # 3. Synthetic data generation
        synthetic_data = model.sample(n_samples)
        
        # 4. Post-processing and validation
        synthetic_data = self.preprocessor.inverse_transform(synthetic_data)
        
        # 5. Quality assessment
        quality_report = self.quality_assessor.assess(
            original=data,
            synthetic=synthetic_data,
            target_column=target_column
        )
        
        return synthetic_data, quality_report
```

### **Quality Assessment Pipeline**
```python
class QualityAssessmentPipeline:
    def __init__(self):
        self.fidelity_assessor = FidelityAssessor()
        self.utility_assessor = UtilityAssessor()
        self.privacy_assessor = PrivacyAssessor()
    
    def comprehensive_assessment(self, real_data, synthetic_data, target_column):
        # Parallel assessment of different quality dimensions
        fidelity_score = self.fidelity_assessor.assess(real_data, synthetic_data)
        utility_score = self.utility_assessor.assess(real_data, synthetic_data, target_column)
        privacy_score = self.privacy_assessor.assess(real_data, synthetic_data)
        
        # Weighted overall score
        overall_score = (
            fidelity_score * 0.4 +
            utility_score * 0.4 +
            privacy_score * 0.2
        )
        
        return {
            'overall_score': overall_score,
            'fidelity_metrics': fidelity_score,
            'utility_metrics': utility_score,
            'privacy_metrics': privacy_score,
            'grade': self.calculate_grade(overall_score),
            'recommendations': self.generate_recommendations(fidelity_score, utility_score, privacy_score)
        }
```

## 📊 Detailed Model Specifications

### **CTGAN Implementation**
```python
class CTGANGenerator:
    """
    Conditional Tabular Generative Adversarial Network
    Optimized for mixed-type tabular data
    """
    
    def __init__(self, 
                 epochs=300,
                 batch_size=500,
                 generator_lr=2e-4,
                 discriminator_lr=2e-4,
                 discriminator_steps=1,
                 log_frequency=True,
                 verbose=True,
                 cuda=True):
        
        self.model = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=verbose,
            cuda=cuda
        )
    
    def fit(self, data):
        """Train the CTGAN model on the provided data"""
        # Automatic discrete column detection
        discrete_columns = self._detect_discrete_columns(data)
        
        # Model training with progress tracking
        self.model.fit(data, discrete_columns)
    
    def sample(self, n_samples):
        """Generate synthetic samples"""
        return self.model.sample(n_samples)
    
    def _detect_discrete_columns(self, data):
        """Automatically detect categorical/discrete columns"""
        discrete_columns = []
        for column in data.columns:
            if data[column].dtype == 'object':
                discrete_columns.append(column)
            elif data[column].dtype in ['int64', 'int32']:
                unique_ratio = data[column].nunique() / len(data)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    discrete_columns.append(column)
        return discrete_columns
```

### **Gaussian Copula Implementation**
```python
class GaussianCopulaGenerator:
    """
    Statistical copula-based synthetic data generation
    Fast and effective for datasets with clear distributions
    """
    
    def __init__(self):
        self.model = GaussianCopula()
        self.metadata = None
    
    def fit(self, data):
        """Fit the Gaussian Copula model"""
        # Automatic metadata detection
        self.metadata = self._detect_metadata(data)
        
        # Model fitting
        self.model.fit(data)
    
    def sample(self, n_samples):
        """Generate synthetic samples"""
        return self.model.sample(n_samples)
    
    def _detect_metadata(self, data):
        """Detect column types and constraints"""
        metadata = {}
        for column in data.columns:
            if data[column].dtype == 'object':
                metadata[column] = {'type': 'categorical'}
            elif data[column].dtype in ['int64', 'int32']:
                metadata[column] = {
                    'type': 'numerical',
                    'subtype': 'integer',
                    'min': data[column].min(),
                    'max': data[column].max()
                }
            else:
                metadata[column] = {
                    'type': 'numerical',
                    'subtype': 'float',
                    'min': data[column].min(),
                    'max': data[column].max()
                }
        return metadata
```

## 🔍 Quality Metrics Deep Dive

### **Fidelity Assessment**
```python
class FidelityAssessor:
    """Comprehensive statistical fidelity assessment"""
    
    def assess(self, real_data, synthetic_data):
        """Calculate comprehensive fidelity metrics"""
        
        # Column-wise statistical tests
        column_scores = {}
        for column in real_data.columns:
            if real_data[column].dtype in ['int64', 'float64']:
                column_scores[column] = self._assess_numerical_column(
                    real_data[column], synthetic_data[column]
                )
            else:
                column_scores[column] = self._assess_categorical_column(
                    real_data[column], synthetic_data[column]
                )
        
        # Correlation preservation
        correlation_score = self._assess_correlation_preservation(real_data, synthetic_data)
        
        # Overall fidelity calculation
        overall_fidelity = np.mean([
            np.mean(list(column_scores.values())),
            correlation_score
        ])
        
        return {
            'overall_fidelity': overall_fidelity,
            'column_scores': column_scores,
            'correlation_preservation': correlation_score,
            'statistical_similarity': np.mean(list(column_scores.values()))
        }
    
    def _assess_numerical_column(self, real_col, synthetic_col):
        """Assess numerical column similarity"""
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = ks_2samp(real_col, synthetic_col)
        
        # Wasserstein distance (normalized)
        wasserstein_dist = wasserstein_distance(real_col, synthetic_col)
        max_range = max(real_col.max() - real_col.min(), 1e-6)
        normalized_wasserstein = 1 - min(wasserstein_dist / max_range, 1)
        
        # Statistical moments comparison
        moments_score = self._compare_moments(real_col, synthetic_col)
        
        # Combined score
        return np.mean([
            1 - ks_statistic,  # Higher KS statistic = lower similarity
            normalized_wasserstein,
            moments_score
        ])
    
    def _assess_categorical_column(self, real_col, synthetic_col):
        """Assess categorical column similarity"""
        # Chi-square test for distribution similarity
        real_counts = real_col.value_counts()
        synthetic_counts = synthetic_col.value_counts()
        
        # Align categories
        all_categories = set(real_counts.index) | set(synthetic_counts.index)
        real_aligned = [real_counts.get(cat, 0) for cat in all_categories]
        synthetic_aligned = [synthetic_counts.get(cat, 0) for cat in all_categories]
        
        # Chi-square test
        chi2_stat, chi2_p_value = chisquare(synthetic_aligned, real_aligned)
        
        # Category coverage (how many real categories are represented)
        coverage = len(set(synthetic_col) & set(real_col)) / len(set(real_col))
        
        # Combined score
        return np.mean([
            min(chi2_p_value * 10, 1),  # Higher p-value = better similarity
            coverage
        ])
    
    def _compare_moments(self, real_col, synthetic_col):
        """Compare statistical moments (mean, std, skewness, kurtosis)"""
        from scipy.stats import skew, kurtosis
        
        real_moments = [
            real_col.mean(),
            real_col.std(),
            skew(real_col),
            kurtosis(real_col)
        ]
        
        synthetic_moments = [
            synthetic_col.mean(),
            synthetic_col.std(),
            skew(synthetic_col),
            kurtosis(synthetic_col)
        ]
        
        # Calculate relative differences
        moment_similarities = []
        for real_moment, synthetic_moment in zip(real_moments, synthetic_moments):
            if abs(real_moment) > 1e-6:
                relative_diff = abs(real_moment - synthetic_moment) / abs(real_moment)
                similarity = max(0, 1 - relative_diff)
            else:
                similarity = 1 if abs(synthetic_moment) < 1e-6 else 0
            moment_similarities.append(similarity)
        
        return np.mean(moment_similarities)
    
    def _assess_correlation_preservation(self, real_data, synthetic_data):
        """Assess how well correlations are preserved"""
        # Only use numerical columns for correlation
        real_numeric = real_data.select_dtypes(include=[np.number])
        synthetic_numeric = synthetic_data.select_dtypes(include=[np.number])
        
        if real_numeric.shape[1] < 2:
            return 1.0  # Perfect score if no correlations to preserve
        
        real_corr = real_numeric.corr()
        synthetic_corr = synthetic_numeric.corr()
        
        # Calculate correlation matrix similarity
        corr_diff = np.abs(real_corr - synthetic_corr)
        correlation_preservation = 1 - np.mean(corr_diff.values[np.triu_indices_from(corr_diff, k=1)])
        
        return max(0, correlation_preservation)
```

### **Utility Assessment**
```python
class UtilityAssessor:
    """Machine learning utility assessment"""
    
    def __init__(self):
        self.classification_models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        self.regression_models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
    
    def assess(self, real_data, synthetic_data, target_column):
        """Comprehensive utility assessment"""
        
        if target_column is None or target_column not in real_data.columns:
            return {'utility_score': None, 'message': 'No target column specified'}
        
        # Determine task type
        task_type = self._determine_task_type(real_data[target_column])
        
        # Prepare data
        X_real, y_real = self._prepare_features_target(real_data, target_column)
        X_synthetic, y_synthetic = self._prepare_features_target(synthetic_data, target_column)
        
        # Select appropriate models
        models = self.classification_models if task_type == 'classification' else self.regression_models
        
        # Calculate utility scores
        utility_scores = []
        model_results = {}
        
        for model_name, model in models.items():
            try:
                # Train on real, test on synthetic (TRTS)
                trts_score = self._train_real_test_synthetic(
                    model, X_real, y_real, X_synthetic, y_synthetic, task_type
                )
                
                # Train on synthetic, test on real (TSTR)
                tstr_score = self._train_synthetic_test_real(
                    model, X_real, y_real, X_synthetic, y_synthetic, task_type
                )
                
                # Combined utility score
                combined_score = min(trts_score, tstr_score)
                utility_scores.append(combined_score)
                
                model_results[model_name] = {
                    'trts_score': trts_score,
                    'tstr_score': tstr_score,
                    'combined_score': combined_score
                }
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        if not utility_scores:
            return {'utility_score': 0, 'message': 'All models failed'}
        
        overall_utility = np.mean(utility_scores)
        
        return {
            'utility_score': overall_utility,
            'task_type': task_type,
            'model_results': model_results,
            'interpretation': self._interpret_utility_score(overall_utility)
        }
    
    def _determine_task_type(self, target_series):
        """Determine if task is classification or regression"""
        if target_series.dtype == 'object':
            return 'classification'
        elif target_series.nunique() <= 10:
            return 'classification'
        else:
            return 'regression'
    
    def _prepare_features_target(self, data, target_column):
        """Prepare features and target for ML models"""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        return X_encoded, y
    
    def _train_real_test_synthetic(self, model, X_real, y_real, X_synthetic, y_synthetic, task_type):
        """Train on real data, test on synthetic data"""
        # Align features
        common_features = X_real.columns.intersection(X_synthetic.columns)
        X_real_aligned = X_real[common_features]
        X_synthetic_aligned = X_synthetic[common_features]
        
        # Train model
        model.fit(X_real_aligned, y_real)
        
        # Predict on synthetic data
        y_pred = model.predict(X_synthetic_aligned)
        
        # Calculate performance
        if task_type == 'classification':
            return accuracy_score(y_synthetic, y_pred)
        else:
            # For regression, convert to relative performance
            mae = mean_absolute_error(y_synthetic, y_pred)
            y_range = y_synthetic.max() - y_synthetic.min()
            return max(0, 1 - mae / max(y_range, 1e-6))
    
    def _train_synthetic_test_real(self, model, X_real, y_real, X_synthetic, y_synthetic, task_type):
        """Train on synthetic data, test on real data"""
        # Align features
        common_features = X_real.columns.intersection(X_synthetic.columns)
        X_real_aligned = X_real[common_features]
        X_synthetic_aligned = X_synthetic[common_features]
        
        # Train model
        model.fit(X_synthetic_aligned, y_synthetic)
        
        # Predict on real data
        y_pred = model.predict(X_real_aligned)
        
        # Calculate performance
        if task_type == 'classification':
            return accuracy_score(y_real, y_pred)
        else:
            # For regression, convert to relative performance
            mae = mean_absolute_error(y_real, y_pred)
            y_range = y_real.max() - y_real.min()
            return max(0, 1 - mae / max(y_range, 1e-6))
    
    def _interpret_utility_score(self, score):
        """Provide interpretation of utility score"""
        if score >= 0.8:
            return "Excellent utility - synthetic data performs very similarly to real data"
        elif score >= 0.6:
            return "Good utility - synthetic data is suitable for most ML tasks"
        elif score >= 0.4:
            return "Moderate utility - synthetic data may be useful for some applications"
        elif score >= 0.2:
            return "Low utility - synthetic data has limited ML usefulness"
        else:
            return "Poor utility - synthetic data is not suitable for ML tasks"
```

### **Privacy Assessment**
```python
class PrivacyAssessor:
    """Privacy risk assessment for synthetic data"""
    
    def assess(self, real_data, synthetic_data):
        """Comprehensive privacy assessment"""
        
        # Distance-based privacy metrics
        distance_privacy = self._assess_distance_privacy(real_data, synthetic_data)
        
        # Membership inference risk
        membership_risk = self._assess_membership_inference(real_data, synthetic_data)
        
        # Attribute inference risk
        attribute_risk = self._assess_attribute_inference(real_data, synthetic_data)
        
        # Overall privacy score
        privacy_score = np.mean([distance_privacy, 1 - membership_risk, 1 - attribute_risk])
        
        return {
            'privacy_score': privacy_score,
            'distance_privacy': distance_privacy,
            'membership_inference_risk': membership_risk,
            'attribute_inference_risk': attribute_risk,
            'risk_level': self._categorize_risk(privacy_score),
            'recommendations': self._generate_privacy_recommendations(privacy_score)
        }
    
    def _assess_distance_privacy(self, real_data, synthetic_data):
        """Assess privacy based on minimum distances between records"""
        # Convert to numerical representation
        real_numeric = self._convert_to_numeric(real_data)
        synthetic_numeric = self._convert_to_numeric(synthetic_data)
        
        # Calculate minimum distances
        min_distances = []
        for i, synthetic_row in synthetic_numeric.iterrows():
            distances = np.sqrt(np.sum((real_numeric - synthetic_row) ** 2, axis=1))
            min_distances.append(np.min(distances))
        
        # Normalize by maximum possible distance
        max_distance = np.sqrt(np.sum((real_numeric.max() - real_numeric.min()) ** 2))
        normalized_distances = np.array(min_distances) / max_distance
        
        # Privacy score based on average minimum distance
        privacy_score = np.mean(normalized_distances)
        
        return min(privacy_score, 1.0)
    
    def _assess_membership_inference(self, real_data, synthetic_data):
        """Assess membership inference attack risk"""
        # Simple membership inference based on nearest neighbor
        real_numeric = self._convert_to_numeric(real_data)
        synthetic_numeric = self._convert_to_numeric(synthetic_data)
        
        # For each synthetic record, find closest real record
        membership_scores = []
        for i, synthetic_row in synthetic_numeric.iterrows():
            distances = np.sqrt(np.sum((real_numeric - synthetic_row) ** 2, axis=1))
            min_distance = np.min(distances)
            
            # If distance is very small, high membership risk
            membership_scores.append(1 / (1 + min_distance))
        
        return np.mean(membership_scores)
    
    def _assess_attribute_inference(self, real_data, synthetic_data):
        """Assess attribute inference attack risk"""
        # Simplified attribute inference assessment
        # Based on how well synthetic data preserves rare attribute combinations
        
        risk_scores = []
        for column in real_data.columns:
            if real_data[column].dtype == 'object':
                # For categorical columns, check rare value preservation
                real_counts = real_data[column].value_counts()
                synthetic_counts = synthetic_data[column].value_counts()
                
                # Focus on rare values (bottom 10%)
                rare_threshold = real_counts.quantile(0.1)
                rare_values = real_counts[real_counts <= rare_threshold].index
                
                if len(rare_values) > 0:
                    # Check if rare values are over-represented in synthetic data
                    rare_risk = 0
                    for rare_value in rare_values:
                        real_freq = real_counts.get(rare_value, 0) / len(real_data)
                        synthetic_freq = synthetic_counts.get(rare_value, 0) / len(synthetic_data)
                        
                        if synthetic_freq > real_freq * 2:  # Over-representation
                            rare_risk += 1
                    
                    risk_scores.append(rare_risk / len(rare_values))
        
        return np.mean(risk_scores) if risk_scores else 0.0
    
    def _convert_to_numeric(self, data):
        """Convert mixed-type data to numeric representation"""
        numeric_data = data.copy()
        
        for column in data.columns:
            if data[column].dtype == 'object':
                # Label encode categorical variables
                le = LabelEncoder()
                numeric_data[column] = le.fit_transform(data[column].astype(str))
        
        # Normalize to [0, 1] range
        scaler = MinMaxScaler()
        numeric_data = pd.DataFrame(
            scaler.fit_transform(numeric_data),
            columns=numeric_data.columns,
            index=numeric_data.index
        )
        
        return numeric_data
    
    def _categorize_risk(self, privacy_score):
        """Categorize privacy risk level"""
        if privacy_score >= 0.8:
            return "Low"
        elif privacy_score >= 0.6:
            return "Medium"
        elif privacy_score >= 0.4:
            return "High"
        else:
            return "Critical"
    
    def _generate_privacy_recommendations(self, privacy_score):
        """Generate privacy improvement recommendations"""
        recommendations = []
        
        if privacy_score < 0.6:
            recommendations.append("Consider adding differential privacy noise")
            recommendations.append("Increase model training epochs for better generalization")
            recommendations.append("Remove or generalize highly identifying features")
        
        if privacy_score < 0.4:
            recommendations.append("Consider k-anonymity or l-diversity techniques")
            recommendations.append("Implement record-level privacy mechanisms")
            recommendations.append("Reduce synthetic dataset size to minimize exposure")
        
        return recommendations
```

## 🚀 Performance Optimization

### **Memory Management**
```python
class MemoryOptimizedPipeline:
    """Memory-efficient processing for large datasets"""
    
    def __init__(self, chunk_size=10000):
        self.chunk_size = chunk_size
    
    def process_large_dataset(self, data):
        """Process large datasets in chunks"""
        if len(data) <= self.chunk_size:
            return self._process_chunk(data)
        
        # Process in chunks
        results = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size]
            chunk_result = self._process_chunk(chunk)
            results.append(chunk_result)
        
        # Combine results
        return pd.concat(results, ignore_index=True)
    
    def _process_chunk(self, chunk):
        """Process individual chunk"""
        # Implement chunk processing logic
        return chunk
```

### **Caching Strategy**
```python
class ModelCache:
    """Intelligent model caching for repeated requests"""
    
    def __init__(self, cache_size=100):
        self.cache = {}
        self.cache_size = cache_size
        self.access_times = {}
    
    def get_model(self, data_hash):
        """Retrieve cached model if available"""
        if data_hash in self.cache:
            self.access_times[data_hash] = time.time()
            return self.cache[data_hash]
        return None
    
    def store_model(self, data_hash, model):
        """Store model in cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            oldest_hash = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_hash]
            del self.access_times[oldest_hash]
        
        self.cache[data_hash] = model
        self.access_times[data_hash] = time.time()
```

---

*This technical specification provides the complete implementation details for the SynData MVP system, including all machine learning models, quality assessment frameworks, and optimization strategies.*