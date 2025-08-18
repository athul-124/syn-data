# SynData MVP - Quick Reference Guide

## 🚀 Quick Start Commands

```bash
# Start the API server
uvicorn main:app --reload --port 8000

# Create test datasets
python test_financial_dataset.py

# Test API functionality
python test_api.py

# Run MVP tools demo
python demo_mvp_tools.py

# Use CLI interface
python mvp_cli.py status
python mvp_cli.py features
python mvp_cli.py report
```

## 📊 Key Quality Metrics

| Metric | Good Score | Excellent Score | Description |
|--------|------------|-----------------|-------------|
| **Overall Quality** | > 70% | > 85% | Combined fidelity, utility, privacy |
| **Fidelity** | > 70% | > 85% | Statistical similarity to real data |
| **Utility** | > 60% | > 80% | ML model performance retention |
| **Privacy** | > 80% | > 90% | Re-identification risk assessment |

## 🤖 Supported ML Models

### Generation Models
- **CTGAN**: Best for complex, mixed-type datasets
- **Gaussian Copula**: Fast generation for simple datasets

### Evaluation Models
- **Classification**: RandomForest, LogisticRegression, GradientBoosting
- **Regression**: RandomForest, LinearRegression, GradientBoosting

## 🔧 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/generate` | POST | Generate synthetic data |
| `/download/{filename}` | GET | Download generated files |

## 📁 File Locations

| File Type | Location | Description |
|-----------|----------|-------------|
| **Generated Data** | `generated/synthetic_*.csv` | Synthetic datasets |
| **Quality Reports** | `generated/reports/` | Quality assessment reports |
| **Test Data** | `*.csv` | Sample datasets for testing |
| **MVP Data** | `mvp_*.json` | MVP tools data files |

## 🎯 MVP Tools Quick Commands

```python
# Feature Prioritization
from mvp_tools.feature_prioritization import FeaturePrioritizer
prioritizer = FeaturePrioritizer()
mvp_features = prioritizer.get_mvp_features()

# User Research
from mvp_tools.user_research import UserResearchManager
research = UserResearchManager()
personas = research.generate_persona_summary()

# Progress Tracking
from mvp_tools.progress_tracker import MVPProgressTracker
tracker = MVPProgressTracker("My Project")
progress = tracker.get_overall_progress()

# Feedback Management
from mvp_tools.feedback_system import FeedbackManager
feedback = FeedbackManager()
analysis = feedback.analyze_feedback()

# Dashboard
from mvp_tools.mvp_dashboard import MVPDashboard
dashboard = MVPDashboard()
summary = dashboard.get_executive_summary()
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Low Utility Score** | Check target column, increase training epochs |
| **Memory Error** | Reduce dataset size, use chunking |
| **API Timeout** | Increase timeout, reduce complexity |
| **Poor Fidelity** | Check data preprocessing, try different model |

## 📈 Performance Benchmarks

| Dataset Size | Generation Time | Memory Usage | Recommended Model |
|--------------|-----------------|--------------|-------------------|
| < 1K rows | < 30 seconds | < 500MB | Either model |
| 1K - 10K rows | 1-5 minutes | 500MB - 2GB | CTGAN preferred |
| 10K - 50K rows | 5-15 minutes | 2GB - 8GB | Gaussian Copula |
| > 50K rows | 15+ minutes | 8GB+ | Chunked processing |

---

*For detailed documentation, see COMPREHENSIVE_DOCUMENTATION.md and TECHNICAL_SPECIFICATIONS.md*