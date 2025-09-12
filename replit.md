# SynData - Synthetic Tabular Data Generation Platform

## Overview

SynData is a comprehensive synthetic tabular data generation platform designed to help startups and small teams generate privacy-safe synthetic datasets with automated quality and utility validation. The platform solves the critical problem of training data scarcity by creating synthetic datasets that preserve statistical properties and ML utility while ensuring privacy compliance.

The system provides CSV upload functionality, synthetic data generation using advanced ML models like CTGAN, comprehensive quality reporting with fidelity and utility metrics, and automated ML model comparison to validate synthetic data effectiveness.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
The backend is built using **FastAPI** with a modular design separating concerns across different components:

- **API Layer**: FastAPI application handling file uploads, generation requests, and downloads
- **Data Generation Engine**: Implements multiple synthetic data generation models including CTGAN and Gaussian Copula
- **Quality Assessment Engine**: Comprehensive reporting system that evaluates statistical fidelity, ML utility, and privacy metrics
- **File Management System**: Handles CSV uploads, temporary file storage, and download management
- **Async Task Management**: Background processing for long-running generation tasks using ThreadPoolExecutor

The system uses a **microservices-oriented approach** with clear separation between data ingestion, processing, quality assessment, and result delivery.

### Frontend Architecture
The frontend is a **React-based single-page application** using modern React patterns:

- **Component-based architecture** with reusable UI components
- **Zustand state management** for global application state
- **React Router** for navigation between dashboard, upload, generation, and task monitoring views
- **Tailwind CSS** for responsive styling and design system
- **Axios** for API communication with the backend

### Data Processing Pipeline
The system implements a **sequential data pipeline**:

1. **Data Validation**: CSV parsing and schema validation
2. **Preprocessing**: Data type inference, missing value handling, and normalization
3. **Model Training**: CTGAN or alternative model training on real data
4. **Synthetic Generation**: Creating new synthetic samples
5. **Quality Assessment**: Multi-dimensional evaluation including fidelity, utility, and privacy
6. **Report Generation**: Comprehensive quality reports with visualizations

### Quality Assessment Framework
The quality reporting system provides **three-pillar evaluation**:

- **Statistical Fidelity**: Kolmogorov-Smirnov tests, correlation preservation, Wasserstein distances
- **ML Utility**: Automated model training comparison (real vs synthetic data)
- **Privacy Assessment**: Re-identification risk analysis and membership inference resistance

### File Management Strategy
The system uses **local filesystem storage** with organized directory structure:
- `uploads/` for incoming CSV files
- `generated/` for synthetic data outputs
- `generated/reports/` for quality assessment reports
- Temporary file cleanup and secure file serving through FastAPI

### MVP Development Tools Integration
The project includes a comprehensive **MVP development toolkit** implementing strategic blueprint methodology:
- Feature prioritization using MoSCoW and RICE scoring
- User research management with personas and user stories
- Progress tracking through defined MVP phases
- Feedback collection and analysis system
- Unified dashboard for project oversight

## External Dependencies

### Core ML Libraries
- **CTGAN**: Primary synthetic data generation model for complex tabular data
- **SDV (Synthetic Data Vault)**: Alternative synthetic data generation framework
- **scikit-learn**: Machine learning utilities, preprocessing, and model evaluation
- **pandas & numpy**: Data manipulation and numerical computing
- **scipy**: Statistical analysis and distance calculations

### Web Framework & API
- **FastAPI**: Modern Python web framework with automatic API documentation
- **uvicorn**: ASGI server for running FastAPI applications
- **python-multipart**: File upload handling
- **aiofiles**: Asynchronous file operations

### Visualization & Reporting
- **matplotlib**: Statistical visualization for quality reports
- **seaborn**: Enhanced statistical plotting
- **Chart utilities**: Custom visualization components for quality metrics

### Frontend Dependencies
- **React ecosystem**: Core React library with modern hooks
- **React Router**: Client-side routing
- **Zustand**: Lightweight state management
- **Tailwind CSS**: Utility-first CSS framework
- **Axios**: HTTP client for API communication
- **react-hot-toast**: User notification system
- **Lucide React**: Icon library

### Development & Testing
- **pytest**: Python testing framework
- **black & flake8**: Code formatting and linting
- **Rich & Click**: Enhanced CLI interfaces for development tools

### Deployment Infrastructure
The system is designed for **cloud deployment** with support for:
- Railway/Render/Heroku for easy deployment
- Docker containerization capabilities
- Environment-based configuration for development/production
- CORS middleware for cross-origin requests in development