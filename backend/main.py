import asyncio
import os
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import json

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app = FastAPI(
    title="SynData API",
    description="Generate high-quality synthetic tabular data with quality assessment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task storage (in production, use Redis or database)
tasks_storage = {}

# Add async processing
executor = ThreadPoolExecutor(max_workers=2)

def enhanced_synthetic_tabular(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Enhanced synthetic data generation with data validation and quality checks"""
    try:
        from ctgan import CTGAN
        
        print("Using enhanced CTGAN with data validation...")
        
        # Clean the input data first
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype in ['object']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown')
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Identify discrete columns more carefully
        discrete_columns = []
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                discrete_columns.append(col)
            elif df_clean[col].dtype in ['int64', 'int32']:
                unique_ratio = df_clean[col].nunique() / len(df_clean)
                if unique_ratio < 0.1 or df_clean[col].nunique() <= 20:  # Low cardinality
                    discrete_columns.append(col)
        
        print(f"Discrete columns identified: {discrete_columns}")
        print(f"Data shape: {df_clean.shape}")
        
        # Enhanced CTGAN configuration
        ctgan = CTGAN(
            epochs=500,  # More training for better quality
            batch_size=min(500, max(64, len(df_clean) // 2)),
            generator_dim=(512, 512),  # Larger network
            discriminator_dim=(512, 512),
            generator_lr=1e-4,  # Slower learning for stability
            discriminator_lr=1e-4,
            discriminator_steps=1,
            log_frequency=True,
            verbose=False,  # Reduce noise
            pac=5  # Helps with mode collapse
        )
        
        # Train the model
        print("Training CTGAN model...")
        ctgan.fit(df_clean, discrete_columns)
        
        # Generate synthetic data
        print("Generating synthetic samples...")
        synth = ctgan.sample(n_rows)
        
        # Post-process to ensure data quality
        synth = _post_process_synthetic_data(synth, df_clean, discrete_columns)
        
        print("✅ Enhanced CTGAN generation successful")
        return synth
        
    except Exception as e:
        print(f"Enhanced CTGAN failed: {e}")
        print("Falling back to correlation-preserving method...")
        return correlation_preserving_synthetic(df, n_rows)

def _post_process_synthetic_data(synth: pd.DataFrame, original: pd.DataFrame, discrete_columns: list) -> pd.DataFrame:
    """Post-process synthetic data to ensure quality and consistency"""
    
    # Ensure all original columns are present
    for col in original.columns:
        if col not in synth.columns:
            print(f"⚠️ Missing column {col} in synthetic data, adding it...")
            if col in discrete_columns:
                # Sample from original for missing categorical columns
                synth[col] = np.random.choice(original[col].dropna(), len(synth), replace=True)
            else:
                # Sample from original for missing numeric columns
                synth[col] = np.random.normal(original[col].mean(), original[col].std(), len(synth))
    
    # Reorder columns to match original
    synth = synth[original.columns]
    
    for col in synth.columns:
        if col in discrete_columns:
            # For categorical columns, ensure values are from original set
            if original[col].dtype == 'object':
                valid_values = set(original[col].unique())
                mode_value = original[col].mode().iloc[0]
                synth[col] = synth[col].apply(lambda x: x if x in valid_values else mode_value)
        else:
            # For numeric columns, clip to reasonable ranges
            if original[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                q1, q99 = original[col].quantile([0.01, 0.99])
                synth[col] = synth[col].clip(lower=q1, upper=q99)
                
                # Ensure integer columns remain integers
                if original[col].dtype in ['int64', 'int32']:
                    synth[col] = synth[col].round().astype(original[col].dtype)
    
    return synth

def correlation_preserving_synthetic(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Fallback method that preserves correlations better than simple sampling"""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import numpy as np
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        synth_data = {}
        
        # Handle numeric columns with PCA to preserve correlations
        if len(numeric_cols) > 0:
            numeric_data = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Standardize
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Apply PCA
            pca = PCA()
            pca_data = pca.fit_transform(scaled_data)
            
            # Generate synthetic data in PCA space
            synthetic_pca = np.random.multivariate_normal(
                mean=np.mean(pca_data, axis=0),
                cov=np.cov(pca_data.T),
                size=n_rows
            )
            
            # Transform back to original space
            synthetic_scaled = pca.inverse_transform(synthetic_pca)
            synthetic_numeric = scaler.inverse_transform(synthetic_scaled)
            
            # Add to synthetic data
            for i, col in enumerate(numeric_cols):
                synth_data[col] = synthetic_numeric[:, i]
        
        # Handle categorical columns
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            synth_data[col] = np.random.choice(
                value_counts.index,
                size=n_rows,
                p=value_counts.values
            )
        
        return pd.DataFrame(synth_data)
        
    except Exception as e:
        print(f"Correlation-preserving fallback failed: {e}")
        return simple_synthetic_tabular(df, n_rows)

def simple_synthetic_tabular(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Simple synthetic data generation fallback"""
    from faker import Faker
    fake = Faker()
    
    synth_data = {}
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Numeric columns: sample from normal distribution
            mean = df[col].mean()
            std = df[col].std()
            synth_data[col] = np.random.normal(mean, std, n_rows)
        elif df[col].dtype == 'object':
            # Categorical columns: sample with replacement
            synth_data[col] = np.random.choice(df[col].dropna().values, n_rows, replace=True)
        else:
            # Other types: repeat random values
            synth_data[col] = np.random.choice(df[col].dropna().values, n_rows, replace=True)
    
    return pd.DataFrame(synth_data)

def create_quality_report(original: pd.DataFrame, synthetic: pd.DataFrame, target_column: str = None) -> dict:
    """Create comprehensive quality assessment report using SynDataQualityReporter"""
    try:
        from quality_reporter import SynDataQualityReporter
        
        # Auto-detect target column if not provided
        if target_column is None:
            # Try to find a suitable target column
            for col in original.columns:
                if col.lower() in ['target', 'label', 'class', 'outcome', 'y']:
                    target_column = col
                    break
            
            # If still None, use the last column as target
            if target_column is None:
                target_column = original.columns[-1]
                print(f"Auto-selected target column: {target_column}")
        
        reporter = SynDataQualityReporter()
        comprehensive_report = reporter.generate_comprehensive_report(
            original, synthetic, target_column
        )
        
        return comprehensive_report
        
    except Exception as e:
        print(f"Quality report error: {e}")
        return {"error": str(e)}

async def generate_async(df: pd.DataFrame, n_rows: int, target_column: str = None):
    """Async wrapper for synthetic data generation"""
    loop = asyncio.get_event_loop()
    
    # Run heavy computation in thread pool
    synth_task = loop.run_in_executor(executor, enhanced_synthetic_tabular, df, n_rows)
    
    # Wait for completion
    synth = await synth_task
    quality_report = create_quality_report(df, synth, target_column)
    
    return synth, quality_report

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and analyze dataset"""
    try:
        print(f"Received file: {file.filename}, size: {file.size}, content_type: {file.content_type}")
        
        # Save upload
        file_id = str(uuid.uuid4())[:8]
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        file_path = f"uploads/{file_id}_{file.filename}"
        print(f"Saving to: {file_path}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            print(f"Saved {len(content)} bytes")
        
        # Analyze the file to get rows and columns
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            rows = len(df)
            columns = len(df.columns)
        except Exception as e:
            print(f"Failed to analyze file: {e}")
            rows = 0
            columns = 0
        
        return {
            "success": True,
            "id": file_id,
            "file_id": file_id,
            "filename": file.filename,
            "size": file.size or len(content),
            "path": file_path,
            "rows": rows,
            "columns": columns
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/preview")
async def generate_preview(request: Dict[str, Any]):
    """Generate small preview of synthetic data"""
    file_id = request.get("file_id")
    n_rows = min(request.get("n_rows", 10), 50)  # Limit preview size
    
    # Find uploaded file
    upload_files = [f for f in os.listdir("uploads") if f.startswith(file_id)]
    if not upload_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = f"uploads/{upload_files[0]}"
    df = pd.read_csv(file_path)
    
    # Generate small synthetic sample
    synth = simple_synthetic_tabular(df, n_rows)
    
    return JSONResponse({
        "success": True,
        "preview_data": synth.to_dict('records'),
        "original_sample": df.head(n_rows).to_dict('records')
    })

@app.post("/generate-async")
async def start_generation_task(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    n_rows: int = Form(1000),
    target_column: Optional[str] = Form(None)
):
    """Start async generation task"""
    task_id = str(uuid.uuid4())
    
    # Initialize task
    task = {
        "id": task_id,
        "status": "PENDING",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "file_id": file_id,
        "n_rows": n_rows,
        "target_column": target_column,
        "result": None,
        "error": None
    }
    
    tasks_storage[task_id] = task
    
    # Start background task
    background_tasks.add_task(run_generation_task, task_id)
    
    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "PENDING"
    })

async def run_generation_task(task_id: str):
    """Background task for data generation"""
    task = tasks_storage[task_id]
    
    try:
        print(f"Starting task {task_id}")  # Add logging
        task["status"] = "RUNNING"
        task["progress"] = 10
        
        # Find uploaded file
        file_id = task["file_id"]
        print(f"Looking for file: {file_id}")  # Add logging
        
        upload_files = [f for f in os.listdir("uploads") if f.startswith(file_id)]
        if not upload_files:
            raise Exception(f"Original file not found for {file_id}")
        
        file_path = f"uploads/{upload_files[0]}"
        print(f"Found file: {file_path}")  # Add logging
        
        df = pd.read_csv(file_path)
        task["progress"] = 30
        
        # Generate synthetic data
        print("Starting synthetic data generation")  # Add logging
        synth = enhanced_synthetic_tabular(df, task["n_rows"])
        task["progress"] = 70
        
        # Create quality report
        quality_report = create_quality_report(df, synth, task["target_column"])
        task["progress"] = 90
        
        # Save results
        os.makedirs("outputs", exist_ok=True)  # Ensure outputs directory exists
        output_path = f"outputs/synth_{task_id}.csv"
        synth.to_csv(output_path, index=False)
        
        task["status"] = "COMPLETED"
        task["progress"] = 100
        task["result"] = {
            "download_url": f"/download/synth_{task_id}.csv",
            "quality_report": quality_report,
            "generated_rows": len(synth)
        }
        print(f"Task {task_id} completed successfully")  # Add logging
        
    except Exception as e:
        print(f"Task {task_id} failed: {str(e)}")  # Add logging
        task["status"] = "FAILED"
        task["error"] = str(e)

@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get task status"""
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return JSONResponse(tasks_storage[task_id])

@app.get("/tasks")
async def get_all_tasks():
    """Get all generation tasks"""
    try:
        return list(tasks.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def analyze_dpdp_compliance(df: pd.DataFrame) -> list:
    """Analyze dataset for DPDP compliance issues"""
    PII_KEYWORDS = {
        "name", "email", "phone", "mobile", "address", "aadhaar", "pan", 
        "voter", "passport", "dob", "date_of_birth", "gender", "caste", "religion"
    }
    
    warnings = []
    
    for column in df.columns:
        column_lower = column.lower()
        for keyword in PII_KEYWORDS:
            if keyword in column_lower:
                warnings.append({
                    "column": column,
                    "issue": f"May contain Personal Information ({keyword})",
                    "recommendation": "Ensure you have valid legal basis under DPDP Act 2023"
                })
                break
    
    return warnings

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "SynData API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "SynData API is running"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated synthetic data"""
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/debug/task/{task_id}")
async def debug_task(task_id: str):
    """Debug endpoint to check task status"""
    task = tasks_storage.get(task_id)
    if not task:
        return {"error": "Task not found"}
    
    # Check if uploads directory exists
    uploads_exist = os.path.exists("uploads")
    upload_files = []
    if uploads_exist:
        upload_files = os.listdir("uploads")
    
    return {
        "task": task,
        "uploads_directory_exists": uploads_exist,
        "upload_files": upload_files,
        "functions_available": {
            "enhanced_synthetic_tabular": "enhanced_synthetic_tabular" in globals(),
            "create_quality_report": "create_quality_report" in globals()
        }
    }

@app.post("/generate-synthetic")
async def generate_synthetic_endpoint(
    file: UploadFile = File(...),
    rows: int = Form(100),
    target_column: str = Form(None)  # Add target column parameter
):
    """Generate synthetic data from uploaded CSV"""
    try:
        # Read uploaded file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        print(f"📊 Original data shape: {df.shape}")
        print(f"🎯 Target column: {target_column}")
        
        # Generate synthetic data
        synthetic_df = enhanced_synthetic_tabular(df, rows)
        
        # Create quality report
        quality_report = create_quality_report(df, synthetic_df, target_column)
        
        # Convert to CSV for download
        output = io.StringIO()
        synthetic_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        return {
            "synthetic_data": csv_content,
            "quality_report": quality_report,
            "original_shape": df.shape,
            "synthetic_shape": synthetic_df.shape
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
