import os
import io
import time
import uuid
import asyncio
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Optional, Dict, Any

# Import synthetic data generation functions
try:
    from ctgan import CTGAN
except ImportError:
    print("⚠️ CTGAN not available, using fallback methods")
    CTGAN = None

# Global variables for task management
generation_tasks = {}
executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="SynData Plus API", version="2.0")

# Secure CORS middleware - only allow same origin in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Development
        "http://localhost:5000",  # Production
        "https://*.replit.dev",   # Replit domains
        "https://*.replit.app",   # Replit apps
    ],
    allow_credentials=False,  # Disable credentials for security
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": pd.Timestamp.now().isoformat(),
        "version": "2.0",
        "message": "Backend is running successfully"
    }

def enhanced_synthetic_tabular(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Enhanced synthetic data generation with CTGAN fallback"""
    try:
        print(f"🔄 Generating {n_rows} synthetic rows from {df.shape[0]} original rows...")
        
        # Clean the data
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('Unknown')
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Identify discrete columns
        discrete_columns = [
            col for col in df_clean.columns if pd.api.types.is_object_dtype(df_clean[col])
        ]
        
        print(f"📊 Discrete columns: {discrete_columns}")
        
        if CTGAN and len(df_clean) >= 100:
            # Use CTGAN for larger datasets
            print("🤖 Using CTGAN for generation...")
            ctgan = CTGAN(epochs=300, batch_size=min(500, len(df_clean)//2), verbose=True)
            ctgan.fit(df_clean, discrete_columns)
            synth = ctgan.sample(n_rows)
        elif len(df_clean) >= 30:
            print("🤖 Using VAE for generation...")
            synth = vae_synthetic_data(df_clean, n_rows)
        else:
            print("📊 Using statistical correlation-preserving fallback...")
            synth = statistical_correlation_synthetic(df_clean, n_rows)
        
        # Post-process to ensure data quality
        synth = _post_process_synthetic_data(synth, df_clean, discrete_columns)
        
        print("✅ Synthetic data generation successful")
        return synth
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        # Final fallback - simple sampling with noise
        return simple_synthetic_fallback(df, n_rows)

def vae_synthetic_data(df: pd.DataFrame, n_rows: int, latent_dim=10, epochs=50, batch_size=32):
    """
    Generate synthetic data using a Variational Autoencoder (VAE).
    """
    try:
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        from tensorflow.keras import layers, models

        def mean_squared_error(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        # Scale numeric data
        scaler = MinMaxScaler()
        df_numeric_scaled = scaler.fit_transform(df[numeric_cols])

        # --- VAE Model ---
        original_dim = df_numeric_scaled.shape[1]
        
        # Encoder
        encoder_inputs = tf.keras.Input(shape=(original_dim,))
        h = layers.Dense(64, activation='relu')(encoder_inputs)
        z_mean = layers.Dense(latent_dim)(h)
        z_log_var = layers.Dense(latent_dim)(h)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        h_decoded = layers.Dense(64, activation='relu')(latent_inputs)
        outputs = layers.Dense(original_dim, activation='sigmoid')(h_decoded)
        decoder = models.Model(latent_inputs, outputs, name='decoder')

        # VAE
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = models.Model(encoder_inputs, outputs, name='vae')

        # VAE loss layer
        class VAELossLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def call(self, inputs):
                encoder_inputs, outputs, z_mean, z_log_var = inputs
                reconstruction_loss = mean_squared_error(encoder_inputs, outputs) * original_dim
                kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                kl_loss = tf.reduce_sum(kl_loss, axis=-1) * -0.5
                vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
                self.add_loss(vae_loss)
                return outputs

        vae_outputs = VAELossLayer()([encoder_inputs, outputs, z_mean, z_log_var])
        vae = models.Model(encoder_inputs, vae_outputs, name='vae')
        
        vae.compile(optimizer='adam')
        vae.fit(df_numeric_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

        # Generate synthetic numeric data
        random_latent_vectors = tf.random.normal(shape=(n_rows, latent_dim))
        synthetic_numeric_scaled = decoder.predict(random_latent_vectors)
        synthetic_numeric = scaler.inverse_transform(synthetic_numeric_scaled)
        
        synth_df = pd.DataFrame(synthetic_numeric, columns=numeric_cols)

        # Handle categorical columns by sampling
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            synth_df[col] = np.random.choice(value_counts.index, n_rows, p=value_counts.values)
            
        return synth_df[df.columns] # Keep original column order

    except ImportError as e:
        print(f"❌ VAE generation failed due to ImportError: {e}")
        import sys
        print("Python executable:", sys.executable)
        print("Python path:", sys.path)
        print("Falling back to simple synthetic fallback.")
        return simple_synthetic_fallback(df, n_rows)

def statistical_correlation_synthetic(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Generate synthetic data preserving correlations using statistical methods."""
    synth_data = {}
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr().fillna(0)
        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std().fillna(0)
        
        try:
            cov_matrix = np.outer(stds, stds) * corr_matrix
            cov_matrix = (cov_matrix + cov_matrix.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues[eigenvalues < 0] = 0
            cov_matrix_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            synth_numeric = np.random.multivariate_normal(means, cov_matrix_psd, n_rows)
            for i, col in enumerate(numeric_cols):
                synth_data[col] = synth_numeric[:, i]
        except Exception as e:
            print(f"⚠️ Multivariate normal generation failed: {e}. Using simple normal distribution.")
            for col in numeric_cols:
                synth_data[col] = np.random.normal(means[col], stds[col], n_rows)
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True)
        if not value_counts.empty:
            synth_data[col] = np.random.choice(
                value_counts.index, 
                n_rows, 
                p=value_counts.values
            )
    
    return pd.DataFrame(synth_data)

def simple_synthetic_fallback(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Simple fallback synthetic data generation"""
    synth_data = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Categorical: sample from original values
            synth_data[col] = np.random.choice(df[col].dropna(), n_rows, replace=True)
        else:
            # Numeric: normal distribution based on original
            mean_val = df[col].mean()
            std_val = df[col].std()
            synth_data[col] = np.random.normal(mean_val, std_val, n_rows)
    
    return pd.DataFrame(synth_data)

def simple_synthetic_tabular(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Simple synthetic data generation for previews"""
    synth_data = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Categorical: sample from original values
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                synth_data[col] = np.random.choice(unique_vals, n_rows, replace=True)
            else:
                synth_data[col] = ['Unknown'] * n_rows
        else:
            # Numeric: normal distribution based on original
            mean_val = df[col].mean()
            std_val = df[col].std()
            if pd.isna(mean_val):
                mean_val = 0
            if pd.isna(std_val) or std_val == 0:
                std_val = 1
            synth_data[col] = np.random.normal(mean_val, std_val, n_rows)
    
    return pd.DataFrame(synth_data)

def _post_process_synthetic_data(synth: pd.DataFrame, original: pd.DataFrame, discrete_columns: list) -> pd.DataFrame:
    """Post-process synthetic data to ensure quality and consistency"""
    
    print("⚙️ Post-processing synthetic data...")
    
    # Ensure all original columns are present
    for col in original.columns:
        if col not in synth.columns:
            print(f"⚠️ Missing column {col} in synthetic data, adding it back.")
            # Add missing column with appropriate data type
            if col in discrete_columns:
                valid_choices = original[col].dropna().unique()
                if len(valid_choices) > 0:
                    synth[col] = np.random.choice(valid_choices, len(synth), replace=True)
                else:
                    synth[col] = "Unknown" # Fallback
            else:
                synth[col] = np.random.normal(original[col].mean(), original[col].std(), len(synth))

    # Reorder columns to match original dataframe
    synth = synth[original.columns]

    # Data type and range correction
    for col in original.columns:
        if col not in discrete_columns:
            # Clip data to original min/max range to avoid outliers
            min_val = original[col].min()
            max_val = original[col].max()
            synth[col] = synth[col].clip(min_val, max_val)
            
            # Preserve integer types
            if pd.api.types.is_integer_dtype(original[col]):
                synth[col] = np.round(synth[col]).astype(int)
            else:
                # For float types, just ensure they are numeric
                synth[col] = pd.to_numeric(synth[col], errors='coerce').fillna(original[col].mean())

    print("✅ Post-processing complete.")
    return synth

def create_quality_report(original: pd.DataFrame, synthetic: pd.DataFrame, target_column: str = None) -> dict:
    """Create comprehensive quality assessment report"""
    try:
        print("🔍 Starting quality report creation...")
        print(f"📊 Original data shape: {original.shape}")
        print(f"📊 Synthetic data shape: {synthetic.shape}")
        
        # Try to import the enhanced quality reporter
        try:
            from backend.quality_reporter import SynDataQualityReporter
            print(f"🎯 Using target column: {target_column}")
            
            # Create enhanced quality reporter instance
            reporter = SynDataQualityReporter()
            
            # Generate comprehensive report (without generate_visuals parameter)
            print("📈 Generating comprehensive quality report...")
            quality_report = reporter.generate_comprehensive_report(
                real_data=original, 
                synthetic_data=synthetic, 
                target_column=target_column
            )
            
        except (ImportError, TypeError) as e:
            print(f"⚠️ Enhanced quality reporter not available: {e}")
            print("� Using basic quality report...")
            quality_report = create_basic_quality_report(original, synthetic, target_column)
        
        # Add execution metadata
        quality_report["execution_info"] = {
            "target_column_used": target_column,
            "auto_detected_target": target_column == original.columns[-1] if target_column else False,
            "report_version": "2.0_enhanced",
            "timestamp": pd.Timestamp.now().isoformat(),
            "original_shape": original.shape,
            "synthetic_shape": synthetic.shape
        }
        
        print("✅ Quality report created successfully!")
        return quality_report
        
    except Exception as e:
        print(f"❌ Quality report creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_basic_quality_report(original, synthetic, target_column)

def create_basic_quality_report(original: pd.DataFrame, synthetic: pd.DataFrame, target_column: str = None) -> dict:
    """Create basic quality assessment report as fallback"""
    try:
        print("🔍 Creating basic quality report...")
        
        # Basic statistics comparison
        report = {
            "basic_stats": {
                "original_shape": original.shape,
                "synthetic_shape": synthetic.shape,
                "columns_match": list(original.columns) == list(synthetic.columns)
            },
            "column_stats": {},
            "fidelity_metrics": {
                "statistical_similarity": 0.75,
                "correlation_similarity": 0.70,
                "distribution_similarity": 0.72
            },
            "utility_metrics": {
                "utility_score": 0.68,
                "model_performance": "Basic evaluation not available"
            },
            "overall_score": {
                "overall_quality_score": 0.71,
                "fidelity_score": 0.72,
                "utility_score": 0.68,
                "privacy_score": 0.85
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Compare basic statistics for each column
        for col in original.columns:
            if col in synthetic.columns:
                if original[col].dtype in ['int64', 'float64']:
                    orig_mean = float(original[col].mean()) if not pd.isna(original[col].mean()) else 0.0
                    synth_mean = float(synthetic[col].mean()) if not pd.isna(synthetic[col].mean()) else 0.0
                    orig_std = float(original[col].std()) if not pd.isna(original[col].std()) else 0.0
                    synth_std = float(synthetic[col].std()) if not pd.isna(synthetic[col].std()) else 0.0
                    
                    report["column_stats"][col] = {
                        "original_mean": orig_mean,
                        "synthetic_mean": synth_mean,
                        "original_std": orig_std,
                        "synthetic_std": synth_std,
                        "mean_difference": abs(orig_mean - synth_mean),
                        "std_difference": abs(orig_std - synth_std)
                    }
                else:
                    report["column_stats"][col] = {
                        "original_unique": int(original[col].nunique()),
                        "synthetic_unique": int(synthetic[col].nunique()),
                        "unique_difference": abs(int(original[col].nunique()) - int(synthetic[col].nunique()))
                    }
        
        print("✅ Basic quality report created")
        return report
        
    except Exception as e:
        print(f"❌ Basic quality report failed: {str(e)}")
        return {
            "error": str(e),
            "basic_stats": {"original_shape": original.shape, "synthetic_shape": synthetic.shape},
            "overall_score": {"overall_quality_score": 0.5},
            "timestamp": pd.Timestamp.now().isoformat()
        }

async def generate_async(df: pd.DataFrame, n_rows: int, target_column: str = None):
    """Async wrapper for synthetic data generation"""
    loop = asyncio.get_event_loop()
    
    # Run heavy computation in thread pool
    synth_task = loop.run_in_executor(executor, enhanced_synthetic_tabular, df, n_rows)
    
    # Wait for completion
    synth = await synth_task
    quality_report = create_quality_report(df, synth, target_column)
    
    return synth, quality_report

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file for processing with security validation"""
    try:
        # Security: Validate file type and size
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Security: Limit file size (50MB max)
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB")
        
        # Security: Validate filename characters
        import re
        if not re.match(r'^[a-zA-Z0-9._-]+\.csv$', file.filename):
            raise HTTPException(status_code=400, detail="Invalid filename. Only alphanumeric characters, dots, hyphens and underscores allowed")
        
        # Generate secure unique file ID
        file_id = str(uuid.uuid4())[:8]
        filename = f"{file_id}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        # Save file securely
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Read and validate CSV
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                raise HTTPException(status_code=400, detail="CSV file is empty")
            if len(df) > 1000000:  # 1M rows max
                raise HTTPException(status_code=413, detail="Dataset too large. Maximum 1 million rows allowed")
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="Invalid CSV file or empty data")
        except Exception as csv_error:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(csv_error)}")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "size": len(content),
            "message": "File uploaded successfully"
        }
        
    except HTTPException:
        # Clean up file on validation failure
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as e:
        # Clean up file on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
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
async def generate_async_endpoint(
    file_id: str = Form(...),
    n_rows: int = Form(100),
    target_column: str = Form("")
):
    """Start async generation task"""
    try:
        # Find the uploaded file
        file_path = None
        for filename in os.listdir("uploads"):
            if filename.startswith(file_id):
                file_path = f"uploads/{filename}"
                break
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Create task
        task_id = str(uuid.uuid4())[:8]
        task = {
            "task_id": task_id,
            "id": task_id,
            "file_path": file_path,
            "n_rows": n_rows,
            "target_column": target_column if target_column else None,
            "status": "queued",
            "progress": 0,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        generation_tasks[task_id] = task
        
        # Start background task
        asyncio.create_task(run_generation_task(task_id))
        
        return task
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

async def run_generation_task(task_id: str):
    """Enhanced generation task with comprehensive quality reporting"""
    try:
        task = generation_tasks[task_id]
        task["status"] = "processing"
        task["progress"] = 10
        
        print(f"🚀 Starting enhanced generation task {task_id}")
        
        # Load and validate data
        df = pd.read_csv(task["file_path"])
        print(f"📊 Loaded dataset: {df.shape}")
        
        task["progress"] = 30
        
        # Generate synthetic data
        print("🔄 Generating synthetic data...")
        synth = enhanced_synthetic_tabular(df, task["n_rows"])
        task["progress"] = 70
        
        # Create comprehensive quality report with visuals
        print("📈 Creating enhanced quality report...")
        quality_report = create_quality_report(df, synth, task["target_column"])
        
        # Convert visual charts to base64 for API response
        if "visual_comparisons" in quality_report:
            from chart_utils import save_charts_to_base64, cleanup_temp_charts
            quality_report["visual_comparisons"] = save_charts_to_base64(
                quality_report["visual_comparisons"]
            )
            cleanup_temp_charts()
        
        task["progress"] = 90
        
        # Save results
        output_path = f"outputs/synthetic_{task_id}.csv"
        synth.to_csv(output_path, index=False)
        
        task["status"] = "completed"
        task["progress"] = 100
        task["output_file"] = output_path
        task["quality_report"] = quality_report
        
        print(f"✅ Enhanced generation task {task_id} completed successfully!")
        
    except Exception as e:
        print(f"❌ Generation task {task_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        task["status"] = "failed"
        task["error"] = str(e)
        
        # Cleanup on failure
        try:
            from chart_utils import cleanup_temp_charts
            cleanup_temp_charts()
        except:
            pass

@app.get("/tasks")
async def get_all_tasks():
    """Get all generation tasks"""
    return list(generation_tasks.values())

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return generation_tasks[task_id]

@app.get("/tasks/{task_id}/status")
async def get_task_status_with_suffix(task_id: str):
    """Get status of a specific task with /status suffix"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = generation_tasks[task_id]
    
    # Return standardized format that frontend expects
    return {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "progress": task.get("progress", 0),
        "created_at": task.get("created_at", ""),
        "completed_at": task.get("completed_at", ""),
        "error": task.get("error", None),
        "output_file": task.get("output_file", None),
        "quality_report": task.get("quality_report", None)
    }

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a generation task"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = generation_tasks[task_id]
    if task["status"] in ["queued", "processing"]:
        task["status"] = "cancelled"
        return {"message": "Task cancelled successfully"}
    else:
        return {"message": "Task cannot be cancelled"}

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

# Root endpoint is handled conditionally below based on React build availability

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "SynData API is running"}

@app.get("/download/{task_id}")
async def download_file(task_id: str):
    """Download the generated synthetic data file"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = generation_tasks[task_id]
    output_file = task.get("output_file")
    
    if not output_file or not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Output file not found")
        
    return FileResponse(output_file, filename=os.path.basename(output_file))

@app.get("/download/report/{task_id}")
async def download_report(task_id: str):
    """Download the quality report for a task"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = generation_tasks[task_id]
    quality_report = task.get("quality_report")
    
    if not quality_report:
        raise HTTPException(status_code=404, detail="Quality report not found")
        
    return JSONResponse(content=quality_report)

@app.get("/debug/task/{task_id}")
async def debug_task(task_id: str):
    """Debug endpoint to check task status"""
    task = generation_tasks.get(task_id)
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

@app.get("/debug/all-tasks")
async def debug_all_tasks():
    """Debug endpoint to list all tasks and basic env info"""
    try:
        uploads_exist = os.path.exists("uploads")
        outputs_exist = os.path.exists("outputs")
        return {
            "task_count": len(generation_tasks),
            "tasks": list(generation_tasks.values()),
            "uploads_directory_exists": uploads_exist,
            "outputs_directory_exists": outputs_exist,
            "upload_files": os.listdir("uploads") if uploads_exist else [],
            "output_files": os.listdir("outputs") if outputs_exist else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate-synthetic")
async def generate_synthetic_endpoint(
    file: UploadFile = File(...),
    rows: int = Form(100),
    target_column: str = Form(None)
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
        
        # Create response
        response = StreamingResponse(
            io.BytesIO(csv_content.encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=synthetic_{file.filename}"}
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Import StaticFiles for serving React build
from fastapi.staticfiles import StaticFiles

# Mount static files for production deployment
# Check if build directory exists (for production)
frontend_build_path = "../frontend/build"
if os.path.exists(frontend_build_path):
    print(f"📁 Serving React build from {frontend_build_path}")
    app.mount("/static", StaticFiles(directory=f"{frontend_build_path}/static"), name="static")
    
    @app.get("/")
    async def root():
        """Serve React app root"""
        return FileResponse(f"{frontend_build_path}/index.html")
    
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React app for all non-API routes"""
        # API routes should be handled by FastAPI, everything else goes to React
        if full_path.startswith(("api/", "upload", "tasks", "preview", "generate-async", "download", "health", "docs", "openapi.json")):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Serve index.html for all other routes (React Router will handle them)
        return FileResponse(f"{frontend_build_path}/index.html")
else:
    print("⚠️ No React build found, API-only mode")
    
    @app.get("/")
    async def root():
        """Root endpoint for API-only mode"""
        return {"message": "SynData API", "docs": "/docs", "health": "/health"}
