"""
Centralized configuration management for SynData platform
"""
import os
from typing import List, Optional
from pydantic import BaseModel, field_validator

class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # API Configuration
    app_name: str = "SynData Plus API"
    app_version: str = "2.1"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 5000
    
    # Database Configuration
    database_url: str = "sqlite:///./syndata.db"
    
    # File Upload Configuration
    max_file_size_mb: int = 100  # Maximum file size in MB
    allowed_file_types: List[str] = ["text/csv", "application/csv", "text/plain"]
    upload_dir: str = "uploads"
    output_dir: str = "outputs"
    
    # Security Configuration  
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5000",
        "https://replit.dev",
        "https://replit.app"
    ]
    cors_allow_credentials: bool = False
    trust_proxy_headers: bool = True  # Set to False in production behind untrusted proxies
    trusted_proxies: List[str] = ["127.0.0.1", "::1"]  # Configure actual proxy/LB IPs in production
    force_https: bool = False  # Set to True in production for HSTS
    csp: str = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none';"  # Content Security Policy
    
    # Rate Limiting Configuration
    rate_limit_requests: int = 100  # Requests per minute
    rate_limit_window: int = 60     # Window in seconds
    
    # Task Management Configuration
    max_workers: int = 2            # ThreadPoolExecutor workers
    task_timeout: int = 3600        # Task timeout in seconds (1 hour)
    task_cleanup_days: int = 7      # Days to keep completed tasks
    
    # Generation Configuration
    default_synthetic_rows: int = 1000
    max_synthetic_rows: int = 50000
    
    # File Cleanup Configuration
    temp_file_cleanup_hours: int = 24
    
    @field_validator('database_url', mode='before')
    @classmethod
    def get_database_url(cls, v):
        """Get database URL from environment or use default"""
        return os.getenv('DATABASE_URL', v)
    
    @field_validator('max_file_size_mb')
    @classmethod
    def validate_file_size(cls, v):
        """Ensure reasonable file size limits"""
        if v < 1 or v > 1000:
            raise ValueError('File size must be between 1MB and 1000MB')
        return v
    
    @field_validator('max_synthetic_rows')
    @classmethod
    def validate_max_rows(cls, v):
        """Ensure reasonable row limits"""
        if v < 100 or v > 1000000:
            raise ValueError('Max synthetic rows must be between 100 and 1,000,000')
        return v
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable if provided as string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = "SYNDATA_"

# Initialize settings with environment variables
def load_settings():
    """Load settings from environment variables"""
    return Settings(
        database_url=os.getenv('DATABASE_URL', 'sqlite:///./syndata.db'),
        debug=os.getenv('DEBUG', 'false').lower() == 'true',
        cors_origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5000,https://*.replit.dev,https://*.replit.app').split(',')
    )

# Global settings instance
settings = load_settings()

# Configuration validation
def validate_config():
    """Validate configuration and create necessary directories"""
    import os
    
    # Create directories if they don't exist
    for directory in [settings.upload_dir, settings.output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    # Validate database connection
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"✅ Database connection successful: {settings.database_url[:50]}...")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
    
    print(f"🚀 Configuration loaded:")
    print(f"   - Server: {settings.host}:{settings.port}")
    print(f"   - Max file size: {settings.max_file_size_mb}MB")
    print(f"   - Max workers: {settings.max_workers}")
    print(f"   - Rate limit: {settings.rate_limit_requests} req/min")

# Security helpers
def is_allowed_file_type(content_type: str) -> bool:
    """Check if file type is allowed"""
    return content_type.lower() in [ft.lower() for ft in settings.allowed_file_types]

def get_max_file_size_bytes() -> int:
    """Get maximum file size in bytes"""
    return settings.max_file_size_mb * 1024 * 1024

# Environment detection
def is_development() -> bool:
    """Check if running in development mode"""
    return settings.debug or os.getenv('ENVIRONMENT', '').lower() in ['dev', 'development']

def is_production() -> bool:
    """Check if running in production mode"""
    return os.getenv('ENVIRONMENT', '').lower() in ['prod', 'production']