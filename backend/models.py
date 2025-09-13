"""
Database models for SynData platform persistence
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import TypeDecorator, CHAR
import os

# Database configuration - use centralized config
def get_database_url():
    """Get database URL from centralized config to avoid split-brain configuration"""
    try:
        from config import settings
        return settings.database_url
    except ImportError:
        # Fallback for when config is not yet available
        return os.getenv("DATABASE_URL", "sqlite:///./syndata.db")

DATABASE_URL = get_database_url()

# Create SQLAlchemy engine  
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(postgresql.UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value

class GenerationTask(Base):
    """Model for tracking data generation tasks"""
    __tablename__ = "generation_tasks"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    task_id = Column(String(255), unique=True, index=True)  # External task ID
    filename = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    progress = Column(Integer, default=0)  # Progress percentage
    n_rows = Column(Integer, nullable=False)
    target_column = Column(String(255), nullable=True)
    result_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Configuration and results
    config = Column(JSON, nullable=True)  # Store generation parameters
    metrics = Column(JSON, nullable=True)  # Store quality metrics

class UploadedFile(Base):
    """Model for tracking uploaded files"""
    __tablename__ = "uploaded_files"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    
    # File validation and metadata
    is_valid = Column(Boolean, default=True)
    validation_errors = Column(JSON, nullable=True)
    columns = Column(JSON, nullable=True)  # Store column information
    rows_count = Column(Integer, nullable=True)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)

class TaskManager:
    """Database-backed task management with proper session handling"""
    
    def __init__(self):
        # Remove shared session - will use per-operation sessions instead
        pass
    
    def create_task(self, task_id: str, filename: str, n_rows: int, 
                   target_column: Optional[str] = None, config: Optional[dict] = None) -> GenerationTask:
        """Create a new generation task with thread-safe session"""
        session = SessionLocal()
        try:
            task = GenerationTask(
                task_id=task_id,
                filename=filename,
                n_rows=n_rows,
                target_column=target_column,
                config=config or {}
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            return task
        finally:
            session.close()
    
    def get_task(self, task_id: str) -> Optional[GenerationTask]:
        """Get a task by task_id with thread-safe session"""
        session = SessionLocal()
        try:
            return session.query(GenerationTask).filter(
                GenerationTask.task_id == task_id
            ).first()
        finally:
            session.close()
    
    def update_task_status(self, task_id: str, status: str, 
                          progress: Optional[int] = None,
                          error_message: Optional[str] = None,
                          result_path: Optional[str] = None,
                          metrics: Optional[dict] = None):
        """Update task status and metadata with thread-safe session"""
        session = SessionLocal()
        try:
            task = session.query(GenerationTask).filter(
                GenerationTask.task_id == task_id
            ).first()
            
            if task:
                task.status = status
                task.updated_at = datetime.utcnow()
                
                if progress is not None:
                    task.progress = progress
                if error_message is not None:
                    task.error_message = error_message
                if result_path is not None:
                    task.result_path = result_path
                if metrics is not None:
                    task.metrics = metrics
                    
                # Set timestamps based on status
                if status == "running" and task.started_at is None:
                    task.started_at = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    task.completed_at = datetime.utcnow()
                    
                session.commit()
                return task
            return None
        finally:
            session.close()
    
    def get_all_tasks(self) -> list[GenerationTask]:
        """Get all tasks, ordered by creation date with thread-safe session"""
        session = SessionLocal()
        try:
            return session.query(GenerationTask).order_by(
                GenerationTask.created_at.desc()
            ).all()
        finally:
            session.close()
    
    def cleanup_old_tasks(self, days: int = 7):
        """Clean up tasks older than specified days with thread-safe session"""
        session = SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            old_tasks = session.query(GenerationTask).filter(
                GenerationTask.created_at < cutoff_date
            ).all()
            
            for task in old_tasks:
                session.delete(task)
            session.commit()
            return len(old_tasks)
        finally:
            session.close()
    
    def close(self):
        """Close database session - no longer needed with per-operation sessions"""
        pass

class FileManager:
    """Database-backed file management with proper session handling"""
    
    def __init__(self):
        # Remove shared session - will use per-operation sessions instead
        pass
    
    def register_upload(self, filename: str, original_filename: str, 
                       file_path: str, file_size: int, content_type: str,
                       columns: Optional[list] = None, rows_count: Optional[int] = None) -> UploadedFile:
        """Register a new uploaded file with thread-safe session"""
        session = SessionLocal()
        try:
            file_record = UploadedFile(
                filename=filename,
                original_filename=original_filename,
                file_path=file_path,
                file_size=file_size,
                content_type=content_type,
                columns=columns,
                rows_count=rows_count
            )
            session.add(file_record)
            session.commit()
            session.refresh(file_record)
            return file_record
        finally:
            session.close()
    
    def get_file(self, filename: str) -> Optional[UploadedFile]:
        """Get file record by filename with thread-safe session"""
        session = SessionLocal()
        try:
            return session.query(UploadedFile).filter(
                UploadedFile.filename == filename
            ).first()
        finally:
            session.close()
    
    def get_all_files(self) -> list[UploadedFile]:
        """Get all uploaded files with thread-safe session"""
        session = SessionLocal()
        try:
            return session.query(UploadedFile).order_by(
                UploadedFile.uploaded_at.desc()
            ).all()
        finally:
            session.close()
    
    def update_last_accessed(self, filename: str):
        """Update last accessed timestamp with thread-safe session"""
        session = SessionLocal()
        try:
            file_record = session.query(UploadedFile).filter(
                UploadedFile.filename == filename
            ).first()
            if file_record:
                file_record.last_accessed = datetime.utcnow()
                session.commit()
        finally:
            session.close()
    
    def close(self):
        """Close database session - no longer needed with per-operation sessions"""
        pass

# Initialize database tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    """FastAPI dependency for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()