"""
Database configuration using SQLAlchemy with PostgreSQL
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://radiox:radiox_password@localhost:5432/radiox_db"
)

# Fallback to SQLite for development/testing
SQLITE_URL = "sqlite:///./radiox_dev.db"

try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )
    # Test connection
    with engine.connect() as conn:
        pass
except Exception:
    print("PostgreSQL not available, using SQLite for development")
    engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
