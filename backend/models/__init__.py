"""
SQLAlchemy ORM Models
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), default=generate_uuid, unique=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(100), default="Médecin")
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    analyses = relationship("Analysis", back_populates="user")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True)
    name = Column(String(255))
    age = Column(Integer)
    gender = Column(String(10))
    dob = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analyses = relationship("Analysis", back_populates="patient")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(50), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)

    # Image
    image_path = Column(String(500))
    heatmap_path = Column(String(500), nullable=True)
    image_filename = Column(String(255))
    image_size = Column(Integer)

    # Results
    predictions = Column(JSON)  # List of {pathology, probability, severity}
    primary_finding = Column(String(100))
    confidence = Column(Float)
    model_version = Column(String(50))
    processing_time_ms = Column(Integer)
    recommendations = Column(JSON)

    # Status
    status = Column(String(20), default="completed")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="analyses")
    patient = relationship("Patient", back_populates="analyses")
