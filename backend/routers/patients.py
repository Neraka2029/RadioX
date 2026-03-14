"""Patients router"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database import get_db
from models import Patient, User
from routers.auth import get_current_user

router = APIRouter()


class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    patient_id: Optional[str] = None


@router.post("/")
async def create_patient(
    patient_data: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    import uuid
    patient = Patient(
        patient_id=patient_data.patient_id or f"PT-{uuid.uuid4().hex[:8].upper()}",
        name=patient_data.name,
        age=patient_data.age,
        gender=patient_data.gender,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return {"id": patient.id, "patient_id": patient.patient_id, "name": patient.name}


@router.get("/")
async def list_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patients = db.query(Patient).order_by(Patient.id.desc()).limit(50).all()
    return [{"id": p.id, "patient_id": p.patient_id, "name": p.name, "age": p.age} for p in patients]
