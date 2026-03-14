"""
RadioX.AI - Backend FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging

from routers import auth, analysis, patients, reports
from database import engine, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RadioX.AI API",
    description="API d'analyse de radiographies thoraciques par intelligence artificielle",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploads
os.makedirs("uploads", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/heatmaps", StaticFiles(directory="heatmaps"), name="heatmaps")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])


@app.get("/api/health")
async def health_check():
    return {
        "status": "operational",
        "service": "RadioX.AI Backend",
        "version": "1.0.0",
        "ml_service": "connected",
    }


@app.get("/")
async def root():
    return {"message": "RadioX.AI API - Documentation: /api/docs"}
