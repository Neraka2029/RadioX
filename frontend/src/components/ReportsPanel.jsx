import { useState, useEffect } from "react";
import pdfService from "../services/pdfService";
import {
  mergeDisplayPredictions,
  buildClinicalConclusion,
} from "../utils/clinicalPriority";
import "../styles/reports.css";

export default function ReportsPanel() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const response = await fetch("http://localhost:8001/reports");
        const data = await response.json();
        setReports(data.reports || []);
      } catch (error) {
        console.error("Failed to fetch reports:", error);
        // Fallback vers données de démonstration
        setReports([
          {
            id: "RPT-DEMO-001",
            title: "Analyse thoracique - Démo",
            patient: "Patient Démo",
            date: "2024-03-11",
            type: "analyse_complete",
            findings: ["Normal", "Aucune anomalie détectée"],
            confidence: 94,
            images: 1,
            status: "completed",
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case "completed": return "#00ffc8";
      case "review": return "#ff9f43";
      case "draft": return "#5352ed";
      default: return "#ff4757";
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case "completed": return "Terminé";
      case "review": return "En révision";
      case "draft": return "Brouillon";
      default: return "Erreur";
    }
  };

  const handleExportPDF = async (report, imageData = null) => {
    setExporting(true);
    
    try {
      // Si aucune image n'est fournie, essayer de la récupérer depuis le backend
      let finalImageData = imageData;
      
      if (!finalImageData && report.analysis_id) {
        try {
          console.log("=== PDF IMAGE DEBUG ===");
          console.log("Analysis ID:", report.analysis_id);
          
          // Récupérer l'image et la Grad-CAM depuis les données d'analyse originales
          const response = await fetch(`http://localhost:8001/analysis/${report.analysis_id}/image`);
          console.log("Image response status:", response.status);
          console.log("Image response ok:", response.ok);
          
          if (response.ok) {
            const imageDataResult = await response.json();
            console.log("Image data result:", imageDataResult);
            if (imageDataResult.success) {
              // Créer un objet avec l'image et la Grad-CAM
              finalImageData = {
                image_base64: imageDataResult.image_base64,
                heatmap_base64: imageDataResult.heatmap_base64 || null  // ← RÉCUPÉRER DU BACKEND
              };
              console.log("Final image data with heatmap:", finalImageData);
            } else {
              console.log("Image data failed:", imageDataResult);
            }
          } else {
            console.log("Image fetch failed with status:", response.status);
          }
        } catch (error) {
          console.warn('Impossible de récupérer l\'image depuis le backend:', error);
        }
      }
      
      // Préparer les données pour le PDF
      const analysisData = {
        patientId: report.patient,
        analysis_id: report.analysis_id,
        date: report.date,
        confidence: report.confidence,
        predictions: report.predictions || [],
        tuberculosis_probability: report.tuberculosis_probability,
        fracture_risk_score: report.fracture_risk_score,
        fracture_mode: report.fracture_mode,
        fracture_detections: report.fracture_detections,
        recommendations: report.recommendations || report.findings || [],
      };
      
      // Générer le PDF
      const result = await pdfService.exportToPDF(analysisData, finalImageData);
      
      if (result.success) {
        console.log(`PDF exporté: ${result.filename}`);
        // Optionnel: afficher une notification de succès
      } else {
        console.error('Erreur export PDF:', result.error);
        // Optionnel: afficher une notification d'erreur
      }
    } catch (error) {
      console.error('Erreur lors de l\'export PDF:', error);
    } finally {
      setExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="reports-panel">
        <div className="reports-header">
          <h2>📄 Rapports médicaux</h2>
        </div>
        <div className="reports-loading">
          <div className="loading-spinner"></div>
          <p>Chargement des rapports...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="reports-panel">
      <div className="reports-header">
        <h2>📄 Rapports médicaux</h2>
        <button className="create-report-btn">
          <span>+</span> Nouveau rapport
        </button>
      </div>

      <div className="reports-grid">
        {reports.map((report) => (
          <div key={report.id} className="report-card" onClick={() => setSelectedReport(report)}>
            <div className="report-header">
              <div className="report-id">{report.id}</div>
              <div className="report-status" style={{ color: getStatusColor(report.status) }}>
                <span className="status-dot"></span>
                {getStatusText(report.status)}
              </div>
            </div>

            <div className="report-title">{report.title}</div>
            
            <div className="report-meta">
              <div className="patient-info">
                <span className="patient-name">👤 {report.patient}</span>
                <span className="report-date">📅 {report.date}</span>
              </div>
              <div className="report-metrics">
                <span className="confidence">🎯 {report.confidence}%</span>
                <span className="image-count">🖼️ {report.images}</span>
              </div>
            </div>

            <div className="report-findings">
              <h4>Conclusion clinique :</h4>
              <ReportClinicalSummary report={report} />
            </div>

            <div className="report-actions">
              <button 
                className="action-btn primary" 
                onClick={() => setSelectedReport(report)}
              >
                Voir le rapport
              </button>
              <button 
                className="action-btn secondary" 
                onClick={() => handleExportPDF(report)}
                disabled={exporting}
              >
                {exporting ? 'Export...' : 'Exporter PDF'}
              </button>
            </div>
          </div>
        ))}
      </div>

      {reports.length === 0 && (
        <div className="empty-reports">
          <div className="empty-icon">📄</div>
          <h3>Aucun rapport disponible</h3>
          <p>Commencez par générer des analyses pour créer des rapports.</p>
        </div>
      )}

      {selectedReport && (
        <div className="report-modal" onClick={() => setSelectedReport(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{selectedReport.title}</h3>
              <button className="close-btn" onClick={() => setSelectedReport(null)}>×</button>
            </div>
            <div className="modal-body">
              <div className="report-details">
                <div className="detail-row">
                  <span className="label">ID Rapport :</span>
                  <span className="value">{selectedReport.id}</span>
                </div>
                <div className="detail-row">
                  <span className="label">Patient :</span>
                  <span className="value">{selectedReport.patient}</span>
                </div>
                <div className="detail-row">
                  <span className="label">Date :</span>
                  <span className="value">{selectedReport.date}</span>
                </div>
                <div className="detail-row">
                  <span className="label">Confiance :</span>
                  <span className="value" style={{ color: getConfidenceColor(selectedReport.confidence) }}>
                    {selectedReport.confidence}%
                  </span>
                </div>
              </div>
              
              <div className="findings-section">
                <h4>Conclusion clinique principale</h4>
                <ReportClinicalSummary report={selectedReport} detailed />
              </div>

              <div className="modal-actions">
                <div className="actions-container">
                  <button 
                    className="action-btn primary" 
                    disabled={exporting}
                    onClick={() => handleExportPDF(selectedReport)}
                  >
                    {exporting ? 'Export...' : 'Télécharger PDF'}
                  </button>
                  <button className="action-btn secondary">Partager</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  function getConfidenceColor(confidence) {
    if (confidence >= 80) return "#00ffc8";
    if (confidence >= 60) return "#ffd32a";
    return "#ff4757";
  }
}

function ReportClinicalSummary({ report, detailed = false }) {
  const analysisLike = {
    predictions: report.predictions || [],
    tuberculosis_probability: report.tuberculosis_probability,
    fracture_risk_score: report.fracture_risk_score,
    fracture_mode: report.fracture_mode,
    fracture_detections: report.fracture_detections,
  };

  const merged = mergeDisplayPredictions(analysisLike);
  const clinical = buildClinicalConclusion(analysisLike, merged);

  if (!clinical.primary) {
    return (
      <ul>
        <li>Aucune conclusion clinique disponible</li>
      </ul>
    );
  }

  const pct = Math.round(
    (clinical.primary.confidence ?? clinical.primary.probability ?? 0) * 100
  );

  return (
    <div className="report-clinical-summary">
      <p className="clinical-primary-line">
        <strong>{clinical.conclusionLine || `${clinical.primary.pathology} (${pct}%)`}</strong>
      </p>
      {detailed && clinical.contributors.length > 0 && (
        <>
          <p className="clinical-contributors-title">Anomalies contributrices :</p>
          <ul>
            {clinical.contributors.map((c) => (
              <li key={c.pathology}>
                {c.pathology} ({Math.round(c.probability * 100)}%)
              </li>
            ))}
          </ul>
        </>
      )}
      {!detailed && clinical.contributors.length > 0 && (
        <ul>
          {clinical.contributors.slice(0, 3).map((c) => (
            <li key={c.pathology}>
              {c.pathology} ({Math.round(c.probability * 100)}%)
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
