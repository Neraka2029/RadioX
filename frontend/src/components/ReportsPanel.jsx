import { useState, useEffect } from "react";
import pdfService from "../services/pdfService";
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
        if (response.ok) {
          const data = await response.json();
          setReports(data.reports || []);
        } else {
          throw new Error("Failed to fetch reports");
        }
      } catch (error) {
        console.warn("Impossible de charger les rapports, utilisation des valeurs par défaut");
        // Données de fallback
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
          const response = await fetch(`http://localhost:8001/analysis/${report.analysis_id}/image`);
          if (response.ok) {
            const imageDataResult = await response.json();
            if (imageDataResult.success) {
              // Convertir base64 en data URL
              finalImageData = `data:image/jpeg;base64,${imageDataResult.image_base64}`;
              console.log('Image récupérée depuis le backend pour le PDF');
            }
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
        recommendations: report.findings || []
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

  const handlePrintReport = (report) => {
    // Créer une fenêtre d'impression avec le contenu du rapport
    const printWindow = window.open('', '_blank');
    
    if (printWindow) {
      const htmlContent = generatePrintHTML(report);
      printWindow.document.write(htmlContent);
      printWindow.document.close();
      
      // Attendre que le contenu soit chargé puis imprimer
      printWindow.onload = () => {
        printWindow.print();
        printWindow.close();
      };
    }
  };

  const generatePrintHTML = (report) => {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Rapport RadioX - ${report.patient}</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          .header { text-align: center; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }
          .patient-info { margin: 20px 0; }
          .findings { margin: 20px 0; }
          .confidence { font-weight: bold; color: #00d4ff; }
          @media print { body { margin: 10px; } }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>RADIOX - Rapport d'Analyse Radiographique</h1>
          <p>Modèle IA: DenseNet-121 NIH ChestX-ray14</p>
          <p>Date: ${report.date}</p>
        </div>
        
        <div class="patient-info">
          <h2>Informations du Patient</h2>
          <p><strong>ID Patient:</strong> ${report.patient}</p>
          <p><strong>ID Analyse:</strong> ${report.analysis_id}</p>
          <p><strong>Confiance:</strong> <span class="confidence">${report.confidence}%</span></p>
        </div>
        
        <div class="findings">
          <h2>Détections Principales</h2>
          <ul>
            ${report.findings ? report.findings.map(f => `<li>${f}</li>`).join('') : '<li>Aucune pathologie détectée</li>'}
          </ul>
        </div>
        
        <div style="margin-top: 30px; font-style: italic; color: #666;">
          <p>Ce rapport a été généré par RadioX - Les résultats doivent être confirmés par un professionnel de santé qualifié.</p>
        </div>
      </body>
      </html>
    `;
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
              <h4>Détections principales :</h4>
              <ul>
                {report.findings.map((finding, index) => (
                  <li key={index}>{finding}</li>
                ))}
              </ul>
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
              <button 
                className="action-btn secondary" 
                onClick={() => handlePrintReport(report)}
              >
                Imprimer
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
                <h4>Détections IA</h4>
                {selectedReport.findings.map((finding, index) => (
                  <div key={index} className="finding-item">
                    <span className="finding-bullet">•</span>
                    {finding}
                  </div>
                ))}
              </div>

              <div className="report-actions-modal">
                <button 
                  className="action-btn primary" 
                  onClick={() => handleExportPDF(selectedReport)}
                  disabled={exporting}
                >
                  {exporting ? 'Export...' : 'Télécharger PDF'}
                </button>
                <button 
                  className="action-btn secondary" 
                  onClick={() => handlePrintReport(selectedReport)}
                >
                  Imprimer
                </button>
                <button className="action-btn secondary">Partager</button>
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
