import { useState, useEffect } from "react";
import "../styles/history.css";

export default function HistoryPanel() {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch("http://localhost:8001/history");
        if (response.ok) {
          const data = await response.json();
          setAnalyses(data.analyses || []);
        } else {
          throw new Error("Failed to fetch history");
        }
      } catch (error) {
        console.warn("Impossible de charger l'historique, utilisation des valeurs par défaut");
        // Données de fallback
        setAnalyses([
          {
            id: "AX-DEMO-001",
            date: "2024-03-11 10:30",
            patient: "Patient Démo",
            primaryFinding: "Normal",
            confidence: 94,
            status: "completed",
            imageCount: 1,
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const filteredAnalyses = analyses.filter(analysis => {
    if (filter === "all") return true;
    return analysis.status === filter;
  });

  const getStatusColor = (status) => {
    switch (status) {
      case "completed": return "#00ffc8";
      case "pending": return "#ff9f43";
      case "failed": return "#ff4757";
      default: return "#5352ed";
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return "#00ffc8";
    if (confidence >= 60) return "#ffd32a";
    return "#ff4757";
  };

  if (loading) {
    return (
      <div className="history-panel">
        <div className="history-header">
          <h2>📋 Historique des analyses</h2>
        </div>
        <div className="history-loading">
          <div className="loading-spinner"></div>
          <p>Chargement de l'historique...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="history-panel">
      <div className="history-header">
        <h2>📋 Historique des analyses</h2>
        <div className="history-filters">
          <button 
            className={`filter-btn ${filter === "all" ? "active" : ""}`}
            onClick={() => setFilter("all")}
          >
            Tout ({analyses.length})
          </button>
          <button 
            className={`filter-btn ${filter === "completed" ? "active" : ""}`}
            onClick={() => setFilter("completed")}
          >
            Terminées ({analyses.filter(a => a.status === "completed").length})
          </button>
        </div>
      </div>

      <div className="history-stats">
        <div className="stat-card">
          <div className="stat-value">{analyses.length}</div>
          <div className="stat-label">Total analyses</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {Math.round(analyses.reduce((sum, a) => sum + a.confidence, 0) / analyses.length)}%
          </div>
          <div className="stat-label">Confiance moyenne</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{analyses.reduce((sum, a) => sum + a.imageCount, 0)}</div>
          <div className="stat-label">Total images</div>
        </div>
      </div>

      <div className="history-list">
        {filteredAnalyses.map((analysis) => (
          <div key={analysis.id} className="history-item">
            <div className="history-item-header">
              <div className="analysis-id">{analysis.id}</div>
              <div className="analysis-date">{analysis.date}</div>
            </div>
            
            <div className="history-item-content">
              <div className="analysis-info">
                <div className="patient-info">
                  <span className="patient-name">{analysis.patient}</span>
                  <span className="finding" style={{ color: getConfidenceColor(analysis.confidence) }}>
                    {analysis.primaryFinding}
                  </span>
                </div>
                <div className="analysis-metrics">
                  <span className="confidence" style={{ color: getConfidenceColor(analysis.confidence) }}>
                    {analysis.confidence}% de confiance
                  </span>
                  <span className="image-count">{analysis.imageCount} image(s)</span>
                </div>
              </div>
              
              <div className="analysis-actions">
                <div className="status-indicator" style={{ color: getStatusColor(analysis.status) }}>
                  <span className="status-dot"></span>
                  {analysis.status === "completed" ? "Terminée" : analysis.status}
                </div>
                <button className="action-btn">Voir détails</button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredAnalyses.length === 0 && (
        <div className="empty-history">
          <div className="empty-icon">📋</div>
          <h3>Aucune analyse trouvée</h3>
          <p>Aucune analyse ne correspond aux filtres sélectionnés.</p>
        </div>
      )}
    </div>
  );
}
