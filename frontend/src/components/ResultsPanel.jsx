import { useState, useEffect } from "react";
import "../styles/results.css";

const PATHOLOGY_COLORS = {
  "Normal": "#00ffc8",
  "Atelectasie": "#ff6b6b",
  "Consolidation": "#ff9f43",
  "Infiltration": "#ffd32a",
  "Pneumothorax": "#ff4757",
  "Oedeme": "#5352ed",
  "Emphyseme": "#ff6348",
  "Fibrose": "#a29bfe",
  "Epanchement pleural": "#00d2d3",
  "Pneumonie": "#ff4466",
  "Epaississement pleural": "#54a0ff",
  "Cardiomegalie": "#48dbfb",
  "Nodule": "#ff9ff3",
  "Masse": "#ee5a24",
  "Hernie": "#c8d6e5",
};

const SEVERITY_LABELS = {
  high: { label: "Élevé", className: "badge-critical" },
  moderate: { label: "Modéré", className: "badge-warning" },
  low: { label: "Faible", className: "badge-normal" },
};

function PathologyBar({ item, index }) {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), index * 100 + 200);
    return () => clearTimeout(t);
  }, [index]);

  const pct = Math.round(item.probability * 100);
  const color = PATHOLOGY_COLORS[item.pathology] || "#4488ff";
  const severity = SEVERITY_LABELS[item.severity] || SEVERITY_LABELS.low;

  return (
    <div className="pathology-bar-row">
      <div className="pathology-bar-header">
        <div className="pathology-name-row">
          <span className="pathology-dot" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
          <span className="pathology-name">{item.pathology}</span>
        </div>
        <div className="pathology-meta">
          <span className={`badge ${severity.className}`}>{severity.label}</span>
          <span className="pathology-pct" style={{ color }}>{pct}%</span>
        </div>
      </div>
      <div className="bar-track">
        <div
          className="bar-fill"
          style={{
            width: animated ? `${pct}%` : "0%",
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            boxShadow: pct > 50 ? `0 0 12px ${color}44` : "none",
          }}
        />
      </div>
    </div>
  );
}

function HeatmapOverlay({ imageUrl, heatmapUrl }) {
  const [opacity, setOpacity] = useState(0.7);
  const [show, setShow] = useState(true);

  // Convertir base64 en data URL si nécessaire
  const getHeatmapDataUrl = (url) => {
    if (!url) return null;
    
    // Si c'est déjà une data URL, la retourner telle quelle
    if (url.startsWith('data:')) {
      return url;
    }
    
    // Si c'est du base64 brut, le convertir en data URL
    if (url.length > 100 && !url.includes('http')) {
      return `data:image/png;base64,${url}`;
    }
    
    return url;
  };

  const heatmapDataUrl = getHeatmapDataUrl(heatmapUrl);
  
  // Debug: voir ce que l'on reçoit en détail
  console.log('=== HEATMAP DEBUG ===');
  console.log('Raw heatmapUrl:', heatmapUrl);
  console.log('Heatmap URL type:', typeof heatmapUrl);
  console.log('Heatmap URL length:', heatmapUrl?.length || 0);
  console.log('Heatmap Data URL:', heatmapDataUrl);
  console.log('===================');

  return (
    <div className="heatmap-section">
      <div className="heatmap-controls">
        <span className="heatmap-label">🔥 Carte de chaleur Grad-CAM</span>
        <div className="heatmap-controls-right">
          <button
            className={`heatmap-toggle ${show ? "active" : ""}`}
            onClick={() => setShow(!show)}
          >
            {show ? "Masquer" : "Afficher"}
          </button>
          {show && (
            <input
              type="range"
              min="0"
              max="100"
              value={Math.round(opacity * 100)}
              onChange={(e) => setOpacity(e.target.value / 100)}
              className="opacity-slider"
            />
          )}
        </div>
      </div>
      <div className="heatmap-viewer">
        <img src={imageUrl} alt="Radiographie" className="heatmap-base-img" />
        {show && heatmapDataUrl && (
          <img 
            src={heatmapDataUrl} 
            alt="Heatmap" 
            className="heatmap-overlay-img"
            style={{
              opacity: opacity * 0.8, // Réduire l'opacité pour ne pas masquer l'image
            }}
          />
        )}
        {show && !heatmapDataUrl && (
          <div
            className="heatmap-overlay"
            style={{
              opacity,
              background: `radial-gradient(ellipse 60% 50% at 45% 55%, rgba(255,68,102,0.8) 0%, rgba(255,170,0,0.5) 35%, rgba(68,136,255,0.2) 60%, transparent 80%)`,
            }}
          />
        )}
        <div className="heatmap-legend">
          <span className="legend-low">Faible</span>
          <div className="legend-gradient" />
          <span className="legend-high">Élevé</span>
        </div>
      </div>
    </div>
  );
}

export default function ResultsPanel({ results, isAnalyzing, uploadedImage }) {
  if (isAnalyzing) {
    return (
      <div className="results-panel results-loading">
        <div className="loading-state">
          <div className="loading-scanner">
            <div className="scanner-ring" />
            <div className="scanner-ring scanner-ring-2" />
            <span className="scanner-icon">🤖</span>
          </div>
          <h3>Analyse en cours...</h3>
          <p>Le modèle DenseNet-121 analyse votre radiographie</p>
          <div className="loading-steps">
            {["Chargement", "Normalisation", "Inférence", "Grad-CAM"].map((s, i) => (
              <div key={i} className="loading-step" style={{ animationDelay: `${i * 0.3}s` }}>
                <div className="step-dot" />
                <span>{s}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-panel results-empty">
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>En attente d'analyse</h3>
          <p>Déposez une radiographie thoracique pour obtenir une analyse IA détaillée des pathologies potentielles.</p>
          <div className="empty-pathology-list">
            {["Pneumonie", "Atelectasie", "Consolidation", "Pneumothorax", "Oedème", "Emphysème", "Fibrose", "Épanchement pleural", "Cardiomégalie", "Nodule", "Masse"].map((p) => (
              <span key={p} className="empty-pathology-tag">{p}</span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const primaryPrediction = results?.predictions?.reduce((a, b) =>
    a.probability > b.probability ? a : b
  );
  const primarySeverity = SEVERITY_LABELS[primaryPrediction?.severity] || SEVERITY_LABELS.low;

  return (
    <div className="results-panel">
      {/* Primary finding header */}
      <div className="results-header">
        <div className="results-finding">
          <div className="finding-tag">Résultat principal</div>
          <h2 className="finding-name">{primaryPrediction?.pathology}</h2>
          <div className="finding-meta">
            <span className={`badge ${primarySeverity.className}`}>
              {primarySeverity.label} risque
            </span>
            <span className="finding-confidence">
              Confiance : {Math.round((results?.confidence || 0.85) * 100)}%
            </span>
          </div>
        </div>
        <div className="results-meta">
          <div className="meta-item">
            <span className="meta-label">ID Analyse</span>
            <span className="meta-value font-mono">{results?.analysis_id}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Temps de traitement</span>
            <span className="meta-value font-mono">{results?.processing_time_ms}ms</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Modèle</span>
            <span className="meta-value font-mono">{results?.model_version}</span>
          </div>
        </div>
      </div>

      <div className="results-content">
        {/* Left column */}
        <div className="results-left">
          {/* Pathology bars */}
          <div className="results-card">
            <h4 className="card-title">
              <span>📈</span> Probabilités des 14 pathologies NIH ChestX-ray14
            </h4>
            <div className="pathology-list">
              {results?.predictions
                ?.sort((a, b) => b.probability - a.probability)
                .map((item, i) => (
                  <PathologyBar key={item.pathology} item={item} index={i} />
                ))}
            </div>
          </div>

          {/* Recommendations */}
          {results?.recommendations?.length > 0 && (
            <div className="results-card recommendations-card">
              <h4 className="card-title">
                <span>💡</span> Recommandations cliniques
              </h4>
              <ul className="recommendations-list">
                {results.recommendations.map((rec, i) => (
                  <li key={i} className="recommendation-item">
                    <span className="rec-bullet">→</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
              <div className="disclaimer">
                ⚠️ Ces résultats sont indicatifs et doivent être confirmés par un professionnel de santé qualifié.
              </div>
            </div>
          )}
        </div>

        {/* Right column - Heatmap */}
        {uploadedImage && (
          <div className="results-right">
            <HeatmapOverlay
              imageUrl={uploadedImage}
              heatmapUrl={results?.heatmap_base64}
            />

            {/* Radial chart */}
            <div className="results-card radial-card">
              <h4 className="card-title"><span>🎯</span> Distribution des risques</h4>
              <div className="risk-gauges">
                {results?.predictions
                  ?.map((item) => {
                    const pct = Math.round(item.probability * 100);
                    const color = PATHOLOGY_COLORS[item.pathology] || "#4488ff";
                    const circumference = 2 * Math.PI * 26;
                    const offset = circumference - (pct / 100) * circumference;
                    return (
                      <div key={item.pathology} className="risk-gauge">
                        <svg width="64" height="64" viewBox="0 0 64 64">
                          <circle cx="32" cy="32" r="26" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="5" />
                          <circle
                            cx="32" cy="32" r="26"
                            fill="none"
                            stroke={color}
                            strokeWidth="5"
                            strokeLinecap="round"
                            strokeDasharray={circumference}
                            strokeDashoffset={offset}
                            transform="rotate(-90 32 32)"
                            style={{ transition: "stroke-dashoffset 1s ease" }}
                          />
                          <text x="32" y="36" textAnchor="middle" fill={color} fontSize="11" fontWeight="700" fontFamily="monospace">
                            {pct}%
                          </text>
                        </svg>
                        <span className="gauge-label">{item.pathology.split(" ")[0]}</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
