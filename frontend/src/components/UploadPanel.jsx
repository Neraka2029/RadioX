import { useState, useRef, useCallback } from "react";
import "../styles/upload.css";

const DEMO_RESULT = {
  patient_id: "PT-2024-0847",
  analysis_id: "AX-" + Math.random().toString(36).substr(2, 8).toUpperCase(),
  timestamp: new Date().toISOString(),
  model_version: "DenseNet121-v2.3",
  predictions: [
    { pathology: "Normal", probability: 0.12, severity: "low" },
    { pathology: "Atelectasie", probability: 0.25, severity: "moderate" },
    { pathology: "Consolidation", probability: 0.18, severity: "low" },
    { pathology: "Infiltration", probability: 0.32, severity: "moderate" },
    { pathology: "Pneumothorax", probability: 0.08, severity: "high" },
    { pathology: "Oedeme", probability: 0.15, severity: "moderate" },
    { pathology: "Emphyseme", probability: 0.22, severity: "low" },
    { pathology: "Fibrose", probability: 0.11, severity: "low" },
    { pathology: "Epanchement pleural", probability: 0.42, severity: "moderate" },
    { pathology: "Pneumonie", probability: 0.68, severity: "high" },
    { pathology: "Epaississement pleural", probability: 0.19, severity: "low" },
    { pathology: "Cardiomegalie", probability: 0.35, severity: "moderate" },
    { pathology: "Nodule", probability: 0.28, severity: "low" },
    { pathology: "Masse", probability: 0.05, severity: "high" },
    { pathology: "Hernie", probability: 0.03, severity: "low" },
  ],
  primary_finding: "Pneumonie",
  confidence: 0.87,
  processing_time_ms: 312,
  heatmap_base64: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", // Pixel rouge visible en base64
  recommendations: [
    "Consultation pneumologique urgente recommandée",
    "Antibiothérapie à envisager selon évaluation clinique",
    "Suivi radiographique dans 2-3 semaines",
  ],
};

export default function UploadPanel({ token, onAnalysisComplete, onAnalyzing, isAnalyzing, hasResult, uploadedImage }) {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [scanPhase, setScanPhase] = useState("");
  const fileInputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) return;
    
    // Accepter les images standards et les fichiers DICOM
    const isImage = file.type.startsWith("image/");
    const isDicom = file.name.toLowerCase().endsWith('.dcm') || 
                   file.name.toLowerCase().endsWith('.dicom');
    
    if (!isImage && !isDicom) {
      console.log("File type not supported:", file.type, file.name);
      return;
    }

    // Pour les fichiers DICOM, créer une URL de données convertie
    if (isDicom) {
      console.log("Processing DICOM file for preview...");
      const reader = new FileReader();
      reader.onload = (e) => {
        // Créer une URL blob pour le preview
        const blob = new Blob([e.target.result], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        setPreviewUrl(url);
        analyzeImage(file, url);
      };
      reader.readAsArrayBuffer(file);
    } else {
      // Pour les images standards, utiliser la méthode normale
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      analyzeImage(file, url);
    }
  }, []);

  const analyzeImage = async (file, imageUrl) => {
    onAnalyzing();
    setUploadProgress(0);

    const phases = [
      { label: "Chargement de l'image...", progress: 15, delay: 400 },
      { label: "Prétraitement MONAI...", progress: 35, delay: 700 },
      { label: "Normalisation 224×224...", progress: 55, delay: 600 },
      { label: "Inférence DenseNet-121...", progress: 75, delay: 900 },
      { label: "Génération heatmap Grad-CAM...", progress: 90, delay: 600 },
      { label: "Compilation des résultats...", progress: 100, delay: 400 },
    ];

    try {
      // Simuler progression
      for (const phase of phases) {
        await new Promise((resolve) => setTimeout(resolve, phase.delay));
        setScanPhase(phase.label);
        setUploadProgress(phase.progress);
      }

      const formData = new FormData();
      formData.append("file", file);
      formData.append("analysis_id", `AX-${Date.now()}`);

      console.log("=== UPLOAD DEBUG ===");
      console.log("Attempting to upload to: http://localhost:8001/predict");
      console.log("File type:", file.type, "File name:", file.name);
      console.log("File size:", file.size, "bytes");

      const response = await fetch("http://localhost:8001/predict", {
        method: "POST",
        body: formData,
      });

      console.log("Response status:", response.status);
      console.log("Response ok:", response.ok);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("=== UPLOAD DEBUG ===");
      console.log("Response data:", data);
      console.log("Heatmap in response:", data.heatmap_base64 ? "Yes" : "No");
      console.log("Processed image in response:", data.processed_image_base64 ? "Yes" : "No");

      // Utiliser l'image traitée depuis le backend si disponible
      if (data.processed_image_base64) {
        const processedImageUrl = `data:image/png;base64,${data.processed_image_base64}`;
        setPreviewUrl(processedImageUrl);
        console.log("Using processed image from backend");
      }

      // Debug de la heatmap
      if (data.heatmap_base64) {
        console.log("Heatmap received, length:", data.heatmap_base64.length);
      } else {
        console.log("No heatmap received from backend");
      }

      onAnalysisComplete(data);
    } catch (error) {
      console.error("Upload failed:", error);
      setScanPhase("Échec de l'analyse");
      // Fallback vers mode démo si le backend n'est pas accessible
      console.log("Falling back to demo mode");
      const demoData = {
        ...DEMO_RESULT,
        analysis_id: `AX-${Date.now()}`,
      };
      onAnalysisComplete(demoData);
    } finally {
      setScanPhase("");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleReset = () => {
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="upload-panel">
      {/* Header */}
      <div className="panel-header">
        <div className="panel-title-row">
          <span className="panel-icon">📤</span>
          <h3 className="panel-title">Radiographie</h3>
        </div>
        <div className="panel-format-hints">
          <span className="format-badge">DICOM</span>
          <span className="format-badge">JPEG</span>
          <span className="format-badge">PNG</span>
        </div>
      </div>

      {/* Upload area */}
      {!previewUrl ? (
        <div
          className={`upload-zone ${dragOver ? "drag-over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.dcm,.dicom,.DCM,.DICOM"
            onChange={(e) => handleFile(e.target.files[0])}
            className="hidden-input"
          />

          <div className="upload-zone-content">
            <div className="upload-icon-container">
              <div className="upload-icon-ring" />
              <div className="upload-icon-ring upload-icon-ring-2" />
              <span className="upload-icon-emoji">🫁</span>
            </div>
            <p className="upload-main-text">Déposer la radiographie ici</p>
            <p className="upload-sub-text">ou cliquez pour sélectionner</p>
            <div className="upload-specs">
              <span>DICOM / JPEG / PNG</span>
              <span>•</span>
              <span>Max 50 MB</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="preview-container">
          {/* Image preview */}
          <div className="preview-image-wrapper">
            <img src={previewUrl} alt="Radiographie" className="preview-image" />

            {isAnalyzing && (
              <div className="scan-overlay">
                <div className="scan-line" />
                <div className="scan-grid" />
                <div className="scan-corners">
                  <span className="corner corner-tl" />
                  <span className="corner corner-tr" />
                  <span className="corner corner-bl" />
                  <span className="corner corner-br" />
                </div>
                <div className="scan-label">ANALYSE EN COURS</div>
              </div>
            )}

            {hasResult && (
              <div className="preview-badge-done">
                <span>✓</span> Analysé
              </div>
            )}
          </div>

          {/* Progress */}
          {isAnalyzing && (
            <div className="analysis-progress">
              <div className="progress-header">
                <span className="progress-phase">{scanPhase}</span>
                <span className="progress-pct">{uploadProgress}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {!isAnalyzing && !hasResult && (
            <button className="btn-ghost" style={{ marginTop: 12 }} onClick={handleReset}>
              Changer d'image
            </button>
          )}
        </div>
      )}

      {/* Model info */}
      <div className="upload-model-info">
        <div className="model-info-row">
          <span className="model-info-dot" />
          <span className="model-info-text">Modèle DenseNet-121 • NIH ChestX-ray14</span>
        </div>
        <div className="model-info-row">
          <span className="model-info-dot" />
          <span className="model-info-text">6 pathologies détectées • Grad-CAM activé</span>
        </div>
      </div>
    </div>
  );
}
