import { useState } from "react";
import Sidebar from "../components/Sidebar";
import StatsBar from "../components/StatsBar";
import UploadPanel from "../components/UploadPanel";
import ResultsPanel from "../components/ResultsPanel";
import HistoryPanel from "../components/HistoryPanel";
import ReportsPanel from "../components/ReportsPanel";
import SettingsPanel from "../components/SettingsPanel";
import "../styles/dashboard.css";
import "../styles/globals.css";

export default function Dashboard({ user, token, onLogout }) {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [activeView, setActiveView] = useState("analysis"); // analysis | history | reports | settings

  const handleAnalysisComplete = (result, imageUrl) => {
    setAnalysisResult(result);
    setUploadedImage(imageUrl);
    setIsAnalyzing(false);
  };

  const handleNewAnalysis = () => {
    setAnalysisResult(null);
    setUploadedImage(null);
  };

  return (
    <div className="dashboard-root">
      <div className="noise-overlay" />

      {/* Sidebar */}
      <Sidebar user={user} token={token} onLogout={onLogout} activeView={activeView} setActiveView={setActiveView} />

      {/* Main content */}
      <main className="dashboard-main">
        {/* Top stats bar */}
        <StatsBar />

        {/* Header */}
        <div className="dashboard-header">
          <div>
            <h1 className="dashboard-title">
              {activeView === "analysis" ? "Analyse Radiographique" : "Historique"}
            </h1>
            <p className="dashboard-subtitle">
              {activeView === "analysis"
                ? "Détection assistée par intelligence artificielle des pathologies thoraciques"
                : "Consultez vos analyses précédentes"}
            </p>
          </div>
          {analysisResult && (
            <button className="btn-ghost" onClick={handleNewAnalysis}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
              Nouvelle analyse
            </button>
          )}
        </div>

        {/* Content based on active view */}
        <div className="dashboard-content">
          {activeView === "analysis" && (
            <div className="dashboard-grid">
              <UploadPanel
                token={token}
                onAnalysisComplete={handleAnalysisComplete}
                onAnalyzing={() => setIsAnalyzing(true)}
                isAnalyzing={isAnalyzing}
                hasResult={!!analysisResult}
                uploadedImage={uploadedImage}
              />
              <ResultsPanel
                results={analysisResult}
                uploadedImage={uploadedImage}
                isAnalyzing={isAnalyzing}
              />
            </div>
          )}
          
          {activeView === "history" && <HistoryPanel />}
          
          {activeView === "reports" && <ReportsPanel />}
          
          {activeView === "settings" && <SettingsPanel user={user} onLogout={onLogout} />}
        </div>
      </main>
    </div>
  );
}
