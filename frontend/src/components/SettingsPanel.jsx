import { useState, useEffect } from "react";
import "../styles/settings.css";

export default function SettingsPanel({ user, onLogout }) {
  const [settings, setSettings] = useState({
    notifications: true,
    autoSave: true,
    confidenceThreshold: 50,
    language: "fr",
    theme: "dark",
    modelAccuracy: true,
    exportFormat: "pdf",
  });
  const [loading, setLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState("");

  useEffect(() => {
    // Charger les paramètres depuis le localStorage ou API
    const savedSettings = localStorage.getItem("radiox_settings");
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  const handleSettingChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSave = async () => {
    setLoading(true);
    setSaveStatus("");
    
    try {
      // Simuler la sauvegarde
      await new Promise(resolve => setTimeout(resolve, 1000));
      localStorage.setItem("radiox_settings", JSON.stringify(settings));
      setSaveStatus("Paramètres sauvegardés avec succès");
    } catch (error) {
      setSaveStatus("Erreur lors de la sauvegarde");
    } finally {
      setLoading(false);
      setTimeout(() => setSaveStatus(""), 3000);
    }
  };

  const handleReset = () => {
    const defaultSettings = {
      notifications: true,
      autoSave: true,
      confidenceThreshold: 50,
      language: "fr",
      theme: "dark",
      modelAccuracy: true,
      exportFormat: "pdf",
    };
    setSettings(defaultSettings);
    setSaveStatus("Paramètres réinitialisés");
    setTimeout(() => setSaveStatus(""), 3000);
  };

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h2>⚙️ Paramètres</h2>
        <div className="settings-actions">
          <button className="reset-btn" onClick={handleReset}>
            Réinitialiser
          </button>
          <button className="save-btn" onClick={handleSave} disabled={loading}>
            {loading ? "Sauvegarde..." : "Sauvegarder"}
          </button>
        </div>
      </div>

      {saveStatus && (
        <div className={`status-message ${saveStatus.includes("succès") ? "success" : "error"}`}>
          {saveStatus}
        </div>
      )}

      <div className="settings-sections">
        {/* Profil utilisateur */}
        <div className="settings-section">
          <h3>👤 Profil utilisateur</h3>
          <div className="setting-item">
            <label>Nom</label>
            <input 
              type="text" 
              value={user?.name || ""} 
              disabled 
              className="disabled-input"
            />
          </div>
          <div className="setting-item">
            <label>Rôle</label>
            <input 
              type="text" 
              value={user?.role || "Médecin"} 
              disabled 
              className="disabled-input"
            />
          </div>
          <div className="setting-item">
            <button className="logout-btn-full" onClick={onLogout}>
              🚪 Se déconnecter
            </button>
          </div>
        </div>

        {/* Préférences IA */}
        <div className="settings-section">
          <h3>🤖 Préférences IA</h3>
          <div className="setting-item">
            <label>Seuil de confiance minimum</label>
            <div className="threshold-control">
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={settings.confidenceThreshold}
                onChange={(e) => handleSettingChange("confidenceThreshold", parseInt(e.target.value))}
              />
              <span className="threshold-value">{settings.confidenceThreshold}%</span>
            </div>
          </div>
          <div className="setting-item">
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                checked={settings.modelAccuracy}
                onChange={(e) => handleSettingChange("modelAccuracy", e.target.checked)}
              />
              Afficher la précision du modèle
            </label>
          </div>
        </div>

        {/* Interface */}
        <div className="settings-section">
          <h3>🎨 Interface</h3>
          <div className="setting-item">
            <label>Langue</label>
            <select 
              value={settings.language}
              onChange={(e) => handleSettingChange("language", e.target.value)}
            >
              <option value="fr">Français</option>
              <option value="en">English</option>
              <option value="es">Español</option>
            </select>
          </div>
          <div className="setting-item">
            <label>Thème</label>
            <select 
              value={settings.theme}
              onChange={(e) => handleSettingChange("theme", e.target.value)}
            >
              <option value="dark">Sombre</option>
              <option value="light">Clair</option>
              <option value="auto">Automatique</option>
            </select>
          </div>
        </div>

        {/* Notifications */}
        <div className="settings-section">
          <h3>🔔 Notifications</h3>
          <div className="setting-item">
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                checked={settings.notifications}
                onChange={(e) => handleSettingChange("notifications", e.target.checked)}
              />
              Activer les notifications
            </label>
          </div>
          <div className="setting-item">
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                checked={settings.autoSave}
                onChange={(e) => handleSettingChange("autoSave", e.target.checked)}
              />
              Sauvegarde automatique des rapports
            </label>
          </div>
        </div>

        {/* Export */}
        <div className="settings-section">
          <h3>📤 Exportation</h3>
          <div className="setting-item">
            <label>Format d'export par défaut</label>
            <select 
              value={settings.exportFormat}
              onChange={(e) => handleSettingChange("exportFormat", e.target.value)}
            >
              <option value="pdf">PDF</option>
              <option value="docx">Word (.docx)</option>
              <option value="html">HTML</option>
              <option value="json">JSON</option>
            </select>
          </div>
        </div>

        {/* Système */}
        <div className="settings-section">
          <h3>💻 Système</h3>
          <div className="setting-item">
            <label>Version de RadioX</label>
            <input type="text" value="1.0.0" disabled className="disabled-input" />
          </div>
          <div className="setting-item">
            <label>Modèle IA actuel</label>
            <input type="text" value="DenseNet-121 NIH" disabled className="disabled-input" />
          </div>
          <div className="setting-item">
            <button className="clear-cache-btn">
              🗑️ Vider le cache
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
