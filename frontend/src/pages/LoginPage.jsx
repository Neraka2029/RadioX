import { useState } from "react";
import "../styles/login.css";

export default function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const formData = new URLSearchParams();
      formData.append("username", email);
      formData.append("password", password);

      const response = await fetch("http://localhost:8000/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Identifiants invalides");
      }

      const data = await response.json();
      onLogin(data.user, data.access_token);
    } catch (err) {
      // Demo login fallback
      if (email === "demo@radiox.ai" && password === "demo123") {
        onLogin(
          { id: 1, name: "Dr. Martin Dubois", email: "demo@radiox.ai", role: "Radiologue", avatar: "MD" },
          "demo-token-123"
        );
      } else {
        setError(err.message || "Connexion impossible. Vérifiez vos identifiants.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-root">
      <div className="noise-overlay" />

      {/* Animated background */}
      <div className="login-bg">
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
        <div className="grid-bg login-grid" />
      </div>

      {/* Lung anatomy decoration */}
      <div className="lung-decoration">
        <svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg" className="lung-svg">
          <path d="M150 40 C150 40 140 50 135 70 C125 90 115 110 110 140 C105 165 108 195 115 215 C120 230 130 240 140 245 C145 247 148 248 150 248"
            stroke="rgba(0,212,255,0.15)" strokeWidth="2" fill="none" strokeLinecap="round" />
          <path d="M150 40 C150 40 160 50 165 70 C175 90 185 110 190 140 C195 165 192 195 185 215 C180 230 170 240 160 245 C155 247 152 248 150 248"
            stroke="rgba(0,212,255,0.15)" strokeWidth="2" fill="none" strokeLinecap="round" />
          <path d="M150 60 L150 40" stroke="rgba(0,212,255,0.2)" strokeWidth="3" strokeLinecap="round" />
          <path d="M150 60 C145 80 130 95 120 115" stroke="rgba(0,212,255,0.12)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
          <path d="M150 60 C155 80 170 95 180 115" stroke="rgba(0,212,255,0.12)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
          <circle cx="150" cy="40" r="4" fill="rgba(0,212,255,0.3)" />
          <circle cx="120" cy="170" r="60" fill="none" stroke="rgba(0,212,255,0.05)" strokeWidth="1" />
          <circle cx="180" cy="170" r="60" fill="none" stroke="rgba(0,212,255,0.05)" strokeWidth="1" />
        </svg>
      </div>

      <div className="login-container">
        {/* Brand */}
        <div className="login-brand">
          <div className="brand-icon">
            <span className="brand-cross">✚</span>
            <div className="brand-ring" />
          </div>
          <div className="brand-text">
            <h1 className="brand-name">RadioX<span className="brand-dot">.</span></h1>
            <p className="brand-tagline">Analyse intelligente de radiographies</p>
          </div>
        </div>

        {/* Card */}
        <div className="login-card">
          <div className="login-card-header">
            <h2>Connexion</h2>
            <p>Accédez à votre espace clinique</p>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            <div className="form-group">
              <label>Email professionnel</label>
              <div className="input-wrapper">
                <span className="input-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                    <circle cx="12" cy="7" r="4" />
                  </svg>
                </span>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="docteur@hopital.fr"
                  required
                  className="form-input"
                  autoComplete="email"
                />
              </div>
            </div>

            <div className="form-group">
              <label>Mot de passe</label>
              <div className="input-wrapper">
                <span className="input-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                  </svg>
                </span>
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  className="form-input"
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  className="input-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? "🙈" : "👁️"}
                </button>
              </div>
            </div>

            {error && (
              <div className="error-message">
                <span>⚠</span> {error}
              </div>
            )}

            <button type="submit" className="login-btn" disabled={isLoading}>
              {isLoading ? (
                <>
                  <span className="btn-spinner" />
                  Connexion en cours...
                </>
              ) : (
                <>
                  <span>→</span>
                  Se connecter
                </>
              )}
            </button>
          </form>

          <div className="login-demo-hint">
            <span>Démo :</span>
            <code onClick={() => { setEmail("demo@radiox.ai"); setPassword("demo123"); }} className="demo-credentials">
              demo@radiox.ai / demo123
            </code>
          </div>
        </div>

        {/* Features preview */}
        <div className="login-features">
          {[
            { icon: "🫁", label: "Détection multi-pathologies" },
            { icon: "🔥", label: "Heatmap Grad-CAM" },
          ].map((f, i) => (
            <div key={i} className="feature-pill" style={{ animationDelay: `${i * 0.15}s` }}>
              <span>{f.icon}</span>
              <span>{f.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
