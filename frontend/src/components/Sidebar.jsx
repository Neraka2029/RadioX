import "../styles/sidebar.css";

const NAV_ITEMS = [
  { id: "analysis", icon: "🫁", label: "Analyse IA" },
  { id: "history", icon: "📋", label: "Historique" },
  { id: "reports", icon: "📄", label: "Rapports" },
  { id: "settings", icon: "⚙️", label: "Paramètres" },
];

export default function Sidebar({ user, token, onLogout, activeView, setActiveView }) {
  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="logo-icon">
          <span className="logo-cross">✚</span>
        </div>
        <div className="sidebar-brand">
          <span className="sidebar-brand-name">RadioX</span>
          <span className="sidebar-brand-suffix">.</span>
        </div>
      </div>

      {/* Status indicator */}
      <div className="sidebar-status">
        <span className="status-dot status-online" />
        <span className="status-text">Système opérationnel</span>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <p className="nav-section-label">Navigation</p>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeView === item.id ? "nav-item-active" : ""}`}
            onClick={() => setActiveView(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
            {activeView === item.id && <span className="nav-active-bar" />}
          </button>
        ))}
      </nav>

      {/* AI Model status */}
      <div className="sidebar-model-status">
        <div className="model-status-header">
          <span className="model-icon">🤖</span>
          <span className="model-name">Modèle IA</span>
        </div>
        <div className="model-info">
          <div className="model-chip">DenseNet-121</div>
          <div className="model-accuracy">AUC: 0.81</div>
        </div>
        <div className="model-bar">
          <div className="model-bar-fill" style={{ width: "81%" }} />
        </div>
      </div>

      {/* User profile */}
      <div className="sidebar-user">
        <div className="user-avatar">
          <span>{user.avatar || user.name?.charAt(0)?.toUpperCase()}</span>
        </div>
        <div className="user-info">
          <p className="user-name">{user.name}</p>
          <p className="user-role">{user.role || "Médecin"}</p>
        </div>
        <button className="logout-btn" onClick={onLogout} title="Déconnexion">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
            <polyline points="16 17 21 12 16 7" />
            <line x1="21" y1="12" x2="9" y2="12" />
          </svg>
        </button>
      </div>
    </aside>
  );
}
