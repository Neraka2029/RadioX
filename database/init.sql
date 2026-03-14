-- RadioX.AI Database Schema
-- PostgreSQL initialization

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(100) DEFAULT 'Médecin',
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Analyses table
CREATE TABLE IF NOT EXISTS analyses (
    id SERIAL PRIMARY KEY,
    analysis_id VARCHAR(50) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    patient_id INTEGER REFERENCES patients(id),
    image_path VARCHAR(500),
    heatmap_path VARCHAR(500),
    image_filename VARCHAR(255),
    image_size INTEGER,
    predictions JSONB,
    primary_finding VARCHAR(100),
    confidence FLOAT,
    model_version VARCHAR(50),
    processing_time_ms INTEGER,
    recommendations JSONB,
    status VARCHAR(20) DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_primary_finding ON analyses(primary_finding);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Views
CREATE OR REPLACE VIEW analysis_stats AS
SELECT
    u.id AS user_id,
    u.name AS user_name,
    COUNT(a.id) AS total_analyses,
    COUNT(CASE WHEN DATE(a.created_at) = CURRENT_DATE THEN 1 END) AS analyses_today,
    AVG(a.confidence) AS avg_confidence,
    MODE() WITHIN GROUP (ORDER BY a.primary_finding) AS most_common_finding
FROM users u
LEFT JOIN analyses a ON a.user_id = u.id
GROUP BY u.id, u.name;

COMMENT ON TABLE analyses IS 'Stores all chest X-ray analysis results including AI predictions and heatmaps';
COMMENT ON TABLE users IS 'Medical staff accounts with authentication credentials';
COMMENT ON TABLE patients IS 'Patient registry linked to analyses';
