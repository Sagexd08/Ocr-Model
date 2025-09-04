-- Enterprise OCR System Database Initialization
-- Creates tables for job management, analytics, and system monitoring

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Jobs table for tracking OCR processing jobs
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    document_name VARCHAR(255),
    document_path TEXT,
    document_size BIGINT,
    document_type VARCHAR(50),
    processing_profile VARCHAR(50) DEFAULT 'balanced',
    processing_mode VARCHAR(50) DEFAULT 'standard',
    max_pages INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    result_path TEXT,
    metadata JSONB
);

-- Processing results table for detailed OCR results
CREATE TABLE IF NOT EXISTS processing_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    page_count INTEGER,
    word_count INTEGER,
    confidence_avg DECIMAL(5,4),
    confidence_min DECIMAL(5,4),
    confidence_max DECIMAL(5,4),
    processing_time_ms INTEGER,
    tokens_extracted INTEGER,
    blocks_detected INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    page_data JSONB
);

-- System metrics table for monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- User sessions table for web interface
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_data JSONB
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_processing_results_job_id ON processing_results(job_id);
CREATE INDEX IF NOT EXISTS idx_processing_results_page_number ON processing_results(page_number);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_activity ON user_sessions(last_activity);

-- Create views for analytics
CREATE OR REPLACE VIEW job_statistics AS
SELECT 
    status,
    COUNT(*) as job_count,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_processing_time_seconds,
    AVG(document_size) as avg_document_size,
    DATE_TRUNC('day', created_at) as date
FROM jobs 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY status, DATE_TRUNC('day', created_at)
ORDER BY date DESC;

CREATE OR REPLACE VIEW processing_performance AS
SELECT 
    j.processing_profile,
    j.processing_mode,
    COUNT(j.id) as total_jobs,
    AVG(pr.confidence_avg) as avg_confidence,
    AVG(pr.processing_time_ms) as avg_processing_time_ms,
    SUM(pr.word_count) as total_words_extracted,
    SUM(pr.tokens_extracted) as total_tokens_extracted
FROM jobs j
LEFT JOIN processing_results pr ON j.id = pr.job_id
WHERE j.status = 'completed'
GROUP BY j.processing_profile, j.processing_mode;

-- Insert initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, metadata) VALUES
('system_startup', 1, 'count', '{"component": "database", "version": "2.0.0"}'),
('database_initialized', 1, 'count', '{"timestamp": "' || CURRENT_TIMESTAMP || '"}');

-- Create function to update last_activity in user_sessions
CREATE OR REPLACE FUNCTION update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_activity = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic session activity updates
CREATE TRIGGER trigger_update_session_activity
    BEFORE UPDATE ON user_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_session_activity();

-- Grant permissions to ocr_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ocr_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ocr_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ocr_user;
