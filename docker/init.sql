-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension loaded
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    RAISE EXCEPTION 'pgvector extension failed to load';
  END IF;
END $$;
