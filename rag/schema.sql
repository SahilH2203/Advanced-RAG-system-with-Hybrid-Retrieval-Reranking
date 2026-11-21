CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS passages (
    passage_id text PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL,
    embedding vector(384)
);
