-- Extensiones
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- PACIENTES
CREATE TABLE patients(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    consent_given BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ENFERMEDADES
CREATE TABLE conditions(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE 
);

-- ENROLLMENTS
CREATE TABLE enrollments(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    condition_id UUID NOT NULL REFERENCES conditions(id),
    enrolled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL
);

-- PLANTILLAS CONTENIDO
CREATE TABLE content_templates(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    condition_id UUID NOT NULL REFERENCES conditions(id),
    day_offset INT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    approved_by_physician BOOLEAN NOT NULL DEFAULT FALSE
);

-- SENDS
CREATE TABLE sends(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id UUID NOT NULL REFERENCES enrollments(id),
    content_template_id UUID NOT NULL REFERENCES content_templates(id),
    sent_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL
);

-- ENGAGEMENT
CREATE TABLE engagement(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    send_id UUID NOT NULL REFERENCES sends(id),
    opened_at TIMESTAMPTZ,
    clicked_at TIMESTAMPTZ
);

-- LIBROS
CREATE TABLE books(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL UNIQUE,
    author TEXT NOT NULL
);

-- CHUNKS
CREATE TABLE chunks(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id UUID NOT NULL REFERENCES books(id),
    content TEXT NOT NULL,
    page_number INT NOT NULL,
    embedding VECTOR(384)
);

-- CONTENT TEMPLATE CHUNKS
CREATE TABLE content_template_chunks(
    content_template_id UUID NOT NULL REFERENCES content_templates(id),
    chunk_id UUID NOT NULL REFERENCES chunks(id),
    PRIMARY KEY (content_template_id, chunk_id)
);

-- TOPICS
CREATE TABLE topics(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL 
);

-- CHUNK_TOPICS
CREATE TABLE chunk_topics(
    chunk_id UUID NOT NULL REFERENCES chunks(id),
    topic_id UUID NOT NULL REFERENCES topics(id),
    PRIMARY KEY (chunk_id, topic_id)
);