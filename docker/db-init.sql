-- Runs once, when the database volume is first created (Postgres' docker-entrypoint-initdb.d).
-- The app database (cv_generator) is created by POSTGRES_DB and gets the vector extension from the
-- Alembic migrations. The test database needs it up front, because the tests create tables directly.
CREATE DATABASE cv_generator_test;
\connect cv_generator_test
CREATE EXTENSION IF NOT EXISTS vector;
