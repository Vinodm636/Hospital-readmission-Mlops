-- ============================================================
-- Hospital Readmission Analytics and Prediction
-- PostgreSQL Database Schema
-- Course: Data Engineering and MLOps
-- ============================================================
--
-- Schema layers:
--   raw       : Original data as loaded from the CSV
--   staging   : Lightly typed copy for validation
--   curated   : Cleaned, feature-engineered data
--   analytics : Dimensional model (fact + dimension tables)
--
-- Run this script once to initialise the database:
--   psql -U postgres -d hospital_readmission -f src/database/schema.sql
-- ============================================================

-- ─────────────────────────────────────────────────────────────
-- Create schemas
-- ─────────────────────────────────────────────────────────────

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS curated;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ─────────────────────────────────────────────────────────────
-- RAW LAYER
-- Stores the original CSV data exactly as loaded.
-- No transformations applied. Preserved for audit purposes.
-- ─────────────────────────────────────────────────────────────

DROP TABLE IF EXISTS raw.encounters CASCADE;
CREATE TABLE raw.encounters (
    -- System columns
    raw_id              SERIAL PRIMARY KEY,
    ingestion_run_id    VARCHAR(36),
    ingestion_timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Original UCI Diabetes dataset columns (all stored as TEXT to preserve
    -- the original values including '?' missing-value codes)
    encounter_id            TEXT,
    patient_nbr             TEXT,
    race                    TEXT,
    gender                  TEXT,
    age                     TEXT,
    weight                  TEXT,
    admission_type_id       TEXT,
    discharge_disposition_id TEXT,
    admission_source_id     TEXT,
    time_in_hospital        TEXT,
    payer_code              TEXT,
    medical_specialty       TEXT,
    num_lab_procedures      TEXT,
    num_procedures          TEXT,
    num_medications         TEXT,
    number_outpatient       TEXT,
    number_emergency        TEXT,
    number_inpatient        TEXT,
    diag_1                  TEXT,
    diag_2                  TEXT,
    diag_3                  TEXT,
    number_diagnoses        TEXT,
    max_glu_serum           TEXT,
    a1cresult               TEXT,
    metformin               TEXT,
    repaglinide             TEXT,
    nateglinide             TEXT,
    chlorpropamide          TEXT,
    glimepiride             TEXT,
    acetohexamide           TEXT,
    glipizide               TEXT,
    glyburide               TEXT,
    tolbutamide             TEXT,
    pioglitazone            TEXT,
    rosiglitazone           TEXT,
    acarbose                TEXT,
    miglitol                TEXT,
    troglitazone            TEXT,
    tolazamide              TEXT,
    examide                 TEXT,
    citoglipton             TEXT,
    insulin                 TEXT,
    glyburide_metformin     TEXT,
    glipizide_metformin     TEXT,
    glimepiride_pioglitazone TEXT,
    metformin_rosiglitazone TEXT,
    metformin_pioglitazone  TEXT,
    change_col              TEXT,   -- renamed from 'change' (SQL reserved word)
    diabetesmed             TEXT,
    readmitted              TEXT
);

COMMENT ON TABLE raw.encounters IS
    'Raw UCI Diabetes encounters loaded directly from CSV. No transformation applied.';

-- ─────────────────────────────────────────────────────────────
-- STAGING LAYER
-- Lightly typed version of raw data (NULLs resolved, basic types cast).
-- ─────────────────────────────────────────────────────────────

DROP TABLE IF EXISTS staging.encounters CASCADE;
CREATE TABLE staging.encounters (
    staging_id              SERIAL PRIMARY KEY,
    ingestion_run_id        VARCHAR(36),
    staging_timestamp       TIMESTAMPTZ DEFAULT NOW(),

    encounter_id            BIGINT,
    patient_nbr             BIGINT,
    race                    VARCHAR(50),
    gender                  VARCHAR(20),
    age                     VARCHAR(10),
    weight                  VARCHAR(20),
    admission_type_id       INTEGER,
    discharge_disposition_id INTEGER,
    admission_source_id     INTEGER,
    time_in_hospital        INTEGER,
    payer_code              VARCHAR(10),
    medical_specialty       VARCHAR(100),
    num_lab_procedures      INTEGER,
    num_procedures          INTEGER,
    num_medications         INTEGER,
    number_outpatient       INTEGER,
    number_emergency        INTEGER,
    number_inpatient        INTEGER,
    diag_1                  VARCHAR(20),
    diag_2                  VARCHAR(20),
    diag_3                  VARCHAR(20),
    number_diagnoses        INTEGER,
    max_glu_serum           VARCHAR(10),
    a1cresult               VARCHAR(10),
    insulin                 VARCHAR(10),
    diabetesmed             VARCHAR(5),
    readmitted              VARCHAR(5)
);

COMMENT ON TABLE staging.encounters IS
    'Staging encounters with basic type casting. Feeds the cleaning pipeline.';

-- ─────────────────────────────────────────────────────────────
-- CURATED LAYER
-- Fully cleaned, feature-engineered data ready for analytics and ML.
-- ─────────────────────────────────────────────────────────────

DROP TABLE IF EXISTS curated.encounters CASCADE;
CREATE TABLE curated.encounters (
    curated_id              SERIAL PRIMARY KEY,
    pipeline_run_id         VARCHAR(36),
    created_at              TIMESTAMPTZ DEFAULT NOW(),

    -- Surrogate keys (privacy-safe)
    encounter_surrogate_id  VARCHAR(16) UNIQUE NOT NULL,
    patient_surrogate_id    VARCHAR(16) NOT NULL,

    -- Original identifiers (kept for join capability — not PII)
    encounter_id            BIGINT,
    patient_nbr             BIGINT,

    -- Demographics
    race                    VARCHAR(50),
    gender                  VARCHAR(20),
    age                     VARCHAR(10),
    age_midpoint            SMALLINT,

    -- Admission / Discharge
    admission_type_id       INTEGER,
    discharge_disposition_id INTEGER,
    admission_source_id     INTEGER,
    time_in_hospital        INTEGER,
    payer_code              VARCHAR(10),
    medical_specialty       VARCHAR(100),

    -- Procedure and lab counts
    num_lab_procedures      INTEGER,
    num_procedures          INTEGER,
    num_medications         INTEGER,
    number_outpatient       INTEGER,
    number_emergency        INTEGER,
    number_inpatient        INTEGER,

    -- Diagnoses
    diag_1                  VARCHAR(20),
    diag_2                  VARCHAR(20),
    diag_3                  VARCHAR(20),
    number_diagnoses        INTEGER,

    -- Lab results
    max_glu_serum           VARCHAR(10),
    a1cresult               VARCHAR(10),

    -- Medications
    metformin               VARCHAR(10),
    repaglinide             VARCHAR(10),
    nateglinide             VARCHAR(10),
    glimepiride             VARCHAR(10),
    glipizide               VARCHAR(10),
    glyburide               VARCHAR(10),
    pioglitazone            VARCHAR(10),
    rosiglitazone           VARCHAR(10),
    insulin                 VARCHAR(10),
    diabetesmed             VARCHAR(5),

    -- Readmission
    readmitted              VARCHAR(5),
    readmission_binary      SMALLINT,   -- 1=<30days, 0=otherwise

    -- Engineered features
    diagnosis_count         SMALLINT,
    medication_count        SMALLINT,
    prior_admission_count   SMALLINT,
    is_prior_admission      SMALLINT,
    length_of_stay          SMALLINT,
    is_long_stay            SMALLINT,
    primary_diagnosis_category VARCHAR(50),
    has_diabetes_drug       SMALLINT,
    total_service_utilisation SMALLINT
);

CREATE INDEX IF NOT EXISTS idx_curated_patient ON curated.encounters (patient_surrogate_id);
CREATE INDEX IF NOT EXISTS idx_curated_readmit ON curated.encounters (readmitted);
CREATE INDEX IF NOT EXISTS idx_curated_age     ON curated.encounters (age);
CREATE INDEX IF NOT EXISTS idx_curated_diag_cat ON curated.encounters (primary_diagnosis_category);

COMMENT ON TABLE curated.encounters IS
    'Cleaned, feature-engineered encounters. Primary analytical source.';

-- ─────────────────────────────────────────────────────────────
-- ANALYTICS LAYER — Dimensional Model
-- ─────────────────────────────────────────────────────────────

-- ── Dimension: Admission ──────────────────────────────────────
DROP TABLE IF EXISTS analytics.dim_admission CASCADE;
CREATE TABLE analytics.dim_admission (
    admission_sk            SERIAL PRIMARY KEY,
    encounter_surrogate_id  VARCHAR(16) NOT NULL,
    patient_surrogate_id    VARCHAR(16) NOT NULL,
    race                    VARCHAR(50),
    gender                  VARCHAR(20),
    age_group               VARCHAR(10),
    age_midpoint            SMALLINT,
    admission_type_id       INTEGER,
    admission_source_id     INTEGER,
    payer_code              VARCHAR(10),
    medical_specialty       VARCHAR(100),
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_adm_patient ON analytics.dim_admission (patient_surrogate_id);
CREATE INDEX IF NOT EXISTS idx_dim_adm_age     ON analytics.dim_admission (age_group);

COMMENT ON TABLE analytics.dim_admission IS
    'Admission dimension: patient demographics and admission characteristics.';

-- ── Dimension: Diagnosis ──────────────────────────────────────
DROP TABLE IF EXISTS analytics.dim_diagnosis CASCADE;
CREATE TABLE analytics.dim_diagnosis (
    diagnosis_sk            SERIAL PRIMARY KEY,
    encounter_surrogate_id  VARCHAR(16) NOT NULL,
    diag_1                  VARCHAR(20),
    diag_2                  VARCHAR(20),
    diag_3                  VARCHAR(20),
    primary_diagnosis_category VARCHAR(50),
    number_diagnoses        INTEGER,
    diagnosis_count         SMALLINT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_diag_encounter ON analytics.dim_diagnosis (encounter_surrogate_id);
CREATE INDEX IF NOT EXISTS idx_dim_diag_category  ON analytics.dim_diagnosis (primary_diagnosis_category);

COMMENT ON TABLE analytics.dim_diagnosis IS
    'Diagnosis dimension: ICD-9 codes and diagnosis categories per encounter.';

-- ── Fact: Patient/Encounter Summary ──────────────────────────
DROP TABLE IF EXISTS analytics.fact_patient_summary CASCADE;
CREATE TABLE analytics.fact_patient_summary (
    summary_sk              SERIAL PRIMARY KEY,
    encounter_surrogate_id  VARCHAR(16) NOT NULL,
    patient_surrogate_id    VARCHAR(16) NOT NULL,
    time_in_hospital        INTEGER,
    length_of_stay          SMALLINT,
    is_long_stay            SMALLINT,
    num_lab_procedures      INTEGER,
    num_procedures          INTEGER,
    num_medications         INTEGER,
    medication_count        SMALLINT,
    number_outpatient       INTEGER,
    number_emergency        INTEGER,
    number_inpatient        INTEGER,
    prior_admission_count   SMALLINT,
    is_prior_admission      SMALLINT,
    total_service_utilisation SMALLINT,
    diagnosis_count         SMALLINT,
    has_diabetes_drug       SMALLINT,
    max_glu_serum           VARCHAR(10),
    a1cresult               VARCHAR(10),
    diabetesmed             VARCHAR(5),
    discharge_disposition_id INTEGER,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_ps_encounter ON analytics.fact_patient_summary (encounter_surrogate_id);
CREATE INDEX IF NOT EXISTS idx_fact_ps_patient   ON analytics.fact_patient_summary (patient_surrogate_id);

COMMENT ON TABLE analytics.fact_patient_summary IS
    'Fact table: encounter metrics, lab counts, medication counts, stay duration.';

-- ── Fact: Readmission ─────────────────────────────────────────
DROP TABLE IF EXISTS analytics.fact_readmission CASCADE;
CREATE TABLE analytics.fact_readmission (
    readmission_sk          SERIAL PRIMARY KEY,
    encounter_surrogate_id  VARCHAR(16) NOT NULL,
    patient_surrogate_id    VARCHAR(16) NOT NULL,
    readmitted              VARCHAR(5),
    readmission_binary      SMALLINT,
    age_group               VARCHAR(10),
    age_midpoint            SMALLINT,
    gender                  VARCHAR(20),
    race                    VARCHAR(50),
    primary_diagnosis_category VARCHAR(50),
    admission_type_id       INTEGER,
    discharge_disposition_id INTEGER,
    length_of_stay          SMALLINT,
    num_medications         INTEGER,
    prior_admission_count   SMALLINT,
    has_diabetes_drug       SMALLINT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_ra_encounter ON analytics.fact_readmission (encounter_surrogate_id);
CREATE INDEX IF NOT EXISTS idx_fact_ra_readmit   ON analytics.fact_readmission (readmitted);
CREATE INDEX IF NOT EXISTS idx_fact_ra_age       ON analytics.fact_readmission (age_group);
CREATE INDEX IF NOT EXISTS idx_fact_ra_diag_cat  ON analytics.fact_readmission (primary_diagnosis_category);
CREATE INDEX IF NOT EXISTS idx_fact_ra_binary    ON analytics.fact_readmission (readmission_binary);

COMMENT ON TABLE analytics.fact_readmission IS
    'Readmission fact table: central analytical table for readmission KPIs.';

-- ── Pipeline Metadata ─────────────────────────────────────────
DROP TABLE IF EXISTS analytics.ingestion_metadata CASCADE;
CREATE TABLE analytics.ingestion_metadata (
    id                      SERIAL PRIMARY KEY,
    run_id                  VARCHAR(36) UNIQUE NOT NULL,
    source_name             TEXT,
    extraction_timestamp    TIMESTAMPTZ,
    file_name               TEXT,
    status                  VARCHAR(10),
    row_count               INTEGER,
    column_count            INTEGER,
    error_message           TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE analytics.ingestion_metadata IS
    'Audit log of every ingestion run: source, timestamp, row count, status.';

-- ── Validation Results ────────────────────────────────────────
DROP TABLE IF EXISTS analytics.validation_results CASCADE;
CREATE TABLE analytics.validation_results (
    id                      SERIAL PRIMARY KEY,
    run_id                  VARCHAR(36),
    dataset_label           VARCHAR(20),
    check_name              VARCHAR(100),
    status                  VARCHAR(10),
    description             TEXT,
    failed_count            INTEGER DEFAULT 0,
    details                 TEXT,
    check_timestamp         TIMESTAMPTZ,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_val_run_id ON analytics.validation_results (run_id);

COMMENT ON TABLE analytics.validation_results IS
    'Results of every automated data-quality check per pipeline run.';

-- ── Rejected Records ──────────────────────────────────────────
DROP TABLE IF EXISTS analytics.rejected_records CASCADE;
CREATE TABLE analytics.rejected_records (
    id                      SERIAL PRIMARY KEY,
    run_id                  VARCHAR(36),
    record_id               TEXT,
    rejection_reason        VARCHAR(100),
    rejection_timestamp     TIMESTAMPTZ,
    details                 TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rej_run_id ON analytics.rejected_records (run_id);
CREATE INDEX IF NOT EXISTS idx_rej_reason ON analytics.rejected_records (rejection_reason);

COMMENT ON TABLE analytics.rejected_records IS
    'Log of records removed during cleaning, with rejection reasons.';

-- ─────────────────────────────────────────────────────────────
-- VIEWS for dashboard queries
-- ─────────────────────────────────────────────────────────────

-- Overall readmission rate
CREATE OR REPLACE VIEW analytics.vw_readmission_rate AS
SELECT
    COUNT(*) AS total_encounters,
    SUM(readmission_binary) AS total_readmitted,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission;

-- Readmission by age group
CREATE OR REPLACE VIEW analytics.vw_readmission_by_age AS
SELECT
    age_group,
    age_midpoint,
    COUNT(*) AS total_encounters,
    SUM(readmission_binary) AS readmitted_count,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY age_group, age_midpoint
ORDER BY age_midpoint;

-- Readmission by diagnosis category
CREATE OR REPLACE VIEW analytics.vw_readmission_by_diagnosis AS
SELECT
    primary_diagnosis_category,
    COUNT(*) AS total_encounters,
    SUM(readmission_binary) AS readmitted_count,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY primary_diagnosis_category
ORDER BY readmission_rate_pct DESC;

-- Length of stay analysis
CREATE OR REPLACE VIEW analytics.vw_length_of_stay AS
SELECT
    age_group,
    ROUND(AVG(fps.length_of_stay), 2) AS avg_los,
    MIN(fps.length_of_stay) AS min_los,
    MAX(fps.length_of_stay) AS max_los,
    COUNT(*) AS encounter_count
FROM analytics.fact_patient_summary fps
JOIN analytics.fact_readmission fra USING (encounter_surrogate_id)
GROUP BY age_group
ORDER BY avg_los DESC;

-- Admission type distribution
CREATE OR REPLACE VIEW analytics.vw_admission_patterns AS
SELECT
    admission_type_id,
    COUNT(*) AS encounter_count,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY admission_type_id
ORDER BY encounter_count DESC;

-- High-risk cohort (readmitted <30 days)
CREATE OR REPLACE VIEW analytics.vw_high_risk_cohort AS
SELECT
    age_group,
    gender,
    race,
    primary_diagnosis_category,
    COUNT(*) AS encounter_count,
    ROUND(AVG(fps.length_of_stay), 2) AS avg_los,
    ROUND(AVG(fps.num_medications::NUMERIC), 2) AS avg_medications,
    ROUND(AVG(fps.prior_admission_count::NUMERIC), 2) AS avg_prior_admissions
FROM analytics.fact_readmission fra
JOIN analytics.fact_patient_summary fps USING (encounter_surrogate_id)
WHERE fra.readmission_binary = 1
GROUP BY age_group, gender, race, primary_diagnosis_category
ORDER BY encounter_count DESC;

-- ─────────────────────────────────────────────────────────────
-- Finalisation
-- ─────────────────────────────────────────────────────────────
DO $$
BEGIN
    RAISE NOTICE 'Hospital Readmission schema created successfully.';
    RAISE NOTICE 'Schemas: raw, staging, curated, analytics';
    RAISE NOTICE 'Tables: raw.encounters, staging.encounters, curated.encounters,';
    RAISE NOTICE '        analytics.dim_admission, analytics.dim_diagnosis,';
    RAISE NOTICE '        analytics.fact_patient_summary, analytics.fact_readmission,';
    RAISE NOTICE '        analytics.ingestion_metadata, analytics.validation_results,';
    RAISE NOTICE '        analytics.rejected_records';
    RAISE NOTICE 'Views: 5 analytical views created';
END $$;
