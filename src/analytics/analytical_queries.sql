-- ============================================================
-- Hospital Readmission Analytics and Prediction
-- Analytical Queries
-- Course: Data Engineering and MLOps
-- ============================================================
-- These queries are used by the Streamlit dashboard and for
-- ad-hoc analytical investigation.
-- All queries run against the analytics schema.
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- 1. Overall KPIs
-- ─────────────────────────────────────────────────────────────

-- Total encounters, readmission rate, average length of stay
SELECT
    COUNT(*)                                        AS total_encounters,
    SUM(readmission_binary)                         AS total_readmitted,
    ROUND(AVG(readmission_binary) * 100, 2)         AS readmission_rate_pct,
    ROUND(AVG(fps.length_of_stay), 2)               AS avg_length_of_stay_days,
    ROUND(AVG(fps.num_medications::NUMERIC), 2)     AS avg_medications,
    ROUND(AVG(fps.num_lab_procedures::NUMERIC), 2)  AS avg_lab_procedures
FROM analytics.fact_readmission fra
JOIN analytics.fact_patient_summary fps USING (encounter_surrogate_id);


-- ─────────────────────────────────────────────────────────────
-- 2. Readmission Rate by Age Group
-- ─────────────────────────────────────────────────────────────

SELECT
    age_group,
    age_midpoint,
    COUNT(*)                                AS total_encounters,
    SUM(readmission_binary)                 AS readmitted_count,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY age_group, age_midpoint
ORDER BY age_midpoint;


-- ─────────────────────────────────────────────────────────────
-- 3. Readmission Rate by Diagnosis Category
-- ─────────────────────────────────────────────────────────────

SELECT
    primary_diagnosis_category,
    COUNT(*)                                AS total_encounters,
    SUM(readmission_binary)                 AS readmitted_count,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY primary_diagnosis_category
ORDER BY readmission_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────
-- 4. Readmission Rate by Gender
-- ─────────────────────────────────────────────────────────────

SELECT
    gender,
    COUNT(*)                                AS total_encounters,
    SUM(readmission_binary)                 AS readmitted_count,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY gender
ORDER BY gender;


-- ─────────────────────────────────────────────────────────────
-- 5. Readmission Rate by Admission Type
-- ─────────────────────────────────────────────────────────────

SELECT
    admission_type_id,
    COUNT(*)                                AS total_encounters,
    SUM(readmission_binary)                 AS readmitted_count,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY admission_type_id
ORDER BY total_encounters DESC;


-- ─────────────────────────────────────────────────────────────
-- 6. Length-of-Stay Analysis
-- ─────────────────────────────────────────────────────────────

-- Average LOS by age group and readmission status
SELECT
    fra.age_group,
    fra.readmission_binary,
    CASE fra.readmission_binary WHEN 1 THEN 'Readmitted <30' ELSE 'Not Readmitted' END AS readmit_label,
    COUNT(*)                              AS encounter_count,
    ROUND(AVG(fps.length_of_stay), 2)     AS avg_los_days,
    MIN(fps.length_of_stay)               AS min_los_days,
    MAX(fps.length_of_stay)               AS max_los_days
FROM analytics.fact_readmission fra
JOIN analytics.fact_patient_summary fps USING (encounter_surrogate_id)
GROUP BY fra.age_group, fra.readmission_binary
ORDER BY fra.age_group, fra.readmission_binary;


-- ─────────────────────────────────────────────────────────────
-- 7. Admission and Discharge Patterns
-- ─────────────────────────────────────────────────────────────

-- Discharge disposition distribution
SELECT
    discharge_disposition_id,
    COUNT(*)                                AS encounter_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY discharge_disposition_id
ORDER BY encounter_count DESC;


-- ─────────────────────────────────────────────────────────────
-- 8. Medication Patterns
-- ─────────────────────────────────────────────────────────────

-- Medication count distribution vs readmission
SELECT
    CASE
        WHEN fps.num_medications <= 5  THEN '1-5'
        WHEN fps.num_medications <= 10 THEN '6-10'
        WHEN fps.num_medications <= 20 THEN '11-20'
        WHEN fps.num_medications <= 30 THEN '21-30'
        ELSE '30+'
    END AS medication_bucket,
    COUNT(*)                                AS encounter_count,
    ROUND(AVG(fra.readmission_binary)*100, 2) AS readmission_rate_pct
FROM analytics.fact_patient_summary fps
JOIN analytics.fact_readmission fra USING (encounter_surrogate_id)
GROUP BY medication_bucket
ORDER BY medication_bucket;


-- ─────────────────────────────────────────────────────────────
-- 9. High-Risk Cohort (readmitted within 30 days)
-- ─────────────────────────────────────────────────────────────

SELECT
    age_group,
    gender,
    primary_diagnosis_category,
    COUNT(*)                                          AS readmitted_encounters,
    ROUND(AVG(fps.length_of_stay), 2)                 AS avg_los,
    ROUND(AVG(fps.num_medications::NUMERIC), 2)       AS avg_medications,
    ROUND(AVG(fps.prior_admission_count::NUMERIC), 2) AS avg_prior_admissions
FROM analytics.fact_readmission fra
JOIN analytics.fact_patient_summary fps USING (encounter_surrogate_id)
WHERE fra.readmission_binary = 1
GROUP BY age_group, gender, primary_diagnosis_category
ORDER BY readmitted_encounters DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────────
-- 10. Diabetes Medication Impact on Readmission
-- ─────────────────────────────────────────────────────────────

SELECT
    fps.has_diabetes_drug,
    CASE fps.has_diabetes_drug WHEN 1 THEN 'Has Diabetes Drug' ELSE 'No Diabetes Drug' END AS drug_label,
    COUNT(*)                                AS encounter_count,
    ROUND(AVG(fra.readmission_binary)*100, 2) AS readmission_rate_pct
FROM analytics.fact_patient_summary fps
JOIN analytics.fact_readmission fra USING (encounter_surrogate_id)
GROUP BY fps.has_diabetes_drug
ORDER BY fps.has_diabetes_drug;


-- ─────────────────────────────────────────────────────────────
-- 11. Readmission by Race
-- ─────────────────────────────────────────────────────────────

SELECT
    race,
    COUNT(*)                                AS total_encounters,
    SUM(readmission_binary)                 AS readmitted_count,
    ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
FROM analytics.fact_readmission
GROUP BY race
ORDER BY total_encounters DESC;


-- ─────────────────────────────────────────────────────────────
-- 12. Prior Admissions vs Readmission Rate
-- ─────────────────────────────────────────────────────────────

SELECT
    fps.is_prior_admission,
    CASE fps.is_prior_admission WHEN 1 THEN 'Has Prior Admissions' ELSE 'No Prior Admissions' END AS prior_label,
    COUNT(*)                                AS encounter_count,
    ROUND(AVG(fra.readmission_binary)*100, 2) AS readmission_rate_pct
FROM analytics.fact_patient_summary fps
JOIN analytics.fact_readmission fra USING (encounter_surrogate_id)
GROUP BY fps.is_prior_admission
ORDER BY fps.is_prior_admission;


-- ─────────────────────────────────────────────────────────────
-- 13. Pipeline Data Quality Summary
-- ─────────────────────────────────────────────────────────────

SELECT
    dataset_label,
    check_name,
    status,
    description,
    failed_count,
    check_timestamp
FROM analytics.validation_results
ORDER BY dataset_label, check_timestamp DESC;


-- ─────────────────────────────────────────────────────────────
-- 14. Rejection Reason Summary
-- ─────────────────────────────────────────────────────────────

SELECT
    rejection_reason,
    COUNT(*) AS rejected_count
FROM analytics.rejected_records
GROUP BY rejection_reason
ORDER BY rejected_count DESC;
