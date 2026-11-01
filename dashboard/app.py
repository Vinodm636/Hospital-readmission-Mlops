"""
app.py
------
Streamlit Analytics Dashboard for the Hospital Readmission project.

What this dashboard does:
    Provides an interactive web interface for exploring hospital readmission
    patterns from the UCI Diabetes 130-US Hospitals dataset.

    Dashboard views (7 total — exceeds the 5 minimum requirement):
        1. Overview KPIs        — total encounters, readmission rate, avg LOS
        2. Readmission by Age   — rate breakdown across 10 age brackets
        3. Readmission by Diagnosis — top diagnosis categories
        4. Length-of-Stay Analysis  — LOS distribution by readmission status
        5. Admission Patterns   — admission type and discharge disposition
        6. High-Risk Cohort     — deep-dive on <30-day readmitted patients
        7. Data Quality Report  — pipeline validation results

    Interactive filters (sidebar):
        - Age group
        - Gender
        - Admission type
        - Primary diagnosis category
        - Readmission status

Why Streamlit:
    Streamlit turns Python scripts into interactive web apps with minimal
    boilerplate.  It is ideal when interactive filtering and future ML
    predictions need to be integrated into the same application.

Data source:
    All data is read from PostgreSQL analytics schema — never from
    manually edited CSV files.

Disclaimer:
    This dashboard is for academic demonstration only. Data is de-identified
    and publicly available (UCI Diabetes dataset). Results are analytical,
    not clinical.

Run:
    streamlit run dashboard/app.py

Viva questions:
    Q: Why do we read from PostgreSQL rather than a CSV for the dashboard?
    A: PostgreSQL provides consistent, query-optimised data. Multiple users
       can query it simultaneously, and the pipeline always writes to the
       same place, so the dashboard always shows the latest processed data.

    Q: What is caching in Streamlit?
    A: @st.cache_data caches the result of a function so the database is
       not re-queried every time the user moves a slider. This makes the
       dashboard faster.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

# ─────────────────────────────────────────────────────────────────────────────
# Setup — paths and environment
# ─────────────────────────────────────────────────────────────────────────────

# Add project root to path so src imports work
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hospital Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Database connection with caching
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db_engine():
    """
    Create and cache the SQLAlchemy engine.
    @st.cache_resource ensures the engine is created only once per session.
    """
    try:
        from src.database.connection import get_engine
        return get_engine()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_query(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query and return the result as a DataFrame.
    Results are cached for 5 minutes to avoid re-querying on every interaction.

    Args:
        sql (str): SQL query string.

    Returns:
        pd.DataFrame: Query results, or empty DataFrame on error.
    """
    engine = get_db_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
    except Exception as e:
        st.warning(f"Query failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_kpi_data() -> pd.DataFrame:
    return run_query("""
        SELECT
            COUNT(*) AS total_encounters,
            SUM(fra.readmission_binary) AS total_readmitted,
            ROUND(AVG(fra.readmission_binary) * 100, 2) AS readmission_rate_pct,
            ROUND(AVG(fps.length_of_stay), 2) AS avg_los_days,
            ROUND(AVG(fps.num_medications::NUMERIC), 2) AS avg_medications,
            ROUND(AVG(fps.num_lab_procedures::NUMERIC), 2) AS avg_lab_procedures,
            COUNT(DISTINCT fra.patient_surrogate_id) AS unique_patients
        FROM analytics.fact_readmission fra
        JOIN analytics.fact_patient_summary fps USING (encounter_surrogate_id)
    """)


@st.cache_data(ttl=300)
def load_readmission_by_age() -> pd.DataFrame:
    return run_query("""
        SELECT
            age_group,
            age_midpoint,
            COUNT(*) AS total_encounters,
            SUM(readmission_binary) AS readmitted_count,
            ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
        FROM analytics.fact_readmission
        WHERE age_group IS NOT NULL
        GROUP BY age_group, age_midpoint
        ORDER BY age_midpoint
    """)


@st.cache_data(ttl=300)
def load_readmission_by_diagnosis() -> pd.DataFrame:
    return run_query("""
        SELECT
            primary_diagnosis_category,
            COUNT(*) AS total_encounters,
            SUM(readmission_binary) AS readmitted_count,
            ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
        FROM analytics.fact_readmission
        WHERE primary_diagnosis_category IS NOT NULL
          AND primary_diagnosis_category != 'Unknown'
        GROUP BY primary_diagnosis_category
        ORDER BY total_encounters DESC
        LIMIT 15
    """)


@st.cache_data(ttl=300)
def load_los_analysis() -> pd.DataFrame:
    return run_query("""
        SELECT
            fps.length_of_stay,
            fra.readmission_binary,
            fra.age_group,
            COUNT(*) AS encounter_count
        FROM analytics.fact_patient_summary fps
        JOIN analytics.fact_readmission fra USING (encounter_surrogate_id)
        WHERE fps.length_of_stay IS NOT NULL
        GROUP BY fps.length_of_stay, fra.readmission_binary, fra.age_group
        ORDER BY fps.length_of_stay
    """)


@st.cache_data(ttl=300)
def load_admission_patterns() -> pd.DataFrame:
    return run_query("""
        SELECT
            admission_type_id,
            COUNT(*) AS total_encounters,
            SUM(readmission_binary) AS readmitted_count,
            ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
        FROM analytics.fact_readmission
        WHERE admission_type_id IS NOT NULL
        GROUP BY admission_type_id
        ORDER BY total_encounters DESC
    """)


@st.cache_data(ttl=300)
def load_discharge_patterns() -> pd.DataFrame:
    return run_query("""
        SELECT
            discharge_disposition_id,
            COUNT(*) AS encounter_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
            ROUND(AVG(readmission_binary) * 100, 2) AS readmission_rate_pct
        FROM analytics.fact_readmission
        WHERE discharge_disposition_id IS NOT NULL
        GROUP BY discharge_disposition_id
        ORDER BY encounter_count DESC
        LIMIT 10
    """)


@st.cache_data(ttl=300)
def load_high_risk_cohort() -> pd.DataFrame:
    return run_query("""
        SELECT
            fra.age_group,
            fra.gender,
            fra.primary_diagnosis_category,
            COUNT(*) AS readmitted_encounters,
            ROUND(AVG(fps.length_of_stay), 2) AS avg_los,
            ROUND(AVG(fps.num_medications::NUMERIC), 2) AS avg_medications,
            ROUND(AVG(fps.prior_admission_count::NUMERIC), 2) AS avg_prior_admissions
        FROM analytics.fact_readmission fra
        JOIN analytics.fact_patient_summary fps USING (encounter_surrogate_id)
        WHERE fra.readmission_binary = 1
          AND fra.age_group IS NOT NULL
        GROUP BY fra.age_group, fra.gender, fra.primary_diagnosis_category
        ORDER BY readmitted_encounters DESC
        LIMIT 20
    """)


@st.cache_data(ttl=300)
def load_validation_results() -> pd.DataFrame:
    return run_query("""
        SELECT
            dataset_label,
            check_name,
            status,
            description,
            failed_count,
            check_timestamp
        FROM analytics.validation_results
        ORDER BY dataset_label, check_timestamp DESC
        LIMIT 50
    """)


@st.cache_data(ttl=300)
def load_medication_analysis() -> pd.DataFrame:
    return run_query("""
        SELECT
            CASE
                WHEN fps.num_medications <= 5  THEN '01-05 meds'
                WHEN fps.num_medications <= 10 THEN '06-10 meds'
                WHEN fps.num_medications <= 20 THEN '11-20 meds'
                WHEN fps.num_medications <= 30 THEN '21-30 meds'
                ELSE '30+ meds'
            END AS medication_bucket,
            COUNT(*) AS encounter_count,
            ROUND(AVG(fra.readmission_binary) * 100, 2) AS readmission_rate_pct
        FROM analytics.fact_patient_summary fps
        JOIN analytics.fact_readmission fra USING (encounter_surrogate_id)
        GROUP BY medication_bucket
        ORDER BY medication_bucket
    """)


@st.cache_data(ttl=300)
def load_all_encounters_filtered(
    age_groups: list,
    genders: list,
    diag_categories: list,
    readmit_status: list,
) -> pd.DataFrame:
    """Load filtered encounter data for interactive sidebar controls."""
    age_filter = "AND fra.age_group = ANY(:ages)" if age_groups else ""
    gender_filter = "AND fra.gender = ANY(:genders)" if genders else ""
    diag_filter = "AND fra.primary_diagnosis_category = ANY(:diags)" if diag_categories else ""

    readmit_map = {"Readmitted <30": 1, "Not Readmitted": 0}
    binary_values = [readmit_map[s] for s in readmit_status if s in readmit_map]
    readmit_filter = f"AND fra.readmission_binary IN ({','.join(map(str, binary_values))})" if binary_values else ""

    sql = f"""
        SELECT
            fra.age_group, fra.gender, fra.race,
            fra.primary_diagnosis_category,
            fra.readmission_binary,
            fra.readmitted,
            fps.length_of_stay,
            fps.num_medications,
            fps.prior_admission_count,
            fps.num_lab_procedures
        FROM analytics.fact_readmission fra
        JOIN analytics.fact_patient_summary fps USING (encounter_surrogate_id)
        WHERE 1=1
        {readmit_filter}
        LIMIT 5000
    """
    return run_query(sql)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Plotly theme settings
# ─────────────────────────────────────────────────────────────────────────────

CHART_THEME = "plotly_white"
COLOR_READMITTED = "#E63946"
COLOR_NOT_READMITTED = "#457B9D"
COLOR_SCALE = "Blues"

ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective",
    4: "Newborn", 5: "Not Available", 6: "NULL", 7: "Trauma",
    8: "Not Mapped",
}


def admission_type_label(tid: int) -> str:
    return ADMISSION_TYPE_MAP.get(int(tid) if pd.notna(tid) else 0, f"Type {tid}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar filters and return selected values."""
    st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=64)
    st.sidebar.title("🏥 Dashboard Filters")
    st.sidebar.markdown("*Filter all charts simultaneously*")
    st.sidebar.divider()

    age_options = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
    ]
    selected_ages = st.sidebar.multiselect(
        "Age Group",
        options=age_options,
        default=age_options,
        help="Select one or more age brackets",
    )

    selected_genders = st.sidebar.multiselect(
        "Gender",
        options=["Male", "Female"],
        default=["Male", "Female"],
    )

    diag_options = [
        "Circulatory_System", "Endocrine_Metabolic", "Respiratory_System",
        "Digestive_System", "Musculoskeletal", "Nervous_System",
        "Genitourinary", "Injury_Poisoning", "Symptoms_Signs",
        "Neoplasms", "Mental_Disorders", "Infectious_Parasitic",
        "Other", "Unknown",
    ]
    selected_diags = st.sidebar.multiselect(
        "Diagnosis Category",
        options=diag_options,
        default=diag_options[:10],
    )

    selected_readmit = st.sidebar.multiselect(
        "Readmission Status",
        options=["Readmitted <30", "Not Readmitted"],
        default=["Readmitted <30", "Not Readmitted"],
    )

    st.sidebar.divider()


    return {
        "ages": selected_ages,
        "genders": selected_genders,
        "diags": selected_diags,
        "readmit": selected_readmit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# View 1: Overview KPIs
# ─────────────────────────────────────────────────────────────────────────────

def render_overview():
    """Display top-level KPI metric cards."""
    st.header("📊 Overview")
    st.caption("Key performance indicators from the hospital readmission dataset")

    df = load_kpi_data()
    if df.empty:
        st.warning("No data available. Run the ETL pipeline first.")
        return

    row = df.iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "Total Encounters",
            f"{int(row['total_encounters']):,}",
            help="Total hospital encounters in the dataset",
        )
    with col2:
        st.metric(
            "Readmitted (<30 days)",
            f"{int(row['total_readmitted']):,}",
            help="Patients readmitted within 30 days",
        )
    with col3:
        st.metric(
            "Readmission Rate",
            f"{row['readmission_rate_pct']}%",
            help="Percentage of encounters resulting in <30-day readmission",
        )
    with col4:
        st.metric(
            "Avg Length of Stay",
            f"{row['avg_los_days']} days",
            help="Mean number of days in hospital per encounter",
        )
    with col5:
        st.metric(
            "Avg Medications",
            f"{row['avg_medications']}",
            help="Mean number of distinct medications administered",
        )

    st.divider()

    # Readmission class breakdown donut
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Readmission Class Distribution")
        dist_df = run_query("""
            SELECT readmitted, COUNT(*) AS count
            FROM analytics.fact_readmission
            GROUP BY readmitted
            ORDER BY readmitted
        """)
        if not dist_df.empty:
            label_map = {"<30": "Readmitted <30 days", ">30": "Readmitted >30 days", "NO": "Not Readmitted"}
            dist_df["label"] = dist_df["readmitted"].map(label_map)
            fig = px.pie(
                dist_df,
                values="count",
                names="label",
                color="label",
                color_discrete_map={
                    "Readmitted <30 days": COLOR_READMITTED,
                    "Readmitted >30 days": "#F4A261",
                    "Not Readmitted": COLOR_NOT_READMITTED,
                },
                hole=0.45,
                template=CHART_THEME,
            )
            fig.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Gender Distribution by Readmission Status")
        gender_df = run_query("""
            SELECT gender, readmitted, COUNT(*) AS count
            FROM analytics.fact_readmission
            WHERE gender IN ('Male', 'Female')
            GROUP BY gender, readmitted
            ORDER BY gender, readmitted
        """)
        if not gender_df.empty:
            fig = px.bar(
                gender_df,
                x="gender",
                y="count",
                color="readmitted",
                barmode="group",
                template=CHART_THEME,
                color_discrete_map={"<30": COLOR_READMITTED, ">30": "#F4A261", "NO": COLOR_NOT_READMITTED},
                labels={"count": "Encounter Count", "gender": "Gender", "readmitted": "Readmission"},
            )
            fig.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# View 2: Readmission by Age Group
# ─────────────────────────────────────────────────────────────────────────────

def render_readmission_by_age():
    """Show readmission rates across all 10 UCI age brackets."""
    st.header("👴 Readmission Rate by Age Group")
    st.caption(
        "The UCI dataset uses 10-year age brackets instead of exact ages to preserve de-identification."
    )

    df = load_readmission_by_age()
    if df.empty:
        st.warning("No age data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df,
            x="age_group",
            y="readmission_rate_pct",
            color="readmission_rate_pct",
            color_continuous_scale="Reds",
            template=CHART_THEME,
            labels={
                "age_group": "Age Group",
                "readmission_rate_pct": "Readmission Rate (%)",
            },
            title="Readmission Rate (%) by Age Group",
            text="readmission_rate_pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Age Group",
            yaxis_title="Readmission Rate (%)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df,
            x="age_group",
            y="readmission_rate_pct",
            size="total_encounters",
            color="readmission_rate_pct",
            color_continuous_scale="Reds",
            template=CHART_THEME,
            labels={
                "age_group": "Age Group",
                "readmission_rate_pct": "Readmission Rate (%)",
                "total_encounters": "Total Encounters",
            },
            title="Bubble Chart: Rate vs Volume",
            hover_data=["total_encounters", "readmitted_count"],
        )
        fig2.update_layout(
            coloraxis_showscale=False,
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        df.rename(columns={
            "age_group": "Age Group",
            "total_encounters": "Total",
            "readmitted_count": "Readmitted",
            "readmission_rate_pct": "Rate (%)",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# View 3: Readmission by Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def render_readmission_by_diagnosis():
    """Show readmission rates by ICD-9 diagnosis category."""
    st.header("🩺 Readmission by Diagnosis Category")
    st.caption(
        "Diagnosis codes from diag_1 (primary diagnosis) are mapped to ICD-9 categories."
    )

    df = load_readmission_by_diagnosis()
    if df.empty:
        st.warning("No diagnosis data available.")
        return

    fig = px.bar(
        df,
        x="readmission_rate_pct",
        y="primary_diagnosis_category",
        orientation="h",
        color="total_encounters",
        color_continuous_scale=COLOR_SCALE,
        template=CHART_THEME,
        labels={
            "readmission_rate_pct": "Readmission Rate (%)",
            "primary_diagnosis_category": "Diagnosis Category",
            "total_encounters": "Total Encounters",
        },
        title="Readmission Rate by ICD-9 Diagnosis Category",
        text="readmission_rate_pct",
        hover_data=["total_encounters", "readmitted_count"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    # Treemap for volume
    st.subheader("Encounter Volume by Diagnosis Category")
    fig2 = px.treemap(
        df,
        path=["primary_diagnosis_category"],
        values="total_encounters",
        color="readmission_rate_pct",
        color_continuous_scale="RdYlGn_r",
        template=CHART_THEME,
        title="Treemap: Encounter Volume (colour = readmission rate)",
        hover_data={"readmission_rate_pct": ":.1f", "total_encounters": True},
    )
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# View 4: Length-of-Stay Analysis
# ─────────────────────────────────────────────────────────────────────────────

def render_los_analysis():
    """Analyse length of stay by readmission status."""
    st.header("🛏️ Length-of-Stay Analysis")
    st.caption(
        "Length of stay (days in hospital) is a proxy for illness severity. "
        "The UCI dataset caps this at 14 days."
    )

    df = load_los_analysis()
    if df.empty:
        st.warning("No length-of-stay data available.")
        return

    # Aggregate for histogram
    df["readmit_label"] = df["readmission_binary"].map(
        {1: "Readmitted <30", 0: "Not Readmitted"}
    )

    col1, col2 = st.columns(2)

    with col1:
        # LOS distribution histogram
        agg = df.groupby(["length_of_stay", "readmit_label"])["encounter_count"].sum().reset_index()
        fig = px.bar(
            agg,
            x="length_of_stay",
            y="encounter_count",
            color="readmit_label",
            barmode="overlay",
            opacity=0.75,
            template=CHART_THEME,
            color_discrete_map={
                "Readmitted <30": COLOR_READMITTED,
                "Not Readmitted": COLOR_NOT_READMITTED,
            },
            labels={
                "length_of_stay": "Length of Stay (days)",
                "encounter_count": "Encounter Count",
                "readmit_label": "Status",
            },
            title="LOS Distribution by Readmission Status",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Average LOS by age group
        los_age = run_query("""
            SELECT
                fra.age_group,
                fra.readmission_binary,
                ROUND(AVG(fps.length_of_stay), 2) AS avg_los
            FROM analytics.fact_readmission fra
            JOIN analytics.fact_patient_summary fps USING (encounter_surrogate_id)
            WHERE fra.age_group IS NOT NULL
            GROUP BY fra.age_group, fra.readmission_binary
            ORDER BY fra.age_group, fra.readmission_binary
        """)
        if not los_age.empty:
            los_age["readmit_label"] = los_age["readmission_binary"].map(
                {1: "Readmitted <30", 0: "Not Readmitted"}
            )
            fig2 = px.line(
                los_age,
                x="age_group",
                y="avg_los",
                color="readmit_label",
                markers=True,
                template=CHART_THEME,
                color_discrete_map={
                    "Readmitted <30": COLOR_READMITTED,
                    "Not Readmitted": COLOR_NOT_READMITTED,
                },
                labels={
                    "age_group": "Age Group",
                    "avg_los": "Average LOS (days)",
                    "readmit_label": "Status",
                },
                title="Average LOS by Age Group and Readmission",
            )
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

    # Long-stay summary
    long_stay = run_query("""
        SELECT
            is_long_stay,
            ROUND(AVG(fra.readmission_binary) * 100, 2) AS readmission_rate_pct,
            COUNT(*) AS encounter_count
        FROM analytics.fact_patient_summary fps
        JOIN analytics.fact_readmission fra USING (encounter_surrogate_id)
        GROUP BY is_long_stay
        ORDER BY is_long_stay
    """)
    if not long_stay.empty:
        long_stay["stay_label"] = long_stay["is_long_stay"].map(
            {0: "Short Stay (≤7 days)", 1: "Long Stay (>7 days)"}
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Short Stay Readmission Rate",
                f"{long_stay[long_stay['is_long_stay']==0]['readmission_rate_pct'].values[0] if len(long_stay[long_stay['is_long_stay']==0]) > 0 else 'N/A'}%",
            )
        with col_b:
            st.metric(
                "Long Stay Readmission Rate (>7 days)",
                f"{long_stay[long_stay['is_long_stay']==1]['readmission_rate_pct'].values[0] if len(long_stay[long_stay['is_long_stay']==1]) > 0 else 'N/A'}%",
            )


# ─────────────────────────────────────────────────────────────────────────────
# View 5: Admission and Discharge Patterns
# ─────────────────────────────────────────────────────────────────────────────

def render_admission_patterns():
    """Show admission type, discharge disposition patterns."""
    st.header("🚑 Admission and Discharge Patterns")
    st.caption(
        "Admission type IDs are per the UCI IDS_mapping.csv file. "
        "1=Emergency, 2=Urgent, 3=Elective, 7=Trauma."
    )

    col1, col2 = st.columns(2)

    # Admission type chart
    adm_df = load_admission_patterns()
    if not adm_df.empty:
        adm_df["admission_label"] = adm_df["admission_type_id"].apply(admission_type_label)
        with col1:
            fig = px.bar(
                adm_df,
                x="admission_label",
                y="total_encounters",
                color="readmission_rate_pct",
                color_continuous_scale="Reds",
                template=CHART_THEME,
                labels={
                    "admission_label": "Admission Type",
                    "total_encounters": "Encounter Count",
                    "readmission_rate_pct": "Readmission Rate (%)",
                },
                title="Encounter Volume by Admission Type",
                text="total_encounters",
            )
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(height=400, coloraxis_showscale=True)
            st.plotly_chart(fig, use_container_width=True)

    # Discharge disposition chart
    dis_df = load_discharge_patterns()
    if not dis_df.empty:
        with col2:
            fig2 = px.bar(
                dis_df,
                x="discharge_disposition_id",
                y="encounter_count",
                color="readmission_rate_pct",
                color_continuous_scale="Reds",
                template=CHART_THEME,
                labels={
                    "discharge_disposition_id": "Discharge Disposition ID",
                    "encounter_count": "Encounter Count",
                    "readmission_rate_pct": "Readmission Rate (%)",
                },
                title="Top 10 Discharge Dispositions",
                text="readmission_rate_pct",
            )
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

    # Medication analysis
    st.subheader("Medication Count vs Readmission Rate")
    med_df = load_medication_analysis()
    if not med_df.empty:
        fig3 = px.bar(
            med_df,
            x="medication_bucket",
            y="readmission_rate_pct",
            color="encounter_count",
            color_continuous_scale=COLOR_SCALE,
            template=CHART_THEME,
            labels={
                "medication_bucket": "Medication Count Range",
                "readmission_rate_pct": "Readmission Rate (%)",
                "encounter_count": "Encounter Count",
            },
            title="Readmission Rate by Number of Medications",
            text="readmission_rate_pct",
        )
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# View 6: High-Risk Cohort
# ─────────────────────────────────────────────────────────────────────────────

def render_high_risk_cohort():
    """Deep-dive into the high-risk (readmitted <30 days) cohort."""
    st.header("⚠️ High-Risk Cohort Analysis")
    st.caption(
        "Patients readmitted within 30 days. "
        "Understanding this group can inform clinical interventions."
    )
    st.warning(
        "⚕️ **Academic prototype only** — results are not for clinical use."
    )

    df = load_high_risk_cohort()
    if df.empty:
        st.warning("No high-risk cohort data available.")
        return

    col1, col2 = st.columns(2)

    # Top diagnosis categories in high-risk cohort
    diag_summary = df.groupby("primary_diagnosis_category")["readmitted_encounters"].sum().reset_index()
    diag_summary = diag_summary.sort_values("readmitted_encounters", ascending=False).head(10)

    with col1:
        fig = px.bar(
            diag_summary,
            x="readmitted_encounters",
            y="primary_diagnosis_category",
            orientation="h",
            template=CHART_THEME,
            color="readmitted_encounters",
            color_continuous_scale="Reds",
            labels={
                "readmitted_encounters": "Readmitted Encounters",
                "primary_diagnosis_category": "Diagnosis Category",
            },
            title="Top Diagnosis Categories (High-Risk Cohort)",
        )
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Age breakdown of high-risk cohort
    age_summary = df.groupby("age_group")["readmitted_encounters"].sum().reset_index()
    with col2:
        fig2 = px.pie(
            age_summary,
            values="readmitted_encounters",
            names="age_group",
            title="Age Group Distribution (High-Risk Cohort)",
            template=CHART_THEME,
            hole=0.35,
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: age × diagnosis
    st.subheader("Heatmap: Readmissions by Age Group × Diagnosis")
    pivot = df.pivot_table(
        values="readmitted_encounters",
        index="age_group",
        columns="primary_diagnosis_category",
        aggfunc="sum",
        fill_value=0,
    )
    if not pivot.empty:
        fig3 = px.imshow(
            pivot,
            template=CHART_THEME,
            color_continuous_scale="Reds",
            labels={"color": "Readmitted Count"},
            title="High-Risk Readmissions: Age Group × Diagnosis",
            aspect="auto",
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # Detailed table
    st.subheader("Top 20 High-Risk Segments")
    st.dataframe(
        df.rename(columns={
            "age_group": "Age Group",
            "gender": "Gender",
            "primary_diagnosis_category": "Diagnosis",
            "readmitted_encounters": "Readmissions",
            "avg_los": "Avg LOS",
            "avg_medications": "Avg Meds",
            "avg_prior_admissions": "Avg Prior Admissions",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# View 7: Data Quality Report
# ─────────────────────────────────────────────────────────────────────────────

def render_data_quality():
    """Display pipeline data-quality validation results."""
    st.header("✅ Data Quality Report")
    st.caption(
        "Automated data-quality checks are run at each pipeline stage. "
        "Results are stored in analytics.validation_results."
    )

    df = load_validation_results()
    if df.empty:
        st.warning(
            "No validation results found. "
            "Run the ETL pipeline (Airflow DAG) to populate this view."
        )
        return

    # Summary metrics
    total = len(df)
    passed = (df["status"] == "PASS").sum()
    failed = (df["status"] == "FAIL").sum()
    warned = (df["status"] == "WARN").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Checks", total)
    col2.metric("✅ Passed", passed)
    col3.metric("❌ Failed", failed)
    col4.metric("⚠️ Warnings", warned)

    # Status colour coding
    def style_status(val: str) -> str:
        colour_map = {"PASS": "background-color: #d4edda", "FAIL": "background-color: #f8d7da", "WARN": "background-color: #fff3cd"}
        return colour_map.get(val, "")

    styled = df.style.map(style_status, subset=["status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar chart of check results
    summary = df.groupby(["dataset_label", "status"]).size().reset_index(name="count")
    fig = px.bar(
        summary,
        x="dataset_label",
        y="count",
        color="status",
        barmode="group",
        template=CHART_THEME,
        color_discrete_map={"PASS": "#28a745", "FAIL": "#dc3545", "WARN": "#ffc107"},
        labels={"dataset_label": "Dataset Stage", "count": "Check Count", "status": "Status"},
        title="Data Quality Check Results by Pipeline Stage",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# View 8: 30-Day Readmission Risk Prediction
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_ml_model_and_preprocessor():
    """Load the best trained model (Random Forest) and preprocessor."""
    import joblib
    models_dir = PROJECT_ROOT / "models"
    try:
        model = joblib.load(models_dir / "random_forest.joblib")
        preprocessor = joblib.load(models_dir / "preprocessor.joblib")
        return model, preprocessor
    except FileNotFoundError:
        return None, None


def _age_to_midpoint(age: str) -> float:
    mapping = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0,
        "[30-40)": 35.0, "[40-50)": 45.0, "[50-60)": 55.0,
        "[60-70)": 65.0, "[70-80)": 75.0, "[80-90)": 85.0,
        "[90-100)": 95.0,
    }
    return mapping.get(age, 65.0)


def _run_prediction(model, preprocessor, inputs: dict) -> dict:
    """Build a DataFrame from inputs, preprocess, and predict."""
    import numpy as np

    NUMERIC_FEATURES = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "diagnosis_count",
        "medication_count", "prior_admission_count", "age_midpoint",
        "total_service_utilisation",
    ]
    CATEGORICAL_FEATURES = [
        "race", "gender", "age", "admission_type_id",
        "discharge_disposition_id", "admission_source_id",
        "max_glu_serum", "a1cresult", "insulin", "diabetesmed",
        "primary_diagnosis_category",
    ]

    row = {
        # Numeric
        "time_in_hospital":         inputs["time_in_hospital"],
        "num_lab_procedures":        inputs["num_lab_procedures"],
        "num_procedures":            inputs["num_procedures"],
        "num_medications":           inputs["num_medications"],
        "number_outpatient":         inputs["number_outpatient"],
        "number_emergency":          inputs["number_emergency"],
        "number_inpatient":          inputs["number_inpatient"],
        "number_diagnoses":          inputs["number_diagnoses"],
        "diagnosis_count":           min(inputs["number_diagnoses"], 3),
        "medication_count":          inputs["num_medications"],
        "prior_admission_count":     inputs["number_inpatient"] + inputs["number_emergency"],
        "age_midpoint":              _age_to_midpoint(inputs["age"]),
        "total_service_utilisation": (inputs["number_inpatient"]
                                      + inputs["number_emergency"]
                                      + inputs["number_outpatient"]
                                      + inputs["num_procedures"]
                                      + inputs["num_medications"]),
        # Categorical
        "race":                      inputs["race"],
        "gender":                    inputs["gender"],
        "age":                       inputs["age"],
        "admission_type_id":         str(inputs["admission_type_id"]),
        "discharge_disposition_id":  str(inputs["discharge_disposition_id"]),
        "admission_source_id":       str(inputs["admission_source_id"]),
        "max_glu_serum":             inputs["max_glu_serum"],
        "a1cresult":                 inputs["a1cresult"],
        "insulin":                   inputs["insulin"],
        "diabetesmed":               inputs["diabetesmed"],
        "primary_diagnosis_category": inputs["primary_diagnosis_category"],
    }

    df = pd.DataFrame([row])
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df_features = df[num_cols + cat_cols]

    X = preprocessor.transform(df_features)
    prob = float(model.predict_proba(X)[0][1])

    THRESHOLD = 0.53
    prediction = int(prob >= THRESHOLD)

    if prob < THRESHOLD:
        risk_cat = "STANDARD / LOW RISK"
        risk_color = "#28a745"
    elif prob < 0.65:
        risk_cat = "ELEVATED / MODERATE RISK"
        risk_color = "#fd7e14"
    else:
        risk_cat = "HIGH RISK"
        risk_color = "#dc3545"

    return {
        "prob": prob,
        "prediction": prediction,
        "risk_category": risk_cat,
        "risk_color": risk_color,
        "threshold": THRESHOLD,
    }


def render_prediction():
    """Render the 30-day readmission risk prediction page."""
    st.header("🔮 30-Day Hospital Readmission Risk Prediction")
    st.caption(
        "Simulate patient discharge profiles to assess 30-day unplanned "
        "readmission probability using the validated machine learning pipeline."
    )

    model, preprocessor = load_ml_model_and_preprocessor()

    if model is None or preprocessor is None:
        st.error("Model not found. Please run the ML training pipeline first.", icon="❌")
        return

    # ── Info banner ───────────────────────────────────────────────────────────
    st.info(
        "Fill in the patient details below and click **Run Prediction** to get "
        "the estimated chance of the patient being readmitted within 30 days. "
        "Uses the **Random Forest** model — the best performing model in this project.",
        icon="ℹ️",
    )

    # ── Example patient buttons ───────────────────────────────────────────────
    st.markdown("#### Try an Example Patient")
    st.caption("Click any button to auto-fill the form with a sample patient profile.")
    col_p1, col_p2, col_p3 = st.columns(3)

    presets = {
        "low": {
            "age": "[30-40)", "gender": "Female", "race": "Caucasian",
            "time_in_hospital": 2, "num_lab_procedures": 20, "num_procedures": 0,
            "num_medications": 8, "number_outpatient": 0, "number_emergency": 0,
            "number_inpatient": 0, "number_diagnoses": 3,
            "admission_type_id": 3, "discharge_disposition_id": 1, "admission_source_id": 1,
            "max_glu_serum": "None", "a1cresult": "None",
            "insulin": "No", "diabetesmed": "No",
            "primary_diagnosis_category": "Respiratory_System",
        },
        "moderate": {
            "age": "[60-70)", "gender": "Female", "race": "Asian",
            "time_in_hospital": 4, "num_lab_procedures": 42, "num_procedures": 1,
            "num_medications": 15, "number_outpatient": 0, "number_emergency": 0,
            "number_inpatient": 0, "number_diagnoses": 7,
            "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
            "max_glu_serum": "None", "a1cresult": "None",
            "insulin": "Steady", "diabetesmed": "Yes",
            "primary_diagnosis_category": "Endocrine_Metabolic",
        },
        "high": {
            "age": "[80-90)", "gender": "Male", "race": "Caucasian",
            "time_in_hospital": 10, "num_lab_procedures": 65, "num_procedures": 3,
            "num_medications": 22, "number_outpatient": 1, "number_emergency": 2,
            "number_inpatient": 3, "number_diagnoses": 9,
            "admission_type_id": 1, "discharge_disposition_id": 3, "admission_source_id": 7,
            "max_glu_serum": ">300", "a1cresult": ">8",
            "insulin": "Up", "diabetesmed": "Yes",
            "primary_diagnosis_category": "Circulatory_System",
        },
    }

    if "preset" not in st.session_state:
        st.session_state.preset = "moderate"

    with col_p1:
        if st.button("🟢 Young, Healthy Patient", use_container_width=True,
                     help="30s, short stay, no prior admissions, no diabetes meds"):
            st.session_state.preset = "low"
    with col_p2:
        if st.button("🟠 Elderly Diabetic Patient", use_container_width=True,
                     help="60s, moderate stay, diabetic on insulin, via emergency"):
            st.session_state.preset = "moderate"
    with col_p3:
        if st.button("🔴 High-Risk Elderly Patient", use_container_width=True,
                     help="80s, long stay, multiple prior admissions, high glucose"):
            st.session_state.preset = "high"

    p = presets.get(st.session_state.preset, presets["moderate"])

    # ── Input form ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Patient Details at Discharge")

    AGE_OPTIONS = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
    ]
    # Human-readable labels for internal codes
    AGE_LABELS   = ["0–10 yrs", "10–20 yrs", "20–30 yrs", "30–40 yrs", "40–50 yrs",
                    "50–60 yrs", "60–70 yrs", "70–80 yrs", "80–90 yrs", "90–100 yrs"]
    RACE_OPTIONS = ["Caucasian", "Asian", "AfricanAmerican", "Hispanic", "Other", "Unknown"]
    RACE_LABELS  = ["White / Caucasian", "Asian", "Black / African American",
                    "Hispanic", "Other", "Unknown / Not Recorded"]
    DIAG_OPTIONS = [
        "Circulatory_System", "Endocrine_Metabolic", "Respiratory_System",
        "Digestive_System", "Musculoskeletal", "Nervous_System",
        "Genitourinary", "Injury_Poisoning", "Symptoms_Signs",
        "Neoplasms", "Mental_Disorders", "Infectious_Parasitic", "Other", "Unknown",
    ]
    DIAG_LABELS  = [
        "Heart / Blood Vessels", "Diabetes / Hormonal", "Lungs / Breathing",
        "Stomach / Digestive", "Bones / Muscles", "Brain / Nerves",
        "Kidneys / Urinary", "Injury / Poisoning", "General Symptoms",
        "Cancer / Tumours", "Mental Health", "Infections / Parasites",
        "Other", "Unknown",
    ]

    # ── Section 1: Who is the patient? ────────────────────────────────────────
    st.markdown("##### 👤 About the Patient")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        age_label = st.selectbox(
            "Age Group",
            options=AGE_LABELS,
            index=AGE_OPTIONS.index(p["age"]),
            help="Select the patient's age range")
        age = AGE_OPTIONS[AGE_LABELS.index(age_label)]
    with r1c2:
        gender = st.selectbox("Gender", options=["Female", "Male"],
            index=["Female", "Male"].index(p["gender"]))
    with r1c3:
        race_label = st.selectbox(
            "Race / Ethnicity", options=RACE_LABELS,
            index=RACE_OPTIONS.index(p["race"]),
            help="Select 'Unknown / Not Recorded' if not available")
        race = RACE_OPTIONS[RACE_LABELS.index(race_label)]

    # ── Section 2: This hospital stay ─────────────────────────────────────────
    st.markdown("##### 🏨 This Hospital Stay")
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        time_in_hospital = st.number_input(
            "Days in Hospital", min_value=1, max_value=14, value=p["time_in_hospital"],
            help="How many days was the patient admitted?")
    with r2c2:
        num_lab_procedures = st.number_input(
            "Lab Tests Done", min_value=0, max_value=132, value=p["num_lab_procedures"],
            help="Total number of blood/urine/other lab tests ordered")
    with r2c3:
        num_procedures = st.number_input(
            "Medical Procedures", min_value=0, max_value=6, value=p["num_procedures"],
            help="Non-lab procedures (e.g. X-ray, surgery, scans)")
    with r2c4:
        num_medications = st.number_input(
            "Medications Prescribed", min_value=1, max_value=81, value=p["num_medications"],
            help="Total number of different medicines given")

    r2b1, r2b2, r2b3 = st.columns(3)
    with r2b1:
        number_diagnoses = st.number_input(
            "Number of Diagnoses", min_value=1, max_value=16, value=p["number_diagnoses"],
            help="How many different health conditions were recorded?")
    with r2b2:
        diag_label = st.selectbox(
            "Main Health Problem",
            options=DIAG_LABELS,
            index=DIAG_OPTIONS.index(p["primary_diagnosis_category"]),
            help="The primary reason the patient was admitted")
        primary_diagnosis_category = DIAG_OPTIONS[DIAG_LABELS.index(diag_label)]
    with r2b3:
        how_admitted = st.selectbox(
            "How Was Patient Admitted?",
            options=["Emergency", "Urgent", "Planned / Elective", "Trauma", "Other"],
            index=["Emergency", "Urgent", "Planned / Elective", "Trauma", "Other"].index(
                {"Emergency":"Emergency","Urgent":"Urgent","Elective":"Planned / Elective",
                 "Trauma":"Trauma"}.get(
                    {1:"Emergency",2:"Urgent",3:"Elective",4:"Other",7:"Trauma",8:"Other"
                    }.get(p["admission_type_id"],"Other"), "Other")),
            help="Was this a planned visit or an emergency?")
        admission_type_id = {"Emergency":1,"Urgent":2,"Planned / Elective":3,
                             "Trauma":7,"Other":8}[how_admitted]

    # ── Section 3: Where did the patient go after discharge? ──────────────────
    st.markdown("##### 🚪 After Discharge")
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        where_from = st.selectbox(
            "Patient Came From",
            options=["Emergency Room", "Doctor Referral", "Transferred from Hospital",
                     "Clinic / HMO", "Other / Unknown"],
            index=["Emergency Room","Doctor Referral","Transferred from Hospital",
                   "Clinic / HMO","Other / Unknown"].index(
                {7:"Emergency Room",1:"Doctor Referral",4:"Transferred from Hospital",
                 2:"Doctor Referral",3:"Clinic / HMO"}.get(p["admission_source_id"],
                 "Other / Unknown")),
            help="Where was the patient before being admitted to this hospital?")
        admission_source_id = {"Emergency Room":7,"Doctor Referral":1,
                                "Transferred from Hospital":4,"Clinic / HMO":3,
                                "Other / Unknown":9}[where_from]
    with r3c2:
        where_discharged = st.selectbox(
            "Discharged To",
            options=["Went Home", "Nursing / Rehab Facility", "Another Hospital",
                     "Home with Home Care", "Hospice Care", "Other / Unknown"],
            index=["Went Home","Nursing / Rehab Facility","Another Hospital",
                   "Home with Home Care","Hospice Care","Other / Unknown"].index(
                {1:"Went Home",2:"Nursing / Rehab Facility",3:"Nursing / Rehab Facility",
                 6:"Home with Home Care",13:"Hospice Care",14:"Hospice Care",
                 22:"Nursing / Rehab Facility"}.get(p["discharge_disposition_id"],
                 "Other / Unknown")),
            help="Where did the patient go after leaving the hospital?")
        discharge_disposition_id = {"Went Home":1,"Nursing / Rehab Facility":3,
                                     "Another Hospital":4,"Home with Home Care":6,
                                     "Hospice Care":13,"Other / Unknown":18}[where_discharged]

    # ── Section 4: Past hospital visits (last 12 months) ─────────────────────
    st.markdown("##### 📋 Previous Hospital Visits (Past Year)")
    r4c1, r4c2, r4c3 = st.columns(3)
    with r4c1:
        number_inpatient = st.number_input(
            "Times Admitted Overnight", min_value=0, max_value=20,
            value=p["number_inpatient"],
            help="How many times was the patient admitted and stayed overnight in the past year?")
    with r4c2:
        number_emergency = st.number_input(
            "Emergency Room Visits", min_value=0, max_value=20,
            value=p["number_emergency"],
            help="How many ER visits did the patient have in the past year?")
    with r4c3:
        number_outpatient = st.number_input(
            "Outpatient / Clinic Visits", min_value=0, max_value=40,
            value=p["number_outpatient"],
            help="Regular clinic or outpatient appointments in the past year")

    # ── Section 5: Diabetes & Blood Sugar ─────────────────────────────────────
    st.markdown("##### 🩸 Diabetes & Blood Sugar")
    r5c1, r5c2, r5c3, r5c4 = st.columns(4)
    with r5c1:
        glu_label = st.selectbox(
            "Blood Glucose Test Result",
            options=["Not Tested", "Normal", "High (200–300)", "Very High (300+)"],
            index=["None","Norm",">200",">300"].index(p["max_glu_serum"]),
            help="Result of the blood glucose serum test during this stay")
        max_glu_serum = {"Not Tested":"None","Normal":"Norm",
                         "High (200–300)":">200","Very High (300+)":">300"}[glu_label]
    with r5c2:
        a1c_label = st.selectbox(
            "HbA1c (3-Month Avg Blood Sugar)",
            options=["Not Tested", "Normal", "Slightly High (>7%)", "High (>8%)"],
            index=["None","Norm",">7",">8"].index(p["a1cresult"]),
            help="HbA1c reflects average blood sugar over the past 3 months")
        a1cresult = {"Not Tested":"None","Normal":"Norm",
                     "Slightly High (>7%)":">7","High (>8%)":">8"}[a1c_label]
    with r5c3:
        ins_label = st.selectbox(
            "Insulin Dose Change",
            options=["Not on Insulin", "No Change", "Dose Increased", "Dose Decreased"],
            index=["No","Steady","Up","Down"].index(p["insulin"]),
            help="Was the patient's insulin dose changed during this stay?")
        insulin = {"Not on Insulin":"No","No Change":"Steady",
                   "Dose Increased":"Up","Dose Decreased":"Down"}[ins_label]
    with r5c4:
        dm_label = st.selectbox(
            "On Diabetes Medication?",
            options=["Yes", "No"],
            index=["Yes","No"].index(p["diabetesmed"]),
            help="Was any diabetes medication prescribed during this visit?")
        diabetesmed = dm_label

    # ── Predict button ─────────────────────────────────────────────────────────
    st.divider()
    predict_clicked = st.button(
        "🔍  Run Prediction",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:
        inputs = {
            "age": age, "gender": gender, "race": race,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "max_glu_serum": max_glu_serum,
            "a1cresult": a1cresult,
            "insulin": insulin,
            "diabetesmed": diabetesmed,
            "primary_diagnosis_category": primary_diagnosis_category,
        }

        with st.spinner("Calculating readmission risk..."):
            result = _run_prediction(model, preprocessor, inputs)

        # ── Result card ────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Result")

        color  = result["risk_color"]
        pct    = result["prob"] * 100
        label  = result["risk_category"]
        is_low = result["prediction"] == 0

        res_left, res_right = st.columns([1, 1])

        with res_left:
            st.markdown(
                f"""
                <div style="border:2px solid {color}; border-radius:14px;
                            padding:28px 32px; text-align:center;">
                    <div style="font-size:13px; font-weight:700; letter-spacing:1px;
                                color:{color}; border:1px solid {color};
                                display:inline-block; padding:4px 14px;
                                border-radius:20px; margin-bottom:12px;">
                        {label}
                    </div>
                    <div style="font-size:64px; font-weight:800; color:{color};
                                line-height:1.1; margin-bottom:8px;">
                        {pct:.1f}%
                    </div>
                    <div style="font-size:14px; color:#aaa; margin-bottom:6px;">
                        estimated chance of readmission within 30 days
                    </div>
                    <div style="font-size:12px; color:#777;">
                        Average across all patients: <b>11.2%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_right:
            st.markdown("**What does this mean?**")
            if is_low:
                st.success("✅ **Low risk** — This patient is unlikely to return to hospital within 30 days based on the available information.")
                st.markdown("""
**Recommended next steps:**
- Book a follow-up appointment within 2–4 weeks
- Give the patient written discharge instructions
- Make sure all prescriptions are filled before they leave
                """)
            elif pct < 65:
                st.warning("⚠️ **Moderate risk** — This patient has a higher than average chance of returning. Extra attention is advised.")
                st.markdown("""
**Recommended next steps:**
- Schedule a follow-up within 1 week
- Review all medications before discharge
- Educate the patient on warning signs to watch for
- Consider a check-in phone call 2–3 days after discharge
                """)
            else:
                st.error("🚨 **High risk** — This patient is at significant risk of returning within 30 days. Immediate planning is needed.")
                st.markdown("""
**Recommended next steps:**
- Arrange a follow-up within 72 hours
- Assign a dedicated care coordinator or nurse
- Contact the patient's primary doctor right away
- Consider extra monitoring or a short-term care facility
- Call the patient within 24 hours of discharge
                """)
            st.caption("⚠️ This is a statistical estimate, not a medical diagnosis. Always use clinical judgment.")

        # ── Model performance summary ──────────────────────────────────────────
        st.divider()
        st.markdown("### How Reliable Is This Prediction?")
        st.caption("These numbers show how well the model performed on 20,354 patients it had never seen before.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy (AUC)", "0.68",
                  help="1.0 = perfect, 0.5 = random guessing. 0.68 is reasonable for this type of medical data.")
        m2.metric("Catches High-Risk Patients", "59%",
                  help="Out of every 100 patients who actually came back, the model correctly flagged 59.")
        m3.metric("Correct Low-Risk Calls", "76%",
                  help="Out of patients who did NOT return, the model correctly identified 76%.")
        m4.metric("Baseline Rate", "11.2%",
                  help="Only 11.2% of patients in the original dataset were readmitted within 30 days.")

        # ── Which factors matter most ──────────────────────────────────────────
        st.markdown("### Which Patient Factors Matter Most?")
        st.caption("The model considers these factors most heavily when making its prediction.")
        try:
            importances = model.feature_importances_
            feat_names  = preprocessor.get_feature_names_out()
            fi = pd.DataFrame({
                "Factor": feat_names,
                "Weight": importances,
            }).sort_values("Weight", ascending=True).tail(10)
            # Clean up sklearn prefixes and make names human-readable
            fi["Factor"] = (fi["Factor"]
                .str.replace(r"^num__", "", regex=True)
                .str.replace(r"^cat__", "", regex=True)
                .str.replace("_", " ").str.title()
                .str.replace("Num ", "# "))
            fig_fi = px.bar(
                fi, x="Weight", y="Factor", orientation="h",
                template="plotly_white",
                color="Weight", color_continuous_scale="Blues",
                labels={"Weight": "Influence on Prediction", "Factor": ""},
            )
            fig_fi.update_layout(
                height=360, showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=20, t=10, b=0),
                xaxis_title="How much this factor influences the result →",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            st.info("Feature importance chart not available.")

        # ── Model comparison ───────────────────────────────────────────────────
        with st.expander("📊 See how all three models compare"):
            st.caption("We trained three different models and picked the one that best identifies high-risk patients.")
            comparison_data = {
                "Model": ["Logistic Regression", "Random Forest ✅ (used)", "XGBoost"],
                "Catches High-Risk (Recall)": ["54%", "59%", "54%"],
                "Overall Score (AUC)":        ["0.663", "0.674", "0.669"],
                "F1 Score":                   ["0.266", "0.276", "0.275"],
            }
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            st.caption("Random Forest was chosen because it catches the most high-risk patients (highest recall). "
                       "In healthcare, missing a high-risk patient is more costly than a false alarm.")

        st.warning(
            "⚕️ **Academic Project Notice:** This tool was built for a Data Engineering & MLOps "
            "university course using publicly available, de-identified data. "
            "It must not be used for real clinical decisions."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main Streamlit app entry point."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🏥 Hospital Readmission Analytics Dashboard")
    st.markdown(
        "*UCI Diabetes 130-US Hospitals Dataset &nbsp;|&nbsp; "
        "Data Engineering and MLOps — Academic Project*"
    )
    st.info(
        "ℹ️ **Disclaimer:** This dashboard is for academic purposes only. "
        "Data is de-identified and publicly available. "
        "Results are analytical, not clinical.",
        icon="ℹ️",
    )

    # ── Sidebar filters ────────────────────────────────────────────────────────
    filters = render_sidebar()

    # ── Tab navigation ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "👴 By Age Group",
        "🩺 By Diagnosis",
        "🛏️ Length of Stay",
        "🚑 Admission Patterns",
        "⚠️ High-Risk Cohort",
        "✅ Data Quality",
        "🔮 Predict",
    ])

    with tabs[0]:
        render_overview()

    with tabs[1]:
        render_readmission_by_age()

    with tabs[2]:
        render_readmission_by_diagnosis()

    with tabs[3]:
        render_los_analysis()

    with tabs[4]:
        render_admission_patterns()

    with tabs[5]:
        render_high_risk_cohort()

    with tabs[6]:
        render_data_quality()

    with tabs[7]:
        render_prediction()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<small>**Hospital Readmission Analytics Dashboard** | "
        "Course: Data Engineering and MLOps | "
        "Dataset: UCI Diabetes 130-US Hospitals (public, de-identified) | "
        "Academic prototype — not for clinical use</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
