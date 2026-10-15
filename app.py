import io
import os
import json
from typing import Any, Dict, TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, END
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Agentic AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):
    df: pd.DataFrame
    original_rows: int
    original_cols: int
    cleaning: Dict[str, Any]
    analysis: Dict[str, Any]
    charts: list
    insights: Dict[str, Any]
    html_report: str
    pdf_report: bytes
    error: str


# ============================================================
# DATA CLEANING AGENT
# ============================================================

def clean_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:

    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    before_rows, before_cols = df.shape

    work = df.copy()

    # Normalize column names
    work.columns = [
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for c in work.columns
    ]

    # Remove empty columns
    empty_cols = [
        c for c in work.columns
        if str(c).strip() in ("", "nan")
    ]

    if empty_cols:
        work = work.drop(columns=empty_cols)

    # Remove duplicates
    duplicate_rows = int(work.duplicated().sum())

    work = (
        work
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Missing values before cleaning
    missing_before = work.isna().sum().to_dict()

    # Detect numeric and categorical columns
    numeric_cols = (
        work
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_cols = (
        work
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )

    # Numeric missing values → median
    for c in numeric_cols:

        if work[c].isna().any():

            median = work[c].median()

            if pd.isna(median):
                work[c] = work[c].fillna(0)
            else:
                work[c] = work[c].fillna(median)

    # Categorical missing values → mode
    for c in categorical_cols:

        if work[c].isna().any():

            mode = work[c].mode(dropna=True)

            if not mode.empty:
                work[c] = work[c].fillna(mode.iloc[0])
            else:
                work[c] = work[c].fillna("Unknown")

    # Missing values after cleaning
    missing_after = work.isna().sum().to_dict()

    info = {
        "before_rows": before_rows,
        "before_cols": before_cols,
        "after_rows": int(work.shape[0]),
        "after_cols": int(work.shape[1]),
        "duplicates_removed": duplicate_rows,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    return work, info


# ============================================================
# DATA ANALYSIS AGENT
# ============================================================

def analyze_dataframe(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    numeric = df.select_dtypes(
        include=np.number
    )

    categorical = df.select_dtypes(
        exclude=np.number
    )

    stats = {}

    for c in numeric.columns:

        s = numeric[c].dropna()

        if not s.empty:

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)

        else:

            q1 = np.nan
            q3 = np.nan

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if not s.empty:

            outliers = int(
                (
                    (s < lower) |
                    (s > upper)
                ).sum()
            )

        else:

            outliers = 0

        stats[c] = {

            "count": int(s.count()),

            "mean":
                float(s.mean())
                if not s.empty
                else None,

            "median":
                float(s.median())
                if not s.empty
                else None,

            "std":
                float(s.std())
                if len(s) > 1
                else 0.0,

            "min":
                float(s.min())
                if not s.empty
                else None,

            "max":
                float(s.max())
                if not s.empty
                else None,

            "q1":
                float(q1)
                if not pd.isna(q1)
                else None,

            "q3":
                float(q3)
                if not pd.isna(q3)
                else None,

            "outliers_iqr":
                outliers,
        }

    # Categorical frequency analysis
    frequencies = {}

    for c in categorical.columns[:15]:

        vc = (
            df[c]
            .astype(str)
            .value_counts()
            .head(10)
        )

        frequencies[c] = {
            str(k): int(v)
            for k, v in vc.items()
        }

    # Correlation
    if numeric.shape[1] >= 2:

        corr = (
            numeric
            .corr(numeric_only=True)
            .round(4)
            .to_dict()
        )

    else:

        corr = {}

    return {

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "dtypes":
            {
                c: str(t)
                for c, t in df.dtypes.items()
            },

        "numeric_stats":
            stats,

        "categorical_frequencies":
            frequencies,

        "correlation":
            corr,
    }


# ============================================================
# VISUALIZATION AGENT
# ============================================================

def create_charts(df: pd.DataFrame):

    charts = []

    numeric = df.select_dtypes(
        include=np.number
    )

    categorical = df.select_dtypes(
        exclude=np.number
    )

    # Numeric charts
    for c in numeric.columns[:4]:

        histogram = px.histogram(
            df,
            x=c,
            title=f"Distribution of {c}",
        )

        charts.append(histogram)

        boxplot = px.box(
            df,
            y=c,
            title=f"Box Plot of {c}",
        )

        charts.append(boxplot)

    # Categorical charts
    for c in categorical.columns[:3]:

        if df[c].nunique(
            dropna=False
        ) <= 15:

            counts = (
                df[c]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                c,
                "count"
            ]

            charts.append(
                px.bar(
                    counts,
                    x=c,
                    y="count",
                    title=f"Category Distribution: {c}",
                )
            )

    # Correlation heatmap
    if numeric.shape[1] >= 2:

        corr = numeric.corr()

        charts.append(
            px.imshow(
                corr,
                text_auto=True,
                title="Correlation Heatmap",
                aspect="auto",
            )
        )

    return charts


# ============================================================
# FALLBACK INSIGHTS
# ============================================================

def deterministic_insights(
    analysis
):

    findings = []

    for c, s in analysis.get(
        "numeric_stats",
        {}
    ).items():

        if s["mean"] is not None:

            findings.append(
                f"{c}: "
                f"mean={s['mean']:.2f}, "
                f"median={s['median']:.2f}, "
                f"range={s['min']:.2f} "
                f"to {s['max']:.2f}, "
                f"IQR outliers="
                f"{s['outliers_iqr']}."
            )

    return {

        "executive_summary":
            f"The cleaned dataset contains "
            f"{analysis['rows']} rows and "
            f"{analysis['columns']} columns.",

        "key_findings":
            findings
            or [
                "No numeric columns were available "
                "for numeric summaries."
            ],

        "recommendations": [

            "Review data quality before using "
            "results for important decisions.",

            "Investigate detected outliers "
            "in their original business context.",
        ],

        "source":
            "Deterministic fallback "
            "(Gemini unavailable).",
    }


# ============================================================
# GEMINI CLIENT
# ============================================================

def gemini_client():

    key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not key:
        return None

    return genai.Client(
        api_key=key
    )


# ============================================================
# GEMINI API CALL
# ============================================================

def call_gemini(
    prompt: str
) -> str:

    client = gemini_client()

    if not client:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    response = client.models.generate_content(

        model=DEFAULT_MODEL,

        contents=prompt,
    )

    text = getattr(
        response,
        "text",
        ""
    )

    if not text:

        return "Gemini returned no text."

    return text


# ============================================================
# INSIGHT GENERATION AGENT
# ============================================================

def insight_agent(
    state: AgentState
) -> AgentState:

    fallback = deterministic_insights(
        state["analysis"]
    )

    try:

        compact = json.dumps(
            state["analysis"],
            default=str
        )[:30000]

        prompt = f"""
You are an analytical assistant.

Use ONLY the supplied calculated dataset context.

Do not invent or alter statistics.

Clearly distinguish:

1. Observed facts
2. AI interpretation
3. Recommendations

Dataset context:

{compact}

Return concise sections:

Executive Summary
Key Findings
Patterns/Anomalies
Data Quality Observations
Business Recommendations
"""

        text = call_gemini(
            prompt
        )

        state["insights"] = {

            "executive_summary":
                text,

            "key_findings": [
                "Generated from the supplied "
                "calculated analysis context."
            ],

            "recommendations": [
                "Review AI recommendations "
                "with human judgment."
            ],

            "source":
                f"Gemini ({DEFAULT_MODEL})",
        }

    except Exception as exc:

        fallback["error"] = str(exc)

        state["insights"] = fallback

    return state


# ============================================================
# AGENT FUNCTIONS
# ============================================================

def cleaning_agent(
    state: AgentState
):

    cleaned, info = clean_dataframe(
        state["df"]
    )

    state["df"] = cleaned
    state["cleaning"] = info

    return state


def analysis_agent(
    state: AgentState
):

    state["analysis"] = (
        analyze_dataframe(
            state["df"]
        )
    )

    return state


def visualization_agent(
    state: AgentState
):

    try:

        state["charts"] = (
            create_charts(
                state["df"]
            )
        )

    except Exception:

        state["charts"] = []

    return state


# ============================================================
# HTML REPORT
# ============================================================

def build_html(
    state: AgentState
) -> str:

    a = state["analysis"]
    c = state["cleaning"]
    i = state["insights"]

    rows = ""

    for k, v in a[
        "numeric_stats"
    ].items():

        rows += (
            f"<tr>"
            f"<td>{k}</td>"
            f"<td>{v}</td>"
            f"</tr>"
        )

    recommendations = (
        "</li><li>"
        .join(
            i.get(
                "recommendations",
                []
            )
        )
    )

    return f"""
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<title>
Agentic AI Data Analysis Report
</title>

<style>

body {{
    font-family: Arial;
    margin: 40px;
    line-height: 1.5;
}}

h1, h2 {{
    color: #222;
}}

table {{
    border-collapse: collapse;
    width: 100%;
}}

td, th {{
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
}}

</style>

</head>

<body>

<h1>
Agentic AI Data Analysis Report
</h1>

<h2>
1. Dataset Overview
</h2>

<p>
Rows: {a['rows']}
|
Columns: {a['columns']}
</p>

<h2>
2. Data Cleaning
</h2>

<p>
Duplicates removed:
{c['duplicates_removed']}
</p>

<p>
Rows before:
{c['before_rows']}
</p>

<p>
Rows after:
{c['after_rows']}
</p>

<h2>
3. Statistical Analysis
</h2>

<table>

<tr>

<th>
Column
</th>

<th>
Statistics
</th>

</tr>

{rows}

</table>

<h2>
4. Outlier Detection
</h2>

<p>
IQR-based outlier counts are included
in the statistical table.
</p>

<h2>
5. Visualization Summary
</h2>

<p>
{len(state.get('charts', []))}
visualizations generated automatically.
</p>

<h2>
6. AI Insights
</h2>

<p>
{i.get('executive_summary', '')}
</p>

<h2>
7. Business Recommendations
</h2>

<ul>

<li>
{recommendations}
</li>

</ul>

<h2>
8. Responsible AI
</h2>

<p>
Results can reflect source-data bias.
Protect privacy, review AI interpretations,
and keep final decisions under human oversight.
</p>

</body>

</html>
"""


# ============================================================
# PDF REPORT
# ============================================================

def build_pdf(
    state: AgentState
) -> bytes:

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    story = [

        Paragraph(
            "Agentic AI Data Analysis Report",
            styles["Title"]
        ),

        Spacer(1, 12),
    ]

    a = state["analysis"]
    c = state["cleaning"]
    i = state["insights"]

    story += [

        Paragraph(
            "1. Dataset Overview",
            styles["Heading2"]
        ),

        Paragraph(
            f"Rows: {a['rows']} | "
            f"Columns: {a['columns']}",
            styles["BodyText"]
        ),

        Paragraph(
            "2. Data Cleaning",
            styles["Heading2"]
        ),

        Paragraph(
            f"Duplicates removed: "
            f"{c['duplicates_removed']}; "
            f"rows before/after: "
            f"{c['before_rows']}/"
            f"{c['after_rows']}",
            styles["BodyText"]
        ),

        Paragraph(
            "3. Statistical Analysis & Outliers",
            styles["Heading2"]
        ),
    ]

    data = [
        [
            "Column",
            "Mean",
            "Median",
            "Min",
            "Max",
            "Outliers",
        ]
    ]

    for k, s in a[
        "numeric_stats"
    ].items():

        data.append(
            [
                k,
                s["mean"],
                s["median"],
                s["min"],
                s["max"],
                s["outliers_iqr"],
            ]
        )

    if len(data) == 1:

        data.append(
            [
                "No numeric columns",
                "",
                "",
                "",
                "",
                "",
            ]
        )

    table = Table(
        data,
        repeatRows=1
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
            ]
        )
    )

    story += [

        table,

        Spacer(1, 12),

        Paragraph(
            "4. AI Insights",
            styles["Heading2"]
        ),

        Paragraph(
            str(
                i.get(
                    "executive_summary",
                    ""
                )
            ),
            styles["BodyText"]
        ),

        Paragraph(
            "5. Responsible AI",
            styles["Heading2"]
        ),

        Paragraph(
            "Use human oversight. "
            "Avoid unnecessary PII. "
            "Review fairness, transparency, "
            "privacy and safety before "
            "high-impact decisions.",
            styles["BodyText"]
        ),
    ]

    doc.build(story)

    return buffer.getvalue()


# ============================================================
# REPORT AGENT
# ============================================================

def report_agent(
    state: AgentState
):

    state["html_report"] = (
        build_html(state)
    )

    state["pdf_report"] = (
        build_pdf(state)
    )

    return state


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "cleaning",
        cleaning_agent
    )

    graph.add_node(
        "analysis",
        analysis_agent
    )

    graph.add_node(
        "visualization",
        visualization_agent
    )

    graph.add_node(
        "insights",
        insight_agent
    )

    graph.add_node(
        "report",
        report_agent
    )

    graph.set_entry_point(
        "cleaning"
    )

    graph.add_edge(
        "cleaning",
        "analysis"
    )

    graph.add_edge(
        "analysis",
        "visualization"
    )

    graph.add_edge(
        "visualization",
        "insights"
    )

    graph.add_edge(
        "insights",
        "report"
    )

    graph.add_edge(
        "report",
        END
    )

    return graph.compile()


# ============================================================
# FILE READER
# ============================================================

def read_uploaded(file):

    name = file.name.lower()

    if name.endswith(".csv"):

        return pd.read_csv(file)

    if name.endswith(
        (".xlsx", ".xls")
    ):

        return pd.read_excel(file)

    raise ValueError(
        "Unsupported file format. "
        "Upload CSV, XLSX or XLS."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.title(
        "📊 Agentic AI Data Analyst"
    )

    st.caption(
        "Automated Data Analysis & Reporting "
        "using Gemini, LangGraph and Streamlit"
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header(
            "Project Control"
        )

        key_present = bool(
            os.getenv(
                "GEMINI_API_KEY"
            )
        )

        if key_present:

            st.write(
                "API connection:",
                "🟢 Configured"
            )

        else:

            st.write(
                "API connection:",
                "🟡 Fallback mode"
            )

        st.write(
            "Gemini model:",
            DEFAULT_MODEL
        )

        st.markdown(
            "**Workflow**"
        )

        st.write(
            "Dataset → Cleaning → "
            "Analysis → Visualization → "
            "Insights → Report"
        )

        st.info(
            "College project: AI-assisted "
            "analytics with responsible "
            "human oversight."
        )

        # ====================================================
        # GEMINI CONNECTION TEST
        # ====================================================

        if st.button(
            "🔎 Test Gemini Connection"
        ):

            try:

                test_reply = call_gemini(
                    "Reply with exactly: "
                    "Gemini connection OK"
                )

                st.success(
                    "Gemini connection successful."
                )

                st.caption(
                    test_reply
                )

            except Exception as exc:

                st.error(
                    "Gemini connection failed."
                )

                # ACTUAL ERROR
                st.code(
                    str(exc)
                )

                st.warning(
                    "Check GEMINI_API_KEY, "
                    "GEMINI_MODEL, API permissions/"
                    "quota, and internet access."
                )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    uploaded = st.file_uploader(
        "Upload dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
    )

    if not uploaded:

        st.info(
            "Upload a CSV or Excel dataset "
            "to begin."
        )

        return

    # ========================================================
    # READ DATASET
    # ========================================================

    try:

        df = read_uploaded(
            uploaded
        )

        if df.empty:

            st.error(
                "The dataset is empty."
            )

            return

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            len(df)
        )

        c2.metric(
            "Columns",
            len(df.columns)
        )

        c3.metric(
            "Dataset",
            uploaded.name
        )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

    except Exception as exc:

        st.error(
            f"Could not read dataset: {exc}"
        )

        return

    # ========================================================
    # RUN AGENTIC ANALYSIS
    # ========================================================

    if st.button(
        "🚀 Run Agentic Analysis",
        type="primary"
    ):

        try:

            with st.status(
                "Running LangGraph agent workflow...",
                expanded=True
            ) as status:

                status.write(
                    "1/5 Data Cleaning Agent"
                )

                status.write(
                    "2/5 Data Analysis Agent"
                )

                status.write(
                    "3/5 Visualization Agent"
                )

                status.write(
                    "4/5 Insight Generation Agent"
                )

                status.write(
                    "5/5 Report Agent"
                )

                result = (
                    build_graph()
                    .invoke(
                        {
                            "df": df,
                            "original_rows":
                                len(df),
                            "original_cols":
                                len(df.columns),
                        }
                    )
                )

                status.update(
                    label="Analysis complete",
                    state="complete"
                )

            st.session_state[
                "result"
            ] = result

        except Exception as exc:

            st.error(
                f"Workflow error: {exc}"
            )

            return

    # ========================================================
    # GET RESULT
    # ========================================================

    result = st.session_state.get(
        "result"
    )

    if not result:

        return

    a = result["analysis"]

    c = result["cleaning"]

    i = result["insights"]

    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(
        [
            "Cleaning",
            "Analysis",
            "Visualizations",
            "AI Insights",
            "Ask Data",
            "Reports",
            "Responsible AI",
        ]
    )

    # ========================================================
    # CLEANING TAB
    # ========================================================

    with tabs[0]:

        st.subheader(
            "Cleaning Summary"
        )

        st.json(
            c
        )

        st.dataframe(
            result["df"].head(50),
            use_container_width=True
        )

    # ========================================================
    # ANALYSIS TAB
    # ========================================================

    with tabs[1]:

        st.metric(
            "Rows after cleaning",
            a["rows"]
        )

        st.json(
            a["numeric_stats"]
        )

        if a["correlation"]:

            st.dataframe(
                pd.DataFrame(
                    a["correlation"]
                )
            )

    # ========================================================
    # VISUALIZATION TAB
    # ========================================================

    with tabs[2]:

        for fig in result.get(
            "charts",
            []
        ):

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ========================================================
    # AI INSIGHTS TAB
    # ========================================================

    with tabs[3]:

        st.subheader(
            "AI Insights"
        )

        st.write(
            i.get(
                "executive_summary",
                ""
            )
        )

        if i.get("error"):

            st.warning(
                "Gemini was unavailable; "
                "deterministic fallback was used."
            )

            with st.expander(
                "Show Gemini error details"
            ):

                st.code(
                    str(
                        i.get(
                            "error"
                        )
                    )
                )

        st.write(
            "**Source:**",
            i.get(
                "source"
            )
        )

    # ========================================================
    # ASK DATA TAB
    # ========================================================

    with tabs[4]:

        question = st.text_input(
            "Ask a question about the dataset"
        )

        if st.button(
            "Ask Gemini / Fallback",
            key="ask"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                context = json.dumps(
                    a,
                    default=str
                )[:30000]

                try:

                    answer = call_gemini(

                        f"""
Answer the user's question using ONLY
this calculated dataset context.

Question:
{question}

Context:
{context}

Do not invent statistics.
"""
                    )

                    st.success(
                        "Gemini response generated."
                    )

                    st.write(
                        answer
                    )

                except Exception as exc:

                    error_text = str(
                        exc
                    )

                    # ========================================
                    # ACTUAL GEMINI ERROR
                    # ========================================

                    st.error(
                        "Gemini request failed."
                    )

                    st.code(
                        error_text
                    )

                    st.warning(
                        "The dataset analysis is still "
                        "available. Check GEMINI_API_KEY, "
                        "GEMINI_MODEL, API permissions/"
                        "quota, and internet access."
                    )

                    st.info(
                        "Fallback mode is active. "
                        "No raw dataset was sent to Gemini."
                    )

                    with st.expander(
                        "Show calculated dataset context"
                    ):

                        st.json(
                            a
                        )

    # ========================================================
    # REPORTS TAB
    # ========================================================

    with tabs[5]:

        st.download_button(
            "Download HTML Report",
            result["html_report"],
            "agentic_ai_report.html",
            "text/html"
        )

        st.download_button(
            "Download PDF Report",
            result["pdf_report"],
            "agentic_ai_report.pdf",
            "application/pdf"
        )

    # ========================================================
    # RESPONSIBLE AI TAB
    # ========================================================

    with tabs[6]:

        st.subheader(
            "Responsible AI"
        )

        st.markdown(
            """
**Fairness:**  
Biased or unrepresentative source data can
produce biased insights.

**Transparency:**  
The workflow exposes cleaning and statistical
calculations rather than hiding everything
behind the LLM.

**Privacy:**  
Avoid confidential or unnecessary personally
identifiable information. The raw dataset is
not intentionally sent to Gemini.

**Safety:**  
AI-generated recommendations should be
reviewed before important or high-impact
decisions.

**Human Oversight:**  
Final decisions remain with the human user.
"""
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()