# Agentic AI Data Analyst

College-level project: **Artificial Intelligence application using Open-source tools**

## Title
**Agentic AI Data Analyst for Automated Data Analysis and Reporting**

## Architecture
User Problem → Prompt → Gemini LLM → LangGraph Agent Workflow → AI Output → Streamlit UI → Responsible AI

Workflow:

Dataset → Data Cleaning → Data Analysis → Visualization → Insight Generation → Report

## Agents
1. Data Cleaning Agent
2. Data Analysis Agent
3. Visualization Agent
4. Insight Generation Agent
5. Report Agent

## Features
- CSV, XLSX and XLS upload
- Missing-value handling
- Duplicate removal
- Column-name normalization
- Statistical profiling
- IQR outlier detection
- Automatic Plotly visualizations
- Gemini-powered insights
- Natural-language Ask Data
- HTML and PDF reports
- Responsible AI section
- Gemini fallback mode: deterministic analysis still works without the API

## Windows PowerShell Setup

Open the project folder in VS Code terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Never commit `.env` to GitHub.

Run:

```powershell
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Gemini
The project uses Google's official Python SDK:

```python
from google import genai
client = genai.Client(api_key=...)
```

The API key is loaded from `GEMINI_API_KEY`. It is never hard-coded.

If Gemini is unavailable or no key is configured, cleaning, statistics, outlier detection, charts, deterministic insights and reports continue to work.

## Project Structure

```text
Agentic_AI_Data_Analyst/
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

## Viva Explanation

The project uses LangGraph to orchestrate five stages. The deterministic agents perform data processing and calculations. Gemini receives compact calculated context instead of the entire raw dataset and produces natural-language interpretation. Streamlit provides the interface, while HTML/PDF report generation produces the final deliverables.

## Responsible AI

- Check fairness and representativeness of source data.
- Do not upload unnecessary confidential/PII data.
- Keep statistical calculations transparent.
- Treat LLM output as interpretation, not ground truth.
- Human users remain responsible for important decisions.
