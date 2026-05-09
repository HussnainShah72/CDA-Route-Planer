# CDA Bus Route Process Mining Project 🚌
**FAST National University | SE4009 Process Mining and Simulation**

A comprehensive process mining and transit analysis tool for the Islamabad CDA Bus network. This project implements a full data pipeline from raw PDF schedules to an interactive, AI-powered analytical dashboard.

## 👥 Group Members
- Hussnain Haider (`23i-0695`)
- Uzair Majeed (`23i-3063`)
- Faez Ahmed (`23i-0598`)
- Hashir Nabeel (`23i-3047`)

---

## 🚀 Key Features
- **Data Extraction:** Automated parsing of official CDA transit PDFs into structured event data.
- **XES Generation:** IEEE Standard XES log construction with full trace mapping.
- **Interactive Dashboard:** Premium Charcoal & Gold themed UI built with Streamlit.
- **Process Discovery:** Dynamic Graphviz-based mapping with Heuristic and Inductive miners.
- **Performance Analytics:** Real-time throughput calculations and automatic bottleneck highlighting.
- **Agentic AI Trip Planner:** Data-grounded RAG assistant (LangChain + Groq/OpenAI) for natural language routing.
- **Personal Route Maps:** Custom commute visualization for all group members.

---

## 🛠️ Setup & Installation

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and provide your API keys for the AI Assistant:
- `GROQ_API_KEY` (Recommended) or `OPENAI_API_KEY`.

---

## 📁 Project Structure
- **`code/`**: Source code repository.
  - `app.py`: Main Streamlit GUI application.
  - `run_pipeline.py`: CLI tool for data processing.
  - `pms_project/`: Core analytical and AI logic modules.
- **`data/`**: Processed datasets and validation logs.
  - `routes.csv`: The structured master schedule.
  - `cda_bus_routes.xes`: The generated event log.
  - `verify_xes.py`: XES validation and screenshot generator.
- **`report/`**: Final academic documentation and screenshots.

---

## 🖥️ Usage

### Launch the Dashboard
```bash
streamlit run code/app.py
```

### Re-run Data Pipeline (Optional)
To re-process the routes and generate a fresh XES log:
```bash
# Re-extract from local PDFs and build log
python3 code/run_pipeline.py all --source-dir data/raw_pdfs --group-dataset

# Or generate only the XES log from existing CSV
python3 code/run_pipeline.py xes
```

### XES Validation
To generate academic process trees and networks for the report:
```bash
python3 data/verify_xes.py
```

---

## 📝 Submission Checklist
Ensure the following are included in the final `.zip`:
1. `report/report.pdf` (Compiled from `report.tex`).
2. `code/` folder with all Python scripts.
3. `data/` folder with `routes.csv` and `cda_bus_routes.xes`.
