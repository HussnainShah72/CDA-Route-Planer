# CDA Bus Route Process Mining Project

FAST National University, SE4009 Process Mining and Simulation.

Group members:

- Hussnain Haider (`i23-0695`)
- Uzair Majeed (`i233063`)
- Faez Ahmed (`i23-0598`)
- Hashir Nabeel (`i233047`)

## What This Project Implements

This repository implements the assignment requirements from `PMS_Project.pdf`:

- Scrape or parse CDA transit schedule PDFs into `data/routes.csv`
- Convert `routes.csv` into a valid XES event log with PM4Py
- Discover a process model with Inductive or Heuristic Miner
- Provide an interactive Streamlit GUI with route filtering
- Annotate transition durations and case frequencies
- Compute throughput statistics and bottlenecks
- Embed a grounded trip-planning AI assistant over `routes.csv`

The original `PMS_Project.pdf` is left unchanged.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the optional OpenAI/LangChain response generator, copy `.env.example` to `.env`
and set `OPENAI_API_KEY`.

## Run The Pipeline

Download and parse the CDA schedule PDFs.

**Group dataset (8 forward routes, `GROUP_DATASET_ROUTE_IDS` in `code/pms_project/config.py`):**

```powershell
python code/run_pipeline.py extract --group-dataset
```

**Arbitrary first N forward PDFs from the CDA page (legacy):**

```powershell
python code/run_pipeline.py extract --limit 8
```

**Explicit routes (comma-separated, forward PDFs only):**

```powershell
python code/run_pipeline.py extract --route-ids FR-01,FR-07,FR-15
```

If you already downloaded route PDFs, parse them from a folder:

```powershell
python code/run_pipeline.py extract --source-dir .\data\raw_pdfs --group-dataset
```

Create the XES log:

```powershell
python code/run_pipeline.py xes
```

Run everything:

```powershell
python code/run_pipeline.py all --group-dataset
```

Launch the GUI:

```powershell
streamlit run code/app.py
```

## Expected Files

- `data/routes.csv`: extracted route schedule rows with `route_id`, `stop_name`,
  `arrival_time`, and `departure_time`
- `data/cda_bus_routes.xes`: XES process mining event log
- `data/raw_pdfs/`: downloaded CDA route schedule PDFs

## Notes For The Report

Include screenshots of:

- `data/routes.csv`
- successful XES import in PM4Py, ProM, Apromore, or Celonis
- all-routes process map
- filtered process map for a route such as `FR-01`
- bottleneck threshold behavior
- top-3 slowest transitions
- the AI trip-planner answering grounded queries

