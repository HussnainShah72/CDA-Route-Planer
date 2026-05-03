"""Project paths and shared constants."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
ROUTES_CSV = DATA_DIR / "routes.csv"
XES_LOG = DATA_DIR / "cda_bus_routes.xes"

CDA_TRANSIT_URL = "https://cda.gov.pk/cdaTransitMap"
DEFAULT_TIMEZONE = "Asia/Karachi"
DEFAULT_SERVICE_DATE = "2026-04-23"

REQUIRED_ROUTE_COLUMNS = [
    "route_id",
    "stop_name",
    "arrival_time",
    "departure_time",
]

