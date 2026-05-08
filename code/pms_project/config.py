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

# Part 6-style group dataset: 4 members → 8 forward routes (2 per member rule of
# thumb in the brief). PMS_Project.pdf is not in this repo—confirm exact Part 6
# wording with your PDF and adjust this list if required.
#
# QUERY (for your report / instructor; not email): Verify against PMS_Project.pdf
# that (a) eight forward schedules satisfy Part 6, and (b) substituting any route
# from the CDA table is allowed if rubric demands different coverage.
# Route names align with the official table at:
# https://www.cda.gov.pk/cdaTransitMap
#
# Coverage intent for this group (homes → Islamabad/Rawalpindi CDA network):
# - G-13/2, G-15/4: FR-01 (NUST Metro–Khanna Pul) + FR-07 (PIMS–Police Foundation)
# - DHA Phase 2: links via Khanna Pul / western corridor (FR-01, FR-05, FR-09, FR-10)
# - Sadiqabad (Rwp): FR-15 (Khanna Pul–T-Chowk) + connecting Khanna Pul routes
GROUP_DATASET_ROUTE_IDS: list[str] = [
    "FR-01",
    "FR-07",
    "FR-15",
    "FR-09",
    "FR-05",
    "FR-10",
    "FR-04",
    "FR-06",
]

REQUIRED_ROUTE_COLUMNS = [
    "route_id",
    "stop_name",
    "arrival_time",
    "departure_time",
]

