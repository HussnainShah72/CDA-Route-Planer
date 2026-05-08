"""Download and parse CDA route schedule PDFs into routes.csv."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin
from urllib.request import urlopen

import pandas as pd
from pypdf import PdfReader

from pms_project.config import CDA_TRANSIT_URL, RAW_PDF_DIR, REQUIRED_ROUTE_COLUMNS


TIME_RE = re.compile(
    r"\b(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)"
    r"(?::(?P<second>[0-5]\d))?\s*(?P<ampm>[AaPp][Mm])?\b"
)
ROUTE_RE = re.compile(
    r"\b(?P<route>(?:FR|BR|OR|GR|BL|RD|YL)-?\d{1,2}[A-Z]?|"
    r"(?:Orange|Blue|Green|Red|Yellow)\s+Line)\b",
    re.IGNORECASE,
)
NOISE_RE = re.compile(
    r"route|arrival|departure|timing|schedule|stop\s*name|"
    r"capital|development|authority|forward|pass",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteEvent:
    """Single stop event extracted from a route schedule."""

    route_id: str
    stop_name: str
    arrival_time: str
    departure_time: str


def download_cda_route_pdfs(
    destination: Path = RAW_PDF_DIR,
    transit_url: str = CDA_TRANSIT_URL,
    limit: int | None = None,
    route_ids: list[str] | None = None,
) -> list[Path]:
    """Download route PDF links discovered on the CDA transit map page."""

    destination.mkdir(parents=True, exist_ok=True)
    html_text = fetch_text(transit_url)
    pdf_urls = discover_pdf_links(html_text, transit_url)

    if route_ids:
        unique_urls = select_forward_pdf_urls_for_route_ids(pdf_urls, route_ids)
    else:
        unique_urls = select_forward_pdf_urls(pdf_urls, limit)

    downloaded: list[Path] = []
    for index, url in enumerate(unique_urls, start=1):
        name = Path(url.split("?")[0]).name or f"route_{index}.pdf"
        path = destination / sanitize_filename(name)
        path.write_bytes(fetch_bytes(url))
        downloaded.append(path)

    if not downloaded:
        raise RuntimeError(
            f"No PDF links were found at {transit_url}. "
            "Download route PDFs manually and use --source-dir."
        )
    return downloaded


def select_forward_pdf_urls(pdf_urls: list[str], limit: int | None) -> list[str]:
    """Select forward-pass route PDFs and exclude maps/backward files."""

    unique_urls = list(dict.fromkeys(pdf_urls))
    route_urls = [
        url for url in unique_urls
        if not re.search(r"backward|reverse|return|transit[-_ ]?map", url, re.IGNORECASE)
    ]
    forward_urls = [
        url for url in route_urls
        if re.search(r"forward|outbound|up[-_ ]?route", url, re.IGNORECASE)
    ]
    selected = forward_urls or route_urls
    if limit is not None:
        selected = selected[:limit]
    return selected


def infer_route_id_from_pdf_url(url: str) -> str | None:
    """Best-effort route id from a CDA schedule PDF URL or filename."""

    name = unquote(Path(url.split("?")[0]).name)
    match = re.search(
        r"(?P<route>(?:FR|ST|FRB|FRG|BL|OR|GR|YL|RD)-?\d{1,2}[A-Z]?)",
        name,
        re.IGNORECASE,
    )
    if not match:
        return None
    return normalize_route_id(match.group("route"))


def select_forward_pdf_urls_for_route_ids(
    pdf_urls: list[str],
    route_ids: list[str],
) -> list[str]:
    """Pick forward-pass PDF URLs matching explicit route ids, in list order."""

    forward = [
        url
        for url in dict.fromkeys(pdf_urls)
        if re.search(r"forward", url, re.IGNORECASE)
        and not re.search(
            r"backward|reverse|return|transit[-_ ]?map", url, re.IGNORECASE
        )
    ]
    by_id: dict[str, str] = {}
    for url in forward:
        rid = infer_route_id_from_pdf_url(url)
        if rid and rid not in by_id:
            by_id[rid] = url

    ordered: list[str] = []
    missing: list[str] = []
    for raw_id in route_ids:
        rid = normalize_route_id(raw_id)
        url = by_id.get(rid)
        if url is None:
            missing.append(rid)
        else:
            ordered.append(url)

    if missing:
        raise RuntimeError(
            "Could not find forward PDF URLs for route(s): "
            f"{', '.join(missing)}. "
            f"Discovered forward route ids: {', '.join(sorted(by_id))}."
        )
    return ordered


def discover_pdf_links(html_text: str, base_url: str) -> list[str]:
    """Discover PDF links with BeautifulSoup when available, otherwise HTMLParser."""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        parser = PdfLinkParser(base_url)
        parser.feed(html_text)
        return parser.links

    soup = BeautifulSoup(html_text, "html.parser")
    pdf_urls: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if ".pdf" in href.lower():
            pdf_urls.append(urljoin(base_url, href))
    return pdf_urls


def fetch_text(url: str) -> str:
    """Fetch text using requests when installed, otherwise urllib."""

    try:
        import requests
    except ImportError:
        with urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def fetch_bytes(url: str) -> bytes:
    """Fetch bytes using requests when installed, otherwise urllib."""

    try:
        import requests
    except ImportError:
        with urlopen(url, timeout=60) as response:
            return response.read()

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


class PdfLinkParser(HTMLParser):
    """Small standard-library parser for PDF anchors."""

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Collect PDF href values from anchor tags."""

        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = (attrs_dict.get("href") or "").strip()
        if ".pdf" in href.lower():
            self.links.append(urljoin(self.base_url, href))


def parse_pdf_folder(source_dir: Path) -> pd.DataFrame:
    """Parse all PDFs in a folder and return the required routes dataframe."""

    pdfs = [
        path for path in sorted(source_dir.glob("*.pdf"))
        if is_forward_schedule_pdf(path)
    ]
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {source_dir}")

    return parse_pdf_files(pdfs)


def parse_pdf_files(pdfs: Iterable[Path]) -> pd.DataFrame:
    """Parse specific route schedule PDFs."""

    events: list[RouteEvent] = []
    for pdf_path in pdfs:
        events.extend(parse_route_pdf(pdf_path))

    return events_to_dataframe(events)


def is_forward_schedule_pdf(pdf_path: Path) -> bool:
    """Return true for likely forward schedule PDFs."""

    name = pdf_path.name
    if re.search(r"backward|reverse|return|transit[-_ ]?map", name, re.IGNORECASE):
        return False
    return True


def parse_route_pdf(pdf_path: Path) -> list[RouteEvent]:
    """Extract forward-pass stop rows from one route schedule PDF."""

    text = extract_pdf_text(pdf_path)
    route_id = infer_route_id(text, pdf_path)
    events: list[RouteEvent] = []

    for line in forward_pass_lines(text):
        event = parse_schedule_line(line, route_id)
        if event is not None:
            events.append(event)

    return dedupe_events(events)


def extract_pdf_text(pdf_path: Path) -> str:
    """Read text from a schedule PDF."""

    reader = PdfReader(str(pdf_path))
    page_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page_text)


def infer_route_id(text: str, pdf_path: Path) -> str:
    """Infer route id from PDF text first, then filename."""

    for source in (text[:1500], pdf_path.stem.replace("_", " ")):
        match = ROUTE_RE.search(source)
        if match:
            return normalize_route_id(match.group("route"))
    return pdf_path.stem.upper().replace(" ", "-")


def forward_pass_lines(text: str) -> list[str]:
    """Return lines belonging to the forward pass when the PDF marks sections."""

    raw_lines = [clean_spacing(line) for line in text.splitlines()]
    raw_lines = [line for line in raw_lines if line]

    forward_indices = [
        i for i, line in enumerate(raw_lines)
        if re.search(r"forward|outbound|up\s*route", line, re.IGNORECASE)
    ]
    if not forward_indices:
        return raw_lines

    start = forward_indices[0]
    end = len(raw_lines)
    for i in range(start + 1, len(raw_lines)):
        if re.search(r"reverse|backward|return|down\s*route", raw_lines[i], re.IGNORECASE):
            end = i
            break
    return raw_lines[start:end]


def parse_schedule_line(line: str, route_id: str) -> RouteEvent | None:
    """Parse one likely stop row from PDF-extracted text."""

    line = split_embedded_times(line)
    times = list(TIME_RE.finditer(line))
    if not times:
        return None

    stop_name = TIME_RE.sub(" ", line)
    stop_name = re.sub(r"^\s*\d+[\).:-]?\s*", "", stop_name)
    stop_name = clean_spacing(stop_name.strip(" -|:\t"))

    if not stop_name or len(stop_name) < 2:
        return None
    if NOISE_RE.fullmatch(stop_name):
        return None

    arrival = normalize_time_text(times[0].group(0))
    departure = normalize_time_text(times[1].group(0) if len(times) > 1 else times[0].group(0))
    repaired = repair_split_hour_suffix(stop_name, times, departure)
    if repaired is not None:
        stop_name, arrival = repaired

    return RouteEvent(
        route_id=route_id,
        stop_name=normalize_stop_name(stop_name),
        arrival_time=arrival,
        departure_time=departure,
    )


def split_embedded_times(line: str) -> str:
    """Separate times that PDF extraction glued to stop names."""

    return re.sub(
        r"([A-Za-z)])(\d{1,2}:\d{2}(?::\d{2})?)\b",
        r"\1 \2",
        line,
    )


def repair_split_hour_suffix(
    stop_name: str,
    times: list[re.Match[str]],
    departure: str,
) -> tuple[str, str] | None:
    """Repair rows extracted as StopNameHH MM:SS HH:MM:SS."""

    suffix_match = re.match(r"^(?P<stop>.+?)(?P<hour>[01]\d|2[0-3])$", stop_name)
    if not suffix_match or len(times) < 2:
        return None

    suffix_hour = int(suffix_match.group("hour"))
    if not departure.startswith(f"{suffix_hour:02d}:"):
        return None

    first_token = times[0].group(0).strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}", first_token):
        return None

    minute_text, second_text = first_token.split(":")
    minute = int(minute_text)
    second = int(second_text)
    if minute > 59 or second > 59:
        return None

    arrival = f"{suffix_hour:02d}:{minute:02d}:{second:02d}"
    return clean_spacing(suffix_match.group("stop")), arrival


def events_to_dataframe(events: Iterable[RouteEvent]) -> pd.DataFrame:
    """Create a clean routes dataframe with the assignment-required columns."""

    rows = [event.__dict__ for event in events]
    frame = pd.DataFrame(rows, columns=REQUIRED_ROUTE_COLUMNS)
    if frame.empty:
        return frame
    frame = frame.dropna().drop_duplicates()
    frame["route_id"] = frame["route_id"].map(normalize_route_id)
    frame["stop_name"] = frame["stop_name"].map(normalize_stop_name)
    return frame[REQUIRED_ROUTE_COLUMNS]


def write_routes_csv(frame: pd.DataFrame, output_path: Path) -> Path:
    """Write routes.csv using the exact required schema."""

    missing = set(REQUIRED_ROUTE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"routes dataframe is missing columns: {sorted(missing)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[REQUIRED_ROUTE_COLUMNS].to_csv(output_path, index=False)
    return output_path


def build_routes_csv(
    output_path: Path,
    source_dir: Path | None = None,
    limit: int | None = None,
    route_ids: list[str] | None = None,
) -> Path:
    """Build routes.csv from local PDFs or freshly downloaded CDA PDFs."""

    pdf_dir = source_dir
    if pdf_dir is None:
        pdfs = download_cda_route_pdfs(
            RAW_PDF_DIR,
            limit=limit,
            route_ids=route_ids,
        )
        frame = parse_pdf_files(pdfs)
    else:
        if route_ids:
            all_pdfs = sorted(source_dir.glob("*.pdf"))
            selected: list[Path] = []
            missing: list[str] = []
            by_id: dict[str, Path] = {}
            for path in all_pdfs:
                if not is_forward_schedule_pdf(path):
                    continue
                rid = infer_route_id_from_pdf_url(path.name)
                if rid and rid not in by_id:
                    by_id[rid] = path
            for raw_id in route_ids:
                rid = normalize_route_id(raw_id)
                p = by_id.get(rid)
                if p is None:
                    missing.append(rid)
                else:
                    selected.append(p)
            if missing:
                raise FileNotFoundError(
                    f"No forward PDF in {source_dir} for: {', '.join(missing)}. "
                    f"Found: {sorted(by_id)}"
                )
            frame = parse_pdf_files(selected)
        else:
            frame = parse_pdf_folder(pdf_dir)

    if frame.empty:
        raise RuntimeError(
            "No route rows were extracted. Check the PDF layout or add a "
            "custom parser for the downloaded schedule format."
        )
    return write_routes_csv(frame, output_path)


def dedupe_events(events: list[RouteEvent]) -> list[RouteEvent]:
    """Preserve row order while removing exact duplicates."""

    seen: set[tuple[str, str, str, str]] = set()
    unique: list[RouteEvent] = []
    for event in events:
        key = (
            event.route_id,
            event.stop_name.casefold(),
            event.arrival_time,
            event.departure_time,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


def normalize_route_id(value: str) -> str:
    """Normalize route labels without losing named CDA lines."""

    cleaned = clean_spacing(value).replace(" ", "-")
    cleaned = re.sub(r"(?i)[_-]?(forward|backward|reverse|return)$", "", cleaned)
    cleaned = re.sub(
        r"(?i)^fr-?(\d+)([a-z]?)$",
        lambda match: f"FR-{int(match.group(1)):02d}{match.group(2).upper()}",
        cleaned,
    )
    return cleaned


def normalize_stop_name(value: str) -> str:
    """Normalize stop names into title-style labels."""

    value = re.sub(r"\s+", " ", value).strip(" -|")
    return value.title().replace("Nust", "NUST").replace("Fast", "FAST")


def normalize_time_text(value: str) -> str:
    """Return HH:MM:SS text from a schedule time token."""

    match = TIME_RE.search(value)
    if not match:
        raise ValueError(f"Invalid time value: {value}")

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second") or 0)
    ampm = match.group("ampm")
    if ampm:
        marker = ampm.lower()
        if marker == "pm" and hour != 12:
            hour += 12
        if marker == "am" and hour == 12:
            hour = 0
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def clean_spacing(value: str) -> str:
    """Collapse whitespace and common PDF table separators."""

    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def sanitize_filename(value: str) -> str:
    """Make a URL filename safe on Windows and Unix."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "route.pdf"
