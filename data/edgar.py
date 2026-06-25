"""SEC EDGAR 13F-HR client.

Pulls institutional manager (13F filer) holdings from SEC EDGAR's public JSON
+ XML endpoints. No API key needed; SEC asks that all programmatic requests
carry an honest ``User-Agent`` header (``"<App> <contact>"``) and that callers
respect the 10 req/s rate limit.

We target the three pieces needed to reverse-engineer a picker (Task 3):

  1. ``list_13f_filings(cik)``     — every 13F-HR submission for a filer.
  2. ``fetch_13f_holdings(...)``   — parse the primary INFORMATION TABLE XML
                                     into a list of ``Holding`` rows.
  3. ``latest_13f_holdings(cik)``  — convenience: most recent filing's
                                     holdings + filing/report metadata.

Per the BRIEF: low-volume polite use only; on rate-limit (HTTP 429 / 403)
we retry with exponential backoff up to ``max_retries``. Anything else
bubbles up as an exception with the EDGAR URL attached so the caller can
reproduce the failure manually.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

log = logging.getLogger(__name__)

# Public EDGAR endpoints (no auth). The data subdomain hosts JSON; the www
# subdomain hosts the raw Archives the JSON points at.
_BASE_DATA = "https://data.sec.gov"
_BASE_WWW = "https://www.sec.gov"

# SEC's published etiquette: identify yourself, respect 10 req/s.
# An override via env keeps the contact configurable without code edits.
_DEFAULT_UA = os.environ.get(
    "EDGAR_USER_AGENT",
    "algo-trading (research; shawntwj12@gmail.com)",
)

# EDGAR's 13F INFORMATION TABLE schema; namespace varies by version.
_NS_PREFIXES = (
    "{http://www.sec.gov/edgar/document/thirteenf/informationtable}",
    "{http://www.sec.gov/edgar/thirteenffiler}",
    "",  # some filings have no namespace at all (older formats).
)


@dataclass(frozen=True)
class Filing:
    """One 13F filing (header-level metadata)."""

    accession: str        # 0001067983-25-000XYZ (with dashes)
    accession_nodash: str # 000106798325000XYZ
    filing_date: str      # YYYY-MM-DD the filing was lodged with the SEC
    report_date: str      # YYYY-MM-DD the report covers (quarter-end)
    form: str             # "13F-HR" or "13F-HR/A"
    primary_document: str # e.g. "primary_doc.xml"


@dataclass(frozen=True)
class Holding:
    """One row of an INFORMATION TABLE."""

    name_of_issuer: str
    cusip: str
    value_usd: float       # SEC reports in thousands; we convert to raw USD.
    shares: float
    share_type: str        # "SH" / "PRN"
    put_call: str | None   # "Put" / "Call" / None for direct stock
    investment_discretion: str | None


# ─── HTTP layer ────────────────────────────────────────────────────────────
def _get(url: str, *, user_agent: str = _DEFAULT_UA,
         max_retries: int = 5, backoff: float = 1.5) -> bytes:
    """Polite GET. Raises after ``max_retries`` consecutive 429/403 responses."""
    req = urlrequest.Request(
        url,
        headers={
            "User-Agent": user_agent,
            # SEC asks specifically for a contact (in UA) + `Accept-Encoding`.
            "Accept-Encoding": "gzip, deflate",
            "Host": url.split("/", 3)[2],
        },
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urlrequest.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                encoding = (resp.headers.get("Content-Encoding") or "").lower()
                if encoding == "gzip":
                    return gzip.decompress(raw)
                if encoding == "deflate":
                    try:
                        return zlib.decompress(raw)
                    except zlib.error:
                        return zlib.decompress(raw, -zlib.MAX_WBITS)
                return raw
        except urlerror.HTTPError as exc:
            last_err = exc
            if exc.code in (429, 403):
                # Rate-limited. SEC's window resets ~1s; back off exponentially.
                wait = backoff ** attempt
                log.warning("EDGAR %s on %s; sleeping %.2fs", exc.code, url, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"EDGAR GET {url} failed: HTTP {exc.code}") from exc
        except urlerror.URLError as exc:
            last_err = exc
            wait = backoff ** attempt
            time.sleep(wait)
    raise RuntimeError(f"EDGAR GET {url} failed after {max_retries} retries: {last_err}")


def _pad_cik(cik: int | str) -> str:
    """EDGAR stores CIKs as zero-padded 10-digit strings."""
    return str(int(cik)).zfill(10)


# ─── Submissions JSON → 13F filings list ───────────────────────────────────
def list_13f_filings(cik: int | str, *, user_agent: str = _DEFAULT_UA) -> list[Filing]:
    """All 13F-HR (and amendments) for ``cik``, newest first.

    Uses the submissions JSON (the same endpoint EDGAR Search uses); the
    `recent` block carries the last ~1000 filings inline. Older filings spill
    into paginated `files` shards — we follow each shard so a 15+ year picker
    history still returns every 13F.
    """
    cik_padded = _pad_cik(cik)
    url = f"{_BASE_DATA}/submissions/CIK{cik_padded}.json"
    payload = json.loads(_get(url, user_agent=user_agent))

    filings: list[Filing] = []
    filings.extend(_parse_filings_block(payload.get("filings", {}).get("recent", {})))
    for shard in payload.get("filings", {}).get("files", []):
        shard_url = f"{_BASE_DATA}/submissions/{shard['name']}"
        shard_payload = json.loads(_get(shard_url, user_agent=user_agent))
        filings.extend(_parse_filings_block(shard_payload))

    # Newest first.
    filings.sort(key=lambda f: f.filing_date, reverse=True)
    return filings


def _parse_filings_block(block: dict[str, Any]) -> list[Filing]:
    """Convert one EDGAR filings dict (columnar arrays) to Filing rows."""
    forms = block.get("form", []) or []
    accs = block.get("accessionNumber", []) or []
    filed = block.get("filingDate", []) or []
    reportd = block.get("reportDate", []) or []
    primary = block.get("primaryDocument", []) or []
    out: list[Filing] = []
    for i, form in enumerate(forms):
        if not form.startswith("13F-HR"):
            continue
        acc = accs[i]
        out.append(
            Filing(
                accession=acc,
                accession_nodash=acc.replace("-", ""),
                filing_date=filed[i],
                report_date=reportd[i],
                form=form,
                primary_document=primary[i],
            )
        )
    return out


# ─── INFORMATION TABLE XML parsing ─────────────────────────────────────────
def _find_information_table_filename(
    cik: int | str, filing: Filing, *, user_agent: str = _DEFAULT_UA
) -> str:
    """The 13F bundle ships several attachments. The INFORMATION TABLE has the
    holdings; the primary doc has only the cover sheet. We list the filing
    directory's index.json and pick the first XML whose name isn't the primary
    doc — that's the table per SEC's filing instructions.
    """
    cik_padded_int = int(cik)
    base = f"{_BASE_WWW}/cgi-bin/browse-edgar"  # not used; index lives elsewhere
    index_url = (
        f"{_BASE_WWW}/Archives/edgar/data/{cik_padded_int}/"
        f"{filing.accession_nodash}/index.json"
    )
    payload = json.loads(_get(index_url, user_agent=user_agent))
    items = payload.get("directory", {}).get("item", [])
    xml_names = [it["name"] for it in items if it["name"].lower().endswith(".xml")]
    if not xml_names:
        raise RuntimeError(f"no XML attachments in {index_url}")
    # Strip the primary document (cover page) — what remains is the table.
    table = [n for n in xml_names if n != filing.primary_document]
    if not table:
        # Some old filings ship only one XML; use it as the table.
        return xml_names[0]
    # Prefer the file that literally contains "table" / "info" in its name.
    preferred = [n for n in table if "table" in n.lower() or "info" in n.lower()]
    return preferred[0] if preferred else table[0]


def fetch_13f_holdings(
    cik: int | str, filing: Filing, *, user_agent: str = _DEFAULT_UA
) -> list[Holding]:
    """Parse the INFORMATION TABLE XML for ``filing`` into ``Holding`` rows.

    The XML's element tree has the same column set across all SEC versions:
    ``nameOfIssuer``, ``cusip``, ``value`` (USD thousands), and
    ``shrsOrPrnAmt`` (``sshPrnamt`` + ``sshPrnamtType``) at minimum, plus an
    optional ``putCall``. We tolerate the three known namespace prefixes.
    """
    cik_padded_int = int(cik)
    table_name = _find_information_table_filename(cik, filing, user_agent=user_agent)
    url = (
        f"{_BASE_WWW}/Archives/edgar/data/{cik_padded_int}/"
        f"{filing.accession_nodash}/{table_name}"
    )
    xml_bytes = _get(url, user_agent=user_agent)
    return parse_information_table(xml_bytes)


def parse_information_table(xml_bytes: bytes) -> list[Holding]:
    """Pure-function XML parser. Public so tests can feed in fixture bytes."""
    root = ET.fromstring(xml_bytes)
    rows: list[Holding] = []
    info_rows = _findall_ns(root, "infoTable")
    for row in info_rows:
        name = _text(row, "nameOfIssuer") or ""
        cusip = (_text(row, "cusip") or "").strip().upper()
        value = float(_text(row, "value") or 0.0)
        # 13F values were reported in thousands until 2023-Q1, then raw USD
        # after the SEC rule update. The XML carries a `value` field without
        # an explicit unit, but SEC's instructions still say "in thousands"
        # — we standardise to raw USD by multiplying by 1000. Document this
        # in IMPROVEMENTS so a future re-read against the rule change can
        # toggle the multiplier per ``report_date``.
        value_usd = value * 1_000.0

        shares_block = _find_ns(row, "shrsOrPrnAmt")
        shares = 0.0
        share_type = ""
        if shares_block is not None:
            shares = float(_text(shares_block, "sshPrnamt") or 0.0)
            share_type = (_text(shares_block, "sshPrnamtType") or "").upper()

        put_call_raw = _text(row, "putCall")
        put_call = put_call_raw.capitalize() if put_call_raw else None
        discretion = _text(row, "investmentDiscretion")

        rows.append(
            Holding(
                name_of_issuer=name,
                cusip=cusip,
                value_usd=value_usd,
                shares=shares,
                share_type=share_type,
                put_call=put_call,
                investment_discretion=discretion,
            )
        )
    return rows


def _findall_ns(root: ET.Element, tag: str) -> list[ET.Element]:
    """Find all ``tag`` elements across the three known 13F namespaces."""
    for prefix in _NS_PREFIXES:
        found = root.findall(f".//{prefix}{tag}")
        if found:
            return found
    return []


def _find_ns(root: ET.Element, tag: str) -> ET.Element | None:
    for prefix in _NS_PREFIXES:
        node = root.find(f".//{prefix}{tag}")
        if node is not None:
            return node
    return None


def _text(root: ET.Element, tag: str) -> str | None:
    node = _find_ns(root, tag)
    if node is None or node.text is None:
        return None
    return node.text.strip()


# ─── Convenience ───────────────────────────────────────────────────────────
def latest_13f_holdings(
    cik: int | str, *, user_agent: str = _DEFAULT_UA
) -> tuple[Filing, list[Holding]]:
    """The most recent 13F-HR for ``cik``, holdings included."""
    filings = list_13f_filings(cik, user_agent=user_agent)
    if not filings:
        raise RuntimeError(f"no 13F filings found for CIK {cik}")
    latest = filings[0]
    holdings = fetch_13f_holdings(cik, latest, user_agent=user_agent)
    return latest, holdings


def filings_as_of(
    filings: list[Filing], as_of_date: str, *, max_filing_delay_days: int = 45
) -> Filing | None:
    """Pick the most recent filing whose ``filing_date <= as_of_date - delay``.

    Used by the 13F-follow variant to honour the realistic public-availability
    lag (BRIEF default: 45 days). Returns ``None`` if no filing pre-dates the
    cutoff (i.e. picker hadn't filed yet).
    """
    import datetime as _dt

    cutoff = _dt.date.fromisoformat(as_of_date) - _dt.timedelta(days=max_filing_delay_days)
    cutoff_iso = cutoff.isoformat()
    eligible = [f for f in filings if f.filing_date <= cutoff_iso]
    if not eligible:
        return None
    return max(eligible, key=lambda f: f.filing_date)


# ─── Public CIK directory (curated) ────────────────────────────────────────
# Hard-coded for the four pickers shipped this task. Source: SEC EDGAR search
# (each CIK verified manually). Keeping this dict in the module avoids a
# round-trip to the SEC's full company_tickers JSON for the common case.
PICKER_CIKS: dict[str, int] = {
    "berkshire": 1067983,         # Berkshire Hathaway Inc — Buffett
    "scion": 1649339,             # Scion Asset Management LLC — Burry
    "pershing_square": 1336528,   # Pershing Square Capital Mgmt LP — Ackman
    "appaloosa": 1656456,         # Appaloosa LP — Tepper
}


def picker_cik(name: str) -> int:
    """Resolve a short picker name to its SEC CIK. KeyError on unknown picker."""
    key = name.lower()
    if key not in PICKER_CIKS:
        raise KeyError(
            f"unknown picker {name!r}. Known: {sorted(PICKER_CIKS.keys())}"
        )
    return PICKER_CIKS[key]


# Re-export dataclass converters for downstream JSON serialisation.
def filing_to_dict(f: Filing) -> dict[str, Any]:
    return dataclasses.asdict(f)


def holding_to_dict(h: Holding) -> dict[str, Any]:
    return dataclasses.asdict(h)
