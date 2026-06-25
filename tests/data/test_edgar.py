"""Offline-safe tests for the EDGAR client.

Network round-trips are gated by ``@pytest.mark.slow`` so the default suite
runs in a few hundred ms. The XML parser is fully exercised with fixture
bytes (no SEC dependency).
"""
from __future__ import annotations

import pytest

from data import edgar


# ─── parse_information_table ────────────────────────────────────────────
FIXTURE_XML_NS = b"""<?xml version='1.0' encoding='UTF-8'?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>5000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>25000000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>25000000</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>594918104</cusip>
    <value>3000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>12000000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <putCall>Put</putCall>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
</informationTable>"""


FIXTURE_XML_NO_NS = b"""<?xml version='1.0' encoding='UTF-8'?>
<informationTable>
  <infoTable>
    <nameOfIssuer>BERKSHIRE HATHAWAY</nameOfIssuer>
    <cusip>084670702</cusip>
    <value>1500</value>
    <shrsOrPrnAmt>
      <sshPrnamt>10</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
</informationTable>"""


def test_parse_information_table_namespaced():
    rows = edgar.parse_information_table(FIXTURE_XML_NS)
    assert len(rows) == 2

    aapl, msft = rows
    assert aapl.name_of_issuer == "APPLE INC"
    assert aapl.cusip == "037833100"
    # SEC reports value in thousands; we convert to raw USD.
    assert aapl.value_usd == 5_000_000 * 1_000.0
    assert aapl.shares == 25_000_000
    assert aapl.share_type == "SH"
    assert aapl.put_call is None
    assert aapl.investment_discretion == "SOLE"

    assert msft.cusip == "594918104"
    assert msft.put_call == "Put"


def test_parse_information_table_no_namespace():
    rows = edgar.parse_information_table(FIXTURE_XML_NO_NS)
    assert len(rows) == 1
    brk = rows[0]
    assert brk.name_of_issuer == "BERKSHIRE HATHAWAY"
    assert brk.value_usd == 1_500 * 1_000.0


def test_parse_information_table_empty_table_returns_empty_list():
    empty = b"<?xml version='1.0'?><informationTable></informationTable>"
    assert edgar.parse_information_table(empty) == []


# ─── filings_as_of (no network) ─────────────────────────────────────────
def _filing(filing_date: str, report_date: str) -> edgar.Filing:
    acc = f"0000000000-{filing_date.replace('-', '')[2:]}-X"
    return edgar.Filing(
        accession=acc,
        accession_nodash=acc.replace("-", ""),
        filing_date=filing_date,
        report_date=report_date,
        form="13F-HR",
        primary_document="primary_doc.xml",
    )


def test_filings_as_of_respects_45_day_delay():
    # 13F for Q2 2024 (report 2024-06-30) was filed 2024-08-14. With a
    # 45-day public-availability delay, callers running an as-of date of
    # 2024-09-25 *should* see it (filed 2024-08-14 ≤ 2024-09-25 - 45d =
    # 2024-08-11... actually no — that's still after cutoff). Test both sides.
    filings = [
        _filing("2024-08-14", "2024-06-30"),  # Q2 2024
        _filing("2024-05-15", "2024-03-31"),  # Q1 2024
        _filing("2024-02-14", "2023-12-31"),  # Q4 2023
    ]

    # On 2024-09-25 the Q1 filing is well past 45d delay, Q2 is not yet 45d old.
    pick = edgar.filings_as_of(filings, "2024-09-25", max_filing_delay_days=45)
    assert pick is not None
    assert pick.report_date == "2024-03-31"

    # On 2024-10-15 the Q2 filing has aged past 45d (filed 2024-08-14,
    # cutoff = 2024-10-15 - 45d = 2024-08-31). Q2 should win.
    pick = edgar.filings_as_of(filings, "2024-10-15", max_filing_delay_days=45)
    assert pick.report_date == "2024-06-30"

    # Very early date — no filing eligible.
    pick = edgar.filings_as_of(filings, "2023-01-01", max_filing_delay_days=45)
    assert pick is None


def test_filings_as_of_zero_delay_picks_most_recent_filed():
    filings = [
        _filing("2024-08-14", "2024-06-30"),
        _filing("2024-05-15", "2024-03-31"),
    ]
    pick = edgar.filings_as_of(filings, "2024-08-14", max_filing_delay_days=0)
    assert pick.report_date == "2024-06-30"


def test_picker_cik_known():
    assert edgar.picker_cik("berkshire") == 1067983
    assert edgar.picker_cik("Berkshire") == 1067983
    assert edgar.picker_cik("scion") == 1649339


def test_picker_cik_unknown_raises():
    with pytest.raises(KeyError):
        edgar.picker_cik("definitely_not_a_picker")


# ─── network smoke (slow, opt-in) ───────────────────────────────────────
@pytest.mark.slow
def test_list_13f_filings_berkshire_smoke():
    """Optional: hit SEC EDGAR for real. Skipped by default (network)."""
    filings = edgar.list_13f_filings(edgar.PICKER_CIKS["berkshire"])
    assert len(filings) > 20  # Berkshire has filed 13Fs since 1999
    assert all(f.form.startswith("13F-HR") for f in filings)
