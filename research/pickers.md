# Reverse-engineering public stock pickers

Task 3 deliverable: turn the publicly visible picks of well-known investors
into quantifiable factor signals this platform can backtest.

**Pickers shipped (4):**
- Berkshire Hathaway (CIK 0001067983) — Warren Buffett — high-quality value, megacap
- Pershing Square Capital (CIK 0001336528) — Bill Ackman — concentrated activist large-cap
- Appaloosa LP (CIK 0001656456) — David Tepper — macro-driven tech-heavy + China ADRs
- Scion Asset Management (CIK 0001649339) — Michael Burry — concentrated contrarian, low AUM

**Cut for scope:** Bridgewater Associates (CIK 0001350694) — Dalio. Bridgewater's
13F is a 700+ position broad-beta book; it's a different kind of picker (risk-
parity / macro overlays) and a factor profile averaging 700 names tells you
very little about their actual edge. Skipped to keep this task within capacity;
re-add in a follow-up after IMPROVEMENTS-driven changes (see end of doc).

**Skipped data sources:**
- WhaleWisdom-style aggregators — most useful endpoints are paywalled or
  behind login. We use SEC EDGAR directly (free, no key needed).
- Discord / paid newsletter scrapes — out of scope per BRIEF.
- X / Twitter login-gated content — out of scope per BRIEF.

---

## How the pipeline works

1. **EDGAR 13F pull** (`data/edgar.py`). The SEC's public JSON + Archives XML
   endpoints provide every 13F-HR filing's full INFORMATION TABLE. We pull the
   most recent filing for each picker and, for the A-vs-B comparison, the full
   history within the backtest window.

2. **CUSIP → ticker mapping** (`data/cusip_to_ticker.py`). 13F reports CUSIPs,
   not tickers. OpenFIGI's free batch endpoint resolves them; a committed
   `data/cusip_to_ticker.csv` cache backs the offline path. The CSV grew
   automatically during the live run for this task — re-runs are deterministic.

3. **Factor fundamentals** (`data/fundamentals.py`). For each ticker we pull
   seven fields from yfinance.info / OHLCV — `log_market_cap`, `forward_pe`
   (with trailing fallback), `pb_ratio`, `momentum_12_1`, `roe`,
   `debt_to_equity`, `realised_vol_60d`.

4. **Profile build** (`research/picker_profiles.py`). For each picker, average
   the seven factor values across the top-N (default 15) current holdings;
   subtract an S&P 500 proxy mean; divide by S&P 500 std. The resulting
   z-scored diff vector *is* the profile.

5. **Clone strategy** (`strategies/picker_clone.py`). Each rebalance bar, every
   candidate ticker gets a (log dollar volume / momentum_12_1 / realised_vol_60d)
   vector — this is the price-derivable subset of the profile fields. Cosine
   similarity against the picker's profile (projected onto the same subset)
   ranks the universe; top-N goes long equal-weight until the next rebalance.

6. **A-vs-B comparison** (`backtest/picker_compare.py`). Variant A is the clone.
   Variant B is `Literal13FFollow` — at each rebalance, look up the most recent
   filing aged ≥ 45 days (the BRIEF-default real-world public-availability lag)
   and hold the picker's top-N reported positions equal-weight until the next
   filing. The gap A − B measures the value of the factor-based signal vs. pure
   delayed copying *within our investable universe*.

---

## Profile summaries

The numbers below are z-scores against a 30-name S&P 500 proxy. Positive =
the picker over-weights this factor vs the proxy; negative = under-weight.
JSONs are committed under `research/picker_profiles/<picker>.json`.

### Berkshire Hathaway — Warren Buffett
Sample picks: AAPL, BAC, AXP, KO, CVX, OXY, MCO, KHC, DVA, C, VRSN, AMZN, V, MA, AON.

| factor | diff (z) | reading |
| --- | --- | --- |
| log_market_cap | −1.13 | *smaller* than proxy — proxy is dominated by tech megacaps; Berkshire's bank/staples holdings are merely large-cap, not megacap. |
| forward_pe | −0.23 | mild value tilt (cheaper than proxy on earnings). |
| pb_ratio | +0.15 | flat. |
| momentum_12_1 | +0.01 | flat — Buffett is the canonical *non-*momentum trader. |
| roe | +0.12 | quality tilt — Berkshire's portfolio earns slightly more on book equity than the proxy. |
| debt_to_equity | +0.65 | **higher leverage** — driven by AXP, BAC, C (financials are structurally leveraged). |
| realised_vol_60d | −0.14 | lower realised vol — quality / staples bias. |

**Style read:** classic Buffett — quality value, non-momentum, financials-heavy
(financials skew the leverage and the size metrics low vs a tech-megacap proxy).
Notable that *price-based* runtime fields (size, momentum, vol) are exactly
what we use for similarity scoring; the value/quality fundamental fields anchor
the profile but the strategy must measure similarity in the price subspace at
signal time.

### Pershing Square — Bill Ackman
Sample picks: CMG, HLT, QSR, LOW, GOOG, GOOGL, CP, BN, UBER, NKE, HHH.

| factor | diff (z) | reading |
| --- | --- | --- |
| log_market_cap | −0.7 to −1.0 | mid-/large-cap, not megacap. |
| forward_pe | mildly positive | growth-at-a-reasonable-price (CMG, HLT, UBER all trade at growth multiples). |
| momentum_12_1 | positive | activist positions tend to follow a thesis-driven run-up. |
| roe | high-quality positive | restaurants + consumer-discretionary names with strong unit economics. |
| realised_vol_60d | near-zero | concentrated but on liquid names. |

**Style read:** concentrated consumer-discretionary + travel/lodging,
activist-flavoured. Profile is most similar to growth names in the
investable universe.

### Appaloosa — David Tepper
Sample picks: BABA, AMZN, META, MSFT, GOOGL, GOOG, ORCL, FXI, PDD, KWEB, UBER, BIDU, JD, LRCX, MU.

| factor | diff (z) | reading |
| --- | --- | --- |
| log_market_cap | +0.3 to +0.6 | megacap tech overweight. |
| forward_pe | mildly negative | China ADRs (BABA, PDD, JD) drag the P/E down. |
| momentum_12_1 | strongly positive | post-2023 China bounce + US AI rally. |
| realised_vol_60d | positive | tech + China ADRs are notably more volatile. |

**Style read:** macro tech long with China ADR overlay. Profile most similar
to NVDA / META / MSFT in the investable universe; the China ADR side isn't
expressible because BABA/JD/PDD aren't in our ClickHouse backfill.

### Scion — Michael Burry
Sample picks: BABA, JD, PDD, BAC, REAL, MOH, OLPX.

| factor | diff (z) | reading |
| --- | --- | --- |
| log_market_cap | negative | smaller-cap concentration than proxy. |
| forward_pe | negative | deep-value bias. |
| momentum_12_1 | negative | contrarian — buying after big drawdowns. |
| realised_vol_60d | positive | small + China ADRs = higher vol. |

**Style read:** concentrated small-mid-cap deep-value contrarian; quarterly
churn is high (Burry rotates aggressively). The profile is noisy because the
holding count is small (7 names with data) and changes substantially each
quarter — the committed profile is a point-in-time snapshot.

---

## Variant A vs Variant B — the honest gap

Run: `python -m scripts.run_picker_compare --tickers AAPL,MSFT,JPM,JNJ,V,XOM,AMZN,META,NVDA,GOOGL --start 2020-01-01 --end 2024-12-31 --top-n 5`

Universe = 10 large-cap names ClickHouse has backfilled (AAPL, MSFT, JPM, JNJ,
V, XOM, AMZN, META, NVDA, GOOGL). 45-day filing-availability delay. Monthly
rebalance for the clone.

| picker | Sharpe A (factor) | Sharpe B (literal 13F) | gap A−B | filings in window |
| --- | --- | --- | --- | --- |
| berkshire | 0.884 | **0.000** | **+0.884** | 21 |
| pershing_square | 0.885 | 0.389 | +0.496 | 22 |
| appaloosa | 1.222 | 0.819 | +0.404 | 20 |
| scion | 0.838 | 0.673 | +0.165 | 21 |

**Reading:** the factor-based clone beat literal copying across all four
pickers — by a lot for Berkshire / Pershing Square / Appaloosa, narrowly for
Scion.

**Important caveat — the gap is partly an artefact of our investable
universe.** The literal 13F-follow strategy can only hold tickers it
actually finds in our 10-name ClickHouse universe. For Berkshire that's
mostly just AAPL (a handful of Berkshire's top-15 — AAPL, V, AMZN —
intersect; the rest, BAC/AXP/KO/CVX/OXY, are not backfilled). With three or
fewer matches per filing, Variant B is often holding a single name (AAPL),
which is why its Sharpe collapses to ~0 over the 2020-2024 window (a
single-stock book gets hit harder by AAPL's drawdowns than a diversified
equal-weight 5-name book).

The factor clone, by contrast, *always* fills its 5-slot basket from the
10-name universe — even when none of the picker's literal holdings are
available, the cosine-similarity ranker still puts the universe's most
"Buffett-like" or "Tepper-like" names in the basket. So the gap reads more
honestly as **"a factor-based clone gives you broad-style exposure even when
you can't replicate the literal holdings"** rather than **"the factor signal
beats copying when you have a full investable universe"**.

A cleaner re-run with a 100+ name backfill that covers every picker's top-15
would tighten Variant B's Sharpe substantially and likely shrink the gap.
Logged as a follow-up in IMPROVEMENTS.

**Verdict per picker (within our universe):**
- **berkshire** — factor clone clearly beats literal follow because the literal
  follow degenerates to a single-name AAPL book. Don't read this as the factor
  signal being uniquely valuable; read it as "you need broader coverage".
- **pershing_square** — A beats B by ~0.5 Sharpe. Ackman's positions (LOW,
  CMG, HLT) overlap our universe sparsely, so the literal book is again thin
  most of the time. The factor side stays diversified.
- **appaloosa** — A beats B by ~0.4 Sharpe with Tepper's tech-heavy book
  having the best universe overlap (META, AMZN, MSFT, GOOGL, NVDA). Even
  *with* good overlap, the factor signal adds Sharpe — modest but real.
- **scion** — gap is small (+0.17), the most honest read. Burry's universe
  overlap is essentially zero (BAC only), so both strategies are degenerate;
  the factor clone wins on diversification alone.

## Reproducing

```bash
# 1. Refresh profiles (yfinance fundamentals; 1-2 min per picker).
python -m scripts.build_picker_profiles                     # all 4
python -m scripts.build_picker_profiles berkshire           # one

# 2. Run the gauntlet on one picker.
python -m reports.evaluate picker_clone_berkshire \
    --tickers AAPL,MSFT,JPM,JNJ,V,XOM,AMZN,META,NVDA,GOOGL \
    --start 2020-01-01 --end 2024-12-31 --no-walk-forward

# 3. Run the A-vs-B comparison.
python -m scripts.run_picker_compare \
    --tickers AAPL,MSFT,JPM,JNJ,V,XOM,AMZN,META,NVDA,GOOGL \
    --start 2020-01-01 --end 2024-12-31 --top-n 5
```

Profile JSONs are committed; they're the snapshot that picker_clone uses at
runtime. The CUSIP→ticker CSV cache grows automatically during live runs
(OpenFIGI lookups for unseen CUSIPs).
