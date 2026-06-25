# arxiv survey — systematic equity strategies (last ~3 years)

Task 2a — literature scan only. Implementation is the job of 2b.

## Methodology

### Search parameters
- **Categories**: q-fin.TR (Trading & Market Microstructure), q-fin.PM (Portfolio Management), q-fin.ST (Statistical Finance).
- **Date range**: submissions from late 2023 through June 2026 (cut-off: today, 2026-06-25). Most candidates are 2025–2026.
- **Query terms** (combined via arxiv API `search_query=cat:X+AND+(abs:Y+OR+abs:Z)`):
  - `abs:sharpe`, `abs:backtest`, `abs:trading strategy`
  - `abs:momentum`, `abs:reversal`, `abs:cross-sectional`, `abs:long-short`
  - `abs:pairs trading`, `abs:statistical arbitrage`
  - `abs:transaction costs`
  - `abs:intraday`, `abs:overnight`, `abs:volatility`
- Sorted by `submittedDate desc`; ~250 abstracts scanned in total across the queries.

### Filter rules (BRIEF gate)
A candidate must pass **all three**:
- **(a) Implementable from public price/volume data** — optionally augmented with basic fundamentals (P/E, book value, market cap) or already-public macro series (VIX, rates). Anything needing news text, 10-K embeddings, institutional flows, options chains, social-media sentiment, or a fitted LLM is **out**.
- **(b) Reproducible signal definition** — the signal can be reconstructed from the abstract plus a few pages of the paper without guessing. "We use a transformer on 47 engineered features, see code" with no released code is out.
- **(c) Realistic transaction costs in the headline result** — at least a basis-point linear cost in the OOS Sharpe number, or the paper does a sensitivity table over costs. Pure gross-Sharpe-only papers are out unless cost robustness is obviously discussed.

A candidate may pass on (c) if the signal turnover is naturally low (e.g. monthly rebalance) and the paper acknowledges costs even if it doesn't bake them in.

### Implementability rubric (1–5)
- **5** — pure price/volume features, fully specified formulas, vectorisable in Polars / pandas / numpy without a research call. One-pass replication.
- **4** — needs basic fundamentals (P/E, market cap, book value) easily fetchable from yfinance or a free vendor. No model fitting harder than OLS / a percentile sort.
- **3** — some ambiguity to resolve (paper handwaves a parameter, mentions "smoothing" without a formula, or relies on a model with non-trivial defaults like ATR-based exits). Replicable but the practitioner has to make and document choices.
- **2** — needs custom estimators (HMM, RL, deep nets) or careful per-asset tuning. Replicable in principle but the failure mode is large.
- **1** — needs alt data, paywalled fundamentals, proprietary news feeds, or LLM scoring. **Excluded from this survey entirely.**

### Reported Sharpe
Quoted at face value as printed in the paper, with universe/window/cost qualifiers. **The honesty pass — does the Sharpe survive our gauntlet — is the job of 2b.** Anything ≥ 2 is treated with suspicion in the notes; anything ≥ 5 is treated as almost-certainly-overfit (see [arxiv:2511.12490] notes).

---

## Candidates

### 1. Discovery of a 13-Sharpe OOS Factor: Drift Regimes Unlock Hidden Cross-Sectional Predictability
- **Authors**: Mainak Singha
- **Year**: 2025
- **arxiv ID / link**: https://arxiv.org/abs/2511.12490
- **Category**: q-fin.TR (cross-listed q-fin.PM)
- **Signal (plain English)**: For each stock on each day, count the number of positive-return days in the trailing 63 days. If that count exceeds 60% (i.e. the stock is in a "drift regime"), combine a value signal and a short-term reversal signal to score the stock cross-sectionally. Long the top-decile composite, short the bottom-decile, frozen parameters, S&P 500 only.
- **Reported Sharpe**: OOS Sharpe > 13, 158.6% annualised return, 12% vol, over 20 years on S&P 500 universe. Author claims "conservative costs and impact modelling" — explicit bp figure not in the abstract.
- **Implementability**: **4/5** — drift gate and reversal signal are pure price; "value signal" is left ambiguous (book/market? P/E? earnings yield?) so one assumption is needed. Otherwise fully specified.
- **Notes / red flags**: Sharpe of 13 is essentially impossible at this scale. Likely candidates for the gap: in-sample parameter selection masquerading as "frozen", lookahead in the value signal, survivorship bias on the S&P 500 constituent list, no realistic borrow/short fee on the short leg, or all of the above. Worth replicating *specifically because* the headline is implausible — if our gauntlet rejects it, that's a clean negative result for the README. **High value as a stress-test of our DSR/PSR machinery.**

### 2. A Volume-Price-Adjusted MACD Trading Strategy with Sensitivity Calibration for U.S. Equity Indices
- **Authors**: Luyun Lin, Lixing Lin, Zhen Zhang, Moxuan Zheng, Yiqing Wang
- **Year**: 2026
- **arxiv ID / link**: https://arxiv.org/abs/2604.26063
- **Category**: q-fin.TR
- **Signal (plain English)**: Standard MACD (12/26/9 EMA pair + signal line) augmented with volume and intraday range terms; a tunable sensitivity parameter shifts the entry threshold so trades trigger earlier. Applied to index ETFs / index futures on S&P 500, Nasdaq-100, DJIA. Calibrated 2018–2022, tested 2023 through Feb 2026.
- **Reported Sharpe**: Abstract says "better profitability, risk-adjusted return, and downside-risk control" than baseline MACD. Specific Sharpe values are in the PDF body (the PDF text layer didn't extract cleanly via WebFetch).
- **Implementability**: **5/5** — MACD plus volume scaling is the most vectorisable strategy in this list. We already have an MA-crossover scaffold in `strategies/`; this is a straight extension.
- **Notes / red flags**: Tested on indices only (3 instruments). MACD literature is the most data-mined corner of TA; the "sensitivity calibration" sounds like an in-sample knob. Need to verify the exact volume weighting before claiming Sharpe.

### 3. Cross-Stock Predictability via LLM-Augmented Semantic Networks
- **Authors**: Yikuan Huang, Zheqi Fan, Kaiqi Hu, Yifan Ye
- **Year**: 2026
- **arxiv ID / link**: https://arxiv.org/abs/2604.19476
- **Category**: q-fin.TR (cross-listed q-fin.PM, q-fin.ST)
- **Signal (plain English)**: Build a sparse stock-stock graph (from 10-K text embeddings, then LLM-filtered for "real" economic edges). Aggregate pair-level mean-reversion signals along graph edges with distance-based weights to produce a stock-level score; long-short on the score. S&P 500 constituents, 2011–2019.
- **Reported Sharpe**: 0.742 (no LLM filter) → 0.820 (LLM filter), long-short OOS. Max DD improved from –10.47% to –7.85%. Cost assumption not disclosed in abstract.
- **Implementability**: **3/5** — the LLM/10-K edge filter is the central novelty, but the *baseline* (sparse correlation-derived graph + pair-level mean-reversion) is implementable from price-only. We'd build the baseline (Sharpe ~0.74) and skip the LLM enhancement; that turns this into a "graph-based stat-arb" replication, not the paper's headline.
- **Notes / red flags**: Headline Sharpe is modest, which is honest. But replicating only the baseline means we're testing a different claim than the paper makes. Use only as a benchmark for our existing mean-reversion code.

### 4. Statistical Arbitrage in Polish Equities Market Using Deep Learning Techniques
- **Authors**: Marek Adamczyk, Michał Dąbrowski
- **Year**: 2025
- **arxiv ID / link**: https://arxiv.org/abs/2512.02037
- **Category**: q-fin.TR
- **Signal (plain English)**: Classical PCA-based statistical arbitrage: extract factor portfolios via PCA on residual returns, build replicating portfolio for each name, trade when the residual (modelled as Ornstein-Uhlenbeck) is z-standard-deviations from zero. WIG20 + mWIG40 universe (Polish equities), 2017–2020. The "deep learning" variant uses an LSTM for replication but the PCA baseline is reported separately.
- **Reported Sharpe**: PCA baseline up to **2.63** annualised, ~20% cumulative return on 2017–2019. 2020 (COVID year) only the ETF-residual variant remained profitable (~5%).
- **Implementability**: **4/5** — PCA stat-arb on residuals is textbook (Avellaneda & Lee 2010); reproducible from the abstract. Polish-market specifics (calendar, trading hours, fees) need translation to a US universe but the algorithm is universe-agnostic.
- **Notes / red flags**: Small universe (~60 stocks). 2020 collapse is reported honestly, which is a good sign. Sharpe is calendar-period-dependent; need to test post-2020. Transaction cost figure not stated explicitly.

### 5. Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals
- **Authors**: Gagan Deep, Akash Deep, William Lamptey
- **Year**: 2025
- **arxiv ID / link**: https://arxiv.org/abs/2512.12924
- **Category**: q-fin.TR
- **Signal (plain English)**: Five hand-crafted microstructure signals (exact list not in abstract — paper body required) tested under a strict walk-forward harness with realistic transaction costs. 100 US equities, 2015–2024. Each signal is interpretable (no neural nets) and combined with RL position-sizing.
- **Reported Sharpe**: **0.33**, 0.55% annualised, max DD –2.76%, p-value 0.34 (statistically not significant). Beta 0.058.
- **Implementability**: **3/5** — paper is *methodologically* well-disciplined but doesn't list the five signals in the abstract. Replication requires reading the PDF body to extract the signals; we'd need to verify they're price/volume-only.
- **Notes / red flags**: This is the *honest* end of the literature — modest returns, transparent about insignificance, strong DD control. Useful as a methodology benchmark (their walk-forward harness mirrors what we built in Task 7c). But the headline Sharpe is too low to be a winner candidate.

### 6. Dynamic Factor Allocation Leveraging Regime-Switching Signals
- **Authors**: Yizhan Shu, John M. Mulvey
- **Year**: 2024
- **arxiv ID / link**: https://arxiv.org/abs/2410.14841
- **Category**: q-fin.PM
- **Signal (plain English)**: Apply a Sparse Jump Model to the active returns of six standard equity style factors (value, size, momentum, quality, low-vol, growth) to detect bull/bear regimes per factor. Combine the regime probabilities with a Black-Litterman optimiser to time the factor sleeve. Long-only style ETF allocation.
- **Reported Sharpe**: Information Ratio ≈ 0.40 vs market (baseline equal-weight 0.05). Per-factor long-short Sharpes "positive across all factors" — no headline number given for a single composite.
- **Implementability**: **2/5** — Sparse Jump Model is a custom estimator (not in sklearn out of the box; authors' implementation exists). Black-Litterman tilt requires a covariance estimate and a view-vector. Six factor sleeves require six factor ETFs (MTUM, QUAL, USMV, VLUE, VTV, VTI growth) — easy to source but the allocation rule is non-trivial.
- **Notes / red flags**: The IR is modest and honestly reported. The methodology is heavy for the claimed edge. Not in our top 3 — sparse-jump-model implementation cost is too high for the expected upside.

### 7. A Multi-Factor Market-Neutral Investment Strategy for New York Stock Exchange Equities
- **Authors**: Georgios M. Gkolemis, Adwin Richie Lee, Amine Roudani
- **Year**: 2024
- **arxiv ID / link**: https://arxiv.org/abs/2412.12350
- **Category**: q-fin.PM
- **Signal (plain English)**: Combine momentum indicators (12-1 month price return, RSI), fundamental factors (P/E, P/B, ROE), and analyst recommendation aggregates via statistical feature selection; build a risk-parity / minimum-variance / equal-weight market-neutral portfolio on NYSE equities.
- **Reported Sharpe**: Abstract claims "higher Sharpe, lower beta, smaller drawdown" vs S&P 500 — specific Sharpe value not disclosed in the abstract.
- **Implementability**: **3/5** — momentum and fundamentals are public. Analyst recommendations (consensus rating) are quasi-public (Yahoo Finance exposes them) but a full point-in-time series is harder. Risk-parity is straightforward.
- **Notes / red flags**: No headline Sharpe number; combination of three signal families means lots of degrees of freedom. Analyst recommendations are a soft dependency that we'd want to skip. Pure momentum + fundamentals version is the implementable subset; the paper doesn't report that ablation.

### 8. Stock Investment: The p-index Approach
- **Authors**: Xinzhao Xie, Bopei Nie, Kuo-Ping Chang
- **Year**: 2026
- **arxiv ID / link**: https://arxiv.org/abs/2606.08569
- **Category**: q-fin.PM (cross-listed q-fin.ST)
- **Signal (plain English)**: Construct a per-stock "p-index" (risk measure derived from European put-option implied volatilities). Form a p-index-efficient frontier and combine with momentum / contrarian sleeves. Tested on SSE 50 (China) and S&P 500, 2018–2023.
- **Reported Sharpe**: Not reported explicitly. Annualised returns: 11.04%–11.93% on Chinese materials sector with momentum, 3.69% on S&P 500 p-index-efficient-momentum.
- **Implementability**: **2/5** — needs option-implied volatilities (or at minimum end-of-day European put prices). This pushes into alt-data territory unless we use ATM IV from Yahoo, which is noisy and only available for liquid names. Drop on (a).
- **Notes / red flags**: Modest US Sharpes (3.69% return is sub-T-bill). Option-data dependency. Excluded.

### 9. Continuous Timing Signals for Growth-Defensive Style Allocation
- **Authors**: Zheli Xiong
- **Year**: 2026
- **arxiv ID / link**: https://arxiv.org/abs/2605.20636
- **Category**: q-fin.PM
- **Signal (plain English)**: Allocate between a "growth" ETF basket (QQQ-style) and a "defensive" ETF basket (XLP/XLU/TLT-style) using three continuous-smooth macro timing signals: rate-relief (recent change in Fed funds), drawdown-depth (rolling max-DD on SPY), and VIX-stress-relief (mean-reversion in VIX). EWMA-smoothed weights, 10 bp transaction cost. June 2017 – May 2026.
- **Reported Sharpe**: **1.01** with 10 bp costs; 19.24% CAGR; improved DD vs static 60/40.
- **Implementability**: **4/5** — three signals are all from public series (FFR, SPY, VIX). EWMA smoothing is standard. Only ambiguity is the specific "stress relief" formula but reasonable defaults exist.
- **Notes / red flags**: Modest Sharpe, modest universe (two ETF baskets). Honest cost figure. Top candidate for replication precisely because the Sharpe is realistic.

### 10. Regime-Based Portfolio Allocation Using Hidden Markov Models and Reinforcement Learning
- **Authors**: Ajay Kumar Verma, Nunik Srikandi Putri, Neo Paul Lesupi
- **Year**: 2026
- **arxiv ID / link**: https://arxiv.org/abs/2605.27848
- **Category**: q-fin.PM
- **Signal (plain English)**: Fit a Hidden Markov Model on SPY / TLT / GLD returns to identify a small number of market regimes (typically 2–4). Train an RL policy (PPO or TD3) to allocate across the three ETFs conditioned on the inferred regime. Daily data 2004–2025.
- **Reported Sharpe**: "Highest Sharpe" vs baselines — no numeric Sharpe in the abstract.
- **Implementability**: **2/5** — HMM fitting is a custom estimator (hmmlearn exists); the RL leg is a non-trivial training loop with its own hyperparameter surface. Replication is plausible but expensive.
- **Notes / red flags**: Three-asset universe is tiny. RL policies on three assets over 20 years have plenty of in-sample fitting room. No headline number.

### 11. From Hypotheses to Factors: Constrained LLM Agents in Cryptocurrency Markets
- **Authors**: Yikuan Huang, Zheqi Fan, Kaiqi Hu, Yifan Ye
- **Year**: 2026
- **arxiv ID / link**: https://arxiv.org/abs/2604.26747
- **Category**: q-fin.TR
- **Signal (plain English)**: LLM-agent generates candidate factor formulas in a point-in-time DSL; ridge-combined into a portfolio. Crypto universe.
- **Reported Sharpe**: 1.55 OOS, 44.55% annualised, 2024–2026.
- **Implementability**: **1/5** — needs an LLM in the loop. Also crypto, not equities. Excluded.

---

## Top 3 (for implementation in 2b)

Scored as `impl × min(reported_sharpe, 3.0)` — we cap the Sharpe input at 3.0 so we don't over-reward implausibly high claims (the 13-Sharpe paper would otherwise dominate the table, and we want the cap to be honest about how seriously to treat the headline). For the VP-MACD paper where no numeric Sharpe is in the abstract, we use 1.0 as a placeholder. For the Polish stat-arb paper, we use the reported 2.63.

| Rank | Title | arxiv | Implementability | Reported Sharpe | Score | Notes |
|------|-------|-------|------------------|-----------------|-------|-------|
| 1    | Statistical Arbitrage in Polish Equities (PCA baseline, adapted to US universe) | [2512.02037](https://arxiv.org/abs/2512.02037) | 4/5 | 2.63 | 10.52 | Classical Avellaneda–Lee stat-arb; algorithm is universe-agnostic, fits cleanly into our existing strategy framework. Honest 2020 collapse reported. |
| 2    | Continuous Timing Signals for Growth-Defensive Style Allocation | [2605.20636](https://arxiv.org/abs/2605.20636) | 4/5 | 1.01 (net 10bp) | 4.04 | Cheap to implement (three macro signals, two ETF baskets), realistic Sharpe with explicit cost figure, complements our existing momentum/MR scaffold with a regime-allocation style. |
| 3    | Discovery of a 13-Sharpe OOS Factor: Drift Regimes | [2511.12490](https://arxiv.org/abs/2511.12490) | 4/5 | 3.0 (capped from 13) | 12.0 | **Picked precisely because the headline is implausible.** Top-of-table on cap-adjusted score. The strategy itself is simple (drift gate + value + short-term reversal) and the most useful Task 2b deliverable is a clean, public negative result showing the gauntlet (PSR/DSR/walk-forward) rejects this — gives the README its first honest "we tried, it didn't survive" entry. |

Honest reading of the table: rank 1 (Polish stat-arb) and rank 2 (growth/defensive timing) are the strategies most likely to *actually backtest cleanly*. Rank 3 is the falsification target. 2b should treat all three as honest tests and not chase numbers.

---

## Excluded

- **arxiv:2601.11958** (Agentic AI Nowcasting) — needs autonomous web search at runtime. Out on (a).
- **arxiv:2510.26228** (ChatGPT in Systematic Investing) — base momentum strategy not specified independently of the LLM enhancement, and the lift comes from the LLM. Out on (b).
- **arxiv:2603.14288** (Beyond Prompting / Agentic AI factor investing) — claimed Sharpe 3.11; signal definition is "the agent figures it out". Out on (a) and (b).
- **arxiv:2604.17327** (Multi-agent LLM stock recommendations) — needs LLM scoring + news ingest. Out on (a).
- **arxiv:2507.04481** (Does Overnight News Explain Overnight Returns?) — needs 2.4M news articles + supervised topic model. Out on (a).
- **arxiv:2604.13458** (Interpretable Systematic Risk around the Clock) — needs LLM-classified news + high-frequency tick data. Out on (a).
- **arxiv:2509.11970** (Sentiment Feedback in Equity Markets) — needs four sentiment proxies (FEARS, AAII, MSI, retail flows). Mostly proprietary. Out on (a).
- **arxiv:2602.18912** (Overreaction-as-momentum on AAPL) — single-ticker case study with Twitter emotion features. Out on (a) and (b).
- **arxiv:2606.09420** (Benchmarking Deep Time Series Models) — paper itself reports negative net Sharpe at 20 bp costs. Out on (c) — the authors acknowledge it doesn't beat costs.
- **arxiv:2606.09478** (HARQ + XGBoost on CSI 300) — China-only; we can't source CSI 300 bars from the existing data layer without new vendor work. Cut for scope, not the BRIEF (a) gate.
- **arxiv:2606.00060** (ML-based Bitcoin Trading) — crypto, not equities. Out of scope.
- **arxiv:2511.08571** (Forecast-to-Fill: Gold Futures) — gold futures, not equities. Out of scope.
- **arxiv:2603.01820** (Deep Learning for Financial Time Series benchmark) — multi-asset futures benchmark, not a single deployable strategy. Useful methodology reference, not implementable as one thing.
- **arxiv:2601.05975** (DeePM macro portfolio) — 50-futures universe; not equities.
- **arxiv:2410.14927** (Hierarchical Reinforced Trader) — needs both market and text signals; RL training loop adds large degrees of freedom. Out on (b) borderline / cut for scope on (a).
- **arxiv:2606.08569** (p-index approach) — needs European put-option implied vols. Out on (a).
- **arxiv:2605.27848** (HMM + RL allocation) — three-asset universe; RL training expensive; no headline Sharpe. Cut for scope.
- **arxiv:2604.26747** (LLM agents in crypto) — LLM + crypto. Out on (a) and out of asset scope.
- **arxiv:2605.04004**, **arxiv:2605.11423** (Mesfin OHLCV / volatility falsification studies) — papers' own conclusion is "no edge survives costs". Included only as honest counter-examples; nothing to replicate.
- **arxiv:2606.24019** (Square-root law of market impact on AAPL) — descriptive empirics, not a strategy.
- **arxiv:2511.06177** (Push-response anomalies in SPY) — tick-level intraday; outside our daily-bar data layer.

---

## Notes on the scan

- The arxiv API returned a strong skew towards LLM-agent / deep-RL papers in 2025–2026; the "implementable from price/volume alone" subset is small (~10–12 out of ~250 abstracts skimmed).
- Several papers list astronomical Sharpes (13+, 6+) — these are concentrated in single-author submissions with no co-authors and no acknowledged data vendor. Treat with maximum suspicion; this is exactly what DSR is for.
- The honest end of the literature (papers reporting Sharpe < 1 with statistical insignificance) is small but exists — Deep et al. 2025 ([2512.12924](https://arxiv.org/abs/2512.12924)) is a good example. These rarely make headlines but their methodology is worth borrowing.
- The "transaction cost" filter (c) is the most aggressive gate. Most papers report gross or "robust to costs" without a numeric bp figure. We let candidates through if turnover is naturally low (monthly) and the paper at least discusses cost sensitivity.
