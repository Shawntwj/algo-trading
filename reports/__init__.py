"""Self-contained HTML evaluation reports (BRIEF Task 7e).

`python -m reports.evaluate <strategy>` produces a single dark-theme HTML file
with every chart, table, and significance test for that strategy. No external
CDNs, no JS framework — matplotlib charts are inlined as base64 PNGs.

`python -m reports.overfit_demo` runs the same gauntlet against the in-sample
sweep winner of a generous grid and verifies the DSR < 0.95 honesty check
flags the curve-fit.
"""

# Lazy re-export: importing `reports.evaluate` here at package load time
# triggers a `runpy` re-entrance warning under `python -m reports.evaluate`.
# Callers that want the API should import from `reports.evaluate` directly.
__all__: list[str] = []
