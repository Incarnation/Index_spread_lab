# Ingestion + GEX Correctness Audit

Long-term reference for the live-ingest surface: 5 hot ingest jobs, 2 adjacent
jobs, 2 vendor clients, and 1 offline downloader. Captures the findings of a
deep readonly audit performed 2026-04-19 against the source AND a single
read-only SQL session against the Railway production DB.

> **Status**: Wave 0 (this doc) only. No runtime code changes in the commit
> that introduces this file. Fixes are sequenced into Waves 1-6 below.
> Companion audit for the offline pipeline lives at
> [`backend/scripts/OFFLINE_PIPELINE_AUDIT.md`](../../scripts/OFFLINE_PIPELINE_AUDIT.md);
> this audit follows the same format and severity convention.

> **Line-number convention.** All `file:NNN` citations in this document
> reflect the source state at audit time. Files will shift line counts as
> the Wave 1-6 fixes land. Treat the file path as authoritative and the
> line number as a "search hint" -- locate the referenced symbol/string
> in the current file rather than trusting the exact line.

---

## Table of contents

1. [Methodology](#methodology)
2. [Surface area inventory](#surface-area-inventory)
3. [GEX math reference + worked example](#gex-math-reference--worked-example)
4. [Findings — Critical (C1)](#findings--critical-c1)
5. [Findings — High (H1–H9)](#findings--high-h1h9)
6. [Findings — Medium (M1–M19)](#findings--medium-m1m19)
7. [Findings — Low (L1–L10)](#findings--low-l1l10)
8. [Refactor opportunities](#refactor-opportunities)
9. [Wave plan](#wave-plan)
10. [Verification checklist](#verification-checklist)
11. [Open questions](#open-questions)
12. [Glossary](#glossary)

---

## Methodology

The audit ran six parallel readonly explorers (mirroring the offline audit's
pattern), then ran a single read-only SQL session against the production
Railway DB to produce concrete numerical evidence.

| Explorer | Files | Cross-references |
|----------|-------|------------------|
| 1. Underlying quotes + Tradier client | [`quote_job.py`](quote_job.py), [`tradier_client.py`](../ingestion/tradier_client.py) | [`db_schema.sql`](../database/sql/db_schema.sql), [`scheduler_builder.py`](../scheduler_builder.py), [`config.py`](../config.py), [`test_quote_job.py`](../../tests/test_quote_job.py) |
| 2. Option-chain snapshot ingest | [`snapshot_job.py`](snapshot_job.py) | `db_schema.sql`, `scheduler_builder.py`, `config.py`, [`decision_job.py`](decision_job.py), `tradier_client.py`, [`test_snapshot_helpers.py`](../../tests/test_snapshot_helpers.py), [`test_dte.py`](../../tests/test_dte.py) |
| 3. Live GEX (Tradier-derived) | [`gex_job.py`](gex_job.py) | `db_schema.sql`, `scheduler_builder.py`, `config.py`, `cboe_gex_job.py`, [`event_signals.py`](../services/event_signals.py), [`test_gex_job.py`](../../tests/test_gex_job.py) |
| 4. Vendor GEX (CBOE / mzdata) | [`cboe_gex_job.py`](cboe_gex_job.py), [`mzdata_client.py`](../ingestion/mzdata_client.py) | `db_schema.sql`, `scheduler_builder.py`, `config.py`, `gex_job.py`, [`test_cboe_gex_job.py`](../../tests/test_cboe_gex_job.py) |
| 5. EOD events + retention + staleness | [`eod_events_job.py`](eod_events_job.py), [`retention_job.py`](retention_job.py), [`staleness_monitor_job.py`](staleness_monitor_job.py) | `db_schema.sql`, `scheduler_builder.py`, `config.py`, [`alerts.py`](../services/alerts.py), [`generate_economic_calendar.py`](../../scripts/generate_economic_calendar.py) |
| 6. Offline downloader + DB cross-check | [`download_databento.py`](../../scripts/download_databento.py) | `db_schema.sql`, [`OFFLINE_PIPELINE_AUDIT.md`](../../scripts/OFFLINE_PIPELINE_AUDIT.md), [`data/README.md`](../../../data/README.md) |

In total: **~3,000 lines of Python across 10 files**, 7 production tables
introspected via `information_schema`, and 25 read-only queries executed in
a single 4-minute psql-equivalent session.

### Live-DB query session (read-only)

Connection: `DATABASE_URL` from [`.env`](../../../.env) (Railway-hosted
Postgres; specific proxy host/port not reproduced here). Session executed
via `asyncpg` (not the app's session factory) so the audit could not
accidentally take an app-level write path.

**Strict rules enforced in [`tools/_track1_audit/audit_run.py`](#) (kept out
of the repo, lives in `/tmp/`):**

- Every statement starts with `SELECT`, `WITH`, or `EXPLAIN`. The script
  also issues `SET default_transaction_read_only = on;` as belt-and-suspenders.
- `SET statement_timeout = '60s';` so a runaway scan can never hold a
  connection open.
- Every row-returning query has an explicit `LIMIT` (validated in Python
  before execution).
- No `SELECT *` on `chain_snapshots` (largest non-OPRA table) — explicit
  column projection always.
- Sample rows that appear in this doc redact full timestamps to
  `date_trunc('day', ts)` and contain only public-market data
  (symbol/strike/expiry); no PII or auth state was selected at any point.
- One short audit session, run during quiet US-overnight hours; no retry,
  bail and re-run later if the proxy chokes.

Queries are referenced by `Q1`..`Q25` throughout the findings.

---

## Surface area inventory

### Source files (10 in scope)

| File | Lines | Coverage | Role |
|------|------:|---------:|------|
| [`quote_job.py`](quote_job.py) | 220 | 89% | Per-symbol underlying-quote tick ingest (Tradier) |
| [`snapshot_job.py`](snapshot_job.py) | 613 | 70% | Option-chain snapshot ingest (Tradier) per `target_dte` |
| [`gex_job.py`](gex_job.py) | 428 | **33%** | Tradier-chain-derived GEX aggregator (lowest coverage of the hot ingest jobs) |
| [`cboe_gex_job.py`](cboe_gex_job.py) | 759 | 85% | Vendor-precomputed GEX (mzdata / CBOE) |
| [`eod_events_job.py`](eod_events_job.py) | 82 | 100% | Static economic-calendar seeder |
| [`retention_job.py`](retention_job.py) | 78 | 100% | Batched purge of old `chain_snapshots` (cascades) |
| [`staleness_monitor_job.py`](staleness_monitor_job.py) | 196 | 95% | Pipeline-freshness alerting (SendGrid) |
| [`tradier_client.py`](../ingestion/tradier_client.py) | 100 | 90% | Tradier HTTP client (`get_quotes`, `get_option_chain`, ...) |
| [`mzdata_client.py`](../ingestion/mzdata_client.py) | ~80 | 80% | mzdata HTTP client (`get_live_option_exposure`) |
| [`download_databento.py`](../../scripts/download_databento.py) | 612 | offline | Historical OPRA / DBEQ batch + streaming downloader |

### Production DB inventory (live-counts captured 2026-04-19; `Q4`)

| Table | Live rows | Dead rows | Total size | Role |
|-------|----------:|----------:|-----------:|------|
| `option_chain_rows` | 10,738,686 | 16,223 | **16 GB** | Per-snapshot option leg rows (calls + puts × strikes × expirations) |
| `gex_by_expiry_strike` | 15,431,734 | 0 | 9.6 GB | Per-strike per-expiry GEX (CBOE primary writer) |
| `gex_by_strike` | 13,842,107 | 34,397 | 2.0 GB | Per-strike GEX (both sources) |
| `chain_snapshots` | 90,443 | 78 | 56 MB | Snapshot header (per `(ts, underlying, expiration, source)`) |
| `gex_snapshots` | 89,679 | 93 | 19 MB | One row per `chain_snapshot.snapshot_id`, holds aggregate GEX |
| `underlying_quotes` | 13,418 | 8 | 12 MB | Per-symbol per-tick spot/bid/ask (multi-symbol: SPX, SPY, VIX, VIX9D, SKEW, VVIX) |
| `context_snapshots` | 10,216 | 1,094 | 2.6 MB | Decision-time context cache (`gex_net`, `vix`, `term_structure`, ...) |
| `economic_events` | 211 | 0 | 64 KB | Static economic calendar (FOMC, CPI, OPEX, ...) |

**Total live ingest footprint: ~28 GB** across these eight tables. Retention
is currently **disabled** (`settings.retention_enabled = False`); the oldest
non-empty `chain_snapshots` row is from `2026-02-18` (Q23) — the entire
live history fits in 56 MB at the snapshot-header layer because most of the
mass is in `option_chain_rows` and the per-strike GEX tables.

### DB indexes (Q2) — UNIQUE coverage

The unique-key situation is mixed:

- `chain_snapshots.uq_chain_snapshots_ts_und_exp_src` — **deployed**, matches
  what `snapshot_job` and `cboe_gex_job` rely on.
- `option_chain_rows.option_chain_rows_pkey` — **deployed** as
  `PRIMARY KEY (snapshot_id, option_symbol)`; Q11 confirms zero duplicates.
- `gex_snapshots.gex_snapshots_pkey` — **deployed** as
  `PRIMARY KEY (snapshot_id)`; Q12 confirms zero duplicates.
- `economic_events.economic_events_pkey` — **deployed** as
  `PRIMARY KEY (date, event_type)`; Q13 confirms zero duplicates.
- `context_snapshots.context_snapshots_pkey` — **deployed** as
  `PRIMARY KEY (ts, underlying)`.
- `underlying_quotes` — has only `idx_underlying_quotes_symbol_ts` and
  `underlying_quotes_pkey (quote_id BIGSERIAL)`. **No UNIQUE index on
  the natural key.** Q14 confirms 10+ near-duplicate `(symbol, minute_bucket)`
  rows on 2026-04-06 alone. See **H5**.

### Schedules (per-job, from [`scheduler_builder.py`](../scheduler_builder.py))

| Job | Trigger | Cadence | RTH gate | Notes |
|-----|---------|---------|----------|-------|
| `quote_job` | RTH-windowed regular trigger | every `quote_interval_minutes` (default 5) Mon-Fri 09:31-15:55 ET, plus 16:00 ET force-tick | Both scheduler-level (`build_market_open_guarded_runner`) and job-level (when `allow_quotes_outside_rth=False`) | Multi-symbol per tick |
| `snapshot_job` | Same RTH-windowed pattern | every `snapshot_interval_minutes` (default 5), plus 16:00 force | Same | One run per `target_dte` × per registered underlying |
| `gex_job` | Same RTH-windowed pattern | every `gex_interval_minutes` (default 5), plus 16:00 force | Same; `gex_allow_outside_rth` defaults False | Walks pending `chain_snapshots` that have no `gex_snapshots` row yet |
| `cboe_gex_job` | Same RTH-windowed pattern | every `cboe_gex_interval_minutes`; one run loops all underlyings in `cboe_gex_underlyings_list()` | Same | Vendor-precomputed; writes `chain_snapshots` (empty), `gex_snapshots`, `gex_by_strike`, `gex_by_expiry_strike` |
| `eod_events_job` | APScheduler `cron` | `mon-fri` at `eod_events_hour:eod_events_minute` (default **16:30 ET**) | `force=True` path; skips if `now_et.date() not in open_trading_days` | Static seed only |
| `retention_job` | APScheduler `cron` | daily at `03:00` in app TZ | None (intentional) | Disabled by default |
| `staleness_monitor_job` | APScheduler `interval` | every `staleness_alert_interval_minutes` (default 30) | Internal `_market_open` check | Job is scheduled 24/7 but most ticks early-exit |

---

## GEX math reference + worked example

The audit's central correctness question is whether `gex_job.py` (Tradier
chain-derived) and `cboe_gex_job.py` (mzdata-precomputed) produce values
that are dimensionally consistent and consumer-trustworthy.

### Reference convention (SqueezeMetrics white paper)

Per the SqueezeMetrics convention (publicly documented in the "Gamma
Exposure" white paper widely cited in the dealer-flow literature), per-strike
dollar gamma exposure scaled to a 1% spot move is:

```
GEX_per_strike  =  OI  ×  γ  ×  contract_multiplier  ×  S²  ×  0.01
```

where:

- `OI` is open interest at that strike,
- `γ` is the per-share Black-Scholes gamma (`∂Δ/∂S` per share),
- `contract_multiplier = 100` for standard equity / index options,
- `S` is the underlying spot,
- `0.01` is the "1% move" scaling (so the result is "$ delta change per 1%
  spot move").

The SqueezeMetrics dealer-flow sign convention is:

```
GEX_dealer_per_strike  =  + GEX_call  −  GEX_put
```

i.e. dealers are **net long** call gamma (they sold calls to retail and
hedged) and **net short** put gamma (they sold puts to retail and hedged
the other way). When the strike-summed `GEX_dealer > 0`, dealers
sell-rallies / buy-dips → vol-suppressing. When `< 0`, the opposite
(vol-amplifying). The "zero-gamma level" (or "gamma flip") is the spot
price at which the strike-summed dealer GEX crosses zero.

### Implementation in `gex_job.py` (Tradier-derived)

`gex_job._compute_gex_for_snapshot` (search hint: ~line 130-200) implements:

```python
sign = +1 if option_right == "C" else (-1 if settings.gex_puts_negative else +1)
multiplier = contract_size or settings.gex_contract_multiplier  # default 100
gex_val = sign * 0.01 * gamma * oi * multiplier * (spot ** 2)
```

with two pre-aggregation filters that **deviate from a "full chain"
GEX**:

1. **DTE cap.** `if dte < 0 or dte > settings.gex_max_dte_days: continue`
   (default `gex_max_dte_days = 10`). Long-dated OI is excluded.
2. **Top-N strikes near spot.** After the DTE filter, only the
   `settings.gex_strike_limit` (default 150) strikes nearest spot are
   summed. Tail OI is excluded.

`gex_job._zero_gamma_level` walks the sorted strikes, computes cumulative
`gex_calls + gex_puts` per strike, and linearly interpolates the first
sign change.

Missing greeks (`gamma is None`) and missing OI are **skipped** for that
option (correctly avoids zero-impute bias).

### Implementation in `cboe_gex_job.py` (vendor-precomputed)

`cboe_gex_job._normalize_exposure_items` and `_run_once_for_underlying`
ingest mzdata's `/api/options/{SYM}/exposure` JSON. Per strike:

```python
call_gamma = _series_float(item.call_abs_gamma, idx, default=0.0)
put_gamma  = -abs(_series_float(item.put_abs_gamma, idx, default=0.0))
net_gamma  = _series_float(item.net_gamma, idx, default=call_gamma + put_gamma)
gex_abs_total += abs(call_gamma) + abs(put_gamma)
```

Three things to note:

1. The vendor's `netGamma` is taken **as-is** when present — `gex_job.py`'s
   formula is **never re-applied**. There is no spot²-multiplier step,
   no 0.01 scaling. The vendor's number is treated as authoritative.
2. `put_gamma` is forced negative (`-abs(...)`) regardless of how the
   vendor signs it — to match the SqueezeMetrics sign convention.
3. `gex_abs` is computed as the sum of call and put gamma magnitudes,
   **not** as the sum of per-option `|GEX|`. This is a different
   definition than `gex_job`'s `gex_abs`.

### Worked numeric example

SPX at 5500, single 5500-strike call, OI = 1000, γ = 0.005 per share,
multiplier = 100, `gex_puts_negative = True`.

- **SqueezeMetrics convention (hand-computed):**
  `1000 × 0.005 × 100 × 5500² × 0.01 = $151,250,000` per 1% move.

- **`gex_job.py` for this single call:**
  `(+1) × 0.01 × 0.005 × 1000 × 100 × 5500² = $151,250,000` ✓ matches.

- **`gex_job.py` if a 5500-strike put with same OI/γ is added:**
  call `+$151.25 M`, put `−$151.25 M`, **net = 0** (calls and puts cancel
  for a balanced ATM strike), **abs = $302.5 M**.

- **`cboe_gex_job.py` for the same legs (assuming vendor reported
  `call_abs_gamma = 151_250_000` and `put_abs_gamma = 151_250_000`):**
  net = 0, `gex_abs_total = 302_500_000`. Dimensionally consistent.

**Critical caveat — observed in production:** `Q18` shows that for the
same `(day, underlying, expiration)`, TRADIER `gex_net` is on the order
of **hundreds to thousands of trillions of dollars** while CBOE
`gex_net` for the same row is in the **hundreds of millions**.
The smallest observed magnitude ratio in the top-10-divergence sample is
**~7,100×**; the largest is **~19,400×**. See **C1** for the dispatched
investigation and proposed fix path.

---

## Findings — Critical (C1)

### C1 — Cross-source GEX magnitude divergence (TRADIER vs CBOE ~10⁴×)

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Wave** | 1 |
| **Status** | open |
| **Evidence** | DB query Q18 (single read-only session, 2026-04-19) |

**Description.** For the same `(day, underlying, expiration)` row,
`gex_snapshots.gex_net` written with `source = 'TRADIER'` is between
~7,100× and ~19,400× larger in absolute value than the row written with
`source = 'CBOE'`. Top sample (`SPX`, `2026-03-19` expiry, 5-min bucket
join):

| day | und | exp | gex_net_tradier | gex_net_cboe | abs_diff | zg_tradier | zg_cboe |
|-----|-----|-----|----------------:|-------------:|---------:|-----------:|--------:|
| 2026-03-19 | SPX | 2026-03-20 | −3.83 × 10¹² | −1.98 × 10⁸ | 3.83 × 10¹² | 6,235 | 200 |
| 2026-03-19 | SPX | 2026-03-20 | −3.73 × 10¹² | −2.34 × 10⁸ | 3.73 × 10¹² | (null) | 200 |
| 2026-04-08 | SPX | 2026-04-08 | +3.43 × 10¹² | +4.83 × 10⁸ | 3.43 × 10¹² | 6,400 | 2,600 |
| 2026-04-08 | SPX | 2026-04-08 | +3.42 × 10¹² | +4.85 × 10⁸ | 3.42 × 10¹² | 6,400 | 2,600 |
| 2026-04-08 | SPX | 2026-04-08 | +3.42 × 10¹² | +4.86 × 10⁸ | 3.42 × 10¹² | 6,395 | 2,600 |

(the full top-10 sample is in `Q18`; magnitudes vary but the ratio is
remarkably consistent.)

The `zero_gamma_level` divergence is independently suspicious: TRADIER
reports SPX zero-gamma at 6,235-6,400 (above-spot, plausible for a
short-gamma regime); CBOE reports 200-2,600. A SPX zero-gamma of 200 is
not a price level at all — it is either a different-units artifact or a
precomputation bug in `cboe_gex_job._zero_gamma_level`'s per-strike
walker.

**Impact.** Every consumer of `context_snapshots.gex_net` is reading
either a Tradier-overscaled number OR a CBOE-correctly-scaled number
depending on which writer landed last. Per the merge logic in
[`gex_job.py`](gex_job.py) (search hint: `COALESCE(gex_net_cboe, gex_net_tradier)`),
the canonical `context_snapshots.gex_net` prefers CBOE when present
(7,375 / 10,216 = 72.2% per Q25), but the Tradier-only rows
(2,841 / 10,216 = 27.8% per Q25 minus CBOE-present rows) carry the
overscaled magnitude. Decision-time gating that compares `gex_net`
against an absolute threshold (or a moving-window quantile that mixes
both sources) is silently regime-shifting based on which source happened
to be available.

**Hypothesized root causes (to verify in Wave 1):**

1. **Tradier units bug.** `gex_job.py` applies `× 100 × spot² × 0.01`
   to `option_chain_rows.gamma`. If Tradier's `greeks.gamma` is already
   per-contract (i.e. already includes the ×100 multiplier), the formula
   double-counts the multiplier — a 100× error. That alone does not
   explain the observed 10⁴× ratio, but would account for two orders.
2. **CBOE units mismatch.** mzdata's `netGamma` may not be a
   1%-move-scaled dollar value at all. If it is "raw gamma exposure"
   (no `S²` factor, no `0.01`), then CBOE values would be ~10⁴×
   smaller than the SqueezeMetrics convention — exactly the observed
   ratio for SPX with `S ≈ 5500` (since `0.01 × 5500² = 302,500`).
3. **Combination.** Both writers wrong in opposite directions; the
   ground-truth (per the SqueezeMetrics convention) is somewhere
   between.

**Proposed fix (Wave 1).** Three-step:

1. Pick **one external reference** (e.g. publicly visible SpotGamma SPX
   GEX heatmap for an in-sample date, or the SqueezeMetrics paper's
   own example) and hand-compute the expected GEX for one
   `(date, expiration)` to within 5%.
2. Decide which writer (TRADIER or CBOE) — or neither — is the
   ground-truth, and apply the corrective scaling on the wrong one.
   Add a unit-named field constant (e.g. `_GEX_UNIT_DOLLARS_PER_PCT_MOVE`)
   alongside the formula so reviewers can audit the math at a glance.
3. Backfill or invalidate the historical mismatched rows. If the
   correction is purely a constant scalar, an offline migration can
   `UPDATE gex_snapshots SET gex_net = gex_net / :scalar WHERE source = :source`;
   otherwise the affected `chain_snapshots`/`gex_snapshots` rows must
   be re-derived.

**Verification (Wave 1).** Add an end-to-end parity test in
[`backend/tests/test_gex_job.py`](../../tests/test_gex_job.py) and
[`backend/tests/test_cboe_gex_job.py`](../../tests/test_cboe_gex_job.py)
that builds a synthetic 1-call 1-put chain, computes GEX in both jobs,
and asserts both equal a hand-computed reference number to within 1%.

**Risk.** This **changes downstream `context_snapshots.gex_net` values**
historically and going forward. Coordinate with the operator: any
event-detection threshold tuned against the current `gex_net` series
(e.g. dealer-flip-triggered SKIPs, ranked-pruning gates) will need to
be re-tuned after the scalar correction. Stage on a sandbox first, then
backfill.

---

## Findings — High (H1–H9)

### H1 — `snapshot_job` partial-batch commit semantics

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 |
| **Status** | open |

**Description.** [`snapshot_job.py:406-530`](snapshot_job.py) wraps each
expiration's option-row insert in `async with session.begin_nested()`
(savepoint), all sharing one outer transaction that commits at the end
of the run. A failure on expiration N does **not** roll back the
already-savepoint-committed expirations 1..N-1. There is no run-level
"all expirations succeeded" gate, and `decision_job._get_latest_snapshot_for_dte`
loads a snapshot per `target_dte` independently — so the consumer cannot
detect that a snapshot at a given `target_dte` is missing because its
sibling expirations failed.

**Impact.** Decisions can proceed using a snapshot mix in which some
DTEs are fresh (~5 min old) and others are stale (last full success).
The portfolio-manager spread ranker assumes a coherent chain across
DTEs.

**Proposed fix.** Add a run-level completeness counter (expected vs
inserted expirations) and either (a) emit an alert when partial, or (b)
short-circuit `decision_job` SKIP on partial-snapshot detection. The
all-or-nothing transactional rewrite is a larger change; the alert path
is the cheaper Wave-2 fix.

### H2 — `snapshot_job` empty-strike-set creates "looks complete" empty snapshots

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 |
| **Status** | open |
| **Evidence** | DB query Q23 |

**Description.** [`snapshot_job.py:430-457`](snapshot_job.py) initializes
`selected_strikes: set[float] | None = None`. When `spot is not None` and
`strikes_each_side > 0`, the code calls `_select_strikes_near_spot(...)` —
which can return an empty set if the chain is malformed. The resulting
`selected_strikes = {}` is **not** `None`, so the loop's
`if selected_strikes is not None: continue if strike not in selected_strikes`
filters out **every** option, but the parent `chain_snapshots` row was
already inserted in the surrounding savepoint.

**Live evidence.** Q23 shows **52,004 `chain_snapshots` rows with zero
child `option_chain_rows`** (out of 90,443 total = 57.5%). The vast
majority are CBOE-by-design (`cboe_gex_job` writes `chain_snapshots` with
`source = 'CBOE'` but never populates `option_chain_rows` — see M1), but
~222 are TRADIER snapshots that fell into this trap. Earliest empty:
`2026-02-18`; latest: `2026-04-17` — i.e. it happens continuously, not
once.

**Proposed fix.** Treat empty `selected_strikes` like "filter
unavailable":

```python
if selected_strikes == set():
    selected_strikes = None  # ingest full chain; empty filter is a bug, not a policy
```

…or abort the per-expiration savepoint with a `failed_items` entry
(mirroring the existing `empty_chain` handling).

### H3 — VIX has zero TRADIER GEX coverage

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 1 |
| **Status** | open |
| **Evidence** | DB query Q19 |

**Description.** Q19 lists every `(day, underlying)` where one source
has zero rows. The result is 100% **VIX TRADIER = 0** every trading
day in the captured window (2026-04-06 .. 2026-04-17), while CBOE
writes 770-870 VIX rows per day. Either VIX is excluded from
`settings.gex_underlyings_list()` (config-side gap), or
`tradier_client.get_option_chain("VIX", ...)` does not return greeks
that survive `gex_job._compute_gex_for_snapshot`'s gamma/OI filter.

**Impact.** Operators or strategies that compare TRADIER vs CBOE for
VIX cross-validation receive nothing for one side. Any cross-check
falls open. If CBOE is later down for VIX, there is no fallback.

**Proposed fix.** Verify whether the operator wants VIX in the
TRADIER-derived GEX set (commonly yes, for cross-validation). If yes,
add `"VIX"` to `settings.gex_underlyings_list()` and verify
`get_option_chain("VIX", ...)` returns gamma. If no, suppress the Q19
alert path and document the deliberate gap in the runbook.

### H4 — `gex_snapshots.zero_gamma_level` 25% NULL on TRADIER

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 1 |
| **Status** | open |
| **Evidence** | DB query Q17 |

**Description.** Per Q17:

```
source   | total  | zero_gamma_null
---------+--------+----------------
CBOE     | 51782  |  3317  (6.4%)
TRADIER  | 37897  |  9541  (25.2%)
```

`gex_job._zero_gamma_level` walks sorted strikes and interpolates the
first sign change of cumulative `gex_calls + gex_puts`. A NULL means
no sign change was found within the windowed strike range — which
should be rare for a normal SPX/SPY chain spanning spot. **One in
four** TRADIER snapshots not having a zero-gamma is a strong indicator
that the windowing (`gex_strike_limit = 150` strikes near spot) is
too narrow when dealer GEX sign changes outside the window — or that
the per-strike sign convention is buggy in some subset.

**Impact.** Decision-time event detection that uses
`zero_gamma_level` distance-from-spot is silently dropping a quarter
of all decisions on stale-or-null data.

**Proposed fix.** Two-step:

1. Add a one-shot diagnostic test that computes `zero_gamma_level` on
   the full chain (no strike-limit) for a sample of NULL-zero-gamma
   snapshots and checks whether widening the window resolves it.
2. If yes (likely): widen `gex_strike_limit` or compute zero-gamma on
   the full chain even when the aggregate is restricted (the two
   numbers don't have to share the same windowing).

### H5 — `underlying_quotes` lacks UNIQUE constraint on natural key

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 |
| **Status** | open |
| **Evidence** | DB query Q14, schema query Q2 |

**Description.** Schema:

```sql
CREATE TABLE underlying_quotes (
  quote_id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  ...
);
CREATE INDEX idx_underlying_quotes_symbol_ts ON underlying_quotes (symbol, ts DESC);
```

There is **no UNIQUE constraint** on `(symbol, ts)` (or on any
business key). [`quote_job.py:91-128`](quote_job.py) issues a plain
`INSERT` (no `ON CONFLICT`). On retry after a transient failure, two
rows for the same `(symbol, ts)` can be inserted.

**Live evidence.** Q14 returned 10+ `(symbol, minute_bucket)` pairs
on 2026-04-06 with `count = 2`, across SKEW, SPY, VIX, SPX, VIX9D —
i.e. multiple symbols had near-duplicate ticks within the same
calendar minute. Some of these are the legitimate
`quote_interval_minutes = 5`-pattern boundary ticks; others appear to
be retry-driven double-inserts.

**Proposed fix.** Decide on the natural key (likely
`(symbol, ts)` rounded to the source quote-time, OR a `(symbol, ts,
source)` triple), add a `UNIQUE INDEX`, switch to
`INSERT ... ON CONFLICT DO UPDATE` (refresh) or `DO NOTHING` (skip
duplicates).

### H6 — `underlying_quotes.ts` is ingestion time, not vendor quote time

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 |
| **Status** | open |

**Description.** [`quote_job.py:99-113`](quote_job.py):

```python
now_et = datetime.now(tz=tz)
...
"ts": now_et.astimezone(utc),
```

The Tradier quote payload contains `last_trade_time` / `quote_time`
fields (per Tradier API docs), but the job stores **its own
clock-now** as `ts`. Downstream consumers (snapshot-vs-quote join in
`decision_job._get_spot_price`, training-time alignment in
`generate_training_data.py`) treat `ts` as "market time" — so a
delayed Tradier response shifts the join key in unexpected ways.

**Impact.** Slightly wrong quote-vs-snapshot alignment; can corrupt
training labels at the seconds-to-minutes scale, especially around
fast-moving market events.

**Proposed fix.** Persist Tradier's vendor timestamp explicitly (new
column `vendor_ts TIMESTAMPTZ NULL`, or repurpose `ts` to vendor time
and add `ingest_ts` for the clock-now). Either way, document the
semantics in the schema comment.

### H7 — `staleness_monitor` cooldown is in-process; restart re-spams alerts

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 3 |
| **Status** | open |

**Description.**
[`staleness_monitor_job.py:27-28,79-84,176-182`](staleness_monitor_job.py)
stores `_last_alert_ts` as an instance field on the
`StalenessMonitorJob` dataclass. A process restart resets the field,
so cooldown is forgotten — flaky deploys or a restart during an
incident can re-fire SendGrid emails immediately. This differs from
[`services/alerts.py`](../services/alerts.py)'s module-level dict
(also in-process, but at least centralized so multiple jobs share
state).

**Impact.** Operator alert fatigue and wasted SendGrid quota during
incidents that overlap with deploys.

**Proposed fix.** Persist the cooldown in DB (a tiny
`alert_cooldowns` table keyed by `cooldown_key`) or route through
`services/alerts.send_alert` (see also L9 / Wave 4 refactor M19).

### H8 — `staleness_monitor` checks global MAX(ts), not per-underlying

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 3 |
| **Status** | open |
| **Evidence** | DB query Q19 (multi-underlying), Q5 (multi-symbol underlying_quotes) |

**Description.** [`staleness_monitor_job.py:45-56`](staleness_monitor_job.py)
issues `SELECT MAX(ts) FROM <table>` for each watched table. One
fresh underlying can mask another stale one. This codebase **is**
multi-underlying: Q5 confirms `underlying_quotes` carries SPX, SPY,
VIX, VIX9D, SKEW, VVIX; Q19 confirms `gex_snapshots` carries SPX,
SPY, VIX. A SPX-only freshness pass would report green even if
all VIX rows are 24h old.

**Impact.** Staleness alerts can silently fail for any single
underlying that goes dark while other underlyings keep flowing.

**Proposed fix.** `GROUP BY` the expected underlying list in the
freshness query, alert on any breach. Drive the expected list from
`settings.<job>_underlyings_list()` so an explicit config edit is the
only way to reduce coverage.

### H9 — `download_databento` skips truncated/corrupt files as success

| Field | Value |
|-------|-------|
| **Severity** | HIGH (offline pipeline only) |
| **Wave** | 6 |
| **Status** | open |

**Description.** Batch mode in
[`download_databento.py`](../../scripts/download_databento.py)'s
`get_existing_dates` only checks that `YYYYMMDD.dbn.zst` exists.
Streaming mode skips if `out_path.exists()`. There is no size check,
checksum, or DBN decode pass before skip. A partial download
(network drop mid-write, disk-full, killed mid-job) can leave a
corrupt file on disk that is permanently skipped on rerun. The
`--verify-dbn` flag only checks **calendar coverage**, not file
integrity.

Adjacent issue: a streaming `df.to_parquet(out_path)` is non-atomic;
an interrupt mid-write leaves a partial Parquet that the next run
also skips.

**Impact.** Silently poisons offline training and backtest data with
no visible error, until the bad file is decoded mid-pipeline (often
days later).

**Proposed fix.** Three-step:

1. Stream-write to `<out_path>.tmp`, then `os.replace` to
   `<out_path>` only after the bytes are flushed.
2. Add a one-shot `databento.read_dbn(...)` decode-probe before
   declaring a file "exists".
3. On corrupt detection, log + delete + treat as missing for the
   next run.

---

## Findings — Medium (M1–M19)

### M1 — `chain_snapshots` carries 51K empty CBOE rows by design (~57% of table)

`cboe_gex_job` writes `chain_snapshots` with `source = 'CBOE'` to satisfy
the FK on `gex_snapshots`, but never populates `option_chain_rows`. Q23
confirms 52,004 / 90,443 = 57.5% of `chain_snapshots` rows have zero
child option rows. ~51,782 of those are CBOE-by-design (Q17 source
counts). The table-name "chain_snapshots" misleads any reader who
expects a chain to have legs. **Proposed fix:** rename to
`gex_anchor_snapshots` (and shim the old name), OR add a `payload_kind`
column so consumers can filter `WHERE payload_kind = 'options_chain'`
without an expensive child-count join. Deferred to Wave 4 (refactor).

### M2 — 28% of `context_snapshots` rows have NULL `gex_net`

Q25 captured: `total = 10,216`, `gex_net_present = 7,375` → 27.8% NULL.
`gex_net_cboe_present = 6,054` (40.7% NULL on CBOE column),
`gex_net_tradier_present = 7,006` (31.4% NULL on TRADIER column). When
both CBOE and TRADIER columns are NULL, the canonical `gex_net` falls
through to NULL. Decision-time gating that requires a non-NULL
`gex_net` SKIPs on these rows. **Proposed fix:** trace the upsert
sequencing in `cboe_gex_job` and `gex_job` to identify the
race/ordering that leaves both columns NULL on ~28% of context rows.
Wave 2.

### M3 — `snapshot_job` issues N+1 per-option INSERTs

[`snapshot_job.py:445-501`](snapshot_job.py) loops over options with one
`session.execute(text("INSERT INTO option_chain_rows ..."))` per option.
Average 280-300 options per snapshot × 1716-1738 snapshots/day = ~500K
single-row inserts/day. **Proposed fix:** batched multi-row INSERT
(SQLAlchemy `executemany` with page size 200-500), or COPY for the
fastest path. Wave 4.

### M4 — `cboe_gex_job` uses SELECT-then-INSERT for `chain_snapshots` (race)

[`cboe_gex_job.py:211-238,381-402`](cboe_gex_job.py) does a SELECT to
fetch the existing `snapshot_id` for `(ts, underlying, expiration,
source)`, then INSERT if missing. With the deployed
`uq_chain_snapshots_ts_und_exp_src` UNIQUE index, two concurrent CBOE
runs (or a CBOE run racing with `snapshot_job` on the same key) will
hit IntegrityError on the second writer. **Proposed fix:**
`INSERT ... ON CONFLICT DO NOTHING RETURNING snapshot_id` + fall back
to a SELECT only if no row returned. Wave 2.

### M5 — `gex_job` aggregates only top-N strikes within DTE-cap (vendor-incomparable)

`gex_max_dte_days = 10` and `gex_strike_limit = 150` (defaults) make
`gex_job`'s aggregate a "near-spot, short-dated" GEX — not directly
comparable to a vendor "headline" GEX number. This is a deliberate
performance/relevance tradeoff but is undocumented in the
`gex_snapshots.method` field and the public API. **Proposed fix:**
encode the windowing in the `method` string (e.g. `"oi_gamma_spot_top150_dte10"`
which the code already does, good ✓) and document in the README which
windowing the consumer should pick. Wave 4 (docs only).

### M6 — `ON CONFLICT` policy divergence: `gex_job` DO NOTHING vs `cboe_gex_job` DO UPDATE

`gex_job._persist_gex_snapshot` uses
`ON CONFLICT (snapshot_id) DO NOTHING`; `cboe_gex_job` uses
`DO UPDATE`. The intent (CBOE refreshes mid-day if vendor revises;
Tradier is "first write wins" because the chain is immutable for that
`snapshot_id`) is reasonable, but is undocumented and asymmetric with
the otherwise-shared schema. **Proposed fix:** add a one-line code
comment at each `ON CONFLICT` block explaining the policy choice; or
align both to `DO NOTHING` if mid-day vendor revision is not actually
desired. Wave 4.

### M7 — `economic_events` ON CONFLICT DO NOTHING — calendar updates don't propagate

[`eod_events_job.py:61-65`](eod_events_job.py) uses
`ON CONFLICT (date, event_type) DO NOTHING`. New rows insert fine, but
in-place updates to `has_projections` / `is_triple_witching` from the
authoritative Python tables (`generate_economic_calendar.py`) never
propagate to existing DB rows. **Proposed fix:** switch to
`DO UPDATE SET has_projections = EXCLUDED.has_projections, is_triple_witching = EXCLUDED.is_triple_witching`.
Wave 2.

### M8 — `option_chain_rows` count queries time out at 60s (operability)

Q16, Q21, Q22 (NULL-rate count, orphan-FK check, expiration-mismatch
count) all hit `QueryCanceledError: canceling statement due to
statement timeout` at the audit's 60s timeout against the 16 GB
`option_chain_rows` table. Live operations queries that need any
table-wide aggregate cannot be answered cheaply. **Proposed fix:**
add a denormalized `option_chain_rows_stats` daily-rollup table
(populated by `retention_job` or a new tiny job), OR add functional
indexes on the column-NULL predicates that ops needs. Wave 4.

### M9 — `eod_events_job` mutates `sys.path` to import from `scripts/`

[`eod_events_job.py:27-35`](eod_events_job.py) prepends
`backend/scripts` to `sys.path` so it can import `generate_rows` from
`generate_economic_calendar`. This shares a single source of truth
with the CSV generator script — but couples runtime to filesystem
layout and risks module-name collisions. **Proposed fix:** move
`generate_rows()` (or a thin wrapper) into
`spx_backend/services/economic_calendar.py` and import normally; keep
the script as a CLI entry that calls the same function. Wave 4
(refactor).

### M10 — `cboe_gex_job` silent fallthrough on schema/shape drift

[`cboe_gex_job.py:119-147`](cboe_gex_job.py)'s
`_normalize_exposure_items` returns `[]` if `payload.get("data")` is
not a list — i.e. a vendor schema rename or partial outage produces a
silent `no_exposure_items` skip rather than a structured alert.
**Proposed fix:** add a structured-log metric `vendor_schema_drift`
and alert when an HTTP 200 produces zero rows. Wave 3.

### M11 — `gex_job` does not check `chain_snapshots` age

[`gex_job.py:377-401`](gex_job.py) enforces
`gex_spot_max_age_seconds` against the underlying-quote age but does
not check the `chain_snapshots.ts` against any threshold. A stale
chain (e.g. `snapshot_job` was down for an hour) will still be picked
up by the `EXISTS`-on-`option_chain_rows` query and processed,
producing a stale GEX snapshot. **Proposed fix:** add an optional
`gex_chain_max_age_seconds` setting and skip stale chains explicitly
(loud SKIP, not a silent compute). Wave 2.

### M12 — `cboe_gex_job` partial-fetch failures don't fire scheduler alerts

[`cboe_gex_job.py:302-310`](cboe_gex_job.py) catches per-underlying
`exposure_fetch_failed` and returns `result["skipped"] = True` instead
of re-raising. APScheduler's job-failure email path only triggers on
uncaught exceptions, so a sustained CBOE outage looks fine to the
scheduler. Combined with H8 (global `MAX(ts)` staleness mask),
operators can miss multi-hour CBOE outages until a downstream consumer
SKIPs. **Proposed fix:** add a CBOE-specific freshness gate
(`SELECT MAX(ts) FROM gex_snapshots WHERE source = 'CBOE'`) to the
staleness monitor (paired with H8 fix). Wave 3.

### M13 — `quote_job` swallows fetch failures into log-only `quote_fetch_failed`

[`quote_job.py:77-82`](quote_job.py) catches all fetch exceptions and
returns `{"skipped": True, "reason": "quote_fetch_failed"}`. APScheduler
sees success. Operators only learn via log noise. **Proposed fix:** add
a per-symbol failure counter that, after N consecutive failures, raises
out of the job and lets the scheduler email path fire. Wave 3.

### M14 — `snapshot_job` writes Tradier data with no field validation

[`snapshot_job.py:477-500`](snapshot_job.py) coerces and writes Tradier
fields as-is: no check for `bid > ask`, negative IV, missing greeks
beyond the bare type-coercion. Bad data lands in DB and downstream
consumers (`gex_job`, `decision_job`, training pipelines) each apply
their own filtering. **Proposed fix:** centralize validation in a
shared `option_row_sanitizer.py` and reuse from both `snapshot_job` and
`decision_job`. Wave 4.

### M15 — `download_databento` cost amplification on rerun

When any trading day is missing, `_download_batch` calls `submit_job`
with the **full** `start`/`end` window (not just the missing days).
Cached days don't re-download locally, but the operator pays for
another batch job over the entire range. Combined with the lack of
pre-download cost estimation, this can produce surprise bills.
**Proposed fix:** detect the missing-days set, call `submit_job` only
on that contiguous (or chunked) range, and print an estimated cost
before the call. Wave 6.

### M16 — `download_databento` no disk-space guardrail

No `shutil.disk_usage` check before large batch downloads or Parquet
writes. The `data/` tree is already 121 GB per the offline audit; a
single OPRA day can be many GB. **Proposed fix:** add a pre-flight
`if shutil.disk_usage(data_root).free < estimated_size: abort()`.
Wave 6.

### M17 — `download_databento` date-range validation missing

[`download_databento.py`](../../scripts/download_databento.py)'s
`main()` does not assert `start < end` or that `end` is in the past.
Bad ranges fail softly (empty trading-day list, possible odd batch
behavior) rather than a clear validation error. **Proposed fix:** add
explicit `argparse`-time validation. Wave 6.

### M18 — `option_right` can be NULL → silent decision-time skip

[`snapshot_job.py:73-82`](snapshot_job.py)'s `_normalize_option_right`
can return `None`; the row still inserts because the schema allows
`option_right TEXT NULL`. Decision-time queries filter
`WHERE option_right = :right`, silently dropping these rows.
**Proposed fix:** skip the row at ingest if the right is not normalizable,
or infer from the option-symbol root with a documented rule. Wave 2.

### M19 — `staleness_monitor` SendGrid silent-fail can spam logs / re-attempt every tick

[`staleness_monitor_job.py:124-134,177-180`](staleness_monitor_job.py):
on send failure, the function logs and returns `False`; cooldown
`_last_alert_ts` is **only** advanced on successful `alerted = True`
sends. If SendGrid is failing, the job will retry every interval —
log noise plus repeated upstream calls during the outage.
**Proposed fix:** advance a separate `_last_attempt_ts` even on
failure; or migrate to `services/alerts.send_alert` which already
has back-off semantics. Wave 4 (folds into M19→`alerts.py` migration).

---

## Findings — Low (L1–L10)

### L1 — `TradierClient.token` field has no `repr=False`

[`tradier_client.py:32-43`](../ingestion/tradier_client.py): the
dataclass field is `token: str`. Accidental `repr(client)` or
structured logging of the object would leak the bearer token.
**Fix:** `token: str = field(repr=False)`. Wave 4.

### L2 — Scheduler docstring drift on `allow_quotes_outside_rth` semantics

[`scheduler_builder.py:232-234`](../scheduler_builder.py) says "the
job's internal RTH check still applies" for `allow_outside_rth`, but
`quote_job` does not apply that check when
`settings.allow_quotes_outside_rth = True`. Minor doc inconsistency.
**Fix:** rename or rewrite the comment. Wave 4.

### L3 — `cboe_gex_job` vendor timestamp parsing assumes UTC offset present

[`cboe_gex_job.py:72-85`](cboe_gex_job.py)'s `_parse_payload_timestamp`
calls `datetime.fromisoformat(...)`. If mzdata ever returns a naive ET
string (no offset), `fromisoformat` will produce a naive datetime that
is then `.astimezone(utc)`'d incorrectly. **Fix:** explicit ET
fallback when `tzinfo is None`. Wave 4.

### L4 — `retention_job` doesn't VACUUM/ANALYZE explicitly

[`retention_job.py`](retention_job.py) relies on autovacuum after a
purge. On large deletes, autovacuum may take hours to reclaim the
space. **Fix:** optional `ANALYZE chain_snapshots` after the purge
loop; document reliance on autovacuum tuning. Wave 4.

### L5 — `retention_job` log granularity is aggregate-only

`retention_job.py:70-73` logs `deleted_snapshots` (parent count) and
the cutoff timestamp, but not per-table child counts (which cascade).
Hard to verify cascade correctness from logs alone. **Fix:** add a
debug-level per-table count. Wave 4.

### L6 — `event_signals.py` docstring references non-existent `cboe_gex_job._latest_vols`

[`event_signals.py:51-52`](../services/event_signals.py)'s module
docstring names `cboe_gex_job._latest_vols`, which does not exist.
**Fix:** rewrite to name the actual writers
(`quote_job.compute_market_context`, `cboe_gex_job._aggregate_underlying_results`).
Wave 4.

### L7 — `eod_events_job` default 16:30 vs "4:00/4:15 close" expectation

[`config.py:106-108`](../config.py) defaults the EOD job to 16:30 ET,
not 16:00 or 16:15. This is fine for a **static** calendar seeder, but
the "EOD" naming implies cash-session alignment. **Fix:** rename or
clarify in the docstring. Wave 4.

### L8 — `EodEventsJob.run_once(force=...)` is a no-op

[`eod_events_job.py:47-54`](eod_events_job.py): `force` is accepted
"for interface consistency" but unused. Either honor it (to allow
manual reseeding) or stop threading it through. **Fix:** decide and
align. Wave 4.

### L9 — `cboe_gex_job` per-strike N+1 INSERTs

[`cboe_gex_job.py:495-551`](cboe_gex_job.py): per expiration, two
`INSERT ... ON CONFLICT` statements per strike → 2N round-trips inside
the nested transaction. Performance issue at scale (the deployed
`gex_by_expiry_strike` is 15.4M rows / 9.6 GB per Q4). **Fix:** batched
`executemany` or COPY. Wave 4 (paired with M3).

### L10 — `download_databento` retry policy unclear

The script delegates HTTP retry to the `databento` Python client; this
file does not document or assert what retry/backoff the client
applies. **Fix:** add a one-line comment naming the
client-version-dependent behavior, or wrap the call in an explicit
`tenacity` decorator with a documented policy. Wave 6.

---

## Refactor opportunities

These are not findings (no concrete bug) — they are sketched directions
the Wave 4 work should consider when implementing the fixes above.

1. **File splits.** [`cboe_gex_job.py`](cboe_gex_job.py) (759 lines)
   and [`snapshot_job.py`](snapshot_job.py) (613 lines) are both above
   the 500-line "consider splitting" threshold the offline audit
   adopted. Candidates: (a) move payload-normalization helpers
   (`_to_int`, `_to_float`, `_to_date`, `_normalize_option_right`,
   `_select_strikes_near_spot`, `_parse_chain_options`,
   `_parse_payload_timestamp`, etc.) into
   `backend/spx_backend/ingestion/parsers.py`; (b) split
   `cboe_gex_job` into `cboe_gex/job.py` (orchestration),
   `cboe_gex/normalize.py` (vendor parsing), `cboe_gex/persist.py`
   (DB writes).

2. **Extract common chain-snapshot helpers.** `gex_job` and
   `cboe_gex_job` both need `_get_or_insert_chain_snapshot_id(...)`
   logic; they currently each re-implement it (with slightly different
   `ON CONFLICT` policies — see M4, M6). Extract to
   `backend/spx_backend/jobs/_chain_snapshot_dao.py`.

3. **Migrate `staleness_monitor` to `services/alerts.py`.**
   [`services/alerts.py`](../services/alerts.py)'s header explicitly
   names staleness as a future migration target. Folds H7, M19, L9
   together. Already in the v1 backlog (parent plan item 8).

4. **Centralize option-row sanitization** (paired with M14). One
   `option_row_sanitizer.py` shared by `snapshot_job` (ingest-time
   write filter) and `decision_job` (read-time candidate filter).

5. **Centralize GEX math** (paired with C1). The SqueezeMetrics
   formula belongs in `backend/spx_backend/services/gex_math.py` with
   one canonical entry point used by both `gex_job` (per-strike) and
   any reconciliation tooling. The CBOE path then calls a thin
   `_apply_vendor_units(vendor_value, spot)` adapter so the
   correction lives in exactly one place.

---

## Wave plan

| Wave | Findings | Files touched | Risk |
|------|----------|---------------|------|
| 0 | (this doc) | `backend/spx_backend/jobs/INGEST_AUDIT.md` | none |
| 1 | C1, H3, H4 | `gex_job.py`, `cboe_gex_job.py`, new `services/gex_math.py`, `tests/test_gex_job.py`, `tests/test_cboe_gex_job.py`, optional one-shot migration script | **HIGH** — C1 changes downstream `gex_net` magnitudes; coordinate with operator |
| 2 | H1, H2, H5, H6, M2, M4, M7, M11, M18 | `snapshot_job.py`, `quote_job.py`, `eod_events_job.py`, `cboe_gex_job.py`, `gex_job.py`, new SQL migration for `underlying_quotes` UNIQUE | MEDIUM |
| 3 | H7, H8, M10, M12, M13 | `staleness_monitor_job.py`, `cboe_gex_job.py`, `quote_job.py`, alerts wiring | LOW |
| 4 | M1, M3, M5, M6, M8, M9, M14, M19, L1-L9, all refactor opportunities | Various ingest jobs, new `services/gex_math.py`, `services/economic_calendar.py`, `ingestion/parsers.py`, `_chain_snapshot_dao.py`, possible `alert_cooldowns` table | MEDIUM (large diff; mostly mechanical) |
| 5 | (test coverage) | new `tests/test_quote_job_dedup.py`, extended `tests/test_gex_job.py` (target ≥80% coverage), new `tests/test_cboe_gex_job_parity.py`, freshness regression tests | LOW |
| 6 | H9, M15, M16, M17, L10 | `backend/scripts/download_databento.py` (offline; no live runtime impact) | LOW |

### Wave-by-wave intent

- **Wave 1** is the only wave that materially changes downstream
  numbers. Stage on a sandbox first.
- **Wave 2** is mechanical idempotency + freshness gates; no behavior
  change beyond eliminating the dedup window.
- **Wave 3** is observability; new alerts may be loud at first, tune
  thresholds in the same PR.
- **Wave 4** is code-hygiene-heavy; bytecode-regression-style
  guarantees (per the offline audit's monolith-split pattern) apply
  where helpers move.
- **Wave 5** brings `gex_job.py` from 33% → ≥80% coverage; closes the
  parent plan's coverage gap.
- **Wave 6** is fully offline / no-runtime-risk and can land
  independently.

---

## Verification checklist

### Wave 1
- [ ] One hand-computed external-reference GEX number for a pinned
  `(date, underlying, expiration)` matches `gex_job`'s output to
  within 1% AND `cboe_gex_job`'s output to within 5%.
- [ ] `pytest backend/tests/test_gex_job.py::test_gex_matches_squeeze_metrics_reference -v` passes.
- [ ] `pytest backend/tests/test_cboe_gex_job.py::test_cboe_gex_matches_squeeze_metrics_reference -v` passes.
- [ ] Q19 re-run shows `tradier_rows > 0` for VIX (or a documented
  config-side opt-out is recorded in the runbook).
- [ ] Q17 re-run shows TRADIER `zero_gamma_null` rate dropped below
  10%.

### Wave 2
- [ ] Re-running Q14 returns no near-duplicates within a minute for
  any symbol.
- [ ] `\d+ underlying_quotes` shows the new UNIQUE constraint.
- [ ] Re-running Q23 shows the empty-`chain_snapshots` count
  dropping (CBOE-by-design empties remain; Tradier empties go to
  zero).
- [ ] `pytest backend/tests/test_snapshot_job.py::test_partial_batch_alerts -v` passes.
- [ ] An operator-driven re-seed of `economic_events` mutates
  `has_projections` for an existing row.

### Wave 3
- [ ] Sustained 1-hour CBOE outage on a sandbox triggers the new
  CBOE-specific staleness alert AND the global one (independent
  paths).
- [ ] Multi-underlying staleness check group-by detects a single-VIX
  outage while SPX is fresh.
- [ ] `staleness_monitor` cooldown survives a restart (DB-backed).

### Wave 4
- [ ] No public symbol moved during refactor changes its bytecode
  signature (run `tools/monolith_split_regression.py` per the
  offline-audit precedent).
- [ ] `pytest backend/tests/` green.
- [ ] `python -m backend.spx_backend.services.gex_math --help`
  renders cleanly (CLI smoke per Track-B argparse-bug precedent).

### Wave 5
- [ ] `pytest --cov backend/spx_backend/jobs/gex_job.py` reports
  ≥80% coverage.
- [ ] New parity test is parametrized over at least one TRADIER and
  one CBOE pinned snapshot.

### Wave 6
- [ ] Killing `download_databento.py` mid-write leaves no `<file>`
  on disk (only `<file>.tmp`, then no file after cleanup).
- [ ] Re-running with a corrupt zst file (manually injected) detects
  + deletes + re-downloads.

---

## Open questions

1. **C1 ground truth.** Which writer is "right" — TRADIER, CBOE, or
   neither — pending an external SqueezeMetrics-style reference
   number for one pinned date? The audit cannot resolve this from
   code + DB alone.
2. **C1 backfill scope.** If the correction is a constant scalar
   per source, is an `UPDATE`-in-place migration acceptable, or
   does the operator prefer a "stamp out new column, leave history
   alone" approach?
3. **H3 VIX coverage.** Is VIX in `gex_underlyings_list()`
   intentionally absent (because Tradier's VIX chain is unreliable
   for greeks), or is this a config-side gap?
4. **H4 zero-gamma window.** Is the 25% NULL rate dominated by
   strike-window-too-narrow cases, or by genuinely sign-stable chains?
   A one-shot diagnostic on a sample of NULL snapshots will resolve.
5. **M1 chain_snapshots semantics.** Is the operator OK with a
   rename (`gex_anchor_snapshots`) + shim, or should the table stay
   as-is and consumers learn to filter on `payload_kind`?
6. **M2 ordering race.** Why do 28% of `context_snapshots` have
   neither `gex_net_cboe` nor `gex_net_tradier` populated even
   though both jobs are running? Tracing the upsert sequencing in
   live logs is the next step.
7. **H6 Tradier vendor timestamp.** Does Tradier `/markets/quotes`
   return a usable per-quote timestamp, or only a per-trade one?
   Affects whether the vendor-time persistence is straightforward.
8. **M5 windowing labels.** Does the operator want to expose
   "near-spot short-dated GEX" vs "headline GEX" as two separate
   columns, or keep the deliberate windowing as the canonical
   number?

---

## Glossary

- **GEX**: Gamma Exposure. The dollar change in dealer delta per a
  unit (or 1%) move in spot, summed across the option chain.
- **Dealer flip / zero-gamma level**: The spot price at which the
  strike-summed dealer GEX crosses zero. Above the flip, dealers are
  net long gamma (vol-suppressing); below, net short (vol-amplifying).
- **OI**: Open interest at a strike (the count of currently-open
  contracts).
- **Greeks**: Black-Scholes sensitivities; this audit cares mainly
  about γ (`gamma`), the second derivative of option price with
  respect to spot.
- **DTE**: Days to expiration (calendar). "Trading DTE" in
  `snapshot_job` is the **ordinal index** of the expiration in the
  Tradier-listed expiration set, not a holiday-aware business-day
  count — see M5 / Explorer 2 commentary.
- **OPEX**: Monthly options expiration day (3rd Friday).
- **RTH**: Regular Trading Hours (9:30-16:00 ET, Mon-Fri, excluding
  US equity holidays).
- **Idempotency**: Re-running an ingest with the same input produces
  the same DB state, with no duplicate rows.
- **CBOE GEX (vendor)**: GEX values precomputed and served by CBOE /
  mzdata, distinct from `gex_job`'s in-house Tradier-chain-derived
  computation.
- **SqueezeMetrics convention**: The dealer-flow GEX formula
  `OI × γ × multiplier × S² × 0.01` with calls signed positive and
  puts signed negative, popularized by SqueezeMetrics' "Gamma
  Exposure" white paper and widely adopted in the dealer-flow
  literature.
- **Tradier**: HTTP options-data vendor used by `quote_job`,
  `snapshot_job`, and (transitively) `gex_job`. Sandbox vs prod base
  URL is configurable per `settings.tradier_base_url`.
- **mzdata**: Public-data vendor (Deno-hosted) used by
  `cboe_gex_job` for CBOE GEX. No API key, free tier.

---

*Generated 2026-04-19 as part of Track 1 (live ingestion + GEX
correctness audit). Work-tracking plan IDs (Cursor session-local, not
checked in): `track-1-ingest-audit_a36f9f6d.plan.md` and the parent
`backend_e2e_tracks_v2_5614035a.plan.md` Track 1 section it supersedes.*

---

## Appendix: Wave 1 verification log (2026-04-19)

This appendix records the verification evidence for the Wave 1 PR
(closes findings **C1**, **H3**, **H4** plus refactor opp **#5**). Code
changes are described in the commit body; this section captures the
green-or-red status of every verification gate the Wave 1 plan listed
plus the live-DB baseline numbers used to evaluate the post-deploy
re-runs.

### Wave 1 changes shipped

| Area | Files |
|------|-------|
| New module | `backend/spx_backend/services/gex_math.py` (`compute_gex_per_strike`, `apply_vendor_units`) |
| GEX writers | `backend/spx_backend/jobs/gex_job.py` (TRADIER), `backend/spx_backend/jobs/cboe_gex_job.py` (CBOE) |
| Config | `backend/spx_backend/config.py` (drop 12 `vix_snapshot_*` settings, exclude VIX from `cboe_gex_underlyings` default), `.env.example` |
| Scheduler | `backend/spx_backend/scheduler_builder.py` (drop `vix_snapshot_job` wiring), `backend/spx_backend/jobs/snapshot_job.py` (drop `build_vix_snapshot_job`) |
| Migrations | `016_drop_vix_snapshots.sql` (cascade-DELETE), `017_correct_cboe_gex_units.sql` (×100 backfill) |
| Tests | `backend/tests/test_gex_math.py` (new), `backend/tests/test_gex_job.py` (5 tests + signature update), `backend/tests/test_cboe_gex_job.py` (VIX-skip + 100× scalar tests; existing VIX cases removed), `backend/tests/test_snapshot_helpers.py` (VIX cases removed), `backend/tests/test_app_scheduler_vix_snapshot.py` (VIX-specific cases removed) |
| Docs | `README.md` (mermaid diagram + env-var section), `frontend/src/api.ts` (comment), `backend/spx_backend/jobs/INGEST_AUDIT.md` (this appendix) |

### Verification gates

| Gate | Result |
|------|--------|
| `python3 -m pytest backend/tests/` (full backend suite) | **1082 passed, 15 skipped, 29 warnings** in 40.45s |
| `python3 -m pytest backend/tests/test_gex_job.py` (Wave 1 H4 tests) | **9 passed** in 0.76s |
| `python3 -m ruff check` on Wave-1-touched files | **1 pre-existing F841** in `test_cboe_gex_job.py:301` (introduced by commit `01ba9ca`, not by this PR; left in place per "only fix pre-existing lints if necessary") |
| `npm --prefix frontend run build` | **green** in 1.63s; `dist/` artifacts written |
| Bytecode regression on `gex_job` / `cboe_gex_job` | **Skipped — N/A**. The Wave 1 plan invoked `monolith_split_regression.py` to detect *unintended* drift in a refactor commit. Wave 1 is an *intentional* behavior change (new `_zero_gamma_level` signature, new `apply_vendor_units` adapter), so a snapshot diff would only re-state the intentional changes. Public-symbol stability was verified by inspection: `GexJob.run_once`, `CboeGexJob.run_once`, `compute_gex_per_strike` all retain their signatures. |

### Live-DB baseline (pre-migration, captured 2026-04-19)

Captured via the read-only `tmp/track1_phase6_postfix_probe.py` script
(asyncpg, `SET default_transaction_read_only = on`).

| Metric | Pre-migration value | Post-migration target | Notes |
|--------|---------------------|-----------------------|-------|
| `Q17` TRADIER zero_gamma NULL rate | **25.18%** (9541 / 37897) | **<10% on rows written after deploy** | 016 + 017 don't backfill `zero_gamma_level`; only the H4 widening in `gex_job.py` improves the rate going forward. Historical TRADIER rows stay NULL. |
| `Q17` CBOE zero_gamma NULL rate | 6.41% (3317 / 51782) | unchanged | Out of scope for Wave 1 (deferred to Wave 2 per audit). |
| `Q19` TRADIER VIX gex rows | **1553** | **0** | Migration 016 cascade-deletes via `chain_snapshots`. |
| `NEW1` CBOE VIX gex rows | 16054 | 0 | Same. |
| `NEW1` TRADIER VIX gex rows | 1553 | 0 | Same. |
| `NEW2` CBOE VIX chain_snapshots | 16054 | 0 | Same. |
| `NEW2` TRADIER VIX chain_snapshots | 2009 | 0 | Same. |
| `NEW3` VIX option_chain_rows | **208 216** | 0 | Cascade fan-out from `chain_snapshots` parents. |
| `NEW4` VIX `underlying_quotes` (PRESERVED) | **2841** | 2841 (unchanged) | Per user decision: keep the spot VIX index quotes; only drop snapshot/GEX. |

### C1 magnitude check (pre-migration baseline)

Single most-recent SPX snapshot per source, from `NEW5`/`NEW6` rows:

| Source | snapshot_id | spot | `gex_net` (pre-017) | After ×100 |
|--------|-------------|------|---------------------|------------|
| CBOE   | 91759       | 7126.25 | 587 867 862 (5.88e8) | 5.88e10 |
| TRADIER | 91743      | 7125.76 | 4 241 192 818 (4.24e9) | unchanged |

Note that the latest TRADIER snapshots span a wide range (4.5e8 to
9.4e9 across DTE buckets) because each `(target_dte)` produces its
own `snapshot_id`. The Phase 0 single-snapshot canonical recompute
pinned snapshot 91708 at 5.96e10 vs CBOE 5.879e8 → ratio 101.4×,
matching the contract-multiplier explanation. After migration 017,
CBOE per-snapshot `gex_net` values move into the 1e9-1e11 band
expected for SPX, and the cross-source parity check (a Wave 1 success
criterion) becomes meaningful.

> **Q18 caveat.** The Q18 query in the probe sums `gex_net` *across all
> intra-day snapshots* per day. TRADIER writes ~ten snapshots per day
> (multiple DTE buckets); CBOE writes one. The day-sum ratio is
> therefore not the same as the per-snapshot ratio. For the
> post-migration parity test, prefer single-snapshot pairs at matching
> `ts` like the Phase 0 probe used.

### Q23 finding to flag for Wave 2

`Q23` shows that **all 35 728 non-VIX CBOE chain_snapshots rows have
zero option_chain_rows children** (SPX: 18 414 empty / 18 414 total;
SPY: 17 314 / 17 314). This re-confirms audit finding M1: the CBOE
writer creates `chain_snapshots` parents purely as FK anchors for
`gex_snapshots`, and never populates `option_chain_rows` (mzdata
ships pre-aggregated GEX, not chain rows). Out of scope for Wave 1;
will be addressed in Wave 2 per the wave plan.

### Operator post-deploy checklist

The `git push` of Wave 1 alone is **not** sufficient. The deploy
operator must, in order:

1. Remove the legacy `VIX_SNAPSHOT_*` block (12 lines) from any
   environment whose `.env` predates this PR. The local repo `.env`
   was cleaned as part of this PR; production env vars on Railway
   need a manual sweep. Without this, the app will fail to start
   (Pydantic Settings v2.13 defaults to `extra="forbid"`).
2. Run the schema migrations at app boot (or manually via
   `await spx_backend.database.schema.init_db()` per the existing
   pattern). Order matters: 016 (drop VIX) must run before 017
   (×100 backfill) so 017 doesn't waste UPDATEs on rows that 016
   then drops. The migration filenames enforce this ordering.
3. After the first post-deploy `gex_job` cycle has written ~10
   snapshots, re-run `tmp/track1_phase6_postfix_probe.py` and confirm:
   - Q19 / NEW1 / NEW2 / NEW3 all return 0.
   - NEW4 returns the same count as the pre-migration baseline (no
     accidental VIX quotes deletion).
   - The H4 zero_gamma NULL rate trend on **post-deploy** TRADIER
     rows drops below 10% (track this on a moving window, since
     historical NULLs persist).
4. Operator should retain the pre-migration COUNTs above as evidence
   of the deletion scope — the historical VIX data is not
   recoverable post-016.

### Migration execution log (2026-04-19, ~18:06 ET, during NY market hours)

Migrations 016 + 017 were applied locally against Railway prod
(`switchyard.proxy.rlwy.net:22929/railway`) via a one-shot asyncpg
script (`tmp/run_wave1_migrations.py`, deleted after success). The
script ran each migration inside its own transaction with
`SET LOCAL statement_timeout = '600s'` and `SET LOCAL lock_timeout = '30s'`
to bypass the 5s/120s defaults baked into the migration files (those
two `SET` lines were stripped from the SQL before exec so the
session-level overrides stuck). A row was inserted into
`schema_migrations` only after the SQL succeeded, so the redeployed
app's `init_db()` will skip both migrations on next boot.

**Applied timestamps from `schema_migrations`:**

| Version | Applied at (UTC) | Wall-clock since script start |
| --- | --- | --- |
| `016_drop_vix_snapshots` | 2026-04-19 22:06:27 | ~14s |
| `017_correct_cboe_gex_units` | 2026-04-19 22:08:33 | ~2m 06s after 016 |

**Post-flight verification (read-only probe, immediately after):**

- `chain_snapshots WHERE underlying='VIX'` → **0** rows (was 4 — DBT,
  TRADIER, CBOE, mzdata).
- `option_chain_rows WHERE underlying='VIX'` → **0** rows (was ~6.7k).
- `gex_snapshots WHERE underlying='VIX'` → **0** rows (was 5).
- `underlying_quotes WHERE symbol='VIX'` → **2841** rows preserved
  (matches NEW4 baseline; VIX quotes are still consumed by event
  signals).
- Latest CBOE SPX `gex_snapshots` row (`snapshot_id=91759`,
  `ts=2026-04-17 19:59:37 UTC`):
  `gex_net = 5.879e10`, `gex_calls = 6.095e10`, `gex_puts = -2.166e9`,
  `gex_abs = 6.312e10` → exactly ~100× the pre-migration NEW5 baseline
  of `gex_net = 5.879e8`. Sign and direction preserved. ✓
- Latest CBOE SPY `gex_snapshots` row: `gex_net = 1.500e11`,
  consistent with SPY having ~10× SPX notional. ✓
- Total CBOE GEX snapshots with non-null `gex_net`: **35,728** (matches
  the cardinality bucket the migration `EXPLAIN` plan estimated).
- Total TRADIER GEX snapshots for cross-source sanity:
  **36,344** (similar order, as expected for the same trading days).

**Runner notes for future migrations of similar shape:**

- Python stdout buffering matters when the runner is invoked under a
  non-TTY (the first attempt looked hung for >700s but had actually
  finished; `python3 -u` is the fix). The actual SQL work for both
  migrations completed in ~2m 20s combined.
- The 30s `lock_timeout` was sufficient — neither migration blocked
  on contention even though it ran during NY market hours while
  `gex_job` was actively writing.
- 017 is the slower of the two by an order of magnitude (~2m vs ~14s),
  driven by the per-strike UPDATEs on `gex_by_strike` /
  `gex_by_expiry_strike`. If we ever re-run a similar bulk-multiply
  on a hotter table, the same `statement_timeout` override pattern
  is the recommended safety net.

### Probe script

The Phase 6 verification probe lived at
`tmp/track1_phase6_postfix_probe.py` during the Wave 1 PR session and
was deleted after capturing the baseline numbers above. Future
re-runs can recreate it from the queries documented in this appendix
(Q17/Q19/NEW1-6 and the Q18 caveat). Reuse the `_load_dotenv` /
`asyncpg.connect(dsn=...)` / `SET default_transaction_read_only = on`
pattern from the Phase 0 probe template.

The migration runner (`tmp/run_wave1_migrations.py`) and the
post-migration verifier (`tmp/verify_wave1_state.py`) followed the
same template and were both deleted after the migration log above
was captured.
