# Offline Pipeline Correctness Audit

Long-term reference for the 13 offline-pipeline scripts under `backend/scripts/` and the
121 GB `data/` tree. Captures the findings of a deep readonly audit performed
2026-04-16 and structures the fix work as Waves 0–5.

> **Status**: Waves 0-5 landed in commit `6093bf6` (2026-04-16). Audit
> documentation sweep landing in the gap-closure follow-up; small open
> findings (M3, M8, M9, M12) and the three monolith splits remain in
> progress. M5 and M11 deferred -- see "Future work" section.
> See [`.cursor/plans/track_b_offline_audit_fixes_73d0812a.plan.md`](../../.cursor/plans/track_b_offline_audit_fixes_73d0812a.plan.md)
> for the original work-tracking plan and
> [`.cursor/plans/offline-pipeline-gap-closure_d993da3d.plan.md`](../../.cursor/plans/offline-pipeline-gap-closure_d993da3d.plan.md)
> for the follow-up gap-closure plan.

> **Line-number convention.** All `file:NNN` citations in this document
> reflect the source state at audit time (commit predecessor of
> `6093bf6`). Files have shifted line counts as Waves 1-5 landed (and
> will shift more as the monolith splits land). Treat the file path as
> authoritative and the line number as a "search hint" -- locate the
> referenced symbol/string in the current file rather than trusting the
> exact line. The Future work section at the end of this doc tracks
> when this convention can be retired.

---

## Table of contents

1. [Methodology](#methodology)
2. [Surface area inventory](#surface-area-inventory)
3. [Findings — Critical (C1–C5)](#findings--critical-c1c5)
4. [Findings — High (H1–H7)](#findings--high-h1h7)
5. [Findings — Medium (M1–M12)](#findings--medium-m1m12)
6. [Findings — Low (L1–L10)](#findings--low-l1l10)
7. [Wave plan](#wave-plan)
8. [Verification checklist](#verification-checklist)
9. [Open questions](#open-questions)
10. [Glossary](#glossary)

---

## Methodology

The audit consumed the source line-by-line via six parallel readonly explorers,
each owning a tightly scoped area:

| Explorer | Files | Cross-references |
|----------|-------|------------------|
| 1. Production-DB exporter | `export_production_data.py`, `generate_economic_calendar.py`, `_env.py`, `_constants.py` | `db_schema.sql` (tables read by exporter), `data/*_export.csv`, `data/production_exports/` |
| 2. Training data generator | `generate_training_data.py` (3261 lines), `_label_helpers.py` | `db_schema.sql`, `pricing.py`, `options.py`, `modeling.extract_candidate_features`, `decision_job._create_trade_from_decision` |
| 3. Backtest layer | `backtest_strategy.py` (3177 lines), `backtest_entry.py` | `decision_job._run_portfolio_managed`, `portfolio_manager`, `pricing.py`, `event_signals.py` |
| 4. ML / optimizer | `xgb_model.py` (1568 lines), `upload_xgb_model.py`, `experiment_tracker.py`, `ingest_optimizer_results.py`, `run_pipeline.py` | `db_schema.sql` (`model_versions`, `optimizer_*`), `modeling.predict_xgb_entry` |
| 5. Regime + SL analysis | `regime_analysis.py`, `regime_utils.py`, `sl_recovery_analysis.py` | `decision_job` regime gating, `event_signals.py`, `db_schema.sql` |
| 6. External data + hygiene | `download_databento.py`, `data/` tree, `.gitignore` | none |

In total: ~456 KB / ~9000 lines of Python across 13 scripts; a 121 GB data
tree comprising 3,788 files (2,798 `.zst`, 954 `.parquet`, 21 `.csv`,
10 `.json`).

---

## Surface area inventory

### Scripts (13 in scope)

| File | Lines | Purpose |
|------|-------|---------|
| [`export_production_data.py`](export_production_data.py) | 529 | DB → CSV/Parquet export |
| [`generate_training_data.py`](generate_training_data.py) | 3261 | Candidate generation + labeling pipeline |
| [`backtest_strategy.py`](backtest_strategy.py) | 3177 | Portfolio simulator + optimizer + walk-forward |
| [`backtest_entry.py`](backtest_entry.py) | 322 | Walk-forward XGBoost entry-model evaluator |
| [`regime_analysis.py`](regime_analysis.py) | 822 | Performance slicing by regime dimensions |
| [`regime_utils.py`](regime_utils.py) | 156 | Threshold classifiers + backtest regime metrics |
| [`sl_recovery_analysis.py`](sl_recovery_analysis.py) | 509 | SL counterfactual + recovery stats |
| [`xgb_model.py`](xgb_model.py) | 1568 | XGBoost training / WF / persistence |
| [`upload_xgb_model.py`](upload_xgb_model.py) | 175 | Insert artifact into `model_versions` |
| [`experiment_tracker.py`](experiment_tracker.py) | 268 | File-based run tracking under `data/experiments/` |
| [`ingest_optimizer_results.py`](ingest_optimizer_results.py) | 309 | Optimizer CSV → `optimizer_*` tables |
| [`run_pipeline.py`](run_pipeline.py) | 343 | Subprocess orchestration of all stages |
| [`download_databento.py`](download_databento.py) | 612 | OPRA + DBEQ batch + streaming downloads |

### Data tree (121 GB / 3,788 files)

| Path | Size | Source |
|------|------|--------|
| `data/databento/` | 120 GB | External (Databento batch + streaming) |
| `data/firstratedata/` | 110 MB | External (FirstRateData converted from raw `.txt`) |
| `data/production_exports/chains/` | 35 MB | DB export (`chain_snapshots` + `option_chain_rows`) |
| `data/production_exports/underlying/` | 5 MB | DB export (`underlying_quotes`) |
| `data/training_candidates.csv` | 297 MB | `generate_training_data.py` output |
| `data/labels_cache/` | 61 MB / 336 files | `generate_training_data.py` per-day Parquet |
| `data/candidates_cache/` | 49 MB / 575 files | `generate_training_data.py` per-day Parquet |
| `data/experiments/` | 54 MB / 3 runs | `experiment_tracker.py` |
| `data/backtest_results.csv` | 54 MB | `backtest_strategy.py` optimizer default |
| `data/optimizer_event_only.csv` | 48 MB | `backtest_strategy.py --optimize-event-only` (likely stale) |
| `data/optimizer_event_only_v2*.csv` | 6 MB combined | `backtest_strategy.py` v2 grid |
| `data/optimizer_selective.csv` | 6 MB | `backtest_strategy.py --optimize-selective` |
| `data/context_snapshots_export.csv` | 1 MB | DB export |
| `data/underlying_quotes_export.csv` | 700 KB | DB export (currently unused downstream) |
| `data/economic_events_export.csv` | 6 KB | DB export (currently unused downstream) |
| `data/economic_calendar.csv` | 6 KB | `generate_economic_calendar.py` (hardcoded source) |
| `data/walkforward_results.csv` | 38 KB | `run_pipeline.py` (currently global path — see H4) |
| `data/pareto_frontier.csv` | 1 KB | `backtest_strategy.run_analysis` |
| `data/pipeline_log_*.json` | 1 KB | `run_pipeline.py` |
| `data/regime_results.csv`, `data/regime_report.csv` | 65 KB / 4 KB | `regime_analysis.py` |

The entire `data/` tree is gitignored via `.gitignore:37` (`data/`) so no large
artifact is tracked. `git ls-files data/` returns 0 files.

---

## Findings — Critical (C1–C5)

These are live-vs-offline parity bugs and model-leakage shaped issues that
either change live trading semantics or inflate reported model performance.
Wave 1 must address all five.

---

### C1 — `term_structure` orientation INVERTED in live event detector

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Wave** | 1 (immediate); permanent fix in 5 (shared evaluator) |
| **Status** | fixed in Wave 5 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

**Description.** The convention everywhere outside the live event detector is
`term_structure = vix9d / vix`. `term_inversion` then fires when the ratio
exceeds 1.0 (i.e. `vix9d > vix` — true backwardation / near-term stress).

But [`backend/spx_backend/services/event_signals.py:270`](../spx_backend/services/event_signals.py)
computes `term_structure = vix_now / latest_vix9d` (the reciprocal). With the
same threshold check at line 155 (`ts > self.term_inversion_threshold`,
default 1.0), the live signal fires when `vix > vix9d` — i.e. **contango / normal
market**. That is the opposite of what every other path treats as "inversion".

Convention sources confirming `vix9d / vix`:

- DB writer [`backend/spx_backend/jobs/quote_job.py:148`](../spx_backend/jobs/quote_job.py): `term_structure = vix9d / vix`
- DB writer [`backend/spx_backend/jobs/cboe_gex_job.py:578`](../spx_backend/jobs/cboe_gex_job.py): same column written via backfill
- Training [`backend/scripts/generate_training_data.py:2335`](generate_training_data.py): `term_structure = (vix9d / vix) if ...`
- Test fixtures: `backend/tests/test_decision_job.py:337-338` uses `vix=16, vix9d=15, term_structure=0.94` ⇒ ratio is `15/16`
- Backtest CSV column: training writes `vix9d/vix`; backtest detector reads the row directly

**Impact.**

- All `term_inversion` events in live trading are firing on a market state
  *opposite* to what offline labels treat as "inversion".
- Models trained offline learned the predictive value of true backwardation;
  live signals fire on contango.
- Highly likely root cause of the operator note in
  [`backend/configs/optimizer/event_only_v2_explore.yaml:4`](../configs/optimizer/event_only_v2_explore.yaml):
  *"term_inversion locked OFF (99.0) based on Phase 1 results showing it loses
  every head-to-head."* The signal is anti-predictive precisely because it is
  inverted.

**Proposed fix.** Replace
```python
term_structure = vix_now / latest_vix9d
```
with
```python
term_structure = latest_vix9d / vix_now
```
in [`event_signals.py:270`](../spx_backend/services/event_signals.py).

**Verification.** Extend [`backend/tests/test_event_signals.py`](../tests/test_event_signals.py)
with two explicit fixtures:

- `vix=18, vix9d=22` (real backwardation) ⇒ `term_inversion` is in `_evaluate(ctx)` output
- `vix=22, vix9d=18` (contango) ⇒ `term_inversion` is NOT in the output

**Risk.** This *changes live trading behavior*. After deploy, `term_inversion`
will start firing during real backwardation (vix9d spikes above vix) and stop
firing during contango. Operator should monitor the first session for
unexpected event-side trade counts, and may want to revisit
`event_only_v2_explore.yaml:4` after fix (the "lock OFF" decision was made
under the inverted-signal regime).

---

### C2 — `train_final_model` deceptive 90/10 split

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Wave** | 1 |
| **Status** | fixed in Wave 1 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

**Description.** [`backend/scripts/xgb_model.py:479-506`](xgb_model.py)
`train_final_model` docstring says *"Train XGBoost on the full dataset (no
holdout)"*. Implementation:

```python
val_split = int(len(X_full) * 0.9)
X_train = X_full.iloc[:val_split]
...
models = train_xgb_models(
    X_train, y_tp50_train, y_pnl_train,
    X_val, y_tp50_val, y_pnl_val,
)
```

The shipped final booster is trained on the first 90% only; the most recent
10% of chronology is used for early stopping and never enters the training
set. Same pattern duplicated in:

- `_run_entry` ([xgb_model.py:1452-1458](xgb_model.py))
- `_run_entry_v2` ([xgb_model.py:1547-1553](xgb_model.py))

**Impact.** Production-shipped models miss the most-recent decile of training
data — the very period whose distribution is most relevant to live inference.

**Proposed fix.** Two-step:

1. Train on 90% with early stopping → record `best_iteration`.
2. Refit on **100%** of data with `n_estimators = best_iteration` and **no**
   early stopping (no `eval_set`). This is the conventional XGBoost recipe
   for "train on all data, use early-stopping run for hyperparameter
   selection only".

Update docstring to describe the two-step recipe accurately.

**Verification.** Unit test asserting the booster's `n_estimators` matches
the recorded `best_iteration` and the trained model has seen `len(X_full)`
rows (e.g. via `model.get_booster().num_boosted_rounds()` and check fit on
all rows by re-evaluating training error on the full set).

---

### C3 — `walk_forward_rolling` threshold tuning leakage

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Wave** | 1 |
| **Status** | fixed in Wave 1 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

**Description.** [`backend/scripts/xgb_model.py:787-828`](xgb_model.py)
`walk_forward_rolling` pools every fold's OOS test prediction and selects the
classifier threshold that maximizes pooled `total_pnl`. This is selecting a
hyperparameter on the very test set whose performance you then report — a
classic test-set leakage / optimistic bias.

**Impact.** Reported metrics for entry-v1 / entry-v2 walk-forward runs are
inflated relative to what a real out-of-sample deployment would see. Any
threshold-based decision rule trained this way will underperform when
deployed.

**Implemented fix.** Pool only the **validation slices** (the last 10% of
each training window, see [`backend/scripts/xgb_model.py`](xgb_model.py)
lines ~870-927) across all walk-forward windows. The recommended threshold
is the argmax over that pooled-val pool (`val_total_pnl`, ties broken
toward the lower threshold for determinism). **Test predictions are never
seen by the threshold tuner**; per-threshold test counters are reported
solely as diagnostics so operators can inspect how the locked threshold
performs OOS.

This is stricter than the original "train+val pool" suggestion because it
also excludes the train rows the model was actually fit on, eliminating
the in-sample optimism on the train side.

**Verification.** Two regression tests in
[`backend/tests/test_xgb_model.py`](../tests/test_xgb_model.py)
(`TestWalkForwardThresholdLeakage`) confirm:

1. The chosen threshold is **identical** with vs without the test rows
   present in the input (i.e. the threshold is a function only of
   train/val rows).
2. The chosen threshold and the entire selection JSON are
   **byte-identical** when only pure-test rows change in any way --
   ensuring no future refactor accidentally re-introduces test-set
   leakage.

---

### C4 — Stop-loss policy divergence (training vs live)

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Wave** | 1 (assert + document); 5+ (full alignment when production switches) |
| **Status** | fixed in Wave 1 (commit 6093bf6) -- hybrid path: startup assertion in generate_training_data.py |
| **Resolved** | 2026-04-16 |

**Description.** The training labeler hardcodes the SL trigger formula:

- [`backend/scripts/generate_training_data.py:1406`](generate_training_data.py): `sl_thr = max_profit * STOP_LOSS_PCT`
- Always uses `max_profit` basis, always SL-enabled.

Production reads from settings:

- [`backend/spx_backend/jobs/decision_job.py:861-868`](../spx_backend/jobs/decision_job.py):
  uses `settings.trade_pnl_stop_loss_basis` (`"max_profit"` or `"max_loss"`)
  and `settings.trade_pnl_stop_loss_enabled`.
- Live formula: `sl_basis_value * settings.trade_pnl_stop_loss_pct` where
  `sl_basis_value = max_loss if basis == "max_loss" else max_profit`.

**Today, by coincidence, both paths agree** because
[`backend/spx_backend/config.py:124-128`](../spx_backend/config.py) defaults
to `basis="max_profit"`, `enabled=True`, `pct=2.00` — the same numbers the
training script hardcodes. **The divergence is latent.** Any environment
override (`TRADE_PNL_STOP_LOSS_BASIS=max_loss`, `TRADE_PNL_STOP_LOSS_ENABLED=false`)
silently breaks the alignment.

The sister helper [`backend/scripts/_label_helpers.py:113-117`](_label_helpers.py)
`evaluate_candidate_outcome` *already supports both bases* — the pipeline
just doesn't use it; it has a duplicate `_evaluate_outcome` in
`generate_training_data.py` (lines 1338–1529) that hardcodes `max_profit`.

**Impact.** If production ever changes basis or disables SL, training labels
will silently misrepresent live trade outcomes; downstream models and
optimizer runs will be calibrated to the wrong policy.

**Proposed fix (Wave 1, hybrid).** Add a startup assertion in
`generate_training_data.py` that fails loudly if live settings drift from the
training constants:

```python
from spx_backend.config import settings as _live_settings

if _live_settings.trade_pnl_stop_loss_basis != "max_profit":
    raise SystemExit(
        "Training labeler hardcodes basis='max_profit' but live config is "
        f"{_live_settings.trade_pnl_stop_loss_basis!r}. See OFFLINE_PIPELINE_AUDIT.md C4."
    )
if not _live_settings.trade_pnl_stop_loss_enabled:
    raise SystemExit("Training simulates SL but live has SL disabled. See audit C4.")
if abs(_live_settings.trade_pnl_stop_loss_pct - STOP_LOSS_PCT) > 1e-9:
    raise SystemExit(
        f"Training STOP_LOSS_PCT={STOP_LOSS_PCT} != live "
        f"trade_pnl_stop_loss_pct={_live_settings.trade_pnl_stop_loss_pct}. See audit C4."
    )
```

The assertion converts a silent latent bug into a loud startup failure the
moment the divergence becomes real.

**Proposed fix (deferred — only when production switches).** Route
`_evaluate_outcome` through the existing `_label_helpers.evaluate_candidate_outcome`
helper (which already supports both bases). This eliminates the duplicate
labeler and lets the pipeline read settings directly. Cost: relabeling
`training_candidates.csv` (297 MB) plus invalidation of `labels_cache/`
(61 MB). Trigger: any env override of the three SL settings.

---

### C5 — `predict_xgb_entry` semantics + missing V2 features

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Wave** | 1 |
| **Status** | fixed in Wave 1 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

**Description.** Two coupled defects in the inference contract that block
safe deployment of an entry-v2 model:

1. **Semantic mismatch.** [`backend/spx_backend/jobs/modeling.py`](../spx_backend/jobs/modeling.py)
   `predict_xgb_entry` always returns `probability_win` from the classifier
   output. Entry-v1's classifier is binary `hit_tp50`, so this is correct.
   Entry-v2's classifier predicts `P(big loss)` (with `scale_pos_weight=7.0`,
   per `_run_entry_v2`). Loading an entry-v2 model would silently
   *inverted-rank* candidates: high `probability_win` would actually mean
   high `P(big loss)`.

2. **Missing V2 features.** `extract_xgb_features` implements only the V1
   set (`vix_x_delta`, `dte_x_credit`, `gex_sign`, etc.). Entry-v2 expects
   the additional V2 extras — `is_call`, `iv_skew_ratio`,
   `credit_to_max_loss`, `gex_abs`, `vix_change_1d`, `recent_loss_rate_5d`
   (per `xgb_model._add_v2_features` lines 611–626). Without these, V2
   inference fills NaN or zero and produces invalid scores.

**Impact.** Today, no production model is loaded so there's no live impact.
But the *contract* is broken: any future `upload_xgb_model.py` of an
entry-v2 artifact would silently mis-rank candidates. This blocks the ML
re-entry path described in `decision_job._run_portfolio_managed`.

**Proposed fix.**

1. Stamp `model_type` in
   [`upload_xgb_model.py:_load_model_artifacts`](upload_xgb_model.py)'s
   `data_snapshot_json` payload. Default to `"xgb_entry_v1"` for backward
   compatibility; entry-v2 uploads must set `"xgb_entry_v2"`.
2. In `predict_xgb_entry`: read `model_payload["model_type"]`.
   - For `xgb_entry_v1`: keep current behavior; classifier output is
     `probability_win`.
   - For `xgb_entry_v2`: classifier output is `P(big loss)`; convert to a
     ranking score. Recommended formulation:
     `utility_score = expected_pnl * (1 - p_big_loss)`. Return both
     `p_big_loss` (the raw classifier output) and `utility_score` so the
     decision_job knows which to rank by.
3. In `extract_xgb_features`: add a `feature_set` parameter (default `"v1"`)
   selecting either the V1 set or V1+V2 extras. Mirror the V2 builder logic
   from `xgb_model._add_v2_features` (lines 611–626):
   - `is_call`: 1 if `spread_side == "call"` else 0
   - `iv_skew_ratio`: `iv_short / iv_long` from candidate legs
   - `credit_to_max_loss`: `entry_credit / max_loss` (guard zero)
   - `gex_abs`: `abs(gex_net)`
   - `vix_change_1d`: passed through from caller (computed externally on
     daily series)
   - `recent_loss_rate_5d`: passed through from caller (rolling 5-day loss
     rate from prior trade outcomes)
4. The decision_job caller selects the feature set based on the model's
   `model_type` (read from `model_versions.data_snapshot_json`).

**Verification.** New tests in [`backend/tests/test_modeling.py`](../tests/test_modeling.py):

- `test_predict_xgb_entry_v1_returns_probability_win`
- `test_predict_xgb_entry_v2_returns_utility_with_inverted_classifier`
- `test_extract_xgb_features_v1_set_matches_existing_columns`
- `test_extract_xgb_features_v2_set_includes_v2_extras`

---

## Findings — High (H1–H7)

Coverage gaps + reproducibility gaps that don't break correctness today but
are silent landmines.

---

### H1 — 2d SPX gap guard missing in backtest detector

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 1 |
| **Status** | fixed in Wave 5 (commit 6093bf6) -- shared evaluator now enforces the guard for both live and backtest |
| **Resolved** | 2026-04-16 |

Live [`event_signals.py:140`](../spx_backend/services/event_signals.py)
suppresses `spx_drop_2d` if `prev_spx_return_2d_gap_days > 4` (a "2-day
return" computed across a 5+ calendar-day gap is not actually a 2-day move).

Backtest [`backtest_strategy.py:617-618`](backtest_strategy.py)
`EventSignalDetector` has no equivalent. Same gap in
[`regime_analysis.py`](regime_analysis.py) `precompute_daily_signals`. The
backtest will fire `spx_drop_2d` across long weekends, holiday weeks, or
data outages — events live trading correctly suppresses.

**Fix.** Mirror the gap calculation in `precompute_daily_signals`; suppress
2d-drop when gap > `_SPX_2D_MAX_CALENDAR_GAP_DAYS` (4). Permanently fixed in
Wave 5 by extracting a shared `EventSignalEvaluator`.

---

### H2 — Optimizer ingest is non-atomic

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 |
| **Status** | fixed in Wave 2 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

[`ingest_optimizer_results.py`](ingest_optimizer_results.py) writes:

1. `optimizer_runs` INSERT inside `with engine.begin():` — atomic.
2. `optimizer_results` via `df.to_sql(...)` — separate connection /
   separate transaction.
3. (Optional) `optimizer_walkforward` via another `to_sql` — another
   transaction.

If step 2 or 3 fails after step 1 commits, you have a `optimizer_runs` row
with no children. Re-running with the same CSV is then **blocked** by the
content-hash duplicate detection (lines 161–173) — you get
`"Duplicate detected: CSV already ingested as <id>. Skipping."` and the
partial state is never repaired.

**Fix.** Wrap all three writes in a single `engine.begin()` block. Or, if
that's infeasible due to `to_sql` semantics, add a repair path: detect "run
exists but `optimizer_results` row count for `run_id` is 0" and reattempt
the children writes (without re-inserting the run row).

---

### H3 — Candidate cache misses input-data versioning

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 |
| **Status** | fixed in Wave 2 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

[`generate_training_data.py:2398-2410`](generate_training_data.py) cache
manifest stores `code_version` (SHA-256 of script bytes) and `grid_hash`.
Neither tracks the underlying Databento `.dbn.zst` or production-export
parquet content for that day.

If a day's source data is silently corrected or replaced (e.g. a Databento
re-issue, a production-export rerun), the candidate cache **does not
invalidate** and stale candidates persist into training rows.

**Fix.** Extend the per-day cache key with at least
`{path: (mtime, size)}` for every input file consumed by that day.
Invalidate when any tuple changes. Optional: full content hash for
ultra-strict cache invalidation; default to mtime+size for performance.

---

### H4 — `walkforward_results.csv` is a global path

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 |
| **Status** | fixed in Wave 2 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

[`run_pipeline.py:147`](run_pipeline.py) hard-codes
`data/walkforward_results.csv`. Multiple concurrent or sequential pipeline
runs (e.g. `run_2026-04-experiment-A` + `run_2026-04-experiment-B`) clobber
each other, while their corresponding `{run_name}_results.csv` files for the
optimize phase are correctly per-run.

**Fix.** Change to `data/{run_name}_walkforward.csv`; update the
`--phase walkforward` write-side and the `--phase ingest` read-side
together.

---

### H5 — `definitions_from_production` uses `Timestamp.now()` for dedup

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 |
| **Status** | fixed in Wave 2 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

[`generate_training_data.py:298-300`](generate_training_data.py):

```python
unique["ts_recv"] = pd.Timestamp.now(tz="UTC")
```

The wall-clock value is then used as a sort key for instrument-id
duplicates. Two runs of the same script over the same data will produce
different sort orders (and possibly different "winning" definitions) for
duplicate `instrument_id` rows.

**Fix.** Use a deterministic sentinel — e.g.
`pd.Timestamp(f"{day_str}T00:00:00Z")` — and document that `ts_recv` is
no longer wall-clock and only exists for dedup tiebreaking.

---

### H6 — SKEW intraday lookahead risk (verification needed)

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 2 (verify) |
| **Status** | fixed in Wave 2 (commit 6093bf6) -- FRD SKEW parquet timestamp lagged to next session |
| **Resolved** | 2026-04-16 |

[`generate_training_data.py:2337-2338`](generate_training_data.py)
`_generate_candidates_for_day` reads `skew_daily.get(day_date)` from
`load_daily_parquet(FRD_SKEW)`. The `load_daily_parquet` helper takes the
**last close** per calendar date. If the FRD SKEW parquet timestamps are
end-of-day, then using `skew_daily[day_date]` at intraday decision times
constitutes **same-day lookahead** — the feature reflects information not
available at the decision instant.

**Fix.** Inspect FRD SKEW parquet `ts` column to determine the timestamp
semantics:

- If timestamps are end-of-session ⇒ shift `skew_daily` index by one trading
  day before per-candidate lookup; document the offset.
- If timestamps are start-of-session ⇒ document explicitly in
  `_generate_candidates_for_day` that the SKEW value is point-in-time and
  no shift is needed.

---

### H7 — Missing production-DB exports for forensics + future ML re-entry

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Wave** | 3 |
| **Status** | fixed in Wave 3 (commit 6093bf6) -- 15+ new tables exported with PII redaction + chunked reads + atomic writes |
| **Resolved** | 2026-04-16 |

[`export_production_data.py:7-12`](export_production_data.py) docstring
explicitly says *"Future ML re-entry will synthesize training rows directly
from `trade_decisions` + `trades` + `trade_marks`"* — but the script does
not export any of those tables.

Tables the exporter currently does NOT cover:

- `trade_decisions`, `trades`, `trade_legs`, `trade_marks` (live trade
  forensics + future label generation)
- `orders`, `fills` (broker-side reconciliation)
- `gex_snapshots`, `gex_by_strike`, `gex_by_expiry_strike` (GEX archaeology;
  currently the offline pipeline relies on the `offline_gex_cache.csv` +
  `context_snapshots_export.csv` merge instead of native GEX exports)
- `portfolio_state`, `portfolio_trades` (portfolio path forensics)
- `optimizer_runs`, `optimizer_results`, `optimizer_walkforward`
  (optimizer-history archaeology)

**Fix.** Extend `export_production_data.py` `--tables` choices and add per-table
exporter functions. Use chunked reads (`pd.read_sql_query` with `chunksize`)
for the heavy tables (`trade_marks`, `gex_by_strike`). PII review: do NOT
export `users`, `auth_audit_log`, or `user_id` columns from
`trade_decisions`/`trades` — replace with a stable anonymous integer
surrogate.

---

## Findings — Medium (M1–M12)

Parity drift, dedup-logic mismatches, dead code, hygiene.

---

### M1 — Daily aggregation `first` vs `last` mismatch

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 3 |
| **Status** | fixed in Wave 3 (commit 6093bf6) -- standardized to 'first' across regime_analysis.py |
| **Resolved** | 2026-04-16 |

- [`regime_analysis.py:80-88`](regime_analysis.py) `enrich_with_daily_features` aggregates per-day VIX/SPX with `last` (sorted by `entry_dt`).
- [`backtest_strategy.py:672-687`](backtest_strategy.py) `precompute_daily_signals` uses `first`.

On days with multiple intraday entries, prior-day returns and VIX changes
differ between explorer and backtest. **Fix:** unify on `first` to match
"snapshot at first decision time" (live always uses the first decision's
context).

---

### M2 — VIX % units inconsistency (percent vs decimal)

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 3 |
| **Status** | fixed in Wave 3 (commit 6093bf6) -- decimal-fraction convention everywhere; bucket thresholds adjusted |
| **Resolved** | 2026-04-16 |

`regime_analysis.py` `prev_vix_pct_change` is in **percent** (e.g. 10 = 10%).
Prod and backtest event configs use **decimal** (e.g. 0.15 = 15%). The script
docstring (lines 73–78) warns about this but doesn't fix it.

**Fix:** normalize to decimal in `regime_analysis` to match prod/backtest.
Operators using existing reports may need a one-line note.

---

### M3 — Index-based dedup in backtest vs leg-identity dedup in live

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 5 (extracted to shared evaluator) |
| **Status** | fixed in gap-closure Phase 2.1 -- shared `candidate_dedupe_key` in `backend/spx_backend/services/candidate_dedupe.py`; live wraps via thin method, backtest checks `seen_leg_keys` per day-window. 14-test parity suite in `backend/tests/test_candidate_dedupe.py`.  **Resolved:** 2026-04-16. |

Live [`decision_job.py:302-362`](../spx_backend/jobs/decision_job.py) uses
`seen_keys = {_candidate_dedupe_key(c) for c in event_candidates}` — i.e.
leg-identity dedup. Backtest
[`backtest_strategy.py:968-972`](backtest_strategy.py) drops scheduled rows
whose **DataFrame index** appears in event picks — index-identity dedup.

Where the backtest grid contains rows with different indices but identical
leg specifications (rare but possible), backtest will not dedupe and live
will. **Fix:** share `_candidate_dedupe_key` between the two paths in
Wave 5.

---

### M4 — `max_margin_pct` dead in backtest `PortfolioManager`

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 3 |
| **Status** | fixed in Wave 3 (commit 6093bf6) |
| **Resolved** | 2026-04-16 |

[`backtest_strategy.py:127-142`](backtest_strategy.py) `PortfolioConfig`
declares `max_margin_pct`, included in optimizer `_config_key` (lines
2528–2531). But `PortfolioManager.compute_lots` (544–552) never references
it. The optimizer is sweeping over a parameter that has no effect.

**Fix:** either implement the cap in `compute_lots` (treat as additional
upper bound on `lots`) or delete the field. Recommendation: implement, since
the optimizer CSV already varies it and operators may believe it is
constraining position size.

---

### M5 — PnL timing mismatch (entry-day vs exit-day)

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 5 (deferred — design change) |
| **Status** | deferred -- moved to Future work section |

Live `record_trade(0.0)` at entry; PnL booked at close (per
`portfolio_manager`). Backtest books **full row PnL on the entry day**
(`backtest_strategy.py:952-980`). This means:

- Backtest equity curves don't show the inter-day drawdown of trades held
  through TP/expiry.
- Monthly stop logic in backtest evaluates based on entry-day attribution,
  not realised exits.

For most spreads this is a small effect (next-day expiries dominate), but
for multi-day holds the divergence is real. **Fix (deferred):** model
exit-day PnL attribution; needs a design change to the backtest engine and
careful invariance check against current optimizer outputs.

---

### M6 — Pareto frontier computed in two places

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 5 |
| **Status** | fixed in Wave 5 (commit 6093bf6) -- single canonical implementation in backend/scripts/_pareto.py |
| **Resolved** | 2026-04-16 |

`extract_pareto_frontier` in
[`backtest_strategy.py:2089-2105`](backtest_strategy.py) writes
`data/pareto_frontier.csv`. `_compute_pareto` in
[`ingest_optimizer_results.py`](ingest_optimizer_results.py) recomputes the
`is_pareto` flag for DB ingest. Two implementations, drift risk.

**Fix:** move one canonical implementation to a shared module; both
consumers import.

---

### M7 — Backtest `EventSignalDetector` duplicated

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 5 |
| **Status** | fixed in Wave 5 (commit 6093bf6) -- shared evaluator in spx_backend/services/event_signals.py |
| **Resolved** | 2026-04-16 |

[`backtest_strategy.py:572-649`](backtest_strategy.py) is a parallel
implementation of the prod `EventSignalDetector` in
[`event_signals.py`](../spx_backend/services/event_signals.py). They are
manually kept similar but already diverge: H1 (gap guard), C1 (term
orientation), M9 (drop-range filters), M11 (per-run-vs-per-day caps).

**Fix:** in Wave 5, make `services/event_signals.py` the single
implementation; backtest constructs the prod evaluator from a row.

---

### M8 — SL counterfactual TP-level mismatch

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 3 |
| **Status** | fixed in gap-closure Phase 2.2 -- `--tp {50,60,75}` CLI flag wired through `compute_trade_pnl` / `evaluate_strategy` / `slice_by_dimension`; main() warns when the matching `min_pnl_before_tp{N}` column is missing. 10 new parametrized tests in `test_sl_recovery_analysis.py::TestTpLevelParametrization`.  **Resolved:** 2026-04-16. |

[`sl_recovery_analysis.py:97-108`](sl_recovery_analysis.py)
`compute_trade_pnl` hardcodes TP50 ordering (uses `min_pnl_before_tp50` for
SL-vs-TP). The training grid's primary TP may be TP60 or TP75
(`generate_training_data.py:1408-1416`). When the grid uses a different
primary TP, the SL counterfactual is mis-modeled.

**Fix:** parameterize TP level via CLI (`--tp 50|60|75`) or read from a
sidecar manifest.

---

### M9 — Backtest detector supports `spx_drop_min/max`; prod does not

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 5 |
| **Status** | fixed in gap-closure Phase 2.3 -- added `event_spx_drop_min` / `event_spx_drop_max` to `Settings` (both default `None` so legacy live behaviour is preserved); wired through `EventSignalDetector.__init__` and `_build_thresholds`; surfaced in `/portfolio/config` and `.env.example`. 8 new tests in `test_event_signals.py::TestSpxDropMagnitudeWindow`.  **Resolved:** 2026-04-16. |

`backtest_strategy.py` `EventSignalDetector` (lines 605–615) supports
`spx_drop_min` / `spx_drop_max` (range filter). Prod `event_signals.py`
does not.

**Fix (when sharing evaluator in Wave 5):** decide whether the range filter
is a real prod feature (then add to prod) or a backtest-only convenience
(then delete from backtest). Likely the former, since it lets the optimizer
sweep "drop magnitude" rather than just "drop threshold".

---

### M10 — OPEX skip default differs

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 3 |
| **Status** | fixed in Wave 3 (commit 6093bf6) -- alignment-warn helper detects future config drift; both defaults are False today |
| **Resolved** | 2026-04-16 |

Live: `settings.decision_avoid_opex` (default `True`).
Backtest: `TradingConfig.avoid_opex` (default `False`).

A backtest run with default config disables OPEX skip; live skips OPEX. This
silently masks the production behavior.

**Fix:** align default in backtest, or print a warning at backtest startup
when the value differs from live.

---

### M11 — Per-run vs per-day cap mismatch

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 5 (model change) |
| **Status** | deferred -- moved to Future work section |

Live: `portfolio_max_trades_per_run` — multiple runs/day possible, cap
applies per run. Backtest: `max_trades_per_day` only.

If live runs the decision job N times per day, backtest cannot model that
cadence. **Fix (deferred):** model multiple runs/day in backtest, or
document the simplification explicitly.

---

### M12 — `download_databento` sample-mode layout undocumented

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Wave** | 3 |
| **Status** | open |

The module docstring describes the **full** mode layout
(`data/databento/<subdir>/YYYYMMDD.dbn.zst`). `--phase sample` actually
writes Parquet (`{start}_{end}.parquet`) — different file format and
naming. Operators inspecting `data/databento/spy/cbbo-1m/` after a sample
run see Parquet, not `.dbn.zst`, with no explanation.

**Fix:** update top-of-file docstring to describe both `--phase full` and
`--phase sample` outputs, OR unify by writing `.dbn.zst` in both modes.

---

## Findings — Low (L1–L10)

Hygiene + refactor opportunities.

---

### L1 — `download_databento` 2026-only holiday calendar

[`download_databento.py`](download_databento.py) `US_EQUITY_HOLIDAYS_2026`
is hardcoded to 2026 only. `--verify-dbn` and expected-day calculations
break for any other year.

**Fix:** add `pandas_market_calendars` to `requirements.txt` and use
`get_calendar("XNYS")`, OR extend the hardcoded set explicitly to 2020–2030
with a comment about the source.

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- expanded to 2020-2030 frozenset.  **Resolved:** 2026-04-16.

---

### L2 — Missing `data/README.md`

121 GB / 3,788 files / 6+ source categories with no in-repo documentation.
A new operator would not know which artifact is regeneratable, which is
external, what the retention policy is, or how to safely clean up.

**Fix:** write `data/README.md` covering each artifact's source, regenerator
command, retention policy, last-known-good size.

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- 240-line README covering producer/consumer/safety per artifact + PII policy + cache invalidation.  **Resolved:** 2026-04-16.

---

### L3 — Stale `optimizer_event_only.csv`

48 MB file (Apr 11) appears superseded by `optimizer_event_only_v2.csv`
(116 KB) + `optimizer_event_only_v2_explore.csv` (6 MB) (Apr 13). Not
referenced in any `.py` file; risk of an operator using the wrong file when
ingesting or comparing runs.

**Fix:** move to `data/archive/optimizer_event_only_v1.csv` with a note in
`data/README.md` documenting the supersession.

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- file moved to data/archive/optimizer_event_only_v1.csv; .gitignore now whitelists `!data/archive/**` (gap-closure Phase 1.4) and a small `data/archive/README.md` tombstone index documents what was archived and the marker-only convention for future archives > 10 MB.  **Resolved:** 2026-04-16.

---

### L4 — Unused exports (`economic_events_export.csv` + `underlying_quotes_export.csv`)

Both are produced by `export_production_data.py --tables all` but are NOT
consumed by `generate_training_data.py` (which uses `economic_calendar.csv`
for events and FRD parquets for quotes).

**Fix:** drop both from default `--tables all`, OR wire them into the
training pipeline. Recommend dropping (the consumers already have
preferred sources).

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- legacy exports excluded from --tables all default.  **Resolved:** 2026-04-16.

---

### L5 — No `DATABENTO_DIR` env variable

The 120 GB databento tree is hardcoded under `data/databento/` in both
`download_databento.py` and `generate_training_data.py`. Cannot be
relocated to external storage (e.g. a dedicated SSD or an NFS mount)
without a code change.

**Fix:** add `databento_dir: Path = Path("data/databento")` to
`Settings` in `backend/spx_backend/config.py`. Both scripts read from
settings.

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- settings.databento_dir + _resolve_databento_dir helper.  **Resolved:** 2026-04-16.

---

### L6 — `regime_utils.py` docstring is incorrect

The module docstring claims `regime_analysis.py` uses it. Grep confirms
`regime_analysis.py` does NOT import `regime_utils`. Only
`backtest_strategy.py` imports `regime_utils.compute_regime_metrics`.

**Fix:** correct the docstring.

**Wave:** 3.

**Status:** fixed in gap-closure Phase 1.3 -- docstring rewritten to list actual consumers (backtest_strategy, generate_training_data cache fingerprint, test_regime_utils) and clarify that regime_analysis.py is a future-adoption candidate, not a current importer.  **Resolved:** 2026-04-16.

---

### L7 — `regime_utils.py` SPX/VIX edges differ from `regime_analysis.py`

`regime_utils.classify_*` uses decimal SPX thresholds (e.g. `-0.02`);
`regime_analysis` uses percent (`-2`). Even when intent is the same.

**Fix:** consolidate to decimal everywhere (linked to M2); ensure both files
use the same edges.

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- decimal-fraction unified everywhere; M2 consolidation.  **Resolved:** 2026-04-16.

---

### L8 — `flat_records` non-deterministic sort in `generate_training_data.py`

`training_candidates.csv` row order follows the cache-hit pattern (line
2810: `labeled = cands_to_label + cached_labeled`). Re-runs with different
cache hit ratios produce row-permuted CSVs even when the multiset is
identical. Hard to diff across runs.

**Fix:** sort `flat_records` by `(day, entry_dt, spread_id)` before writing
the CSV.

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- explicit sort by (day, entry_dt, spread_id) before write.  **Resolved:** 2026-04-16.

---

### L9 — Non-atomic CSV writes in `export_production_data.py`

Several `df.to_csv(path)` calls write directly to the destination path. A
crash mid-write leaves a truncated CSV that downstream consumers may
interpret as a complete file.

**Fix:** write to `<path>.tmp` then `os.replace(<path>.tmp, <path>)`.

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- atomic temp+rename for all CSV/Parquet writes.  **Resolved:** 2026-04-16.

---

### L10 — Optimizer rows non-deterministic under parallel runs

`backtest_strategy.py` optimizer uses `Pool.imap_unordered` (line 1652).
Test comparison (`test_backtest_strategy.py:1550-1558`) sorts before
comparing seq vs par — but the *output CSV* row order is non-deterministic.

**Fix:** sort optimizer rows by `_config_key` before `df.to_csv(...)`.

**Wave:** 3.

**Status:** fixed in Wave 3 (commit 6093bf6) -- deterministic sort by `_config_key` before write.  **Resolved:** 2026-04-16.

---

## Wave plan

| Wave | Findings | Files touched | Risk |
|------|----------|---------------|------|
| 0 | (this doc) | `backend/scripts/OFFLINE_PIPELINE_AUDIT.md` | none |
| 1 | C1, C2, C3, C4, C5, H1 | `event_signals.py`, `xgb_model.py`, `modeling.py`, `upload_xgb_model.py`, `generate_training_data.py`, `backtest_strategy.py`, `regime_analysis.py`, 4 test files | **HIGH** — C1 changes live trading behavior |
| 2 | H2, H3, H4, H5, H6 | `ingest_optimizer_results.py`, `generate_training_data.py`, `run_pipeline.py` | LOW–MEDIUM |
| 3 | H7, M1, M2, M4, M8, M10, M12, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10 | `export_production_data.py`, `regime_analysis.py`, `regime_utils.py`, `backtest_strategy.py`, `download_databento.py`, `config.py`, new `data/README.md`, `data/archive/` | LOW |
| 4 | (test coverage) | new `test_regime_utils.py`, extended `test_sl_recovery_analysis.py`, new `test_backtest_entry.py`, extended `test_xgb_model.py` | LOW |
| 5 | M3, M5 (defer), M6, M7, M9, M11 (defer); refactor + ML contract | `backend/scripts/training/`, `backend/scripts/backtest/`, `backend/scripts/xgb/` packages; `event_signals.py` shared evaluator; new `OFFLINE_ML.md` | MEDIUM (large diff) |

---

## Verification checklist

### Wave 1
- [ ] `pytest backend/tests/test_event_signals.py -v` — new direction tests pass
- [ ] `pytest backend/tests/test_xgb_model.py::test_train_final_model_uses_all_rows -v` passes
- [ ] `pytest backend/tests/test_xgb_model.py::test_walk_forward_threshold_excludes_test -v` passes
- [ ] `pytest backend/tests/test_modeling.py::test_predict_xgb_entry_v2_inverts_p_big_loss -v` passes
- [ ] `pytest backend/tests/test_backtest_strategy.py::test_2d_drop_suppressed_on_long_gap -v` passes
- [ ] `python -m backend.scripts.generate_training_data --max-days 1 --quiet` raises `SystemExit` when `TRADE_PNL_STOP_LOSS_BASIS=max_loss`
- [ ] `python -m backend.scripts.generate_training_data --max-days 1 --quiet` succeeds with default settings
- [ ] Full `pytest backend/tests/` green

### Wave 2
- [ ] `pytest backend/tests/test_ingest_optimizer_results.py::test_partial_failure_recoverable -v` passes
- [ ] Two consecutive runs of `generate_training_data.py` over the same date range produce *byte-identical* `training_candidates.csv` (deterministic `ts_recv`)
- [ ] Two `run_pipeline.py` invocations with different `--run-name` no longer clobber each other's walkforward output
- [ ] Modifying a `.dbn.zst` mtime triggers candidate cache invalidation for that day

### Wave 3
- [ ] `python -m backend.scripts.export_production_data --tables all` produces all H7 tables
- [ ] `data/README.md` lists every file currently in `data/`
- [ ] `python -c "from spx_backend.config import settings; print(settings.databento_dir)"` returns expected path
- [ ] `pytest backend/tests/` green

### Wave 4
- [ ] `pytest backend/tests/test_regime_utils.py -v` passes
- [ ] `pytest backend/tests/test_sl_recovery_analysis.py::test_evaluate_strategy -v` passes
- [ ] `pytest backend/tests/test_backtest_entry.py::test_main_smoke -v` passes
- [ ] Coverage report shows new files run

### Wave 5
- [ ] `python -m backend.scripts.training.cli --max-days 5` produces byte-identical output to pre-refactor `generate_training_data.py --max-days 5` (pinned date range)
- [ ] `python -m backend.scripts.backtest.cli --csv ...` produces byte-identical optimizer output to pre-refactor `backtest_strategy.py`
- [ ] `python -m backend.scripts.xgb.cli ...` produces identical model artifacts
- [ ] `pytest backend/tests/` green
- [ ] Backtest `EventSignalDetector` is a thin wrapper around `services/event_signals.py`
- [ ] `OFFLINE_ML.md` reviewed by operator

---

## Open questions

1. **C1 deploy window.** The fix changes live trading behavior. Coordinate
   with the operator: deploy outside trading hours? Stage on a sandbox first?
2. **C5 V2 features schema.** What exactly does the live decision-job
   feature-builder pass as `vix_change_1d` and `recent_loss_rate_5d`? These
   require *external* daily-level inputs (not present in a single candidate
   row). The decision-job will need a daily context cache or a query to
   `context_snapshots` + recent `trade_decisions` outcomes.
3. **H6 SKEW timestamp.** Need to inspect actual FRD SKEW parquet to
   confirm timestamp semantics (EOD vs SOD). Likely a 1–2 cell investigation.
4. **H7 PII surrogate scheme.** What's the right anonymization key for
   `trade_decisions.user_id`? Use `hashlib.sha256(user_id || salt)`? Use a
   stored `user_anon_id BIGINT` mapping table? Operator decision.
5. **Wave 5 refactor sequencing.** Do all three monolith splits land in
   one PR or three sub-PRs? Recommendation: three sub-PRs, each with a
   byte-identical regression check on a pinned date range.

---

## Future work (deferred from main audit)

These findings were reviewed and intentionally not fixed in the main
audit waves because each requires a design-level decision rather than a
mechanical fix. Tracked here so they remain visible without blocking the
audit doc from being marked complete.

### M5 — PnL timing mismatch (entry-day vs exit-day)

**Why deferred.** Choosing entry-day vs exit-day attribution materially
changes how every downstream Sharpe/DD metric is computed and how live
P&L is reconciled against the broker. Operator-level decision; not a
correctness bug under either convention.

**Trigger to revisit.** When daily P&L attribution is added to the
operator dashboard, OR when an MTM-based reporting layer is introduced
that requires a single canonical timing convention.

### M11 — Per-run vs per-day cap mismatch

**Why deferred.** Backtest enforces caps per simulated *run*; live
enforces per *trading day*. Aligning these requires deciding whether
backtest should mimic live's daily semantics (changes optimizer search
space) or live should switch to per-run accounting (riskier).

**Trigger to revisit.** When the optimizer's daily-cap dimension shows
material disagreement between backtest-Sharpe and live-Sharpe on the
same parameter set.

---

## Glossary

- **CBBO**: Consolidated Best Bid Offer (Databento OPRA schema).
- **CBOE GEX**: Cboe-published Gamma Exposure (alternative to internal GEX
  computation).
- **Entry-v1 / entry-v2**: First and second iterations of the XGBoost entry
  model in `xgb_model.py`. V2 adds extra features and uses
  `scale_pos_weight=7` to upweight the rare "big loss" class.
- **Hold-vs-close mode**: A walk-forward variant that compares "hold to TP"
  vs "close at signal" PnLs.
- **OPEX**: Monthly options expiration day (3rd Friday).
- **Pareto frontier**: Set of optimizer configs that are not dominated on
  both `sharpe` and `-max_dd_pct`.
- **Term structure**: VIX9D / VIX ratio. Backwardation when ratio > 1.0
  (near-term vol stress); contango when ratio < 1.0 (normal market).
- **Triple witching**: Quarterly OPEX where stock options, index options,
  and futures all expire on the same day.

---

*Generated 2026-04-16 as part of Track B (offline pipeline correctness audit
+ fix waves). See [`.cursor/plans/track_b_offline_audit_fixes_73d0812a.plan.md`](../../.cursor/plans/track_b_offline_audit_fixes_73d0812a.plan.md)
for the work-tracking plan and
[`.cursor/plans/backend_e2e_tracks_v2_5614035a.plan.md`](../../.cursor/plans/backend_e2e_tracks_v2_5614035a.plan.md)
for the Track B section it supersedes.*

*Status sweep + small-finding closure landed via the gap-closure plan
[`.cursor/plans/offline-pipeline-gap-closure_d993da3d.plan.md`](../../.cursor/plans/offline-pipeline-gap-closure_d993da3d.plan.md).
File-path citations remain authoritative; line numbers reflect the
audit-time snapshot (pre-`6093bf6`) and are intentionally not refreshed
on every commit -- search for the referenced symbol in the current file
instead.*
