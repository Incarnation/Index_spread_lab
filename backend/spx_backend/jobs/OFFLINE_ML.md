# Offline ML Re-entry Contract

> **Purpose.** Define the end-to-end contract between the offline ML
> training pipeline and the live decision job (`_run`)
> so a future XGBoost re-activation lands cleanly. Authored as part of
> Wave 5 in [`backend/scripts/OFFLINE_PIPELINE_AUDIT.md`](../../scripts/OFFLINE_PIPELINE_AUDIT.md).
> See C5 in that audit for the underlying defects this contract closes.

The current production decision path (`decision_job._run`)
is **portfolio-managed** with no ML scoring layer: candidates are ranked
by `credit_to_width`, capped by `PortfolioManager`, and filtered by
event signals from `services/event_signals.py`. ML re-entry is an
explicit future opt-in. This document spells out exactly what the
offline pipeline must produce so that opt-in is a configuration change
and **not** a code change.

---

## 1. Scope

This contract covers the boundary between:

| Side                | Owner                                                    | File(s) |
|---------------------|----------------------------------------------------------|---------|
| **Producer (offline)** | Walk-forward training + artifact upload                | [`backend/scripts/xgb/cli.py`](../../scripts/xgb/cli.py), [`backend/scripts/xgb/training.py`](../../scripts/xgb/training.py) (`save_model`), [`backend/scripts/upload_xgb_model.py`](../../scripts/upload_xgb_model.py) |
| **Consumer (live)** | Inference inside the decision job                        | [`backend/spx_backend/jobs/modeling.py`](modeling.py) (`predict_xgb_entry`, `extract_xgb_features`) |
| **Storage**         | DB row carrying the artifact + selection metadata        | `model_versions` table (`data_snapshot_json`, `algorithm`, `is_active`) |

> **Producer-path note.** Wave 5 of the offline pipeline gap-closure
> (see [`OFFLINE_PIPELINE_AUDIT.md`](../../scripts/OFFLINE_PIPELINE_AUDIT.md))
> split the original 1568-line `backend/scripts/xgb_model.py` monolith
> into the [`backend/scripts/xgb/`](../../scripts/xgb/) package
> (`features.py`, `training.py`, `walkforward.py`, `cli.py`).
> [`backend/scripts/xgb_model.py`](../../scripts/xgb_model.py) remains
> as a back-compat shim that re-exports the package's public surface
> -- callers that still `from xgb_model import ...` keep working --
> but new code should import from the package paths above.

The decision job itself (`_run`) is **not** modified by
this contract today. The hooks below describe the smallest patch that
re-activates ML scoring without changing the portfolio-management,
event-signal, or dedupe semantics.

---

## 2. Supported model types

`predict_xgb_entry` dispatches on `model_payload["model_type"]`. Every
artifact uploaded to `model_versions` **must** stamp one of these
values; an unknown value raises in `upload_xgb_model._load_model_artifacts`.

| `model_type`        | Classifier target  | Inference semantics                              | Feature set |
|---------------------|--------------------|--------------------------------------------------|-------------|
| `xgb_entry_v1` (default for legacy artifacts) | `hit_tp50` (binary) | `probability_win = clf(x)`                       | V1 only |
| `xgb_v1` (alias for `xgb_entry_v1`) | `hit_tp50`            | Same as above                                    | V1 only |
| `xgb_entry_v2`      | `is_big_loss`      | `probability_win = 1 − clf(x)`; also returns `p_big_loss` | V1 + V2 extras |

`utility_score = probability_win * max(expected_pnl, 0)` for **all**
types. Callers rank by `utility_score` (or `1 − p_big_loss` if they
want a pure loss-avoidance ordering for V2).

> **Critical.** Loading a V2 artifact under V1 semantics silently inverts
> the candidate ranking (high `probability_win` would actually mean high
> `P(big_loss)`). The dispatch above is the only thing standing between
> a re-uploaded V2 model and a wrong-sign trade book — do not remove it.

---

## 3. Producer side: `xgb/` package → `upload_xgb_model.py`

### 3.1 Walk-forward training ([`backend/scripts/xgb/`](../../scripts/xgb/))

The producer is the post-Wave-5 [`backend/scripts/xgb/`](../../scripts/xgb/)
package; the legacy `xgb_model.py` shim re-exports the same public
surface for back-compat. CLI entry point:
[`backend/scripts/xgb/cli.py`](../../scripts/xgb/cli.py) (or
`python -m scripts.xgb_model …` via the shim).

* Walk-forward windows are sized by
  [`walk_forward_rolling`](../../scripts/xgb/walkforward.py). Each window
  trains on a `train + val` slice and **evaluates** on a strictly later
  `test` slice. The recommended threshold returned for the window is
  selected on the validation rows only — never on test rows. (See C3 in
  the audit; locked-in by `tests/test_xgb_model.py::TestWalkForwardThresholdLeakage`.)
* [`train_final_model`](../../scripts/xgb/training.py) re-fits on
  **100% of available rows** without a validation split (per C2). The
  artifact written to disk is the full-data refit, not the last
  walk-forward checkpoint.
* The classifier and regressor are trained as separate XGBoost Boosters
  and saved as `classifier.json` and `regressor.json` JSON dumps via
  [`save_model`](../../scripts/xgb/training.py).
* `metadata.json` accompanies the model files and **must include**:
  * `model_type`: one of the strings in §2.
  * `feature_names`: ordered list matching the DMatrix column order
    used at training time.
  * `cls_params` / `reg_params`: the XGBoost params dict.

### 3.2 Upload (upload_xgb_model.py)

`_load_model_artifacts` reads the three JSON files and returns a
`payload` dict that is embedded under
`model_versions.data_snapshot_json -> model_payload`. The keys consumed
downstream by `predict_xgb_entry` are:

| Key              | Source                        | Required |
|------------------|-------------------------------|----------|
| `model_type`     | `metadata.json::model_type`   | yes      |
| `classifier_json`| `classifier.json` (raw text)  | yes      |
| `regressor_json` | `regressor.json` (raw text)   | yes      |
| `feature_names`  | `metadata.json::feature_names`| yes      |
| `cls_params`     | `metadata.json::cls_params`   | optional (for audit) |
| `reg_params`     | `metadata.json::reg_params`   | optional (for audit) |

`upload_model` also stamps `model_versions.algorithm` with the same
`model_type`, so the row-level selection
(`SELECT … WHERE model_name = ? AND is_active`) and the inference-time
dispatch (`payload["model_type"]`) cannot drift apart.

`--activate` flips `rollout_status='active'` / `is_active=True` and
**deactivates the previous active row for the same `model_name`** in a
single transaction. Operators must explicitly opt-in by running
`upload_xgb_model … --activate` — uploads default to shadow.

---

## 4. Consumer side: `predict_xgb_entry` + `extract_xgb_features`

### 4.1 Feature extraction (`extract_xgb_features`)

The function builds a feature dict from a candidate's
`chosen_legs_json` payload + market context. The output dict's keys are
matched against `model_payload["feature_names"]` at inference time.

> **Missing-feature semantics (verified against
> [`modeling.py:1147`](modeling.py)).** The DMatrix row builder is:
> ```python
> row = [float(features.get(fn) or 0) if features.get(fn) is not None else float("nan")
>        for fn in feature_names]
> ```
> So:
> * **Key absent** (`features.get(fn) is None`) → `NaN`. XGBoost handles
>   NaN natively via its default split direction.
> * **Key present but falsy** (`0`, `0.0`, `""`, `False`) → `0.0` via
>   the `or 0` guard. This is the path V1 callers hit for V2-extra
>   keys *if and only if* they explicitly set them to `0` upstream.
>
> A V1 caller that omits the V2 extras therefore gets `NaN` for those
> columns -- not `0.0`. V1/V2 isolation is enforced primarily by
> `feature_names` length-matching (the DMatrix only has the columns
> the model was trained on), not by zero-padding. Any caller that wants
> deterministic 0.0 for "feature unavailable" must set the key
> explicitly.

V2-only inference requires the V1 features **plus** the V2 extras
listed below (sourced from `xgb_model._add_v2_features`):

| Feature              | Definition                                           | Source                          |
|----------------------|------------------------------------------------------|---------------------------------|
| `is_call`            | `1` if `spread_side == "call"` else `0`              | `chosen_legs_json["spread_side"]` |
| `iv_skew_ratio`      | `iv_short / iv_long` (zero-guarded)                  | candidate legs                   |
| `credit_to_max_loss` | `entry_credit / max_loss` (zero-guarded)             | candidate legs                   |
| `gex_abs`            | `abs(offline_gex_net)`                               | day context                      |
| `vix_change_1d`      | `VIX(today) − VIX(yesterday)`                        | passed in by caller (computed on the daily series) |
| `recent_loss_rate_5d`| Rolling 5-day fraction of trades with `hold_realized_pnl < BIG_LOSS_THRESHOLD`, **shifted by 1** to exclude same-day | passed in by caller (rolling stat from prior trade outcomes) |

The `vix_change_1d` and `recent_loss_rate_5d` values are **caller-supplied**
because the inference path does not have direct access to history. The
re-entry hook in `_run` (see §5) is where these
values must be sourced from.

### 4.2 Inference (`predict_xgb_entry`)

Returns the unified payload below regardless of `model_type`:

```python
{
    "source": str,            # "xgb_entry_v1" | "xgb_entry_v2"
    "probability_win": float, # always 0..1; V2 = 1 - p_big_loss
    "expected_pnl": float,    # regressor output, dollars per contract
    "utility_score": float,   # probability_win * max(expected_pnl, 0)
    # V2-only extras:
    "p_big_loss": float,      # raw classifier output (V2 only)
    # V1-shape compatibility (always present, zeroed under XGB):
    "bucket_key": None,
    "bucket_level": None,
    "bucket_count": 0,
    "tail_loss_proxy": 0.0,
    "pnl_std": 0.0,
    "margin_usage": 0.0,
}
```

Callers **must not** access `probability_win` without first checking
`source` (or `model_type`) — the field is normalised but the meaning
of "high score" differs across models.

---

## 5. Re-entry hook in `_run`

Today, `_run` ranks candidates by
`credit_to_width` and skips ML entirely. The minimal patch to wire ML
back in is:

1. **Load the active model.** At the top of the method, after
   `event_det.detect`, query `model_versions WHERE model_name = ? AND is_active = TRUE`
   for a configurable `model_name`. If no row is returned, fall through
   to the current behavior (no ML).
2. **Compute the contextual features once per run.**
   * `vix_change_1d`: pull the latest two daily VIX values from
     `underlying_quotes` and diff.
   * `recent_loss_rate_5d`: pull the last 5 days of `trades.hold_realized_pnl`
     from the production DB (or `hold_realized_pnl` proxy from
     `trade_marks`) and compute the loss rate on rows older than today.
3. **Score each candidate.** For every `c in candidates`, build a
   feature dict via `extract_xgb_features(c["chosen_legs_json"], …)`,
   merge in the contextual `vix_change_1d` and `recent_loss_rate_5d`,
   and call `predict_xgb_entry(payload, features)`.
4. **Filter by score.** Drop candidates whose `utility_score` is below
   a configurable floor (default: `0.0` to retain current behavior;
   raise to `> 0` to enable a true gate). Candidates surviving the
   floor are ranked by `utility_score` descending **before** the
   existing event-vs-scheduled placement loop runs.
5. **Telemetry.** Persist the score on the decision row via
   `_insert_decision(strategy_params_json={..., "utility_score": …,
   "p_big_loss": …, "model_version_id": mvid})`. This makes
   re-evaluation possible offline against the same candidate.

> The patch is intentionally **additive**: portfolio risk limits, event
> dedupe, and OPEX skip all run unchanged. The only behavioral change
> is the ranking input. This means a regression can be reverted by
> deactivating the model row (`is_active=FALSE`) — no code rollback
> needed.

---

## 6. Verification checklist (pre-activation)

Before flipping a new model to `is_active=TRUE` in production:

- [ ] `metadata.json::model_type` matches one of `xgb_entry_v1`,
      `xgb_v1`, `xgb_entry_v2`. Spot-check by reading the file.
- [ ] `classifier.json` and `regressor.json` load via
      `xgb.Booster().load_model(...)` in a Python REPL.
- [ ] `feature_names` length and order match what
      `extract_xgb_features` produces for that `model_type`.
- [ ] `pytest backend/tests/test_modeling.py -v` is green, especially:
      - `TestPredictXgbEntrySemantics::test_v1_returns_probability_win`
      - `TestPredictXgbEntrySemantics::test_v2_inverts_p_big_loss`
- [ ] `pytest backend/tests/test_xgb_model.py -v` is green, especially:
      - `TestWalkForwardThresholdLeakage::test_threshold_byte_identical_when_only_pure_test_rows_change`
- [ ] Upload was performed via `python -m scripts.upload_xgb_model …`
      (so `algorithm` and `model_payload.model_type` agree).
- [ ] Shadow-mode review: with the model uploaded but
      `is_active=FALSE`, run `_run` (or a forecast
      job) for at least one full trading week and diff the proposed
      ranking against the credit-to-width ranking. Investigate any
      candidate whose ML rank differs by more than 3 positions.

---

## 7. Open follow-ups (deferred)

* **`chosen_legs_json` shape mismatch (Track B follow-up).** Live
  [`decision_job._build_candidate`](decision_job.py) writes the
  selected legs at the **top level** of `chosen_legs_json`:
  `{"short": {...}, "long": {...}, ...}` (see
  [`decision_job.py:678-715`](decision_job.py)). But
  [`extract_xgb_features`](modeling.py) reads them from the
  **nested** `legs.short` / `legs.long` keys, with a flat
  `short_iv` / `long_iv` fallback (see
  [`modeling.py:995-1006`](modeling.py)). Neither path matches the
  live shape, so leg-level features (`short_iv`, `long_iv`,
  `short_delta`, `long_delta`, `iv_skew_ratio`, `credit_to_max_loss`)
  would silently fall through to `None` → `NaN` if XGB scoring were
  re-enabled today. **Recommended fix:** extend `extract_xgb_features`
  to also try the top-level shape (`candidate_json.get("short")`)
  before falling back to the nested or flat forms. Safer than changing
  the live `_build_candidate` payload, which is consumed by
  downstream telemetry and the trade row. Track this before flipping
  any V1/V2 model to `is_active=TRUE`.
* **Per-side models.** The current contract uses a single classifier +
  regressor pair. A future iteration may want separate (call vs put)
  models — that requires an additional `model_payload["side"]` key and
  a small loop in the decision job. Not in scope here.
* **A/B canary.** `model_versions.rollout_status` already supports
  `shadow` / `canary` / `active`. A canary plan (e.g. 25% of runs
  scored by canary, 75% by active) is outside this contract.
* **Re-training cadence.** Decided per-operator. The producer side
  above runs whenever `python -m scripts.run_pipeline --xgb` is
  invoked; cadence is a CI / cron concern.
* **Bucket-empirical fallback.** `model_type = "bucket_empirical_v1"`
  is also accepted by `predict_with_bucket_model` (a sibling of
  `predict_xgb_entry`). Not part of the XGB re-entry path.
