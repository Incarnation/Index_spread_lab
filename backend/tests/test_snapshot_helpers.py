from spx_backend.jobs.snapshot_job import _select_strikes_near_spot


def test_select_strikes_near_spot_balances_below_and_above() -> None:
    options = [{"strike": s} for s in [6800, 6810, 6820, 6830, 6840, 6850, 6860]]
    selected = _select_strikes_near_spot(options, spot=6835.0, each_side=2)

    # Expect two nearest below and two nearest above-or-at insertion point.
    assert selected == {6820.0, 6830.0, 6840.0, 6850.0}


def test_select_strikes_near_spot_returns_empty_when_no_strikes() -> None:
    selected = _select_strikes_near_spot(options=[{"strike": None}], spot=6835.0, each_side=2)

    assert selected == set()
