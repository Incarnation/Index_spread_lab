from spx_backend.jobs.gex_job import GexJob


def test_zero_gamma_level_interpolates_crossing() -> None:
    job = GexJob()
    per_strike = {
        6800.0: {"gex_calls": 100.0, "gex_puts": -300.0},  # cumulative: -200
        6825.0: {"gex_calls": 250.0, "gex_puts": -20.0},   # cumulative: +30 (cross)
    }

    zero = job._zero_gamma_level(per_strike)

    assert zero is not None
    assert 6800.0 < zero < 6825.0


def test_zero_gamma_level_returns_none_without_cross() -> None:
    job = GexJob()
    per_strike = {
        6800.0: {"gex_calls": 200.0, "gex_puts": -50.0},
        6825.0: {"gex_calls": 150.0, "gex_puts": -40.0},
    }

    zero = job._zero_gamma_level(per_strike)

    assert zero is None
