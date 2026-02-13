from datetime import date

from spx_backend.dte import (
    choose_expiration_for_trading_dte,
    closest_expiration_for_trading_dte,
    trading_dte_lookup,
)


def test_trading_dte_lookup_starts_at_zero_when_asof_is_expiration() -> None:
    expirations = [
        date(2026, 2, 12),
        date(2026, 2, 13),
        date(2026, 2, 17),
        date(2026, 2, 18),
    ]
    lookup = trading_dte_lookup(expirations, as_of=date(2026, 2, 12))

    assert lookup[date(2026, 2, 12)] == 0
    assert lookup[date(2026, 2, 13)] == 1
    assert lookup[date(2026, 2, 17)] == 2
    assert lookup[date(2026, 2, 18)] == 3


def test_trading_dte_lookup_starts_at_one_when_asof_not_expiration() -> None:
    expirations = [
        date(2026, 2, 13),
        date(2026, 2, 17),
        date(2026, 2, 18),
    ]
    lookup = trading_dte_lookup(expirations, as_of=date(2026, 2, 12))

    assert lookup[date(2026, 2, 13)] == 1
    assert lookup[date(2026, 2, 17)] == 2
    assert lookup[date(2026, 2, 18)] == 3


def test_choose_expiration_for_trading_dte_exact_match() -> None:
    expirations = [
        date(2026, 2, 12),
        date(2026, 2, 13),
        date(2026, 2, 17),
        date(2026, 2, 18),
        date(2026, 2, 19),
    ]
    exp = choose_expiration_for_trading_dte(
        expirations=expirations,
        target_dte=3,
        as_of=date(2026, 2, 12),
        tolerance=0,
    )

    assert exp == date(2026, 2, 18)


def test_choose_expiration_for_trading_dte_honors_tolerance() -> None:
    expirations = [
        date(2026, 2, 12),
        date(2026, 2, 13),
        date(2026, 2, 17),
        date(2026, 2, 18),
        date(2026, 2, 19),
    ]
    # DTE 4 has no exact match in this small set if we remove Feb 19.
    exp = choose_expiration_for_trading_dte(
        expirations=expirations[:-1],
        target_dte=4,
        as_of=date(2026, 2, 12),
        tolerance=1,
    )

    assert exp == date(2026, 2, 18)


def test_closest_expiration_for_trading_dte_returns_nearest() -> None:
    expirations = [
        date(2026, 2, 12),
        date(2026, 2, 13),
        date(2026, 2, 17),
        date(2026, 2, 18),
    ]
    exp = closest_expiration_for_trading_dte(expirations, target_dte=5, as_of=date(2026, 2, 12))

    assert exp == date(2026, 2, 18)
