"""During a long test run the trading mode chosen at the first start survives restarts."""

from unittest.mock import Mock

from tests.test_portfolio_risk import _strategy


def _results(**passes):
    return {name: Mock(passes_threshold=ok) for name, ok in passes.items()}


def test_locked_mode_is_reused_after_restart_even_if_backtest_changes(temp_db):
    import main
    strats = [_strategy("A"), _strategy("B")]
    first = main.resolve_trading_mode(strats, _results(A=False, B=False),
                                      allow_unvalidated=False, lock=True)
    assert first[1:] == ("observe", "OBSERVE")

    # After a restart strategy A now passes its (re-run) backtest...
    second = main.resolve_trading_mode(strats, _results(A=True, B=False),
                                       allow_unvalidated=False, lock=True)
    assert second[1:] == ("observe", "OBSERVE")          # ...but the test keeps its mode
    assert [s.name for s in second[0]] == ["A", "B"]


def test_without_lock_the_mode_follows_the_backtest(temp_db):
    import main
    strats = [_strategy("A"), _strategy("B")]
    main.resolve_trading_mode(strats, _results(A=False, B=False), False, lock=False)
    active, book, mode = main.resolve_trading_mode(strats, _results(A=True, B=False), False, lock=False)
    assert mode == "TRADE" and [s.name for s in active] == ["A"]
