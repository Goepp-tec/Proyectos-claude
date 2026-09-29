"""A5: WebSocket reconnects with capped exponential backoff, then falls back to REST."""

import logging

from binance_client import BinanceWebSocketClient


def _client(outcomes):
    """outcomes: list of Exception (failed attempt) or None (connected, then closed)."""
    ws = BinanceWebSocketClient(["btcusdt"])
    ws._running = True
    waits = []
    seq = iter(outcomes)

    def connect_once(url):
        try:
            outcome = next(seq)
        except StopIteration:
            ws._running = False
            return
        if outcome is not None:
            raise outcome
        ws._ever_connected = True

    ws._connect_once = connect_once
    ws._wait = waits.append
    return ws, waits


def test_persistent_block_backs_off_exponentially_then_uses_rest_only(caplog):
    ws, waits = _client([ConnectionResetError("reset by peer")] * 50)
    with caplog.at_level(logging.WARNING, logger="binance_client"):
        ws._run()

    assert waits == [5, 10, 20, 40, 80, 160, 300]       # capped at 300 s
    assert ws.rest_only is True
    assert ws._running is False
    ws_logs = [r for r in caplog.records if "WebSocket" in r.getMessage()]
    assert len(ws_logs) == 8                               # 8 attempts -> exactly one log each
    assert "REST polling only" in ws_logs[-1].getMessage()


def test_successful_connection_resets_backoff():
    err = ConnectionResetError("reset")
    ws, waits = _client([err, err, None, err])
    ws._run()
    assert waits == [5, 10, 5, 5]
    assert ws.rest_only is False
