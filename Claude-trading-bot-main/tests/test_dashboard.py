"""Dashboard: no client/WebSocket leak on refresh; learning tab renders."""

from unittest.mock import patch

import database as db


def test_kpi_refresh_reuses_one_price_fetcher_and_never_opens_websockets(temp_db):
    import binance_client
    from dashboard import app as dash_app

    dash_app._price_fetcher = None
    with patch.object(binance_client, "BinancePublicDataFetcher") as fetcher, \
         patch.object(binance_client, "BinanceWebSocketClient") as ws:
        fetcher.return_value.get_current_price.return_value = 50_000.0
        for _ in range(5):
            dash_app.update_kpis(0)
    assert fetcher.call_count <= 1
    ws.assert_not_called()
