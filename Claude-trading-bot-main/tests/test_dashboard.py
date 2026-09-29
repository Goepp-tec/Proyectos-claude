"""Dashboard: no client/WebSocket leak on refresh; learning tab renders."""

from unittest.mock import patch

import database as db


def _dump(component) -> str:
    from plotly.io.json import to_json_plotly
    return to_json_plotly(component)


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


def test_learning_tab_renders_audit_and_learner_vs_baseline(temp_db):
    import json
    from dashboard import app as dash_app

    db.set_meta("trading_mode", "OBSERVE")
    db.set_meta("learned_params:EMA5_Momentum", json.dumps({"ema_period": 4}))
    db.record_balance(10_050, 50, 0, {"EMA5_Momentum": {"capital": 1300}}, book="observe")
    db.record_balance(10_010, 10, 0, {"EMA5_Momentum": {"capital": 1260}}, book="baseline")
    db.record_learning_audit(ts="2026-09-29T03:53:24+00:00", strategy_name="EMA5_Momentum",
                             decision="applied", reason="validation PF 1.09->1.30",
                             param="ema_period", old_value=5, new_value=4,
                             metrics_before={"profit_factor": 1.09, "max_drawdown": 0.02, "trades": 10},
                             metrics_after={"profit_factor": 1.30, "max_drawdown": 0.02, "trades": 10},
                             book="observe")
    db.record_learning_audit(ts="2026-09-29T03:53:27+00:00", strategy_name="PriceMomentum_25",
                             decision="rejected", reason="validation: only 4 trades (< 10)",
                             param="lookback", old_value=25, new_value=26, book="observe")

    html_out = _dump(dash_app._render_learning())
    for text in ("OBSERVE", "EMA5_Momentum", "ema_period", "applied", "rejected",
                 "only 4 trades", "10,050", "10,010"):
        assert text in html_out, text


def test_learning_tab_is_registered():
    from dashboard import app as dash_app
    assert "tab-learning" in _dump(dash_app.app.layout)
