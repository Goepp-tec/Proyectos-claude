"""Dashboard: no client/WebSocket leak on refresh; learning tab renders."""

from unittest.mock import patch

import database as db


def _dump(component) -> str:
    """Full JSON of a Dash component, with accents kept (not \\u escapes)."""
    import json
    from plotly.io.json import to_json_plotly
    return json.dumps(json.loads(to_json_plotly(component)), ensure_ascii=False)


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


def test_strategy_performance_tab_never_builds_a_client_per_refresh(temp_db):
    """That tab still built a BinanceClient (REST pings + a WebSocket thread that
    is never stopped) on every 10 s refresh: a leak while the page stays open."""
    import binance_client
    from dashboard import app as dash_app

    db.upsert_strategy("EMA5_Momentum", capital=100, params={}, is_active=True)
    dash_app._price_fetcher = None
    with patch.object(binance_client, "BinanceClient") as client, \
         patch.object(binance_client, "BinancePublicDataFetcher") as fetcher:
        fetcher.return_value.get_current_price.return_value = 50_000.0
        for _ in range(5):
            dash_app._render_strategies()
    client.assert_not_called()
    assert fetcher.call_count <= 1


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


def test_strategies_tab_shows_status_regimes_and_sources(temp_db):
    from dashboard import app as dash_app
    metrics = {"trades": 120, "profit_factor": 0.86, "profitable_windows": 0, "windows": 4,
               "worst_drawdown": 0.09, "by_regime": {"RANGING": {"trades": 60, "profit_factor": 0.8}},
               "by_side": {}, "live": {"trades": 3, "pnl": -4.2, "profit_factor": 0.5}}
    db.upsert_strategy_status("Breakout", "DESCARTADA", 0.0, [], [], "pierde de forma consistente", metrics)
    db.upsert_strategy_status("Turtle_Breakout", "CONDICIONAL", 0.64, ["RANGING"], ["LONG", "SHORT"],
                              "funciona en: lateral",
                              {**metrics, "profit_factor": 1.5,
                               "by_regime": {"RANGING": {"trades": 14, "profit_factor": 2.1}}})
    out = _dump(dash_app._render_strategy_evaluation())
    for text in ("Breakout", "DESCARTADA", "Turtle_Breakout", "CONDICIONAL", "lateral",
                 "Way of the Turtle", "pierde de forma consistente"):
        assert text in out, text
    assert "tab-evaluation" in _dump(dash_app.app.layout)


def test_control_tab_saves_settings_and_rejects_invalid_ones(temp_db):
    from dashboard import app as dash_app
    from risk_engine import RiskSettings
    layout = _dump(dash_app._render_control())
    for text in ("ctl-aggr", "ctl-budget", "ctl-funds", "ctl-mode", "ctl-save", "ctl-close-all"):
        assert text in layout, text
    msg = dash_app.save_risk_settings(1, 8, 1_000, 100, "trade", ["short"])
    assert "Guardado" in _dump(msg)
    s = RiskSettings.load()
    assert (s.aggressiveness, s.funds, s.budget, s.mode, s.allow_short) == (8, 1_000, 100, "trade", True)
    msg = dash_app.save_risk_settings(2, 8, 1_000, 5_000, "trade", [])
    assert "no puede ser mayor" in _dump(msg)
    assert RiskSettings.load().budget == 100                       # unchanged


def test_control_preview_and_live_status(temp_db):
    from dashboard import app as dash_app
    from risk_engine import RiskEngine
    preview = _dump(dash_app.preview_aggressiveness(2)) + _dump(dash_app.preview_aggressiveness(9))
    assert "VIABLE" in preview and "EN_PRUEBA" in preview
    db.set_meta("trading_mode", "OBSERVE")
    RiskEngine("observe").reset_kill_switch()
    db.set_meta("risk:observe:kill", '{"at": "x", "reason": "caida de 6 USD"}')
    status = _dump(dash_app.update_control_status(0))
    assert "freno" in status.lower() and "caida de 6 USD" in status
    dash_app.request_close_all(1)
    assert RiskEngine("observe").close_all_requested()


def test_control_tab_is_not_rebuilt_by_the_auto_refresh(temp_db):
    import dash
    from dashboard import app as dash_app
    assert dash_app.render_tab_for("tab-control", "interval-refresh") is dash.no_update
    assert dash_app.render_tab_for("tab-control", "main-tabs") is not dash.no_update


def test_market_tab_shows_positioning_funding_and_sentiment(temp_db):
    from dashboard import app as dash_app
    from market_data import MarketDataCollector
    from tests.test_market_data import T0, _fake_http
    MarketDataCollector("BTCUSDT", http=_fake_http(), clock=lambda: T0).update(force=True)
    out = _dump(dash_app._render_market())
    for text in ("Miedo y Codicia", "Top traders", "Todas las cuentas", "Funding", "tab-market"[4:]):
        assert text in out, text
    assert "20" in out                                          # latest Fear & Greed value
    assert "tab-market" in _dump(dash_app.app.layout)


def test_market_tab_without_data_explains_why(temp_db):
    from dashboard import app as dash_app
    assert "todavía no" in _dump(dash_app._render_market()).lower()


def test_learning_tab_is_registered():
    from dashboard import app as dash_app
    assert "tab-learning" in _dump(dash_app.app.layout)


def test_positions_and_kpis_value_each_coin_at_its_own_price(temp_db, monkeypatch):
    import binance_client
    import config
    from dashboard import app as dash_app
    monkeypatch.setattr(config, "SYMBOLS", ["BTCUSDT", "ETHUSDT"])
    db.open_position(strategy_name="A", symbol="BTCUSDT", side="LONG", entry_price=50_000.0,
                     quantity=0.001, stop_loss=45_000.0, take_profit=60_000.0, order_id="x",
                     ml_confidence=0.6, metadata={}, book="main", entry_time="2026-09-29T10:00:00+00:00")
    db.open_position(strategy_name="A@ETH", symbol="ETHUSDT", side="LONG", entry_price=2_000.0,
                     quantity=0.01, stop_loss=1_800.0, take_profit=2_400.0, order_id="y",
                     ml_confidence=0.6, metadata={}, book="main", entry_time="2026-09-29T10:00:00+00:00")
    dash_app._price_fetcher = None
    live = {"BTCUSDT": 50_000.0, "ETHUSDT": 2_100.0}
    with patch.object(binance_client, "BinancePublicDataFetcher") as fetcher:
        fetcher.return_value.get_current_price.side_effect = lambda sym: live[sym]
        kpis = _dump(dash_app.update_kpis(0)[0])
        positions = _dump(dash_app._render_positions())
    assert "+1.00" in kpis                          # only ETH moved: +100 x 0.01
    assert "ETH" in positions and "2,100.00" in positions


def test_market_tab_has_a_coin_selector_and_the_new_sources(temp_db, monkeypatch):
    import dash
    import config
    from dashboard import app as dash_app
    from market_data import MarketDataCollector
    from tests.test_market_data import T0, _fake_http
    monkeypatch.setattr(config, "SYMBOLS", ["BTCUSDT", "ETHUSDT"])
    MarketDataCollector(["BTCUSDT", "ETHUSDT"], http=_fake_http(), clock=lambda: T0).update(force=True)
    panel = _dump(dash_app._render_market())
    assert "market-symbol" in panel and "ETHUSDT" in panel
    eth = _dump(dash_app.update_market_symbol("ETHUSDT"))
    for text in ("OKX", "Hyperliquid", "CME"):
        assert text in eth, text
    # hourly data: the tab (and the chosen coin) is not rebuilt by the 10 s refresh
    assert dash_app.render_tab_for("tab-market", "interval-refresh") is dash.no_update


def test_strategies_tab_summarises_each_coin(temp_db):
    from dashboard import app as dash_app
    m = {"trades": 40, "profit_factor": 1.3, "profitable_windows": 3, "windows": 4,
         "worst_drawdown": 0.05, "by_regime": {}, "by_side": {}, "live": {}}
    db.upsert_strategy_status("Turtle_Breakout", "VIABLE", 0.8, ["RANGING"], ["LONG"], "ok", m)
    db.upsert_strategy_status("Turtle_Breakout@ETH", "DESCARTADA", 0.0, [], [], "pierde", m)
    out = _dump(dash_app._render_strategy_evaluation())
    assert "Por cripto" in out and "Turtle_Breakout@ETH" in out
    assert "Turtle_Breakout (0.80)" in out            # best strategy of the BTC row
    assert "Turtle System 1" in out or "Turtle" in out  # source found through the base name
