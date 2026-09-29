# Evidencias – corrida demo (paper trading) 2026-09-29

- Entorno: contenedor cloud Linux, Python 3.11.15, venv + `pip install -r requirements.txt` OK.
- Acceso a Binance: `api.binance.com` y `testnet.binance.vision` → **HTTP 451** (región restringida).
  `data-api.binance.vision` (mirror oficial de datos públicos de Binance) → 200.
  El bot usó **precios reales de Binance** vía ese mirror (override `BINANCE_PUBLIC_BASE` en `.env`).
  Sin ese override el bot no arranca: `ConnectionError` → `sys.exit(1)` (no existe fallback a datos simulados).
- Modo: PAPER_TRADING=true, USE_TESTNET=true, INITIAL_CAPITAL=10000, sin API keys (ver `env_usado.txt`).
- Corrida de main.py: 01:39:25 → 01:52:13 UTC (~13 min), detenido con SIGTERM ("Shutdown complete").

| Archivo | Contenido |
|---|---|
| pytest_output.txt | pytest -v: 10 passed, 2 failed (SyntaxError en strategies/ml_adaptive.py:199) |
| backtest_output.txt | salida de run_backtest_only.py (tabla por estrategia) |
| ../backtest_results.html | curvas de equity regeneradas |
| run_demo.log / run_demo_sin_colores.log | stdout+stderr de main.py |
| trading_bot.log | log propio del bot |
| monitor.txt | chequeos cada 2 min: proceso, HTTP dashboard, trades, errores, balance |
| dashboard_http.txt | último código HTTP de http://localhost:8050 antes de detener |
| trades.csv / positions.csv / balance_history.csv / strategies.csv / journal_entries.csv | export de trading_bot.db |
| trading_bot_snapshot.db | copia SQLite de la DB antes de detener |
| pip_freeze.txt / python_version.txt | versiones exactas |
