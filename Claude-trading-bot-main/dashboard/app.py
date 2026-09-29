"""
Trading Bot Dashboard – Dash / Plotly
──────────────────────────────────────
Five tabs:
  1. Portfolio Overview  – equity curve, balance, unrealized P&L
  2. Strategy Performance – per-strategy metrics and equity curves
  3. Open Positions       – live table with unrealized P&L
  4. Trade History        – filterable trade log
  5. Trade Journal        – individual entries with reflections

Reads all data from the SQLite database; auto-refreshes every 15 s.
Run standalone:  python dashboard/app.py
Or imported by main.py for in-process startup.
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import config
import database as db

# ─── App bootstrap ────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="BTC Trading Bot",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ─── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#0d1117",
    "card":     "#161b22",
    "border":   "#30363d",
    "green":    "#3fb950",
    "red":      "#f85149",
    "yellow":   "#d29922",
    "blue":     "#58a6ff",
    "purple":   "#bc8cff",
    "text":     "#c9d1d9",
    "subtext":  "#8b949e",
}

STRATEGY_PALETTE = [
    "#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff"
]

# ─── Layout helpers ───────────────────────────────────────────────────────────

def _metric_card(title: str, value: str, color: str = "text",
                 subtitle: str = "") -> dbc.Card:
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
            html.H4(value, style={"color": COLORS[color], "fontWeight": "bold", "margin": 0}),
            html.Small(subtitle, style={"color": COLORS["subtext"]}) if subtitle else None,
        ], style={"padding": "12px 16px"}),
    ], style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
              "borderRadius": "8px"})


def _empty_fig(msg: str = "No data yet") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font={"color": COLORS["subtext"], "size": 14})
    fig.update_layout(**_dark_layout())
    return fig


def _dark_layout(title: str = "") -> dict:
    return dict(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["card"],
        font_color=COLORS["text"],
        title=title,
        title_font_color=COLORS["blue"],
        xaxis=dict(gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["border"], zeroline=False),
        legend=dict(bgcolor=COLORS["card"], bordercolor=COLORS["border"]),
        margin=dict(l=40, r=20, t=40, b=40),
    )


# ─── Main layout ──────────────────────────────────────────────────────────────

app.layout = dbc.Container(fluid=True, style={"backgroundColor": COLORS["bg"],
                                               "minHeight": "100vh", "padding": "0"}, children=[

    # Auto-refresh interval
    dcc.Interval(id="interval-refresh", interval=config.DASHBOARD_UPDATE_MS, n_intervals=0),
    dcc.Store(id="store-price"),
    dcc.Store(id="store-balance"),

    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Navbar(
        dbc.Container([
            html.Span("₿ BTC Trading Bot", style={
                "color": COLORS["yellow"], "fontWeight": "bold", "fontSize": "1.2rem"
            }),
            html.Span(id="header-mode", style={"color": COLORS["subtext"], "fontSize": "0.85rem"}),
            html.Span(id="header-time", style={"color": COLORS["subtext"], "fontSize": "0.8rem"}),
        ], fluid=True, style={"display": "flex", "justifyContent": "space-between",
                              "alignItems": "center"}),
        color=COLORS["card"], dark=True,
        style={"borderBottom": f"1px solid {COLORS['border']}", "padding": "8px 20px"}
    ),

    # ── Top KPI row ───────────────────────────────────────────────────────────
    dbc.Row(id="kpi-row", className="g-2 my-2 mx-2"),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dbc.Tabs(id="main-tabs", active_tab="tab-overview", style={"margin": "0 12px"},
             children=[
        dbc.Tab(label="Portfolio Overview",    tab_id="tab-overview"),
        dbc.Tab(label="Strategy Performance",  tab_id="tab-strategies"),
        dbc.Tab(label="Open Positions",        tab_id="tab-positions"),
        dbc.Tab(label="Trade History",         tab_id="tab-history"),
        dbc.Tab(label="Trade Journal",         tab_id="tab-journal"),
        dbc.Tab(label="Aprendizaje",           tab_id="tab-learning"),
        dbc.Tab(label="Estrategias",           tab_id="tab-evaluation"),
        dbc.Tab(label="Mercado",               tab_id="tab-market"),
        dbc.Tab(label="Control",               tab_id="tab-control"),
    ]),

    html.Div(id="tab-content", style={"padding": "12px 12px 30px"}),
])

# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("header-time", "children"),
    Output("header-mode", "children"),
    Input("interval-refresh", "n_intervals"),
)
def update_header(_):
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mode = "● TESTNET" if config.USE_TESTNET else "● LIVE"
    color = COLORS["yellow"] if config.USE_TESTNET else COLORS["red"]
    trading_mode = db.get_meta("trading_mode")
    if trading_mode:
        mode += f" · {trading_mode}"
    return now, html.Span(mode, style={"color": color, "marginLeft": "12px"})


_price_fetcher = None


def _get_price_fetcher():
    """One REST price reader for the dashboard. Building a BinanceClient on every
    refresh used to ping Binance twice and start a WebSocket thread that was
    never stopped (one leaked connection every 10 s while the page was open)."""
    global _price_fetcher
    if _price_fetcher is None:
        import binance_client
        _price_fetcher = binance_client.BinancePublicDataFetcher()
    return _price_fetcher


@app.callback(
    Output("kpi-row", "children"),
    Output("store-balance", "data"),
    Input("interval-refresh", "n_intervals"),
)
def update_kpis(_):
    bal  = db.get_latest_balance() or {}
    total   = bal.get("total_balance", config.INITIAL_CAPITAL)
    real    = bal.get("realized_pnl", 0.0)

    # Compute unrealized P&L LIVE from current open positions + latest price
    # (don't trust stale DB values that may be minutes old)
    try:
        current_price = _get_price_fetcher().get_current_price(config.SYMBOL)
        unreal = 0.0
        for pos in db.get_open_positions():
            ep = float(pos["entry_price"])
            qty = float(pos["quantity"])
            if pos["side"] == "LONG":
                unreal += (current_price - ep) * qty
            else:
                unreal += (ep - current_price) * qty
    except Exception:
        # Fallback to DB value if live calculation fails
        unreal = bal.get("unrealized_pnl", 0.0)

    total_pnl = real + unreal
    pct_chg   = (total_pnl / config.INITIAL_CAPITAL) * 100 if config.INITIAL_CAPITAL else 0

    stats   = db.get_trade_stats(include_backtest=config.SHOW_BACKTEST_DATA)
    wins    = int(stats.get("wins") or 0)
    total_t = int(stats.get("total_trades") or 0)
    wr_str  = f"{wins}/{total_t}" if total_t > 0 else "0/0"
    wr_pct  = stats.get("win_rate", 0)

    cards = [
        dbc.Col(_metric_card("Total Balance", f"${total:,.2f}",
                             "green" if total >= config.INITIAL_CAPITAL else "red"), width=2),
        dbc.Col(_metric_card("Unrealized P&L", f"${unreal:+,.2f}",
                             "green" if unreal >= 0 else "red",
                             subtitle="Open positions"), width=2),
        dbc.Col(_metric_card("Realized P&L", f"${real:+,.2f}",
                             "green" if real >= 0 else "red",
                             subtitle="Closed trades"), width=2),
        dbc.Col(_metric_card("Total Return", f"{pct_chg:+.2f}%",
                             "green" if pct_chg >= 0 else "red",
                             subtitle=f"from ${config.INITIAL_CAPITAL:,.0f}"), width=2),
        dbc.Col(_metric_card("Win Rate", f"{wr_pct*100:.1f}%",
                             "green" if wr_pct >= 0.5 else "yellow",
                             subtitle=f"{wr_str} trades"), width=2),
        dbc.Col(_metric_card("Open Positions", str(len(db.get_open_positions())),
                             color="blue"), width=2),
    ]
    
    # Add "Live Since" card if available
    live_since = db.get_live_since()
    if live_since:
        # Extract just the date
        live_since_date = live_since.split("T")[0] if "T" in live_since else live_since
        cards.append(dbc.Col(_metric_card("Live Since", live_since_date,
                             color="purple"), width=2))
    
    return cards, bal


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("interval-refresh", "n_intervals"),
)
def render_tab(active_tab, _):
    ctx = dash.callback_context
    trigger = ctx.triggered_id if ctx.triggered else None
    return render_tab_for(active_tab, trigger)


def render_tab_for(active_tab, trigger):
    if active_tab == "tab-control":
        # A form: rebuilding it every 10 s would wipe what the user is typing.
        # Its live status box has its own refresh callback.
        return dash.no_update if trigger == "interval-refresh" else _render_control()
    if active_tab == "tab-overview":
        return _render_overview()
    elif active_tab == "tab-strategies":
        return _render_strategies()
    elif active_tab == "tab-positions":
        return _render_positions()
    elif active_tab == "tab-history":
        return _render_history()
    elif active_tab == "tab-journal":
        return _render_journal()
    elif active_tab == "tab-learning":
        return _render_learning()
    elif active_tab == "tab-evaluation":
        return _render_strategy_evaluation()
    elif active_tab == "tab-market":
        return _render_market()
    return html.Div("Select a tab")


# ─── Tab renderers ────────────────────────────────────────────────────────────

def _render_overview():
    # Get balance history - filter out backtest data based on config
    history = db.get_balance_history(days=90, include_backtest=config.SHOW_BACKTEST_DATA)
    if history:
        df = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["recorded_at"], y=df["total_balance"],
            name="Portfolio Balance", fill="tozeroy",
            line=dict(color=COLORS["blue"], width=2),
            fillcolor="rgba(88,166,255,0.12)",
        ))
        fig.add_hline(y=config.INITIAL_CAPITAL, line_dash="dot",
                      line_color=COLORS["subtext"], annotation_text="Initial Capital")
        fig.update_layout(**_dark_layout("Portfolio Equity Curve"), height=320)

        # Drawdown
        eq = df["total_balance"].values
        peak = pd.Series(eq).cummax().values
        dd   = (peak - eq) / (peak + 1e-8) * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df["recorded_at"], y=-dd,
            fill="tozeroy", name="Drawdown %",
            line=dict(color=COLORS["red"], width=1),
            fillcolor="rgba(248,81,73,0.15)",
        ))
        fig_dd.update_layout(**_dark_layout("Drawdown (%)"), height=160,
                             yaxis_ticksuffix="%")
    else:
        fig    = _empty_fig("No balance history yet. Waiting for first data point.")
        fig_dd = _empty_fig("No drawdown data")

    # Strategy allocation pie
    active = db.get_active_strategies()
    if active:
        names  = [s["name"] for s in active]
        caps   = [s.get("capital", config.INITIAL_CAPITAL / config.MAX_STRATEGIES) for s in active]
        fig_pie = go.Figure(go.Pie(
            labels=names, values=caps, hole=0.5,
            marker=dict(colors=STRATEGY_PALETTE[:len(names)]),
        ))
        fig_pie.update_layout(**_dark_layout("Capital Allocation"), height=260,
                              showlegend=True)
    else:
        fig_pie = _empty_fig("No active strategies")

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}), width=9),
            dbc.Col(dcc.Graph(figure=fig_pie, config={"displayModeBar": False}), width=3),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_dd, config={"displayModeBar": False}), width=12),
        ], className="g-2 mt-1"),
    ])


def _render_strategies():
    strategies = db.get_active_strategies()
    if not strategies:
        return html.Div("No active strategies.", style={"color": COLORS["subtext"], "padding": "20px"})

    rows = []
    charts = []

    # Get current price for unrealized P&L calculation
    try:
        current_price = _get_price_fetcher().get_current_price(config.SYMBOL)
    except Exception:
        current_price = 0.0

    for i, strat in enumerate(strategies):
        name  = strat["name"]
        stats = db.get_trade_stats(name, include_backtest=config.SHOW_BACKTEST_DATA)
        db_capital = strat.get("capital", 0)  # Free capital from DB
        wr    = float(stats.get("win_rate") or 0)
        realized_pnl = float(stats.get("total_pnl") or 0)
        n     = int(stats.get("total_trades") or 0)
        bt_cagr = float(strat.get("backtest_cagr") or 0)
        bt_wr   = float(strat.get("backtest_win_rate") or 0)

        # Calculate committed notional and unrealized P&L for this strategy
        open_pos = db.get_open_positions(name)
        committed = 0.0
        unrealized_pnl = 0.0
        for pos in open_pos:
            ep = float(pos["entry_price"])
            qty = float(pos["quantity"])
            committed += ep * qty
            if current_price > 0:
                if pos["side"] == "LONG":
                    unrealized_pnl += (current_price - ep) * qty
                else:
                    unrealized_pnl += (ep - current_price) * qty

        # Total Capital = Initial Share + Realized P&L + Unrealized P&L
        # Free Capital = Total Capital - Committed Notional (locked in open positions)
        #
        # db_capital = initial share allocated to strategy
        initial_share = db_capital
        true_total_cap = initial_share + realized_pnl + unrealized_pnl
        free_cap = true_total_cap - committed  # After subtracting locked-in positions
        total_pnl = realized_pnl + unrealized_pnl

        rows.append({
            "Strategy": name,
            "Total Cap": f"${true_total_cap:,.2f}",
            "Free Cap": f"${free_cap:,.2f}",
            "Committed": f"${committed:,.2f}" if committed > 0 else "—",
            "Realized P&L": f"${realized_pnl:+.2f}" if realized_pnl != 0 else "$0.00",
            "Unrealized P&L": f"${unrealized_pnl:+.2f}" if unrealized_pnl != 0 else "$0.00",
            "Total P&L": f"${total_pnl:+.2f}" if total_pnl != 0 else "$0.00",
            "Closed Trades": n,
            "Win Rate": f"{wr*100:.1f}%",
            "BT CAGR": f"{bt_cagr*100:.1f}%",
        })

        # Mini equity curve per strategy
        perf = db.get_strategy_performance_history(name, days=60)
        color = STRATEGY_PALETTE[i % len(STRATEGY_PALETTE)]
        if perf:
            pdf = pd.DataFrame(perf)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pdf["date"], y=pdf["capital"],
                name=name, line=dict(color=color, width=2), fill="tozeroy",
                fillcolor=f"rgba{tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0,2,4)) + (0.10,)}",
            ))
            fig.update_layout(**_dark_layout(name))
            fig.update_layout(height=200, showlegend=False, margin=dict(l=30, r=10, t=30, b=30))
        else:
            fig = _empty_fig(f"{name}: no history")
            fig.update_layout(height=200)

        charts.append(dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}), md=4))

    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": COLORS["card"], "color": COLORS["blue"],
                      "fontWeight": "bold", "border": f"1px solid {COLORS['border']}"},
        style_cell={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}", "fontSize": "13px",
                    "padding": "8px 12px"},
        style_data_conditional=[
            # Realized P&L coloring
            {"if": {"filter_query": '{Realized P&L} contains "+"'}, "color": COLORS["green"]},
            {"if": {"filter_query": '{Realized P&L} contains "-"'}, "color": COLORS["red"]},
            # Unrealized P&L coloring
            {"if": {"filter_query": '{Unrealized P&L} contains "+"'}, "color": COLORS["green"], "fontWeight": "bold"},
            {"if": {"filter_query": '{Unrealized P&L} contains "-"'}, "color": COLORS["red"], "fontWeight": "bold"},
            # Total P&L coloring
            {"if": {"filter_query": '{Total P&L} contains "+"'}, "color": COLORS["green"]},
            {"if": {"filter_query": '{Total P&L} contains "-"'}, "color": COLORS["red"]},
        ],
    )

    return html.Div([
        html.H6("Strategy Metrics", style={"color": COLORS["blue"], "marginBottom": "10px"}),
        table,
        html.H6("Equity Curves", style={"color": COLORS["blue"], "margin": "16px 0 8px"}),
        dbc.Row(charts, className="g-2"),
    ])


def _render_positions():
    positions = db.get_open_positions()
    if not positions:
        return html.Div([
            html.P("No open positions.", style={"color": COLORS["subtext"], "padding": "20px"}),
        ])

    # Need current price for unrealized PnL
    # Try to read from DB or use last stored balance
    last_bal = db.get_latest_balance()
    try:
        bd = last_bal.get("strategy_breakdown", {}) if last_bal else {}
        # Try reading a stored price from a JSON field
        current_price = None
        for v in bd.values():
            if isinstance(v, dict) and "current_price" in v:
                current_price = v["current_price"]
                break
    except Exception:
        current_price = None

    rows = []
    for p in positions:
        ep  = float(p["entry_price"])
        qty = float(p["quantity"])
        sl  = float(p["stop_loss"] or 0)
        tp  = float(p["take_profit"] or 0)
        ml  = float(p.get("ml_confidence") or 0.5)

        unreal = "N/A"
        unreal_pct = "N/A"
        if current_price:
            if p["side"] == "LONG":
                ur = (current_price - ep) * qty
            else:
                ur = (ep - current_price) * qty
            unreal     = f"${ur:+.2f}"
            unreal_pct = f"{(ur / (ep * qty)) * 100:+.2f}%"

        cur_price_str = f"${current_price:,.2f}" if current_price else "N/A"

        rows.append({
            "ID": p["id"],
            "Strategy": p["strategy_name"],
            "Side": p["side"],
            "Entry Price": f"${ep:,.2f}",
            "Current Price": cur_price_str,
            "Qty (BTC)": f"{qty:.5f}",
            "Notional": f"${ep * qty:,.2f}",
            "Stop Loss": f"${sl:,.2f}",
            "Take Profit": f"${tp:,.2f}",
            "Unrealized P&L": unreal,
            "Unrealized %": unreal_pct,
            "ML Conf.": f"{ml:.2f}",
            "Entry Time": p.get("entry_time", "")[:16],
        })

    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": COLORS["card"], "color": COLORS["blue"],
                      "fontWeight": "bold", "border": f"1px solid {COLORS['border']}"},
        style_cell={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}", "fontSize": "12px",
                    "padding": "6px 10px", "whiteSpace": "nowrap"},
        style_data_conditional=[
            {"if": {"filter_query": '{Side} = "LONG"'},  "color": COLORS["green"]},
            {"if": {"filter_query": '{Side} = "SHORT"'}, "color": COLORS["red"]},
            # Current Price: green when winning (LONG up / SHORT down), red otherwise
            {"if": {"filter_query": '{Unrealized P&L} contains "+"',
                    "column_id": "Current Price"},
             "color": COLORS["green"], "fontWeight": "bold"},
            {"if": {"filter_query": '{Unrealized P&L} contains "-"',
                    "column_id": "Current Price"},
             "color": COLORS["red"], "fontWeight": "bold"},
        ],
    )

    return html.Div([
        html.H6(f"{len(positions)} Open Position(s)",
                style={"color": COLORS["blue"], "marginBottom": "10px"}),
        table,
    ])


def _render_history():
    # Get trades - filter out backtest data based on config
    trades = db.get_trades(limit=200, include_backtest=config.SHOW_BACKTEST_DATA)
    if not trades:
        return html.Div("No closed trades yet.", style={"color": COLORS["subtext"], "padding": "20px"})

    rows = []
    for t in trades:
        pnl     = float(t["pnl"])
        pnl_pct = float(t["pnl_pct"]) * 100
        rows.append({
            "Date": t.get("exit_time", "")[:16],
            "Strategy": t["strategy_name"],
            "Side": t["side"],
            "Entry $": f"{float(t['entry_price']):,.2f}",
            "Exit $": f"{float(t['exit_price']):,.2f}",
            "Qty": f"{float(t['quantity']):.5f}",
            "P&L $": f"{pnl:+.2f}",
            "P&L %": f"{pnl_pct:+.2f}%",
            "Fees": f"${float(t['fees_paid']):.2f}",
            "Duration": f"{float(t['duration_hours']):.1f}h",
            "Exit Reason": t.get("exit_reason", ""),
        })

    # P&L distribution histogram
    pnls = [float(t["pnl"]) for t in trades]
    fig_hist = go.Figure(go.Histogram(
        x=pnls, nbinsx=30,
        marker_color=[COLORS["green"] if p >= 0 else COLORS["red"] for p in pnls],
        opacity=0.8,
    ))
    fig_hist.update_layout(**_dark_layout("P&L Distribution"), height=200,
                           xaxis_title="P&L ($)", yaxis_title="Count",
                           bargap=0.05)

    # Cumulative PnL chart
    cum_pnl = []
    running = 0
    dates   = []
    for t in reversed(trades):
        running += float(t["pnl"])
        cum_pnl.append(running)
        dates.append(t.get("exit_time", ""))
    fig_cum = go.Figure(go.Scatter(
        x=dates, y=cum_pnl,
        fill="tozeroy", line=dict(color=COLORS["blue"], width=2),
        fillcolor="rgba(88,166,255,0.1)",
    ))
    fig_cum.update_layout(**_dark_layout("Cumulative Realized P&L"), height=200,
                          yaxis_tickprefix="$")

    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": COLORS["card"], "color": COLORS["blue"],
                      "fontWeight": "bold", "border": f"1px solid {COLORS['border']}"},
        style_cell={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}", "fontSize": "12px",
                    "padding": "5px 9px", "whiteSpace": "nowrap"},
        style_data_conditional=[
            {"if": {"filter_query": '{P&L $} contains "+"'}, "color": COLORS["green"]},
            {"if": {"filter_query": '{P&L $} contains "-"'}, "color": COLORS["red"]},
        ],
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cum, config={"displayModeBar": False}), width=7),
            dbc.Col(dcc.Graph(figure=fig_hist, config={"displayModeBar": False}), width=5),
        ], className="g-2 mb-3"),
        html.H6(f"Trade History ({len(rows)} trades)",
                style={"color": COLORS["blue"], "marginBottom": "8px"}),
        table,
    ])


def _render_journal():
    # Get journal entries - filter out backtest entries based on config
    entries = db.get_journal_entries(limit=50, include_backtest=config.SHOW_BACKTEST_DATA)
    if not entries:
        return html.Div("No journal entries yet. Entries are created after each closed trade.",
                        style={"color": COLORS["subtext"], "padding": "20px"})

    cards = []
    for e in entries:
        pnl     = float(e.get("pnl") or 0)
        pnl_pct = float(e.get("pnl_pct") or 0) * 100
        won     = pnl > 0
        border_color = COLORS["green"] if won else COLORS["red"]
        badge_color  = "success" if won else "danger"
        badge_text   = f"+{pnl_pct:.2f}%" if won else f"{pnl_pct:.2f}%"

        cards.append(dbc.Card([
            dbc.CardHeader([
                html.Span(e.get("strategy_name", ""), style={"fontWeight": "bold",
                          "color": COLORS["blue"]}),
                html.Span("  "),
                dbc.Badge(e.get("side", ""), color="info", className="me-2"),
                dbc.Badge(badge_text, color=badge_color, className="me-2"),
                html.Small(e.get("created_at", "")[:16],
                           style={"color": COLORS["subtext"], "float": "right"}),
            ], style={"backgroundColor": COLORS["card"]}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P([html.Strong("Setup: "), e.get("setup_summary", "")],
                               style={"fontSize": "13px", "color": COLORS["text"]}),
                        html.P([html.Strong("Outcome: "), e.get("outcome_analysis", "")],
                               style={"fontSize": "13px", "color": COLORS["text"]}),
                    ], md=5),
                    dbc.Col([
                        html.P([html.Strong("Reflection: ")],
                               style={"fontSize": "13px", "color": COLORS["yellow"],
                                      "marginBottom": "2px"}),
                        html.P(e.get("reflection", ""),
                               style={"fontSize": "12px", "color": COLORS["text"],
                                      "fontStyle": "italic", "borderLeft":
                                      f"3px solid {COLORS['border']}",
                                      "paddingLeft": "10px"}),
                    ], md=4),
                    dbc.Col([
                        html.P([html.Strong("Lessons: ")],
                               style={"fontSize": "13px", "color": COLORS["purple"],
                                      "marginBottom": "2px"}),
                        html.P(e.get("lessons", ""),
                               style={"fontSize": "12px", "color": COLORS["text"]}),
                    ], md=3),
                ]),
            ], style={"backgroundColor": COLORS["bg"], "padding": "12px 16px"}),
        ], style={"border": f"1px solid {border_color}", "borderRadius": "6px",
                  "marginBottom": "12px"}))

    return html.Div([
        html.H6("Trade Journal", style={"color": COLORS["blue"], "marginBottom": "16px"}),
        html.Div(cards),
    ])


def _table(rows, page_size=15, conditional=None):
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": COLORS["card"], "color": COLORS["blue"],
                      "fontWeight": "bold", "border": f"1px solid {COLORS['border']}"},
        style_cell={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}", "fontSize": "12px",
                    "padding": "5px 9px", "whiteSpace": "normal", "textAlign": "left"},
        style_data_conditional=conditional or [],
    )


def _render_learning():
    """Learner vs frozen baseline, current params and the learning audit trail."""
    from strategies import ALL_STRATEGIES

    trading_mode = db.get_meta("trading_mode") or "?"
    learner_book = "observe" if trading_mode == "OBSERVE" else "main"
    learn_bal = db.get_latest_balance(book=learner_book) or {}
    base_bal  = db.get_latest_balance(book="baseline") or {}
    learn_eq  = learn_bal.get("total_balance", config.INITIAL_CAPITAL)
    base_eq   = base_bal.get("total_balance", config.INITIAL_CAPITAL)
    audit     = db.get_learning_audit(limit=300)
    counts    = {d: sum(1 for r in audit if r["decision"] == d)
                 for d in ("applied", "rejected", "rollback")}

    notice = None
    if trading_mode == "OBSERVE":
        notice = dbc.Alert(
            "Modo OBSERVACIÓN: ninguna estrategia pasó el backtest, así que no se abren "
            "posiciones en el libro principal. Las operaciones de abajo son teóricas "
            "(libro 'observe'). Para operar igualmente: ALLOW_UNVALIDATED_STRATEGIES=true.",
            color="warning", style={"fontSize": "13px"})

    diff = learn_eq - base_eq
    cards = dbc.Row([
        dbc.Col(_metric_card("Modo", trading_mode, "yellow",
                             subtitle=f"libro que aprende: {learner_book}"), width=2),
        dbc.Col(_metric_card("Equity: aprende", f"${learn_eq:,.2f}",
                             "green" if learn_eq >= config.INITIAL_CAPITAL else "red"), width=2),
        dbc.Col(_metric_card("Equity: baseline", f"${base_eq:,.2f}",
                             "green" if base_eq >= config.INITIAL_CAPITAL else "red",
                             subtitle="parámetros congelados"), width=2),
        dbc.Col(_metric_card("Aprende − baseline", f"${diff:+,.2f}",
                             "green" if diff >= 0 else "red"), width=2),
        dbc.Col(_metric_card("Ajustes aplicados", str(counts["applied"]), "blue",
                             subtitle=f"{counts['rollback']} revertidos"), width=2),
        dbc.Col(_metric_card("Propuestas rechazadas", str(counts["rejected"]), "subtext"), width=2),
    ], className="g-2 mb-3")

    fig = go.Figure()
    for book, label, color in ((learner_book, "Aprende", COLORS["blue"]),
                               ("baseline", "Baseline (congelado)", COLORS["subtext"])):
        hist = db.get_balance_history(days=90, include_backtest=True, book=book)
        if hist:
            df = pd.DataFrame(hist)
            fig.add_trace(go.Scatter(x=df["recorded_at"], y=df["total_balance"],
                                     name=label, line=dict(color=color, width=2)))
    if fig.data:
        fig.add_hline(y=config.INITIAL_CAPITAL, line_dash="dot", line_color=COLORS["border"])
        fig.update_layout(**_dark_layout("Equity: aprende vs baseline"), height=300)
    else:
        fig = _empty_fig("Sin historial de equity todavía")

    learn_bd = learn_bal.get("strategy_breakdown", {})
    base_bd  = base_bal.get("strategy_breakdown", {})
    param_rows = []
    for S in ALL_STRATEGIES:
        default = S()
        saved = json.loads(db.get_meta(f"learned_params:{default.name}") or "{}")
        for p, spec in default.TUNABLE_PARAMS.items():
            current = saved.get(p, default.params[p])
            param_rows.append({
                "Estrategia": default.name,
                "Parámetro": p,
                "Actual (aprende)": current,
                "Baseline": default.params[p],
                "Rango": f"{spec.min} – {spec.max} (paso {spec.step})",
                "Equity aprende": f"${learn_bd.get(default.name, {}).get('capital', 0):,.2f}",
                "Equity baseline": f"${base_bd.get(default.name, {}).get('capital', 0):,.2f}",
            })

    audit_rows = []
    for r in audit:
        mb, ma = r["metrics_before"], r["metrics_after"]
        audit_rows.append({
            "Fecha (UTC)": (r["ts"] or "")[:16].replace("T", " "),
            "Estrategia": r["strategy_name"],
            "Parámetro": r["param"] or "",
            "Antes → después": (f"{r['old_value']:g} → {r['new_value']:g}"
                                if r["old_value"] is not None and r["new_value"] is not None
                                else ""),
            "Decisión": r["decision"],
            "Motivo": r["reason"] or "",
            "PF antes/después": (f"{mb.get('profit_factor', 0):.2f} / {ma.get('profit_factor', 0):.2f}"
                                 if ma else ""),
            "MaxDD antes/después": (f"{mb.get('max_drawdown', 0):.1%} / {ma.get('max_drawdown', 0):.1%}"
                                    if ma else ""),
            "Trades (valid.)": ma.get("trades", "") if ma else "",
        })

    decision_colors = [
        {"if": {"filter_query": '{Decisión} = "applied"'}, "color": COLORS["green"]},
        {"if": {"filter_query": '{Decisión} = "rollback"'}, "color": COLORS["red"]},
        {"if": {"filter_query": '{Decisión} = "rejected"'}, "color": COLORS["subtext"]},
    ]
    return html.Div([
        notice,
        cards,
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.H6("Parámetros ajustables: actual vs baseline",
                style={"color": COLORS["blue"], "margin": "16px 0 8px"}),
        _table(param_rows, page_size=12),
        html.H6(f"Auditoría del aprendizaje ({len(audit_rows)} propuestas)",
                style={"color": COLORS["blue"], "margin": "16px 0 8px"}),
        _table(audit_rows, conditional=decision_colors) if audit_rows else html.Div(
            "Todavía no hay propuestas. El aprendizaje corre cada "
            f"{config.LEARNING_INTERVAL_HOURS:g} h.", style={"color": COLORS["subtext"]}),
    ])


STATUS_STYLE = {
    "VIABLE":      ("✅ VIABLE", "green"),
    "CONDICIONAL": ("🟡 CONDICIONAL", "yellow"),
    "EN_PRUEBA":   ("🔍 EN PRUEBA", "blue"),
    "DESCARTADA":  ("⛔ DESCARTADA", "red"),
}


def _render_strategy_evaluation():
    """Strategy evaluator: viability, when (market regime) and how (side) per strategy."""
    from strategies import ALL_STRATEGIES, CANDIDATE_STRATEGIES
    from strategy_evaluator import REGIMES, REGIME_ES

    catalog = {S().name: ("registrada", S) for S in ALL_STRATEGIES}
    catalog.update({S().name: ("catálogo", S) for S in CANDIDATE_STRATEGIES})
    statuses = db.get_all_strategy_status()
    if not statuses:
        return html.Div("El evaluador todavía no calificó ninguna estrategia (corre cada "
                        f"{config.EVAL_INTERVAL_HOURS:g} h, la primera vez al arrancar).",
                        style={"color": COLORS["subtext"], "padding": "20px"})

    counts = {k: sum(s["status"] == k for s in statuses) for k in STATUS_STYLE}
    cards = dbc.Row([
        dbc.Col(_metric_card(label, str(counts[key]), color), width=3)
        for key, (label, color) in STATUS_STYLE.items()
    ], className="g-2 mb-3")

    ordered = sorted(statuses, key=lambda s: -s["score"])
    fig_score = go.Figure(go.Bar(
        x=[s["strategy_name"] for s in ordered], y=[s["score"] for s in ordered],
        marker_color=[COLORS[STATUS_STYLE[s["status"]][1]] for s in ordered],
        text=[STATUS_STYLE[s["status"]][0].split(" ", 1)[1] for s in ordered],
    ))
    fig_score.update_layout(**_dark_layout("Puntaje de viabilidad (0 = descartada)"), height=300,
                            yaxis_range=[0, 1])

    # "When": profit factor per market regime (text = trades)
    z, text = [], []
    for s in ordered:
        row_z, row_t = [], []
        for r in REGIMES:
            st = s["metrics"].get("by_regime", {}).get(r)
            row_z.append(min(st["profit_factor"], 3.0) if st and st["trades"] else None)
            row_t.append(f"PF {st['profit_factor']:.2f}<br>{st['trades']} trades" if st and st["trades"] else "-")
        z.append(row_z)
        text.append(row_t)
    fig_regime = go.Figure(go.Heatmap(
        z=z, x=[REGIME_ES[r] for r in REGIMES], y=[s["strategy_name"] for s in ordered],
        text=text, texttemplate="%{text}", zmin=0.5, zmax=1.5, zmid=1.0,
        colorscale=[[0, COLORS["red"]], [0.5, COLORS["card"]], [1, COLORS["green"]]],
        colorbar=dict(title="PF"),
    ))
    fig_regime.update_layout(**_dark_layout("¿Cuándo funciona? Profit factor por tipo de mercado "
                                            "(backtest walk-forward)"), height=60 + 28 * len(ordered))

    rows = []
    for s in ordered:
        m = s["metrics"]
        live = m.get("live", {})
        origin, S = catalog.get(s["strategy_name"], ("?", None))
        rows.append({
            "Estrategia": s["strategy_name"],
            "Origen": origin,
            "Estado": STATUS_STYLE[s["status"]][0],
            "Puntaje": f"{s['score']:.2f}",
            "Trades (hist.)": m.get("trades", 0),
            "PF": f"{m.get('profit_factor', 0):.2f}",
            "Ventanas rentables": f"{m.get('profitable_windows', 0)}/{m.get('windows', 0)}",
            "Peor DD": f"{m.get('worst_drawdown', 0):.1%}",
            "Opera en": ", ".join(REGIME_ES.get(r, r) for r in s["allowed_regimes"]) or "—",
            "Lados": ", ".join(s["allowed_sides"]) or "—",
            "Lab en vivo": f"{live.get('trades', 0)} trades, ${live.get('pnl', 0):+,.2f}",
            "Motivo": s["reason"] or "",
            "Fuente": (S.SOURCE if S is not None else "") or "estrategia registrada del bot",
            "Evaluada": (s["updated_at"] or "")[:16].replace("T", " "),
        })
    status_colors = [
        {"if": {"filter_query": '{Estado} contains "DESCARTADA"'}, "color": COLORS["red"]},
        {"if": {"filter_query": '{Estado} contains "VIABLE"'}, "color": COLORS["green"]},
        {"if": {"filter_query": '{Estado} contains "CONDICIONAL"'}, "color": COLORS["yellow"]},
    ]
    return html.Div([
        dbc.Alert(
            "El evaluador califica cada estrategia con backtests en 4 ventanas de "
            f"{config.EVAL_WINDOW_DAYS} días, por tipo de mercado y por dirección, más sus trades en "
            "vivo del libro 'lab'. La versión que aprende solo opera VIABLES, o CONDICIONALES en su "
            "tipo de mercado. DESCARTADA = valor 0: no se vuelve a evaluar ni a operar. "
            "Es evidencia histórica, no garantía.",
            color="secondary", style={"fontSize": "13px"}),
        cards,
        dcc.Graph(figure=fig_score, config={"displayModeBar": False}),
        dcc.Graph(figure=fig_regime, config={"displayModeBar": False}),
        html.H6("Detalle por estrategia", style={"color": COLORS["blue"], "margin": "16px 0 8px"}),
        _table(rows, page_size=20, conditional=status_colors),
    ])


# ─── Market: what top traders, the crowd and sentiment are doing ─────────────

def _fng_label(v: float) -> str:
    if v <= 24:
        return "Miedo extremo"
    if v <= 44:
        return "Miedo"
    if v <= 55:
        return "Neutral"
    if v <= 75:
        return "Codicia"
    return "Codicia extrema"


def _render_market():
    from market_data import load_series
    sym = config.SYMBOL
    s = {m: load_series(sym, m) for m in ("fng", "funding_rate", "top_pos_ratio", "global_ratio",
                                          "taker_ratio", "oi_value")}
    if all(x.empty for x in s.values()):
        return html.Div("Todavía no hay datos de mercado: el bot los recoge cada hora "
                        "(la primera vez tarda unos minutos).",
                        id="market-panel", style={"color": COLORS["subtext"], "padding": "20px"})

    long_pct = lambda r: r / (1 + r) * 100         # long/short ratio -> % of longs
    last = lambda x: float(x.iloc[-1]) if len(x) else None
    cards = []
    if (v := last(s["fng"])) is not None:
        cards.append(_metric_card("Miedo y Codicia", f"{v:.0f}", "red" if v <= 24 else "green" if v >= 76 else "yellow",
                                  subtitle=_fng_label(v)))
    if (v := last(s["top_pos_ratio"])) is not None:
        cards.append(_metric_card("Top traders en largo", f"{long_pct(v):.1f}%", "blue",
                                  subtitle=f"ratio {v:.2f} (por posición)"))
    if (v := last(s["global_ratio"])) is not None:
        cards.append(_metric_card("Todas las cuentas en largo", f"{long_pct(v):.1f}%", "purple",
                                  subtitle=f"ratio {v:.2f} (la masa)"))
    if (v := last(s["funding_rate"])) is not None:
        cards.append(_metric_card("Funding (cada 8 h)", f"{v * 100:+.4f}%", "green" if v >= 0 else "red",
                                  subtitle=f"≈ {v * 3 * 365 * 100:+.1f}% anual; > 0 = los largos pagan"))
    if (v := last(s["taker_ratio"])) is not None:
        cards.append(_metric_card("Compradores / vendedores", f"{v:.2f}", "green" if v >= 1 else "red",
                                  subtitle="volumen agresivo (taker)"))
    if (v := last(s["oi_value"])) is not None:
        cards.append(_metric_card("Open interest", f"${v / 1e9:,.2f} B", "text", subtitle="futuros BTCUSDT"))

    figs = []
    if len(s["top_pos_ratio"]) or len(s["global_ratio"]):
        fig = go.Figure()
        for key, name, color in (("top_pos_ratio", "Top traders (posición)", COLORS["blue"]),
                                 ("global_ratio", "Todas las cuentas", COLORS["purple"])):
            x = s[key]
            if len(x):
                fig.add_trace(go.Scatter(x=x.index, y=long_pct(x), name=name, line=dict(color=color, width=2)))
        fig.add_hline(y=50, line_dash="dot", line_color=COLORS["border"])
        fig.update_layout(**_dark_layout("% en largo: top traders vs la masa (Binance Futures)"),
                          height=300, yaxis_ticksuffix="%")
        figs.append(fig)
    if len(s["fng"]):
        x = s["fng"].iloc[-365:]
        fig = go.Figure(go.Scatter(x=x.index, y=x, line=dict(color=COLORS["yellow"], width=2), name="F&G"))
        fig.add_hrect(y0=0, y1=24, fillcolor=COLORS["red"], opacity=0.08, line_width=0)
        fig.add_hrect(y0=76, y1=100, fillcolor=COLORS["green"], opacity=0.08, line_width=0)
        fig.update_layout(**_dark_layout("Miedo y Codicia (último año)"), height=260, yaxis_range=[0, 100])
        figs.append(fig)
    if len(s["funding_rate"]):
        x = s["funding_rate"].iloc[-270:]            # ~90 days of 8-hour fundings
        fig = go.Figure(go.Bar(x=x.index, y=x * 100,
                               marker_color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in x]))
        fig.update_layout(**_dark_layout("Funding rate por periodo de 8 h (últimos ~90 días)"), height=240,
                          yaxis_ticksuffix="%")
        figs.append(fig)

    return html.Div(id="market-panel", children=[
        dbc.Alert("Datos públicos y gratuitos: posicionamiento de los top traders y de todas las cuentas "
                  "de Binance Futures, funding, volumen agresivo y el índice de Miedo y Codicia. Binance "
                  "solo guarda 30 días de los ratios: la historia se acumula desde que el bot los recoge. "
                  "Las estrategias modernas usan estos datos y el evaluador decide si sirven.",
                  color="secondary", style={"fontSize": "13px"}),
        dbc.Row([dbc.Col(c, md=2) for c in cards], className="g-2 mb-3"),
        *[dcc.Graph(figure=f, config={"displayModeBar": False}) for f in figs],
    ])


# ─── Control: aggressiveness, budget, mode (risk_engine.py) ──────────────────

def _learner_book() -> str:
    return "observe" if db.get_meta("trading_mode") == "OBSERVE" else "main"


def _render_control():
    from risk_engine import RiskSettings
    s = RiskSettings.load()
    label = {"fontSize": "13px", "color": COLORS["subtext"], "marginBottom": "4px"}
    form = dbc.Card(dbc.CardBody([
        html.H6("Agresividad (1 = prudente, 10 = agresivo)", style={"color": COLORS["blue"]}),
        dcc.Slider(id="ctl-aggr", min=1, max=10, step=1, value=s.aggressiveness,
                   marks={i: str(i) for i in range(1, 11)}),
        dbc.Row([
            dbc.Col([html.Div("Fondos de la cuenta (paper, USD)", style=label),
                     dbc.Input(id="ctl-funds", type="number", min=1, value=s.funds)], md=3),
            dbc.Col([html.Div("Presupuesto que el bot puede usar (USD)", style=label),
                     dbc.Input(id="ctl-budget", type="number", min=1, value=s.budget)], md=3),
            dbc.Col([html.Div("Modo", style=label),
                     dbc.RadioItems(id="ctl-mode", value=s.mode, inline=True, options=[
                         {"label": "Operar", "value": "trade"},
                         {"label": "Solo cierre", "value": "close_only"}])], md=3),
            dbc.Col([html.Div("Cortos (ventas en corto)", style=label),
                     dbc.Checklist(id="ctl-short", value=["short"] if s.allow_short else [],
                                   options=[{"label": "Permitir", "value": "short"}], switch=True)], md=3),
        ], className="g-3 my-2"),
        dbc.Button("Guardar", id="ctl-save", color="primary", className="me-2"),
        html.Div(id="ctl-save-msg", className="mt-2"),
    ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}"})

    actions = dbc.Card(dbc.CardBody([
        html.H6("Acciones", style={"color": COLORS["blue"]}),
        dbc.Button("Reactivar (quitar freno de emergencia)", id="ctl-reset-kill",
                   color="secondary", className="me-2"),
        dcc.ConfirmDialogProvider(
            dbc.Button("Cerrar todas las posiciones ahora", color="danger"),
            id="ctl-close-all",
            message="¿Cerrar TODAS las posiciones del bot al precio actual y pasar a 'Solo cierre'? "
                    "(paper trading, sin dinero real)"),
        html.Div(id="ctl-reset-msg", className="mt-2"),
        html.Div(id="ctl-close-msg", className="mt-2"),
    ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}"})

    return html.Div([
        dbc.Alert("Paper trading: todo es simulado con precios reales de Binance. El presupuesto limita "
                  "cuánto usa la versión que aprende; mayor agresividad = operaciones más grandes, más "
                  "estrategias permitidas y límites de pérdida más amplios.",
                  color="secondary", style={"fontSize": "13px"}),
        html.Div(id="ctl-status", children=update_control_status(0)),
        dbc.Row([dbc.Col(form, md=8), dbc.Col(actions, md=4)], className="g-2 my-2"),
        html.H6("Qué significa este nivel", style={"color": COLORS["blue"], "margin": "16px 0 8px"}),
        html.Div(id="ctl-preview", children=preview_aggressiveness(s.aggressiveness)),
    ])


@app.callback(Output("ctl-preview", "children"), Input("ctl-aggr", "value"),
              Input("ctl-budget", "value"), prevent_initial_call=True)
def preview_aggressiveness(level, budget=None):
    from risk_engine import RiskSettings, profile
    p = profile(level or 1)
    try:
        budget = float(budget) if budget else RiskSettings.load().budget   # value being typed
    except (TypeError, ValueError):
        budget = RiskSettings.load().budget
    rows = [
        ("Riesgo por operación (pérdida si toca el stop)", f"{p['risk_per_trade']:.2%} = ${budget * p['risk_per_trade']:,.2f}"),
        ("Tamaño máximo de una posición", f"{p['max_position']:.0%} = ${budget * p['max_position']:,.2f}"),
        ("Posiciones abiertas a la vez", str(p["max_open"])),
        ("Exposición máxima (suma de posiciones)", f"{p['max_exposure']:.0%} = ${budget * p['max_exposure']:,.2f}"),
        ("Límite de pérdida diaria", f"{p['daily_loss']:.1%} = ${budget * p['daily_loss']:,.2f}"),
        ("Freno de emergencia (caída desde el máximo)", f"{p['max_drawdown']:.0%} = ${budget * p['max_drawdown']:,.2f}"),
        ("Confianza mínima de la señal", f"{p['min_confidence']:.2f}"),
        ("Estrategias que puede usar", ", ".join(p["statuses"])),
    ]
    return _table([{"Parámetro": a, f"Nivel {p['level']} (presupuesto ${budget:,.0f})": b}
                   for a, b in rows], page_size=10)


@app.callback(Output("ctl-save-msg", "children"), Input("ctl-save", "n_clicks"),
              State("ctl-aggr", "value"), State("ctl-funds", "value"), State("ctl-budget", "value"),
              State("ctl-mode", "value"), State("ctl-short", "value"), prevent_initial_call=True)
def save_risk_settings(_, aggr, funds, budget, mode, short):
    from risk_engine import RiskSettings
    try:
        RiskSettings(funds=float(funds or 0), budget=float(budget or 0), aggressiveness=int(aggr or 0),
                     mode=mode, allow_short="short" in (short or [])).save()
    except (ValueError, TypeError) as e:
        return dbc.Alert(f"No se guardó: {e}", color="danger")
    return dbc.Alert("Guardado. El bot lo aplica en menos de 30 segundos.", color="success")


@app.callback(Output("ctl-reset-msg", "children"), Input("ctl-reset-kill", "n_clicks"),
              prevent_initial_call=True)
def reset_kill_switch(_):
    from risk_engine import RiskEngine
    RiskEngine(_learner_book()).reset_kill_switch()
    return dbc.Alert("Freno quitado: el bot vuelve a poder abrir posiciones.", color="success")


@app.callback(Output("ctl-close-msg", "children"), Input("ctl-close-all", "submit_n_clicks"),
              prevent_initial_call=True)
def request_close_all(_):
    from risk_engine import RiskEngine
    RiskEngine(_learner_book()).request_close_all()
    return dbc.Alert("Pedido enviado: el bot cierra todo en menos de 30 s y pasa a 'Solo cierre'.",
                     color="warning")


@app.callback(Output("ctl-status", "children"), Input("interval-refresh", "n_intervals"))
def update_control_status(_):
    """Live risk state read from the DB (the bot refreshes it every ~20 s)."""
    from risk_engine import RiskSettings, profile
    s = RiskSettings.load()
    p = profile(s.aggressiveness)
    book = _learner_book()
    positions = db.get_open_positions(book=book)
    exposure = sum(float(x["entry_price"]) * float(x["quantity"]) for x in positions)
    unreal = 0.0
    if positions:
        try:
            price = _get_price_fetcher().get_current_price(config.SYMBOL)
            unreal = sum(((price - x["entry_price"]) if x["side"] == "LONG" else (x["entry_price"] - price))
                         * x["quantity"] for x in positions)
        except Exception:
            pass
    pnl = float(db.get_trade_stats(book=book).get("total_pnl") or 0) + unreal
    day = json.loads(db.get_meta(f"risk:{book}:day") or "{}")
    day_pnl = pnl - day["start_pnl"] if day else 0.0
    kill = json.loads(db.get_meta(f"risk:{book}:kill") or "null")
    mode = "⛔ FRENO DE EMERGENCIA" if kill else ("Solo cierre" if s.mode == "close_only" else "Operar")
    daily_limit = p["daily_loss"] * s.budget
    cards = dbc.Row([
        dbc.Col(_metric_card("Modo", mode, "red" if kill else ("yellow" if s.mode == "close_only" else "green"),
                             subtitle=f"agresividad {p['level']}/10"), width=3),
        dbc.Col(_metric_card("P&L total", f"${pnl:+,.2f}", "green" if pnl >= 0 else "red",
                             subtitle=f"{pnl / s.budget:+.2%} del presupuesto ${s.budget:,.0f}"), width=3),
        dbc.Col(_metric_card("P&L de hoy", f"${day_pnl:+,.2f}", "green" if day_pnl >= 0 else "red",
                             subtitle=f"límite diario -${daily_limit:,.2f}"), width=3),
        dbc.Col(_metric_card("Presupuesto en uso", f"${exposure:,.2f}", "blue",
                             subtitle=f"de ${p['max_exposure'] * s.budget:,.2f} · {len(positions)}/{p['max_open']} posiciones"),
                width=3),
    ], className="g-2")
    alert = dbc.Alert(f"Freno de emergencia activado: {kill['reason']}. No abre posiciones nuevas hasta que "
                      "pulses 'Reactivar'.", color="danger") if kill else None
    return html.Div([alert, cards])


# ─── Entry point ──────────────────────────────────────────────────────────────

def run_dashboard(debug: bool = False):
    db.init_db()
    app.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=debug,
    )


if __name__ == "__main__":
    run_dashboard(debug=True)
