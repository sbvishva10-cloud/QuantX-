import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import streamlit as st
import yfinance as yf

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="QuantX Pro",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TRADING_DAYS = 252

ASSETS = {
    "Gold (GC=F)": "GC=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "NVIDIA (NVDA)": "NVDA",
    "S&P 500 ETF (SPY)": "SPY",
}


# =========================================================
# PROFESSIONAL DARK FINTECH UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            #15253f 0%,
            #080d17 38%,
            #050810 100%
        );
}

[data-testid="stSidebar"] {
    background: #070b13;
    border-right: 1px solid rgba(255,255,255,.08);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
}

.hero {
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(67,217,173,.10),
            rgba(92,135,255,.08)
        );

    border: 1px solid rgba(255,255,255,.08);
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
}

.hero-subtitle {
    color: #8d9ab0;
    margin-top: 5px;
}

div[data-testid="stMetric"] {
    background: rgba(13,20,34,.92);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
}

.small {
    color: #8d9ab0;
    font-size: .82rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA ENGINE
# =========================================================

@st.cache_data(ttl=3600)
def download_market_data(tickers, start, end):

    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    return data


def get_close(data, ticker):

    if data.empty:
        return pd.Series(dtype=float)

    try:

        if isinstance(data.columns, pd.MultiIndex):

            if ticker in data.columns.get_level_values(0):
                series = data[ticker]["Close"]

            elif ticker in data.columns.get_level_values(1):
                series = data["Close"][ticker]

            else:
                return pd.Series(dtype=float)

        else:
            series = data["Close"]

    except Exception:
        return pd.Series(dtype=float)

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    series.index = pd.to_datetime(series.index)

    return series.sort_index()


def build_asset_returns(data):

    result = {}

    for name, ticker in ASSETS.items():

        close = get_close(data, ticker)

        if not close.empty:
            result[name] = close.pct_change().rename(name)

    if not result:
        return pd.DataFrame()

    return pd.concat(
        result.values(),
        axis=1
    ).dropna(how="all")


# =========================================================
# QUANTITATIVE METRICS
# =========================================================

def calculate_metrics(returns, risk_free=0.04):

    returns = pd.Series(returns).dropna()

    if returns.empty:
        return {}

    wealth = (1 + returns).cumprod()

    years = max(
        len(returns) / TRADING_DAYS,
        1 / TRADING_DAYS
    )

    total_return = wealth.iloc[-1] - 1

    annual_return = (
        wealth.iloc[-1] ** (1 / years) - 1
    )

    volatility = (
        returns.std() *
        np.sqrt(TRADING_DAYS)
    )

    daily_rf = (
        (1 + risk_free) **
        (1 / TRADING_DAYS)
    ) - 1

    excess = returns - daily_rf

    sharpe = (
        excess.mean() /
        returns.std() *
        np.sqrt(TRADING_DAYS)
        if returns.std() > 0
        else np.nan
    )

    downside = returns[
        returns < daily_rf
    ] - daily_rf

    downside_dev = (
        downside.std() *
        np.sqrt(TRADING_DAYS)
        if len(downside) > 1
        else np.nan
    )

    sortino = (
        (annual_return - risk_free) /
        downside_dev
        if downside_dev and downside_dev > 0
        else np.nan
    )

    drawdown = (
        wealth /
        wealth.cummax()
        - 1
    )

    max_drawdown = drawdown.min()

    return {
        "total": total_return,
        "annual": annual_return,
        "vol": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "drawdown": max_drawdown,
        "win_rate": (returns > 0).mean(),
    }


def rolling_sharpe(
    returns,
    window=30,
    risk_free=0.04
):

    daily_rf = (
        (1 + risk_free) **
        (1 / TRADING_DAYS)
    ) - 1

    excess = returns - daily_rf

    return (
        excess.rolling(window).mean() /
        excess.rolling(window).std() *
        np.sqrt(TRADING_DAYS)
    )


# =========================================================
# STRATEGIES
# =========================================================

def generate_signal(
    price,
    strategy,
    fast,
    slow,
    momentum_window,
    mean_window,
    z_entry
):

    # SMA CROSSOVER

    if strategy == "SMA Crossover":

        fast_ma = price.rolling(fast).mean()
        slow_ma = price.rolling(slow).mean()

        signal = (
            fast_ma > slow_ma
        ).astype(float)

        return signal, fast_ma, slow_ma


    # EMA TREND

    if strategy == "EMA Trend":

        fast_ema = price.ewm(
            span=fast,
            adjust=False
        ).mean()

        slow_ema = price.ewm(
            span=slow,
            adjust=False
        ).mean()

        signal = (
            fast_ema > slow_ema
        ).astype(float)

        return signal, fast_ema, slow_ema


    # MOMENTUM

    if strategy == "Momentum":

        momentum = price.pct_change(
            momentum_window
        )

        signal = (
            momentum > 0
        ).astype(float)

        return signal, momentum, pd.Series(
            index=price.index,
            dtype=float
        )


    # MEAN REVERSION

    mean = price.rolling(
        mean_window
    ).mean()

    std = price.rolling(
        mean_window
    ).std()

    z_score = (
        price - mean
    ) / std.replace(0, np.nan)

    signal = pd.Series(
        0.0,
        index=price.index
    )

    position = 0

    for i in range(len(price)):

        z = z_score.iloc[i]

        if pd.isna(z):
            signal.iloc[i] = position
            continue

        if position == 0 and z < -z_entry:
            position = 1

        elif position == 1 and z >= 0:
            position = 0

        signal.iloc[i] = position

    return signal, mean, z_score


# =========================================================
# BACKTEST ENGINE
# =========================================================

def run_backtest(
    price,
    strategy,
    fast,
    slow,
    momentum_window,
    mean_window,
    z_entry,
    initial_capital,
    transaction_cost,
    position_size
):

    if strategy == "Buy & Hold":

        signal = pd.Series(
            1.0,
            index=price.index
        )

    else:

        signal, _, _ = generate_signal(
            price,
            strategy,
            fast,
            slow,
            momentum_window,
            mean_window,
            z_entry
        )

    # CRITICAL:
    # Signal is shifted one day to avoid
    # look-ahead bias.

    position = (
        signal.shift(1)
        .fillna(0)
        * position_size
    )

    asset_return = (
        price.pct_change()
        .fillna(0)
    )

    turnover = (
        position
        .diff()
        .abs()
        .fillna(position.abs())
    )

    costs = (
        turnover *
        transaction_cost
    )

    strategy_return = (
        position *
        asset_return
        - costs
    )

    equity = (
        initial_capital *
        (1 + strategy_return).cumprod()
    )

    benchmark = (
        initial_capital *
        (1 + asset_return).cumprod()
    )

    trades = (
        position.diff()
        .fillna(position)
        .abs() > 0
    )

    return pd.DataFrame({
        "Price": price,
        "Signal": signal,
        "Position": position,
        "Return": strategy_return,
        "Equity": equity,
        "Benchmark": benchmark,
        "Cost": costs,
        "Trade": trades,
    })


# =========================================================
# MARKET REGIMES
# =========================================================

def classify_market_regimes(
    price,
    trend_window=100,
    volatility_window=30
):

    returns = price.pct_change()

    volatility = (
        returns
        .rolling(volatility_window)
        .std()
        * np.sqrt(TRADING_DAYS)
    )

    trend = price.pct_change(
        trend_window
    )

    volatility_threshold = volatility.median()

    regimes = pd.Series(
        "Unknown",
        index=price.index
    )

    for idx in price.index:

        if (
            pd.isna(volatility.loc[idx])
            or pd.isna(trend.loc[idx])
        ):
            continue

        if (
            trend.loc[idx] >= 0
            and volatility.loc[idx]
            < volatility_threshold
        ):
            regimes.loc[idx] = "Bull / Low Vol"

        elif trend.loc[idx] >= 0:
            regimes.loc[idx] = "Bull / High Vol"

        elif volatility.loc[idx] >= volatility_threshold:
            regimes.loc[idx] = "Bear / High Vol"

        else:
            regimes.loc[idx] = "Bear / Low Vol"

    return regimes, volatility, trend


def regime_statistics(
    backtest,
    regimes,
    risk_free
):

    rows = []

    for regime in [
        "Bull / Low Vol",
        "Bull / High Vol",
        "Bear / Low Vol",
        "Bear / High Vol",
    ]:

        sample = backtest.loc[
            regimes == regime,
            "Return"
        ]

        if len(sample) < 5:
            continue

        m = calculate_metrics(
            sample,
            risk_free
        )

        rows.append({
            "Regime": regime,
            "Days": len(sample),
            "Return": m["total"],
            "Volatility": m["vol"],
            "Sharpe": m["sharpe"],
            "Max Drawdown": m["drawdown"],
        })

    return pd.DataFrame(rows)


# =========================================================
# PORTFOLIO OPTIMIZATION
# =========================================================

def portfolio_statistics(
    weights,
    returns,
    risk_free
):

    expected_return = (
        returns.mean()
        * TRADING_DAYS
    )

    covariance = (
        returns.cov()
        * TRADING_DAYS
    )

    annual_return = float(
        weights @ expected_return
    )

    annual_volatility = float(
        np.sqrt(
            weights
            @ covariance.values
            @ weights
        )
    )

    sharpe = (
        (annual_return - risk_free)
        / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    return (
        annual_return,
        annual_volatility,
        sharpe
    )


def calculate_weights(
    returns,
    method,
    risk_free
):

    n = returns.shape[1]

    if method == "Equal Weight":

        return pd.Series(
            np.repeat(1 / n, n),
            index=returns.columns
        )


    if method == "Inverse Volatility":

        volatility = returns.std()

        inverse = (
            1 /
            volatility.replace(
                0,
                np.nan
            )
        )

        weights = (
            inverse /
            inverse.sum()
        )

        return weights.fillna(1 / n)


    # Maximum historical Sharpe

    initial = np.repeat(
        1 / n,
        n
    )

    bounds = [
        (0, 1)
        for _ in range(n)
    ]

    constraint = {
        "type": "eq",
        "fun": lambda x:
            np.sum(x) - 1
    }

    result = minimize(

        lambda x:
            -portfolio_statistics(
                x,
                returns,
                risk_free
            )[2],

        initial,

        method="SLSQP",

        bounds=bounds,

        constraints=constraint,

        options={
            "maxiter": 1000,
            "ftol": 1e-10
        }
    )

    weights = (
        result.x
        if result.success
        else initial
    )

    return pd.Series(
        weights,
        index=returns.columns
    )


# =========================================================
# ROBUSTNESS TEST
# =========================================================

def robustness_matrix(
    price,
    fast_values,
    slow_values,
    initial,
    cost,
    risk_free
):

    matrix = []

    for fast in fast_values:

        row = []

        for slow in slow_values:

            if fast >= slow:

                row.append(np.nan)
                continue

            bt = run_backtest(
                price,
                "SMA Crossover",
                fast,
                slow,
                20,
                30,
                1.5,
                initial,
                cost,
                1.0
            )

            m = calculate_metrics(
                bt["Return"],
                risk_free
            )

            row.append(
                m["sharpe"]
            )

        matrix.append(row)

    return np.array(matrix)


# =========================================================
# MONTE CARLO
# =========================================================

def monte_carlo(
    returns,
    initial,
    days,
    simulations,
    seed
):

    returns = (
        pd.Series(returns)
        .dropna()
    )

    rng = np.random.default_rng(
        seed
    )

    mean = returns.mean()
    sigma = returns.std()

    random_returns = rng.normal(
        mean,
        sigma,
        size=(days, simulations)
    )

    paths = (
        initial *
        np.cumprod(
            1 + random_returns,
            axis=0
        )
    )

    paths = np.vstack([
        np.full(
            simulations,
            initial
        ),
        paths
    ])

    return paths


# =========================================================
# AI RESEARCH REPORT
# =========================================================

def create_report(
    asset,
    metrics_asset,
    metrics_strategy,
    strategy,
    regime_table
):

    def pct(x):

        if pd.isna(x):
            return "N/A"

        return f"{x * 100:.2f}%"

    regime_text = "No sufficient regime observations."

    if not regime_table.empty:

        lines = []

        for _, row in regime_table.iterrows():

            lines.append(
                f"- {row['Regime']}: "
                f"return {pct(row['Return'])}, "
                f"volatility {pct(row['Volatility'])}, "
                f"Sharpe {row['Sharpe']:.2f}, "
                f"max drawdown "
                f"{pct(row['Max Drawdown'])}"
            )

        regime_text = "\n".join(lines)

    return f"""
# QuantX Quantitative Research Report

## Asset

**{asset}**

## Historical Profile

- Total return: **{pct(metrics_asset['total'])}**
- Annualized return: **{pct(metrics_asset['annual'])}**
- Annualized volatility: **{pct(metrics_asset['vol'])}**
- Sharpe ratio: **{metrics_asset['sharpe']:.2f}**
- Sortino ratio: **{metrics_asset['sortino']:.2f}**
- Maximum drawdown: **{pct(metrics_asset['drawdown'])}**
- Positive-return days: **{pct(metrics_asset['win_rate'])}**

## Strategy

Selected strategy:

**{strategy}**

- Strategy return: **{pct(metrics_strategy['total'])}**
- Strategy volatility: **{pct(metrics_strategy['vol'])}**
- Strategy Sharpe: **{metrics_strategy['sharpe']:.2f}**
- Strategy maximum drawdown: **{pct(metrics_strategy['drawdown'])}**

## Market Regimes

{regime_text}

## Quantitative Interpretation

The results describe historical observations under the selected
sample period, strategy parameters and transaction-cost assumptions.

Strategy performance should be considered together with volatility,
drawdown, benchmark performance, market regime and parameter robustness.

## Research Limitations

Historical backtest results do not guarantee future performance.

Results can change with:

- sample period
- transaction costs
- parameter choices
- market conditions
- data quality
- execution assumptions

This platform is intended for quantitative research and historical
analysis rather than personalized investment advice.
"""


def optional_ai_rewrite(report):

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    model = os.getenv(
        "OPENAI_MODEL"
    )

    if (
        not api_key
        or not model
        or OpenAI is None
    ):
        return None

    try:

        client = OpenAI(
            api_key=api_key
        )

        response = client.responses.create(

            model=model,

            input=(
                "Rewrite this quantitative research "
                "report professionally. Preserve all "
                "numbers. Do not give personalized "
                "investment advice or guaranteed "
                "future predictions.\n\n"
                + report
            )
        )

        return response.output_text

    except Exception:

        return None


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## ◈ QUANTX PRO"
)

st.sidebar.caption(
    "Quantitative Financial Intelligence"
)

asset_name = st.sidebar.selectbox(
    "Primary Asset",
    list(ASSETS.keys())
)

ticker = ASSETS[
    asset_name
]

period = st.sidebar.selectbox(
    "Research Window",
    [
        "1Y",
        "3Y",
        "5Y",
        "10Y",
        "Custom"
    ],
    index=1
)

today = date.today()

if period == "1Y":
    start = today - timedelta(days=365)

elif period == "3Y":
    start = today - timedelta(days=1095)

elif period == "5Y":
    start = today - timedelta(days=1825)

elif period == "10Y":
    start = today - timedelta(days=3650)

else:

    start = st.sidebar.date_input(
        "Start",
        date(2020, 1, 1),
        max_value=today
    )

end = st.sidebar.date_input(
    "End",
    today,
    max_value=today
)

st.sidebar.divider()

st.sidebar.markdown(
    "### Strategy"
)

strategy = st.sidebar.selectbox(
    "Strategy",
    [
        "SMA Crossover",
        "EMA Trend",
        "Momentum",
        "Mean Reversion",
        "Buy & Hold"
    ]
)

fast = st.sidebar.slider(
    "Fast MA",
    5,
    100,
    20
)

slow = st.sidebar.slider(
    "Slow MA",
    20,
    300,
    50
)

momentum_window = st.sidebar.slider(
    "Momentum Window",
    5,
    120,
    20
)

mean_window = st.sidebar.slider(
    "Mean Reversion Window",
    10,
    120,
    30
)

z_entry = st.sidebar.slider(
    "Mean Reversion Z Entry",
    0.5,
    3.0,
    1.5,
    0.1
)

st.sidebar.markdown(
    "### Trading Assumptions"
)

initial = st.sidebar.number_input(
    "Initial Capital (₹)",
    1000.0,
    1e9,
    100000.0,
    10000.0
)

transaction_cost = st.sidebar.number_input(
    "Transaction Cost (%)",
    0.0,
    5.0,
    0.10,
    0.05
)

position_size = st.sidebar.slider(
    "Position Size (%)",
    10,
    100,
    100
)

risk_free = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    0.0,
    20.0,
    4.0,
    0.25
)

st.sidebar.markdown(
    "### Monte Carlo"
)

mc_days = st.sidebar.slider(
    "Simulation Days",
    30,
    504,
    252
)

mc_simulations = st.sidebar.slider(
    "Simulations",
    100,
    5000,
    1000,
    100
)

mc_seed = st.sidebar.number_input(
    "Random Seed",
    1,
    999999,
    42
)

if start >= end:

    st.error(
        "Start date must be before End date."
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
◈ QuantX Pro
</div>

<div class="hero-subtitle">
Multi-Asset Financial Intelligence • Risk Analytics •
Backtesting • Portfolio Lab • Monte Carlo • AI Research
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

with st.spinner(
    "Loading historical market data..."
):

    data = download_market_data(
        tuple(ASSETS.values()),
        str(start),
        str(end + timedelta(days=1))
    )

price = get_close(
    data,
    ticker
)

if price.empty:

    st.error(
        "No market data returned."
    )

    st.stop()

returns = price.pct_change()

risk_free_decimal = (
    risk_free / 100
)

asset_metrics = calculate_metrics(
    returns,
    risk_free_decimal
)


# =========================================================
# BACKTEST
# =========================================================

backtest = run_backtest(
    price,
    strategy,
    fast,
    slow,
    momentum_window,
    mean_window,
    z_entry,
    initial,
    transaction_cost / 100,
    position_size / 100
)

strategy_metrics = calculate_metrics(
    backtest["Return"],
    risk_free_decimal
)


# =========================================================
# KPI HEADER
# =========================================================

columns = st.columns(6)

columns[0].metric(
    "Total Return",
    f"{asset_metrics['total'] * 100:.2f}%"
)

columns[1].metric(
    "Annualized",
    f"{asset_metrics['annual'] * 100:.2f}%"
)

columns[2].metric(
    "Volatility",
    f"{asset_metrics['vol'] * 100:.2f}%"
)

columns[3].metric(
    "Sharpe",
    f"{asset_metrics['sharpe']:.2f}"
)

columns[4].metric(
    "Max Drawdown",
    f"{asset_metrics['drawdown'] * 100:.2f}%"
)

columns[5].metric(
    "Trades",
    int(backtest["Trade"].sum())
)


# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "📊 Market Intelligence",
    "🧪 Strategy Lab",
    "🔬 Robustness",
    "🌐 Market Regimes",
    "🧩 Portfolio",
    "🎲 Monte Carlo",
    "🤖 AI Research"
])


# =========================================================
# MARKET INTELLIGENCE
# =========================================================

with tabs[0]:

    st.subheader(
        "Market Intelligence"
    )

    chart = go.Figure()

    chart.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            name="Price",
            line=dict(width=2)
        )
    )

    chart.add_trace(
        go.Scatter(
            x=price.index,
            y=price.rolling(fast).mean(),
            name=f"SMA {fast}"
        )
    )

    chart.add_trace(
        go.Scatter(
            x=price.index,
            y=price.rolling(slow).mean(),
            name=f"SMA {slow}"
        )
    )

    chart.add_trace(
        go.Scatter(
            x=price.index,
            y=price.ewm(
                span=fast,
                adjust=False
            ).mean(),
            name=f"EMA {fast}"
        )
    )

    chart.add_trace(
        go.Scatter(
            x=price.index,
            y=price.ewm(
                span=slow,
                adjust=False
            ).mean(),
            name=f"EMA {slow}"
        )
    )

    chart.update_layout(
        title=f"{asset_name} Price & Indicators",
        template="plotly_dark",
        height=520,
        hovermode="x unified"
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        cumulative = (
            1 + returns.fillna(0)
        ).cumprod() - 1

        fig = go.Figure(
            go.Scatter(
                x=cumulative.index,
                y=cumulative * 100,
                name="Return"
            )
        )

        fig.update_layout(
            title="Cumulative Return",
            yaxis_title="%",
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        wealth = (
            1 + returns.fillna(0)
        ).cumprod()

        drawdown = (
            wealth /
            wealth.cummax()
            - 1
        )

        fig = go.Figure(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                fill="tozeroy",
                name="Drawdown"
            )
        )

        fig.update_layout(
            title="Maximum Drawdown Curve",
            yaxis_title="%",
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    rolling_vol = (
        returns
        .rolling(30)
        .std()
        * np.sqrt(TRADING_DAYS)
    )

    rolling_sh = rolling_sharpe(
        returns,
        30,
        risk_free_decimal
    )

    fig = make_subplots(
        specs=[
            [{"secondary_y": True}]
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol * 100,
            name="Rolling Volatility"
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_sh.index,
            y=rolling_sh,
            name="Rolling Sharpe"
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="30-Day Rolling Risk",
        template="plotly_dark",
        height=380
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# STRATEGY LAB
# =========================================================

with tabs[1]:

    st.subheader(
        "Strategy Backtesting Engine"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Final Value",
        f"₹{backtest['Equity'].iloc[-1]:,.0f}"
    )

    b.metric(
        "Strategy Return",
        f"{strategy_metrics['total'] * 100:.2f}%"
    )

    c.metric(
        "Strategy Sharpe",
        f"{strategy_metrics['sharpe']:.2f}"
    )

    d.metric(
        "Strategy Drawdown",
        f"{strategy_metrics['drawdown'] * 100:.2f}%"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["Equity"],
            name="Strategy",
            line=dict(width=3)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["Benchmark"],
            name="Buy & Hold"
        )
    )

    fig.update_layout(
        title="Strategy vs Buy & Hold",
        yaxis_title="Portfolio Value (₹)",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    entries = (
        backtest["Position"].diff() > 0
    )

    exits = (
        backtest["Position"].diff() < 0
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            name="Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=price.index[entries],
            y=price[entries],
            mode="markers",
            name="Entry",
            marker=dict(
                symbol="triangle-up",
                size=11
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=price.index[exits],
            y=price[exits],
            mode="markers",
            name="Exit",
            marker=dict(
                symbol="triangle-down",
                size=11
            )
        )
    )

    fig.update_layout(
        title="Entry / Exit Signals",
        template="plotly_dark",
        height=380
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Backtest CSV",
        backtest.reset_index()
        .to_csv(index=False)
        .encode(),
        "quantx_backtest.csv",
        "text/csv"
    )


# =========================================================
# ROBUSTNESS
# =========================================================

with tabs[2]:

    st.subheader(
        "Strategy Robustness Lab"
    )

    fast_values = [
        10,
        20,
        30,
        40,
        50
    ]

    slow_values = [
        50,
        75,
        100,
        150,
        200
    ]

    matrix = robustness_matrix(
        price,
        fast_values,
        slow_values,
        initial,
        transaction_cost / 100,
        risk_free_decimal
    )

    heatmap = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[
                str(x)
                for x in slow_values
            ],
            y=[
                str(x)
                for x in fast_values
            ],
            text=np.round(
                matrix,
                2
            ),
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar_title="Sharpe"
        )
    )

    heatmap.update_layout(
        title="SMA Parameter Robustness",
        xaxis_title="Slow SMA",
        yaxis_title="Fast SMA",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    st.info(
        "This surface shows how historical Sharpe changes "
        "when strategy parameters are modified. It helps "
        "identify sensitivity and possible over-optimization."
    )


# =========================================================
# MARKET REGIMES
# =========================================================

with tabs[3]:

    st.subheader(
        "Market Regime Analysis"
    )

    regimes, volatility, trend = (
        classify_market_regimes(price)
    )

    regime_table = regime_statistics(
        backtest,
        regimes,
        risk_free_decimal
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            name="Price"
        )
    )

    for regime in regime_table["Regime"]:

        mask = (
            regimes == regime
        )

        fig.add_trace(
            go.Scatter(
                x=price.index[mask],
                y=price[mask],
                mode="markers",
                name=regime,
                marker=dict(size=5)
            )
        )

    fig.update_layout(
        title="Historical Market Regimes",
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if not regime_table.empty:

        display = regime_table.copy()

        for column in [
            "Return",
            "Volatility",
            "Max Drawdown"
        ]:

            display[column] *= 100

        st.dataframe(
            display.round(2),
            hide_index=True,
            use_container_width=True
        )


# =========================================================
# PORTFOLIO
# =========================================================

with tabs[4]:

    st.subheader(
        "Multi-Asset Portfolio Allocation"
    )

    asset_returns = build_asset_returns(
        data
    ).dropna()

    if asset_returns.shape[1] < 2:

        st.warning(
            "At least two assets are needed."
        )

    else:

        method = st.radio(
            "Allocation Method",
            [
                "Equal Weight",
                "Inverse Volatility",
                "Maximum Sharpe"
            ],
            horizontal=True
        )

        weights = calculate_weights(
            asset_returns,
            method,
            risk_free_decimal
        )

        expected_return, portfolio_vol, portfolio_sharpe = (
            portfolio_statistics(
                weights.values,
                asset_returns,
                risk_free_decimal
            )
        )

        portfolio_returns = (
            asset_returns[weights.index]
            @ weights
        )

        portfolio_metrics = calculate_metrics(
            portfolio_returns,
            risk_free_decimal
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Expected Annual Return",
            f"{expected_return * 100:.2f}%"
        )

        b.metric(
            "Annual Volatility",
            f"{portfolio_vol * 100:.2f}%"
        )

        c.metric(
            "Portfolio Sharpe",
            f"{portfolio_sharpe:.2f}"
        )

        d.metric(
            "Historical Max DD",
            f"{portfolio_metrics['drawdown'] * 100:.2f}%"
        )

        left, right = st.columns(
            [1, 2]
        )

        with left:

            allocation = pd.DataFrame({
                "Asset": weights.index,
                "Weight": [
                    f"{x * 100:.2f}%"
                    for x in weights
                ]
            })

            st.dataframe(
                allocation,
                hide_index=True,
                use_container_width=True
            )

            pie = go.Figure(
                go.Pie(
                    labels=weights.index,
                    values=weights.values,
                    hole=.55
                )
            )

            pie.update_layout(
                title="Portfolio Allocation",
                template="plotly_dark",
                height=350
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        with right:

            wealth = (
                1 + portfolio_returns
            ).cumprod() - 1

            fig = go.Figure(
                go.Scatter(
                    x=wealth.index,
                    y=wealth * 100,
                    name="Portfolio"
                )
            )

            fig.update_layout(
                title="Historical Portfolio Growth",
                yaxis_title="Cumulative Return (%)",
                template="plotly_dark",
                height=430
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        correlation = (
            asset_returns.corr()
        )

        fig = go.Figure(
            go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                text=np.round(
                    correlation.values,
                    2
                ),
                texttemplate="%{text}"
            )
        )

        fig.update_layout(
            title="Cross-Asset Correlation",
            template="plotly_dark",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# MONTE CARLO
# =========================================================

with tabs[5]:

    st.subheader(
        "Monte Carlo Scenario Engine"
    )

    st.caption(
        "Scenario simulation based on historical daily returns. "
        "It is not a guaranteed future forecast."
    )

    paths = monte_carlo(
        returns,
        initial,
        mc_days,
        mc_simulations,
        int(mc_seed)
    )

    terminal_values = paths[-1]

    p5, p50, p95 = np.percentile(
        terminal_values,
        [5, 50, 95]
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "5th Percentile",
        f"₹{p5:,.0f}"
    )

    b.metric(
        "Median",
        f"₹{p50:,.0f}"
    )

    c.metric(
        "95th Percentile",
        f"₹{p95:,.0f}"
    )

    d.metric(
        "Above Initial",
        f"{(
            terminal_values > initial
        ).mean() * 100:.1f}%"
    )

    fig = go.Figure()

    x = np.arange(
        paths.shape[0]
    )

    for i in range(
        min(100, paths.shape[1])
    ):

        fig.add_trace(
            go.Scatter(
                x=x,
                y=paths[:, i],
                mode="lines",
                opacity=.2,
                line=dict(width=.7),
                showlegend=False
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=np.median(
                paths,
                axis=1
            ),
            mode="lines",
            name="Median",
            line=dict(width=3)
        )
    )

    fig.update_layout(
        title=(
            f"{mc_simulations:,} Monte Carlo "
            f"Scenario Paths"
        ),
        xaxis_title="Trading Day",
        yaxis_title="Portfolio Value (₹)",
        template="plotly_dark",
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    histogram = go.Figure(
        go.Histogram(
            x=terminal_values,
            nbinsx=60
        )
    )

    histogram.add_vline(
        x=initial,
        line_dash="dash",
        annotation_text="Initial Capital"
    )

    histogram.update_layout(
        title="Terminal Value Distribution",
        template="plotly_dark",
        height=380
    )

    st.plotly_chart(
        histogram,
        use_container_width=True
    )


# =========================================================
# AI RESEARCH
# =========================================================

with tabs[6]:

    st.subheader(
        "AI Quantitative Research Report"
    )

    report = create_report(
        asset_name,
        asset_metrics,
        strategy_metrics,
        strategy,
        regime_table
    )

    ai_report = optional_ai_rewrite(
        report
    )

    final_report = (
        ai_report
        if ai_report
        else report
    )

    if ai_report:

        st.success(
            "AI-assisted research report generated."
        )

    else:

        st.info(
            "Deterministic quantitative report active. "
            "Optional LLM rewriting can be enabled with "
            "OPENAI_API_KEY and OPENAI_MODEL."
        )

    st.markdown(
        final_report
    )

    st.download_button(
        "⬇ Download Research Report",
        final_report.encode(),
        "quantx_research_report.md",
        "text/markdown"
    )


# =========================================================
# RESEARCH INTEGRITY
# =========================================================

st.divider()

st.subheader(
    "Research Integrity"
)

st.markdown("""
✓ One-period signal lag to reduce look-ahead bias  
✓ Transaction costs included  
✓ Configurable position sizing  
✓ Buy & Hold benchmark  
✓ Parameter robustness testing  
✓ Historical market-regime analysis  
✓ Backtest results presented as simulations, not guarantees
""")

st.caption(
    "QUANTX PRO • Quantitative research prototype • "
    "Historical analysis only; not investment advice."
)
