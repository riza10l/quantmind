IMPORTANT PRODUCT CONTEXT:
QuantMind is a quantitative finance research laboratory, not a magic trading bot. The UI must communicate serious data-driven research: data ingestion, 113 technical indicators, ML prediction, backtesting, Monte Carlo testing, risk management, paper/testnet execution, dashboard reporting, and 104 automated tests. Make it feel like TradingView + quant research terminal, with a strong risk-first and research-first identity.

# ROLE
Act as a senior product designer, frontend engineer, and financial dashboard UX specialist.

You are assigned to redesign and improve the UI/UX of my app called **QuantMind**.

QuantMind is a quantitative research / trading intelligence dashboard that displays market data, charts, AI signals, strategy insights, watchlists, indicators, backtest results, and market summaries.

The current UI looks too generic, too AI-generated, and not professional enough. I want it to feel like a premium fintech terminal/dashboard inspired by TradingView, Bloomberg-lite, and modern quant research tools.

IMPORTANT:
Do not blindly copy TradingView. Use it only as a reference for layout quality, spacing, hierarchy, interactivity, chart-first design, and professional market dashboard feel.

---

# MAIN OBJECTIVE

Redesign QuantMind so it looks:

- Professional
- Clean
- Interactive
- Data-dense but readable
- Fintech-grade
- Similar in spirit to TradingView
- Not obviously AI-generated
- Suitable for serious traders, quant researchers, and technical users

The final UI should look like a real production-grade market platform, not a demo project.

---

# REFERENCE STYLE

Use TradingView-style design inspiration:

1. Large chart area as the main visual focus.
2. Clean top navigation.
3. Market summary cards.
4. Watchlist / indices panel.
5. Asset detail header with symbol, price, change, status, and exchange.
6. Tabs such as:
   - Overview
   - Chart
   - Signals
   - Strategies
   - Backtest
   - News
   - Technicals
   - Components
7. Minimal but strong typography.
8. Good spacing and clear hierarchy.
9. Green/red financial color semantics.
10. Smooth hover states and interactive cards.
11. Light and dark mode support.
12. Responsive layout for desktop and laptop screens.

---

# DATA CONTEXT

QuantMind may receive data like:

- Index/asset name
- Symbol/ticker
- Price
- Percentage change
- Volume
- Open
- Previous close
- Day range
- Market status
- Chart candles or time-series data
- AI sentiment score
- Signal confidence
- Strategy performance
- Backtest metrics
- Drawdown
- Sharpe ratio
- Win rate
- News headlines
- Watchlist assets
- Macro indicators
- Crypto/forex/stocks/futures sections

You must inspect the available data structure in the codebase or sample data and design the UI around the actual data. Do not create fake UI sections that cannot be powered by the existing data unless you clearly mark them as optional/future features.

---

# DESIGN DIRECTION

Create a premium dashboard with this layout:

## 1. Top Navigation Bar

Include:
- QuantMind logo
- Global search bar
- Navigation links:
  - Dashboard
  - Markets
  - Strategies
  - Backtests
  - Signals
  - Research
  - Settings
- Theme toggle
- User/profile button

The navbar should feel clean and fintech-grade.

## 2. Market / Asset Header

At the top of the main content, show:

- Asset name
- Symbol
- Exchange/source
- Current price
- Currency
- Daily change
- Percentage change
- Market status
- Last updated time

Example layout:

S&P 500  
SPX · US Index  
7,483.22 USD  
-16.13 / -0.22%  

For QuantMind, adapt this to the selected asset or strategy.

## 3. Main Chart Area

The chart area should be the visual center.

Requirements:
- Large card container
- Candlestick or line chart
- Time range selector:
  - 1D
  - 5D
  - 1M
  - 6M
  - YTD
  - 1Y
  - All
- Chart controls:
  - Fullscreen
  - Indicators
  - Compare
  - Screenshot/export
- Crosshair-like hover behavior if possible
- Tooltip with date, price, volume, and signal info
- Clean axis and grid styling

If the project currently uses a charting library, improve it.
If it does not, recommend and implement a suitable one such as Lightweight Charts, Recharts, ECharts, or TradingView Lightweight Charts depending on the stack.

## 4. Right Sidebar / Watchlist

Create a right panel similar to market watch.

Show:
- Major indices
- Crypto
- Forex
- Stocks
- User watchlist
- Price
- Daily percentage change
- Small status badge

Cards should use:
- Red for negative movement
- Green for positive movement
- Neutral gray for flat data

## 5. Market Summary Cards

Below or beside the chart, add compact cards:

- Market sentiment
- AI signal
- Volatility
- Volume
- Risk score
- Strategy confidence
- Sharpe ratio
- Max drawdown
- Win rate
- Latest backtest result

Each card should have:
- Label
- Main value
- Small change indicator
- Optional mini sparkline
- Clear visual hierarchy

## 6. AI / Quant Insight Panel

Since this app is QuantMind, add a section that feels unique:

Title examples:
- QuantMind Signal
- AI Market Read
- Strategy Intelligence
- Model Confidence
- Regime Detection

It should display:
- Bullish / Bearish / Neutral signal
- Confidence percentage
- Key reasoning bullets
- Risk warning
- Suggested scenario
- Invalidated if price breaks a certain level

Make it look like a serious quant dashboard, not a chatbot.

Avoid chat bubble UI unless necessary.

## 7. Strategy / Backtest Section

Create a professional table or card layout showing:

- Strategy name
- Asset
- Timeframe
- Return
- Sharpe
- Max drawdown
- Win rate
- Number of trades
- Status
- Last run

The table should support:
- Sorting
- Filtering
- Hover states
- Clean badges
- Responsive behavior

## 8. News / Research Feed

Add a feed section:

- Headline
- Source
- Time
- Related ticker
- Sentiment badge
- Short summary

Make it look similar to a financial news terminal.

## 9. Technical Indicators Section

Show indicator summary:

- RSI
- MACD
- Moving Average
- Trend strength
- Volatility
- Volume profile
- Support/resistance

Use badge states:
- Strong Sell
- Sell
- Neutral
- Buy
- Strong Buy

Do not overdecorate. Keep it clean.

---

# VISUAL STYLE REQUIREMENTS

Use this visual style:

## Light Mode
- Background: near-white / soft gray
- Cards: white
- Borders: subtle gray
- Text: deep black / slate
- Muted text: gray
- Positive: green
- Negative: red
- Accent: blue, cyan, or violet, but use sparingly

## Dark Mode
- Background: deep navy / near black
- Cards: slightly lighter dark surface
- Borders: subtle slate
- Text: white / light gray
- Positive: green
- Negative: red
- Accent: cyan / blue / violet

## Typography
Use modern sans-serif typography:
- Inter
- Geist
- Satoshi
- Plus Jakarta Sans
- IBM Plex Sans

Hierarchy:
- Big price numbers
- Strong section titles
- Small metadata labels
- Compact table text

## Spacing
Use consistent spacing:
- 8px system
- Cards with 16–24px padding
- Rounded corners but not too bubbly
- Subtle shadows only
- Avoid childish gradients
- Avoid excessive glassmorphism

## Interactions
Add:
- Hover states
- Active tab states
- Smooth transitions
- Skeleton loading
- Empty states
- Error states
- Tooltip states
- Responsive collapse behavior

---

# IMPORTANT UI QUALITY RULES

Avoid these mistakes:

- Do not make it look like a generic AI SaaS landing page.
- Do not use random gradients everywhere.
- Do not use huge empty cards with little data.
- Do not use childish icons.
- Do not make the chart too small.
- Do not overuse purple glow effects.
- Do not make every card the same size if the data hierarchy is different.
- Do not hide important market numbers.
- Do not create fake data unless clearly labeled as mock data.
- Do not make the UI feel like a template.

The UI must feel like a real market dashboard used by serious traders.

---

# TASK

Analyze the current QuantMind codebase and do the following:

1. Inspect the existing pages, components, styles, chart components, data fetching, and layout.
2. Identify why the current UI looks bad or amateur.
3. Redesign the information architecture.
4. Improve the visual hierarchy.
5. Create or refactor reusable components:
   - Navbar
   - AssetHeader
   - MarketSummaryCard
   - ChartPanel
   - WatchlistPanel
   - MetricCard
   - SignalPanel
   - StrategyTable
   - NewsFeed
   - TechnicalSummary
   - Tabs
   - Badge
   - SkeletonLoader
6. Make the UI responsive.
7. Make the UI support dark mode if possible.
8. Make the design data-driven.
9. Keep the code clean and maintainable.
10. Explain all major changes after implementation.

---

# EXPECTED OUTPUT

Return the result in this structure:

## 1. UI Audit
Explain what is wrong with the current UI.

## 2. Design Plan
Explain the new layout and component structure.

## 3. Component Architecture
List all components that will be created or modified.

## 4. Implementation
Provide the actual code changes.

## 5. Styling System
Explain colors, typography, spacing, cards, badges, tables, and chart styling.

## 6. Responsive Behavior
Explain how the layout adapts to:
- Desktop
- Laptop
- Tablet
- Mobile

## 7. Final Result Description
Describe how the finished UI should look.

## 8. Next Improvements
Suggest future upgrades such as:
- Real-time websocket data
- Strategy comparison
- Portfolio dashboard
- Alert system
- Economic calendar
- Multi-chart layout
- AI research notebook

---

# IMPLEMENTATION RULES

- Use the existing tech stack of the project.
- Do not rewrite the whole app unless necessary.
- Prefer incremental refactor.
- Preserve existing functionality.
- Improve UI without breaking data flow.
- Use real existing data whenever available.
- If data is missing, create clearly separated mock data with comments.
- Keep components reusable and scalable.
- Write clean production-grade code.
- Use TypeScript if the project already uses TypeScript.
- Use Tailwind CSS if the project already uses Tailwind.
- Use existing styling conventions unless they are the reason the UI looks bad.
- Do not add unnecessary dependencies unless justified.

---

# FINAL QUALITY CHECK

Before finishing, check:

- Does the UI look premium?
- Does it feel like a fintech/market platform?
- Is the chart the main focus?
- Is the hierarchy clear?
- Are red/green values consistent?
- Is the spacing clean?
- Is it responsive?
- Does it avoid generic AI dashboard aesthetics?
- Does it use the actual QuantMind data?
- Does every section have a purpose?

If not, improve it again before giving the final answer.

# PRODUCT IDENTITY — WHAT QUANTMIND ACTUALLY IS

QuantMind is not a “magic trading bot” and not a get-rich-quick app.

QuantMind is a personal quantitative finance research laboratory built to study how markets such as stocks and crypto move, then test trading strategies using data, statistics, machine learning, and risk management.

The UI must communicate this identity clearly:

- QuantMind is a research platform.
- QuantMind is a learning and portfolio project.
- QuantMind is built around scientific testing, not guessing.
- QuantMind helps evaluate whether a trading idea is valid before risking real money.
- QuantMind should feel like a small hedge-fund research terminal, not a casual crypto bot.
- QuantMind must look serious, technical, and credible.

Avoid copywriting or UI patterns that imply guaranteed profit, easy money, or “AI predicts the market perfectly.”

Use language like:
- Research
- Backtest
- Signal confidence
- Risk exposure
- Market regime
- Strategy validation
- Historical simulation
- Model performance
- Paper trading
- Testnet
- Portfolio lab
- Quant research

Avoid language like:
- Auto profit
- Guaranteed win
- Rich bot
- Perfect prediction
- No-risk trading

---

# QUANTMIND CORE STORY

The app should be designed around this story:

“QuantMind is a small financial research laboratory for learning how markets move, collecting market data, calculating technical indicators, training ML models, backtesting strategies, managing risk, and simulating execution before any real money is used.”

The design should make users immediately understand that QuantMind has a real research workflow:

1. Data Engineering
2. Feature Engineering
3. Machine Learning
4. Backtesting
5. Risk Management
6. Paper Trading / Testnet Execution
7. Dashboard & Reporting

This workflow should be visible somewhere in the product, either as:
- a dashboard pipeline status section,
- a sidebar navigation structure,
- a workflow timeline,
- or a research module overview.

---

# CORE MODULES THAT SHOULD APPEAR IN THE UI

Design QuantMind around these main modules:

## 1. Data Engineering Module

Purpose:
Automatically fetch historical market data such as Bitcoin, stocks, indices, or crypto pairs from sources like Yahoo Finance, Binance, or other market APIs.

UI should show:
- Data source
- Asset symbol
- Time range
- Last data update
- Number of candles/rows
- Data quality status
- Missing values
- Download/sync status

Suggested component names:
- DataIngestionPanel
- DataSourceCard
- DatasetStatusCard
- MarketDataTable

## 2. Feature Engineering / Indicators Module

Purpose:
Transform raw OHLCV data into technical features.

Important context:
QuantMind can calculate around 113 technical indicators, including RSI, moving averages, MACD, volatility indicators, momentum indicators, and others.

UI should show:
- Total indicators generated
- Indicator categories
- Enabled/disabled features
- Latest indicator values
- Indicator health/status
- Feature importance if available

Suggested component names:
- IndicatorExplorer
- FeatureMatrixPanel
- TechnicalIndicatorGrid
- FeatureImportanceChart

## 3. Machine Learning Module

Purpose:
Train models such as XGBoost, LightGBM, or similar ML models to classify or estimate whether price may go up/down based on historical patterns.

UI should show:
- Model name
- Training status
- Accuracy / precision / recall / F1 if available
- Prediction direction
- Prediction confidence
- Latest prediction
- Model version
- Last trained timestamp
- Feature importance

Important:
The UI must not present the model output as guaranteed truth. It should show it as probabilistic research output.

Suggested labels:
- Model Signal
- Confidence
- Probability
- Historical Accuracy
- Model Version
- Last Training Run

Suggested component names:
- ModelPerformanceCard
- PredictionSignalPanel
- MLTrainingStatus
- FeatureImportancePanel

## 4. Backtesting Module

Purpose:
Test a strategy on historical data before using it in real or simulated markets.

Important context:
Backtesting is one of the most important parts of QuantMind.

The UI should show:
- Initial capital
- Final equity
- Total return
- Sharpe ratio
- Max drawdown
- Win rate
- Number of trades
- Transaction cost
- Slippage
- Equity curve
- Drawdown chart
- Trade history
- Monte Carlo simulation result if available

Example real result that may appear:
“EMA Crossover on Bitcoin from Oct 2023 to Jun 2026: initial capital 10 million became 25 million (+150%), with maximum drawdown around -27%.”

Important:
Present this as historical simulation only. Add clear text that good backtest performance does not guarantee future profit.

Suggested component names:
- BacktestSummaryCard
- EquityCurveChart
- DrawdownChart
- TradeHistoryTable
- MonteCarloPanel
- StrategyComparisonTable

## 5. Risk Management Module

Purpose:
Act as the safety layer before any trade or simulated order is executed.

UI should show:
- Position size
- Risk per trade
- Max daily loss
- Max drawdown limit
- Stop loss logic
- Circuit breaker status
- Exposure limit
- Risk score
- Whether trading is allowed or blocked

The risk panel should feel like a “safety gate” or “risk control room.”

Suggested component names:
- RiskGuardPanel
- CircuitBreakerStatus
- PositionSizingCard
- ExposureMeter
- RiskLimitTable

## 6. Execution Module

Purpose:
Support both paper trading and live/testnet execution.

Execution modes:
- Paper Trading: simulated trades using virtual money.
- Testnet / Live Trading: connected to real exchange APIs, but testnet should be used first until the system is proven safe.

UI should show:
- Current mode: Research / Paper / Testnet / Live
- API connection status
- Open positions
- Recent orders
- Order status
- Execution latency if available
- Safety warnings

Important:
Live trading should not be visually encouraged too aggressively. The UI should make Paper/Testnet mode feel like the responsible default.

Suggested component names:
- ExecutionModeBadge
- PaperTradingPanel
- OrderHistoryTable
- ExchangeConnectionStatus
- LiveRiskWarning

## 7. Dashboard & Reporting Module

Purpose:
Show all research results in a readable web interface instead of raw terminal logs.

UI should show:
- Market overview
- Asset chart
- Strategy performance
- AI/ML signal
- Risk status
- Latest backtest
- News/research feed
- System health
- Test status

Suggested component names:
- QuantDashboard
- ResearchOverview
- StrategyPerformancePanel
- SystemHealthPanel
- ResearchReportCard

---

# UI POSITIONING

The final design should feel like:

“TradingView meets a quant research lab.”

It should combine:
- TradingView-style chart-first market dashboard
- Hedge fund research terminal style
- Clean SaaS-level usability
- Developer portfolio-level polish
- AI/ML system transparency
- Risk-first trading interface

Do not make the app look like:
- A generic AI chatbot
- A crypto casino dashboard
- A meme coin tracker
- A basic student project
- A random admin panel template
- A landing page with cards but no real product depth

---

# HOMEPAGE / DASHBOARD HERO IDEA

The main dashboard should immediately communicate:

QuantMind  
Quantitative Market Research Lab

Subtitle:
“Data-driven trading research platform for market data, technical indicators, machine learning signals, backtesting, and risk-controlled paper trading.”

Primary status badges:
- Research Mode
- 113 Indicators
- ML Pipeline Active
- Risk Guard Enabled
- 104 Automated Tests
- Paper Trading Ready

Do not make the hero too big. This is a dashboard, not a marketing landing page.

---

# SUGGESTED MAIN DASHBOARD LAYOUT

Use this dashboard structure:

## Top Area
- QuantMind logo
- Global search
- Market selector
- Navigation
- Theme toggle
- Current mode badge: Research / Paper / Testnet / Live

## Asset / Strategy Header
Show:
- Selected asset
- Symbol
- Current price
- Daily change
- Market status
- Last updated
- Current strategy
- Current model signal

## Main Content Grid

Left / Center:
- Large chart panel
- Equity curve / asset price chart
- Timeframe selector
- Indicator overlay controls
- Signal markers

Right:
- Watchlist
- Current AI signal
- Risk status
- Model confidence
- Circuit breaker status

Below:
- Backtest summary
- Strategy table
- Risk metrics
- Indicator summary
- News / research feed
- System health / automated tests

---

# SPECIAL QUANTMIND UI SECTIONS

Add these sections if supported by available data:

## Research Pipeline Status

A horizontal or vertical pipeline showing:

Data Ingestion → Feature Engineering → ML Training → Backtest → Risk Check → Paper Execution → Report

Each step should have:
- Status: success / warning / failed / running
- Last run time
- Output summary

## Strategy Lab

A section where strategies are displayed like research experiments.

Each strategy card should show:
- Strategy name
- Asset
- Timeframe
- Hypothesis
- Return
- Drawdown
- Sharpe
- Win rate
- Status: Promising / Risky / Failed / Needs More Data

## Model Intelligence Panel

A clean AI/ML insight section showing:
- Prediction: Bullish / Bearish / Neutral
- Confidence
- Top features influencing the signal
- Market regime
- Reasoning summary
- Risk warning

Important:
This should not look like a chatbot response. It should look like a structured research output.

## Risk Guard Panel

A serious-looking panel that answers:
“Is this strategy safe enough to simulate or execute?”

Show:
- Risk per trade
- Max drawdown
- Circuit breaker
- Position sizing
- Exposure
- Status: Allowed / Blocked / Warning

## Test & Reliability Panel

Since QuantMind has 104 automated tests, show reliability as part of the product quality.

Display:
- Automated tests: 104
- Last test run
- Passed / failed
- System health
- Data pipeline health
- Backtest engine health
- Risk engine health

This makes the app look more professional as a software engineering portfolio project.

---

# COPYWRITING STYLE

Use professional but understandable wording.

Good examples:
- “Historical simulation only — not a guarantee of future performance.”
- “Model output is probabilistic and should be validated through backtesting.”
- “Risk guard is active. Position size is limited by configured exposure rules.”
- “Strategy is promising historically, but drawdown remains significant.”
- “Paper trading mode is recommended before live execution.”

Avoid childish or hype wording:
- “AI says buy now!”
- “Guaranteed profit”
- “This strategy will make money”
- “Bot auto rich”
- “100% accurate prediction”

---

# IMPORTANT FINANCE DISCLAIMER UI

Add a small but visible disclaimer in appropriate places:

“QuantMind is a research and simulation tool. Backtest results are historical and do not guarantee future performance. Use paper trading or testnet before considering live execution.”

This disclaimer should appear:
- In the backtest section
- In the ML signal section
- Near execution mode when switching to testnet/live

Keep it clean and professional, not scary.

---

# COMPONENTS TO CREATE OR IMPROVE

Create or refactor these components if they fit the codebase:

- QuantNavbar
- GlobalSearch
- ModeBadge
- AssetHeader
- MarketChartPanel
- TimeframeSelector
- WatchlistPanel
- MarketSummaryCard
- MetricCard
- ResearchPipeline
- DataIngestionStatus
- IndicatorExplorer
- FeatureMatrixPanel
- ModelSignalPanel
- ModelPerformanceCard
- FeatureImportanceChart
- BacktestSummaryCard
- EquityCurveChart
- DrawdownChart
- StrategyLab
- StrategyTable
- RiskGuardPanel
- CircuitBreakerStatus
- PositionSizingCard
- ExecutionPanel
- OrderHistoryTable
- NewsFeed
- TechnicalSummary
- SystemHealthPanel
- AutomatedTestBadge
- EmptyState
- LoadingSkeleton
- ErrorState

---

# DATA-DRIVEN UI REQUIREMENTS

Before designing the UI, inspect the project data shape.

Look for:
- market data schema
- OHLCV data
- indicator outputs
- ML prediction outputs
- backtest result objects
- risk metrics
- strategy configs
- execution mode
- test/system health data

The UI must adapt to the actual available fields.

If certain data is missing, create clearly marked mock/demo data only inside a dedicated mock data file.

Do not hardcode fake values directly inside UI components.

---

# PORTFOLIO QUALITY GOAL

QuantMind should be strong enough to show in a portfolio for:

- Fintech engineering
- Data science
- AI engineering
- Quantitative research
- Backend/data engineering
- Full-stack engineering

The UI should make the project look like serious engineering work, not just a script with charts.

The final product should tell the story:

“This person can build a full quantitative research system: data pipeline, indicators, ML, backtesting, risk engine, execution simulation, dashboard, and automated testing.”

---

# FINAL DESIGN CHECKLIST FOR QUANTMIND

Before finishing the UI redesign, verify:

- Does the UI clearly show QuantMind is a quant research lab?
- Does it avoid sounding like a guaranteed-profit trading bot?
- Is the chart still the main visual focus?
- Are backtest metrics easy to read?
- Is risk management visible and serious?
- Are ML predictions shown as probabilities, not certainties?
- Are the 113 indicators represented professionally?
- Are the 104 automated tests/system reliability visible?
- Is the workflow from data → features → ML → backtest → risk → execution clear?
- Does it look premium like a fintech dashboard?
- Does it avoid generic AI SaaS design?
- Does it feel portfolio-worthy?