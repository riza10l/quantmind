# 🧠 QuantMind
**Personal Institutional-Grade Systematic Trading & Research Lab**

QuantMind is an end-to-end quantitative research, machine learning, and systematic trading environment. It is built to bridge the gap between academic algorithmic trading research and production-grade execution by leveraging modern data engineering, machine learning pipelines, and reinforcement learning.

The architecture takes inspiration from the tech stacks of industry leaders (QuantConnect, Two Sigma, Jane Street) and combines them into a modular, highly scalable local laboratory.

---

## 🏗️ Architecture & Core Components

QuantMind is designed strictly around event-driven paradigms, ensuring that backtesting perfectly mirrors live execution. The system is decoupled into four primary modules:

### 1. Data Engineering & ETL (`src/data/`)
A robust pipeline that handles historical and live data ingestion.
- **Providers:** CCXT (Binance), yfinance (Yahoo Finance), and Alternative.me (Fear & Greed Index).
- **Storage:** SQLite/PostgreSQL with `INSERT OR IGNORE` UPSERT semantics to ensure idempotency.
- **Validation:** 9-step automated data cleaning (handling missing ticks, negative prices, volume anomalies).

### 2. Feature Store & Engineering (`src/features/`)
A dynamic registry (`@feature_registry.register`) for on-the-fly feature computation.
- **Indicators:** 150+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.).
- **Statistical & Microstructure:** Shannon Entropy, Hurst Exponent, Amihud Illiquidity, Kyle's Lambda.
- **Feature Selection:** Automated ranking via SHAP, Mutual Information, RFE, and PCA using Reciprocal Rank Fusion.

### 3. Machine Learning Lab (`src/models/`)
Predictive modeling utilizing cross-validation techniques specifically designed for financial time-series to prevent look-ahead bias.
- **Models:** Gradient Boosting (XGBoost, LightGBM, CatBoost) and a Meta-Learner Stacking Ensemble.
- **Cross-Validation:** Purged K-Fold and Walk-Forward (expanding window) CV.
- **Regime Detection:** 
  - *Gaussian Hidden Markov Models (HMM)* for latent state detection (Bull/Bear/Neutral).
  - *HDBSCAN & GMM Clustering* for multidimensional regime discovery based on volatility and momentum.
- **Explainable AI (XAI):** SHAP (TreeExplainer & KernelExplainer) integrated to generate human-readable narratives for every trading signal.
- **Hyperparameter Optimization:** Automated tuning via Optuna.
- **Experiment Tracking:** Full integration with MLflow for tracking parameters, metrics, and model artifacts.

### 4. Strategy & Optimization (`src/strategy/` & `src/backtest/`)
- **Genetic Algorithm:** DEAP-inspired optimizer that evolves strategy parameters using Tournament Selection, Uniform Crossover, and Gaussian Mutation.
- **Backtesting Metrics:** Comprehensive evaluation including Sharpe Ratio, Profit Factor, Max Drawdown, and Win Rate.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- (Optional but recommended) Docker for TimescaleDB, Redis, Grafana, and MLflow.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/riza10l/quantmind.git
cd quantmind
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up the environment variables (copy `.env.example` to `.env` and fill in the keys).

### CLI Usage

QuantMind provides a CLI for managing the pipeline:

```bash
# 1. Initialize the local database
python src/cli.py setup

# 2. Download historical data
python src/cli.py download --symbols BTC/USDT --provider binance --timeframe 1d

# 3. Compute and store features
python src/cli.py features --symbol BTC/USDT

# 4. Perform feature selection
python src/cli.py select --symbol BTC/USDT --top-k 30

# 5. Train the ML models with Optuna auto-tuning
python src/cli.py train --symbol BTC/USDT --models xgboost,lightgbm --trials 10
```

### Research Notebooks
For interactive EDA and manual feature analysis, use the provided Jupyter templates:
- `notebooks/01_research_template.ipynb`
- `notebooks/02_feature_analysis.ipynb`

---

## 🗺️ Project Roadmap

- [x] **Phase 1:** Core Infrastructure, Data Engineering, and Feature Store.
- [x] **Phase 2:** Machine Learning Classifiers, Regime Detection (HMM/Clustering), XAI, and GA Optimizer.
- [ ] **Phase 3:** Full Backtesting Engine, Strategy Templates, and RL (Reinforcement Learning) Environment.
- [ ] **Phase 4:** Portfolio Risk Optimization, Circuit Breakers, Live Broker Execution (CCXT), and Streamlit Dashboard.

---

## 🛡️ License
MIT License. See `LICENSE` for more information.

## 🤝 Author
Built by [Riza Wahyu Nugraha](https://github.com/riza10l) as a comprehensive laboratory for systematic trading research.
