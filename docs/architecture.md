# QuantMind System Architecture

## Overview

QuantMind is a **modular, event-driven** quantitative trading platform designed as a personal institutional-grade research lab. The system follows a microservices-inspired architecture where each module communicates through a shared event bus and database.

## System Architecture Diagram

```mermaid
flowchart TB
    subgraph External["External Data Sources"]
        BN["🔶 Binance API"]
        YF["📈 Yahoo Finance"]
        FG["😱 Fear & Greed API"]
        PG["📊 Polygon.io"]
    end

    subgraph Ingestion["Data Ingestion Layer"]
        DP["Data Pipeline<br/>ETL Orchestrator"]
        DV["Data Validator<br/>Quality Checks"]
        DC["Data Cleaner<br/>Auto-Fix"]
    end

    subgraph Storage["Storage Layer"]
        DB[("SQLite / TimescaleDB<br/>OHLCV + Metadata")]
        FS[("Feature Store<br/>300+ Features")]
        RD[("Redis<br/>Real-time Cache")]
    end

    subgraph Research["Research & Feature Layer"]
        FE["Feature Engineering<br/>Technical · Statistical<br/>Microstructure · Sentiment"]
        FSel["Feature Selection<br/>SHAP · MI · RFE<br/>Permutation · PCA"]
        MLF["MLflow<br/>Experiment Tracking"]
    end

    subgraph ML["Machine Learning Layer"]
        GB["Gradient Boosting<br/>XGBoost · LightGBM · CatBoost"]
        DL["Deep Learning<br/>LSTM · Transformer · TFT"]
        REG["Regime Detection<br/>HMM · HDBSCAN · GMM"]
        XAI["Explainable AI<br/>SHAP · LIME"]
    end

    subgraph Strategy["Strategy Layer"]
        RLA["RL Agent<br/>PPO · SAC · DQN"]
        GA["Genetic Algorithm<br/>DEAP · Optuna"]
        SG["Strategy Generator<br/>Rule Mining"]
    end

    subgraph Backtest["Backtesting Layer"]
        BTE["Backtest Engine<br/>Event-Driven"]
        VBT["VectorBT<br/>Fast Vectorized"]
        WF["Walk-Forward"]
        MC["Monte Carlo"]
    end

    subgraph Portfolio["Portfolio & Risk Layer"]
        PO["Portfolio Optimizer<br/>Kelly · Risk Parity · CVaR"]
        BL["Black-Litterman"]
        RE["Risk Engine<br/>VaR · MaxDD · Ruin"]
        CB["Circuit Breaker<br/>Auto-Stop"]
    end

    subgraph Execution["Execution Layer"]
        PP["Paper Trading<br/>Simulated"]
        LT["Live Trading<br/>CCXT Connector"]
        OM["Order Manager<br/>State Machine"]
        LM["Latency Monitor"]
    end

    subgraph Monitor["Monitoring Layer"]
        ST["Streamlit Dashboard<br/>Research UI"]
        GR["Grafana<br/>Real-time Metrics"]
    end

    %% Data Flow
    BN & YF & FG & PG --> DP
    DP --> DV --> DC --> DB
    DB --> FE --> FS
    FS --> FSel
    FSel --> GB & DL & RLA
    GB & DL --> REG
    GB & DL --> XAI
    REG --> SG & GA
    RLA --> BTE
    SG & GA --> BTE
    BTE & VBT --> WF & MC
    WF & MC --> PO & BL
    PO & BL --> RE --> CB
    CB --> PP & LT
    PP & LT --> OM --> LM
    LM --> ST & GR
    GB & DL & RLA --> MLF

    %% Styling
    classDef external fill:#4A90D9,color:#fff,stroke:none
    classDef ingestion fill:#7B68EE,color:#fff,stroke:none
    classDef storage fill:#2E8B57,color:#fff,stroke:none
    classDef research fill:#E6783E,color:#fff,stroke:none
    classDef ml fill:#DC143C,color:#fff,stroke:none
    classDef strategy fill:#20B2AA,color:#fff,stroke:none
    classDef backtest fill:#DAA520,color:#fff,stroke:none
    classDef portfolio fill:#8B008B,color:#fff,stroke:none
    classDef execution fill:#FF6347,color:#fff,stroke:none
    classDef monitor fill:#FFD700,color:#000,stroke:none

    class BN,YF,FG,PG external
    class DP,DV,DC ingestion
    class DB,FS,RD storage
    class FE,FSel,MLF research
    class GB,DL,REG,XAI ml
    class RLA,GA,SG strategy
    class BTE,VBT,WF,MC backtest
    class PO,BL,RE,CB portfolio
    class PP,LT,OM,LM execution
    class ST,GR monitor
```

## Module Dependency Graph

```mermaid
graph LR
    Core["core/"] --> Data["data/"]
    Core --> Features["features/"]
    Core --> Models["models/"]
    Core --> RL["rl/"]
    Core --> Backtest["backtest/"]
    Core --> Portfolio["portfolio/"]
    Core --> Execution["execution/"]
    Core --> Dashboard["dashboard/"]

    Data --> Features
    Features --> Models
    Features --> RL
    Features --> Backtest
    Models --> Backtest
    RL --> Backtest
    Backtest --> Portfolio
    Portfolio --> Execution
    Execution --> Dashboard
    Models --> Dashboard
```

## Data Flow

### Phase 1: Data Acquisition
1. **Providers** fetch raw data from APIs (Binance, Yahoo, etc.)
2. **Validator** checks data quality (gaps, nulls, OHLC logic)
3. **Cleaner** fixes issues (duplicates, negative volumes, OHLC violations)
4. **Storage** writes clean data to database with upsert semantics

### Phase 2: Feature Engineering
1. **Registry** discovers all registered feature functions
2. **Technical** computes 150+ TA indicators (SMA, RSI, MACD, BB, etc.)
3. **Statistical** computes advanced features (entropy, Hurst, Garman-Klass)
4. **Microstructure** computes market structure proxies (Amihud, Kyle's λ)
5. **Sentiment** integrates Fear & Greed, funding rate
6. **Store** persists features to the feature store

### Phase 3: Feature Selection & ML
1. **Selection** runs SHAP, MI, RFE, Permutation, PCA
2. **Consensus** ranking via Reciprocal Rank Fusion
3. **ML Lab** trains gradient boosting and deep learning models
4. **Regime Detection** labels market states (HMM, clustering)
5. **XAI** generates human-readable explanations

### Phase 4: Strategy & Execution
1. **RL Agent** learns trading policy via PPO/SAC/DQN
2. **Backtest** validates strategies with walk-forward + Monte Carlo
3. **Portfolio** optimizes allocation (Kelly, Risk Parity, CVaR)
4. **Risk Engine** enforces limits (VaR, MaxDD, circuit breaker)
5. **Execution** submits orders via paper or live broker

## Key Design Decisions

| Decision | Rationale |
|:---|:---|
| **SQLite → TimescaleDB** | Zero-setup local dev, production-ready upgrade path |
| **Feature Registry pattern** | Extensible, auto-documented, selective computation |
| **Event Bus** | Decoupled modules, easy to swap for Redis/Kafka |
| **Pydantic configs** | Type-safe, validated, IDE-friendly |
| **Abstract base classes** | Swappable implementations per module |
| **Upsert semantics** | Safe incremental data loading |

## Database Schema

```mermaid
erDiagram
    OHLCV {
        int id PK
        datetime timestamp
        string symbol
        string timeframe
        float open
        float high
        float low
        float close
        float volume
        string source
    }

    FEATURE_STORE {
        int id PK
        datetime timestamp
        string symbol
        string timeframe
        string feature_name
        float value
        string version
    }

    SENTIMENT {
        int id PK
        datetime timestamp
        string indicator
        float value
        string label
        string source
    }

    TRADES_LOG {
        int id PK
        string trade_id
        datetime timestamp
        string symbol
        string side
        float quantity
        float entry_price
        float exit_price
        float pnl
        string explanation
        string mode
    }

    SELECTED_FEATURES {
        int id PK
        string selection_run_id
        string method
        string feature_name
        float importance_score
        int rank
    }

    OHLCV ||--o{ FEATURE_STORE : generates
    OHLCV ||--o{ TRADES_LOG : triggers
    FEATURE_STORE ||--o{ SELECTED_FEATURES : selects
```
