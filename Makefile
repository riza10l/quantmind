# ============================================================
# QuantMind Makefile
# Usage: make <target>
# ============================================================

.PHONY: help setup download-data features select train backtest dashboard test lint clean

PYTHON = python
PIP = pip

help: ## Show this help message
	@echo QuantMind - Systematic Trading Research Lab
	@echo ============================================
	@echo.
	@findstr /R "^[a-zA-Z_-]*:.*##" Makefile

setup: ## Create virtual env, install dependencies, init database
	$(PIP) install -e ".[all]"
	$(PYTHON) scripts/setup_db.py
	@echo Setup complete!

setup-minimal: ## Install only Fase 1 dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PYTHON) scripts/setup_db.py
	@echo Minimal setup complete!

download-data: ## Download market data (use SYMBOLS=BTC/USDT,ETH/USDT)
	$(PYTHON) scripts/download_data.py $(if $(SYMBOLS),--symbols $(SYMBOLS),)

features: ## Generate features from downloaded data
	$(PYTHON) -m src.features.store

select: ## Run feature selection
	$(PYTHON) -m src.features.selection

train: ## Train ML models (use MODEL=xgboost)
	$(PYTHON) scripts/train_model.py $(if $(MODEL),--model $(MODEL),)

backtest: ## Run backtesting (use STRATEGY=ema_cross)
	$(PYTHON) scripts/run_backtest.py $(if $(STRATEGY),--strategy $(STRATEGY),)

dashboard: ## Launch Streamlit dashboard
	streamlit run src/dashboard/app.py

mlflow-ui: ## Launch MLflow tracking UI
	mlflow ui --host 0.0.0.0 --port 5000

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html

lint: ## Run linter and type checker
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean: ## Remove build artifacts and caches
	@if exist __pycache__ rd /s /q __pycache__
	@if exist .pytest_cache rd /s /q .pytest_cache
	@if exist .mypy_cache rd /s /q .mypy_cache
	@if exist htmlcov rd /s /q htmlcov
	@if exist dist rd /s /q dist
	@if exist build rd /s /q build
	@if exist *.egg-info rd /s /q *.egg-info
	@echo Cleaned!

docker-up: ## Start all services (DB, Redis, MLflow, Grafana)
	docker-compose -f docker/docker-compose.yml up -d

docker-down: ## Stop all services
	docker-compose -f docker/docker-compose.yml down
