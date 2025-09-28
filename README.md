# AlgoTrendy: Advanced ML Futures Trading Platform

[![Isolated smoke tests](https://github.com/KenyBoi/algotrendy/actions/workflows/isolated-tests.yml/badge.svg)](https://github.com/KenyBoi/algotrendy/actions/workflows/isolated-tests.yml)


## Quick Launch Instructions

### Launch the Trading Interface
```bash
cd c:/Users/kenne/algotrendy
python src/main.py interface
```
Launches the unified menu with access to ML tools, backtesting, monitoring, and live integrations.

### Available Commands

#### Main Interface
```bash
python src/main.py interface          # Launch unified trading interface (recommended)
```

#### Individual AI/ML Commands
```bash
python src/main.py advanced-train          # Advanced ML training (>80% accuracy)
python src/main.py discover-indicators     # AI indicator discovery
python src/main.py crypto-strategies       # AI crypto strategy discovery
python src/main.py futures-strategies      # AI futures strategy discovery
```

#### Trading System Commands
```bash
python src/main.py crypto-scalp           # Start crypto scalping (24/7)
python src/main.py futures-auto           # Start automated futures trading
```

#### Testing & Development
```bash
python src/main.py backtest               # Run backtests
python src/main.py replay-demo            # Market replay testing
python src/main.py qc-setup               # Setup QuantConnect
python src/main.py qc-projects            # List QuantConnect projects
python src/main.py qc-deploy              # Deploy to QuantConnect
```

Smoke tests badge: The badge above shows the status of the fast smoke-suite that runs `pytest isolated_tests` on each PR and pushes to `main`.

## System Overview
AlgoTrendy combines machine learning, backtesting, and live integrations for equities, futures, and crypto. The platform provides:
- Feature-rich ML pipelines with monitoring and automated retraining
- Market replay and quant tooling (QuantConnect, Alpaca)
- Unified interface and CLI for demos, live trading, and diagnostics

## Repository Structure (high level)
- `src/algotrendy/` � core package (config, data management, ML traders, monitoring, orchestration)
- `examples/` � runnable demos and setup scripts
- `docs/` � documentation (architecture, security, monitoring)
- `tests/` � pytest suites for core components

See `docs/overview.md` and `docs/security/credential_rotation.md` for additional details on architecture and operational procedures.
