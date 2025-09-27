# AlgoTrendy Trading Platform - Replit Setup

## Project Overview
AlgoTrendy is a comprehensive algorithmic trading platform that combines machine learning, market replay testing, and cloud deployment capabilities. The system supports both stocks and futures trading with advanced risk management and automated execution.

## Architecture
- **Backend**: Python FastAPI server (ai_orchestrator_api.py) running on port 5000
- **CLI Interface**: Interactive trading interface (main.py) with multiple trading systems
- **AI Orchestration**: Multi-AI provider orchestration with OpenAI, Anthropic, and GitHub integrations
- **Trading Systems**: ML models, backtesting, market replay, automated futures trading, crypto scalping

## Recent Changes (September 27, 2025)
1. Cleaned up duplicate dependencies in src/requirements.txt
2. Fixed missing `import os` in src/ai_orchestrator.py (line 13)
3. Fixed missing `import os` in src/main.py (line 11)
4. Set up FastAPI server workflow on port 5000 with webview output
5. Configured deployment for VM target (stateful backend)

## Project Structure
- `/src/` - Main source code directory
  - `ai_orchestrator_api.py` - FastAPI server entry point
  - `main.py` - CLI interface entry point
  - `ai_orchestrator.py` - AI provider orchestration
  - `trading_interface.py` - Unified trading interface
  - Various trading modules (backtester, data_manager, etc.)
- `/examples/` - Demo scripts and examples
- `/docs/` - Documentation files

## User Preferences
- Project successfully imported from GitHub
- Ready for algorithmic trading development and testing
- Both CLI and web API interfaces available

## Current State
- ✅ All dependencies installed successfully
- ✅ FastAPI server running on port 5000
- ✅ CLI interface working
- ✅ Deployment configured for production
- ✅ Import completed successfully