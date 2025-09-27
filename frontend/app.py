"""
Modular Frontend Server for AlgoTrendy Trading Platform
This module provides the web interface without modifying core trading system files.
"""
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import httpx
import uvicorn

app = FastAPI(title="AlgoTrendy Frontend", description="Web Interface for Trading Platform")

# Get the frontend directory path
frontend_dir = Path(__file__).parent
static_dir = frontend_dir / "static"
templates_dir = frontend_dir / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=templates_dir)

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main trading dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/status")
async def get_backend_status():
    """Get backend API status"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/")
            return response.json()
    except Exception as e:
        return {"status": "backend_unreachable", "error": str(e)}

@app.get("/api/health")
async def frontend_health():
    """Frontend health check"""
    return {"status": "healthy", "service": "AlgoTrendy Frontend"}

# Trading system proxy endpoints
@app.get("/api/trading/models")
async def get_trading_models():
    """Get ML models from backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/trading/models")
            return response.json()
    except Exception as e:
        return {"error": str(e), "models": [], "total_models": 0}

@app.get("/api/trading/strategies")
async def get_trading_strategies():
    """Get trading strategies from backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/trading/strategies")
            return response.json()
    except Exception as e:
        return {"error": str(e), "strategies": [], "total_strategies": 0}

@app.get("/api/trading/backtests")
async def get_backtest_results():
    """Get backtest results from backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/trading/backtests")
            return response.json()
    except Exception as e:
        return {"error": str(e), "backtests": [], "total_backtests": 0}

@app.post("/api/trading/backtests/run")
async def run_backtest(request: Request):
    """Run a new backtest"""
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{BACKEND_URL}/trading/backtests/run", json=data)
            return response.json()
    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    # Run the frontend server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )