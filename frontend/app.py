"""
Modular Frontend Server for AlgoTrendy Trading Platform
This module provides the web interface without modifying core trading system files.
"""
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import httpx
import uvicorn

app = FastAPI(title="AlgoTrendy Frontend", description="Web Interface for Trading Platform")

# Allow cross-origin requests from local SPA dev servers (Vite, etc.)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during local development enable any origin; tighten in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

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
            # Prefer a health endpoint if available
            health_url = f"{BACKEND_URL}/api/health"
            try:
                resp = await client.get(health_url, timeout=5.0)
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except Exception:
                        return {"status": "backend_ok"}
            except Exception:
                # fall through to root check
                pass

            # Fallback: make a HEAD request to backend root to check reachability
            resp = await client.head(BACKEND_URL, timeout=5.0)
            if resp.status_code < 400:
                return {"status": "backend_ok"}
            else:
                return {"status": "backend_unreachable", "code": resp.status_code}
    except Exception as e:
        return {"status": "backend_unreachable", "error": str(e)}

@app.get("/api/health")
async def frontend_health():
    """Frontend health check"""
    return {"status": "healthy", "service": "AlgoTrendy Frontend"}


@app.get('/api/spa/info')
async def spa_info():
    """Small SPA discovery endpoint: returns backend availability + version hints"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BACKEND_URL}/api/health", timeout=3.0)
            backend = resp.json() if resp.status_code == 200 else {"status": "unknown"}
    except Exception:
        backend = {"status": "unreachable"}

    return {
        "frontend": {"status": "healthy", "version": "0.1"},
        "backend": backend,
        "proxy_prefix": "/api/proxy"
    }

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


@app.get('/htmx/models', response_class=HTMLResponse)
async def htmx_models(request: Request):
    """Render model list as an HTMX fragment."""
    data = await get_trading_models()
    models = data.get('models', []) if isinstance(data, dict) else []
    return templates.TemplateResponse(
        'partials/models_table.html',
        {
            'request': request,
            'models': models,
        },
    )


@app.get('/htmx/settings', response_class=HTMLResponse)
async def htmx_settings(request: Request):
    """Render settings panel."""
    return templates.TemplateResponse('partials/settings_panel.html', {
        'request': request,
    })
@app.get("/api/trading/backtests")
async def get_backtest_results():
    """Get backtest results from backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/trading/backtests")
            return response.json()
    except Exception as e:
        return {"error": str(e), "backtests": [], "total_backtests": 0}


# Generic proxy endpoints used by the frontend UI (portfolio, positions, performance)
@app.get('/api/proxy/portfolio')
async def proxy_portfolio():
    """Proxy portfolio overview from backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/portfolio")
            return response.json()
    except Exception as e:
        return {
            "error": str(e),
            "total_value": 0,
            "cash_balance": 0,
            "invested_amount": 0,
            "unrealized_pnl": 0,
            "realized_pnl": 0,
            "day_change": 0,
            "day_change_percent": 0,
            "risk_metrics": {},
            "portfolio_performance": {},
            "last_updated": None,
        }


@app.get('/api/proxy/positions')
async def proxy_positions():
    """Proxy current positions from backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/positions")
            return response.json()
    except Exception as e:
        return {"error": str(e), "positions": [], "total_positions": 0, "total_market_value": 0}


@app.get('/api/proxy/portfolio/performance')
async def proxy_portfolio_performance():
    """Proxy portfolio performance data from backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/portfolio/performance")
            return response.json()
    except Exception as e:
        return {"error": str(e), "summary": {}, "performance_data": []}


# Module configuration endpoints used by the frontend
@app.get('/api/config/modules')
async def get_module_config():
    """Fetch module configuration from backend or return sensible defaults"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/config/modules")
            return response.json()
    except Exception:
        # Return a simple default modules map the frontend can use
        return {
            "modules": {
                "portfolio": {"id": "portfolio", "name": "Portfolio", "enabled": True},
                "trading": {"id": "trading", "name": "Trading", "enabled": True},
                "market_data": {"id": "market_data", "name": "Market Data", "enabled": True},
                "strategies": {"id": "strategies", "name": "Strategies", "enabled": True},
                "backtesting": {"id": "backtesting", "name": "Backtesting", "enabled": True},
                "ml_models": {"id": "ml_models", "name": "ML Models", "enabled": True},
                "stocks": {"id": "stocks", "name": "Stocks", "enabled": True},
                "futures": {"id": "futures", "name": "Futures", "enabled": True},
                "crypto": {"id": "crypto", "name": "Crypto", "enabled": True},
                "signals": {"id": "signals", "name": "Trading Signals", "enabled": True}
            }
        }


@app.post('/api/config/modules/{module_id}/toggle')
async def toggle_module(module_id: str):
    """Toggle a module on the backend or simulate toggle when backend is unreachable"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{BACKEND_URL}/config/modules/{module_id}/toggle")
            return response.json()
    except Exception as e:
        return {"error": str(e), "status": "failed", "module_id": module_id}

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


@app.post('/api/proxy/positions/{symbol}/close')
async def proxy_close_position(symbol: str):
    """Proxy to close a position on the backend; returns a simulated response if backend unreachable."""
    candidates = [
        f"{BACKEND_URL}/trading/positions/{symbol}/close",
        f"{BACKEND_URL}/positions/{symbol}/close",
    ]
    try:
        async with httpx.AsyncClient() as client:
            for url in candidates:
                try:
                    resp = await client.post(url, timeout=5.0)
                    if resp.status_code < 400:
                        try:
                            return resp.json()
                        except Exception:
                            return {"status": "closed", "symbol": symbol}
                except Exception:
                    continue
    except Exception as e:
        return {"error": str(e), "status": "failed", "symbol": symbol}

    # Backend unreachable or endpoints not implemented — simulate success so UI can proceed
    return {"status": "simulated_closed", "symbol": symbol}


@app.post('/api/proxy/positions/{symbol}/reopen')
async def proxy_reopen_position(symbol: str):
    """Proxy to reopen a previously closed position. Tries known endpoints then simulates on failure."""
    candidates = [
        f"{BACKEND_URL}/trading/positions/{symbol}/reopen",
        f"{BACKEND_URL}/positions/{symbol}/reopen",
    ]
    try:
        async with httpx.AsyncClient() as client:
            for url in candidates:
                try:
                    resp = await client.post(url, timeout=5.0)
                    if resp.status_code < 400:
                        try:
                            return resp.json()
                        except Exception:
                            return {"status": "reopened", "symbol": symbol}
                except Exception:
                    continue
    except Exception as e:
        return {"error": str(e), "status": "failed", "symbol": symbol}

    return {"status": "simulated_reopened", "symbol": symbol}

# Backend Control Endpoints
@app.post("/api/backend/start")
async def start_backend():
    """Start the backend API server"""
    try:
        # This would restart the backend workflow
        import subprocess
        result = subprocess.run([
            "replit", "workflow", "restart", "Trading Backend API"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return {"status": "success", "message": "Backend starting..."}
        else:
            return {"status": "error", "message": f"Failed to start backend: {result.stderr}"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "Backend start timeout - it may still be starting"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to start backend: {str(e)}"}

@app.post("/api/backend/stop") 
async def stop_backend():
    """Stop the backend API server"""
    try:
        # This would stop the backend workflow
        import subprocess
        result = subprocess.run([
            "replit", "workflow", "stop", "Trading Backend API"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return {"status": "success", "message": "Backend stopped"}
        else:
            return {"status": "error", "message": f"Failed to stop backend: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to stop backend: {str(e)}"}

if __name__ == "__main__":
    # Run the frontend server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
