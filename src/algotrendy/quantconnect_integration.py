"""
QuantConnect Integration for AlgoTrendy
Connects to QuantConnect cloud platform for advanced backtesting and live trading
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

from .config import CONFIG

logger = logging.getLogger(__name__)

class QuantConnectIntegration:
    """
    Comprehensive QuantConnect API integration for AlgoTrendy
    """

    def __init__(self, user_id: str = None, api_token: str = None):
        """
        Initialize QuantConnect integration

        Args:
            user_id: QuantConnect user ID
            api_token: QuantConnect API token
        """
        self.user_id = user_id or os.getenv('QC_USER_ID')
        self.api_token = api_token or os.getenv('QC_API_TOKEN')

        # Do not raise on import/initialization when credentials are missing. Many
        # callers only want an object to hold methods and will call authenticate()
        # explicitly. If strict behavior is required, pass require_credentials=True.
        self.require_credentials = False
        if not self.user_id or not self.api_token:
            logger.warning("QuantConnect credentials not found - create object and call authenticate() later.")

        # API endpoints
        self.base_url = "https://www.quantconnect.com/api/v2"
        self.live_url = "https://live.quantconnect.com/api/v2"

        # Authentication
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }

        # Setup a requests Session with a small default timeout and retry strategy
        # to make the integration more robust against transient network issues.
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.3,
                        status_forcelist=(429, 500, 502, 503, 504))
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

        # Default request timeout (seconds)
        self._timeout = 10

        # Cache
        self.projects_cache = {}
        self.backtests_cache = {}

        logger.info("QuantConnect integration initialized")

    def _make_request(self, endpoint: str, method: str = 'GET', data: Dict = None,
                     use_live: bool = False, params: Dict = None) -> Dict:
        """
        Make authenticated API request

        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data for POST/PUT
            use_live: Use live trading API

        Returns:
            API response data
        """
        base = self.live_url if use_live else self.base_url
        url = f"{base.rstrip('/')}/{str(endpoint).lstrip('/')}"

        try:
            resp = self.session.request(method=method.upper(), url=url,
                                        headers=self.headers, json=data,
                                        params=params, timeout=self._timeout)
            resp.raise_for_status()
            # Prefer JSON, but gracefully fallback to raw text if response isn't JSON
            try:
                return resp.json()
            except ValueError:
                return {'text': resp.text}

        except requests.exceptions.RequestException as e:
            logger.error(f"QuantConnect API request failed: {e}")
            # Re-raise for callers that want to handle errors; keep behavior
            # predictable for higher-level code.
            raise

    def authenticate(self) -> bool:
        """
        Test authentication and connection

        Returns:
            True if authentication successful
        """
        try:
            # Test with projects endpoint
            response = self._make_request("projects")
            logger.info("QuantConnect authentication successful")
            return True
        except Exception as e:
            logger.error(f"QuantConnect authentication failed: {e}")
            return False

    def get_projects(self, refresh: bool = False) -> List[Dict]:
        """
        Get list of user's projects

        Args:
            refresh: Force refresh from API

        Returns:
            List of project dictionaries
        """
        if not refresh and self.projects_cache:
            return list(self.projects_cache.values())

        try:
            response = self._make_request("projects")
            projects = response.get('projects', [])

            # Cache projects
            self.projects_cache = {p['projectId']: p for p in projects}

            logger.info(f"Retrieved {len(projects)} QuantConnect projects")
            return projects

        except Exception as e:
            logger.error(f"Failed to get projects: {e}")
            return []

    def create_project(self, name: str, language: str = 'Python') -> Optional[int]:
        """
        Create a new QuantConnect project

        Args:
            name: Project name
            language: Programming language (Python, C#)

        Returns:
            Project ID if successful, None otherwise
        """
        try:
            data = {
                'name': name,
                'language': language
            }

            response = self._make_request("projects/create", method='POST', data=data)
            project_id = response.get('projectId') if isinstance(response, dict) else None

            logger.info(f"Created QuantConnect project: {name} (ID: {project_id})")
            return project_id

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None

    def update_algorithm_code(self, project_id: int, filename: str, code: str) -> bool:
        """
        Update algorithm code in a project

        Args:
            project_id: QuantConnect project ID
            filename: Algorithm filename
            code: Algorithm code content

        Returns:
            True if successful
        """
        try:
            data = {
                'fileName': filename,
                'fileContent': code
            }

            self._make_request(f"projects/{project_id}/files/update", method='POST', data=data)
            logger.info(f"Updated algorithm code in project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update algorithm code: {e}")
            return False

    def create_backtest(self, project_id: int, compile_id: str, backtest_name: str) -> Optional[str]:
        """
        Create a backtest for a project

        Args:
            project_id: QuantConnect project ID
            compile_id: Compilation ID from successful compile
            backtest_name: Name for the backtest

        Returns:
            Backtest ID if successful
        """
        try:
            data = {
                'compileId': compile_id,
                'backtestName': backtest_name
            }

            response = self._make_request(f"projects/{project_id}/backtests", method='POST', data=data)
            backtest_id = response.get('backtestId') if isinstance(response, dict) else None

            logger.info(f"Created backtest: {backtest_name} (ID: {backtest_id})")
            return backtest_id

        except Exception as e:
            logger.error(f"Failed to create backtest: {e}")
            return None

    def get_backtest_results(self, project_id: int, backtest_id: str) -> Optional[Dict]:
        """
        Get backtest results

        Args:
            project_id: QuantConnect project ID
            backtest_id: Backtest ID

        Returns:
            Backtest results dictionary
        """
        try:
            response = self._make_request(f"projects/{project_id}/backtests/{backtest_id}")
            return response if isinstance(response, dict) else {'result': response}

        except Exception as e:
            logger.error(f"Failed to get backtest results: {e}")
            return None

    def deploy_live_algorithm(self, project_id: int, compile_id: str,
                            server_type: str = 'LIVE', base_currency: str = 'USD') -> Optional[str]:
        """
        Deploy algorithm to live trading

        Args:
            project_id: QuantConnect project ID
            compile_id: Compilation ID
            server_type: Server type (LIVE, PAPER)
            base_currency: Base currency for account

        Returns:
            Deployment ID if successful
        """
        try:
            data = {
                'compileId': compile_id,
                'serverType': server_type,
                'baseCurrency': base_currency
            }

            response = self._make_request(f"live/{project_id}", method='POST', data=data)
            deploy_id = response.get('deployId') if isinstance(response, dict) else None

            logger.info(f"Deployed live algorithm (ID: {deploy_id})")
            return deploy_id

        except Exception as e:
            logger.error(f"Failed to deploy live algorithm: {e}")
            return None

    def get_live_algorithm_status(self, project_id: int, deploy_id: str) -> Optional[Dict]:
        """
        Get live algorithm status

        Args:
            project_id: QuantConnect project ID
            deploy_id: Deployment ID

        Returns:
            Live algorithm status
        """
        try:
            response = self._make_request(f"live/{project_id}/{deploy_id}", use_live=True)
            return response if isinstance(response, dict) else {'status': response}

        except Exception as e:
            logger.error(f"Failed to get live algorithm status: {e}")
            return None

    def stop_live_algorithm(self, project_id: int, deploy_id: str) -> bool:
        """
        Stop a live algorithm

        Args:
            project_id: QuantConnect project ID
            deploy_id: Deployment ID

        Returns:
            True if successful
        """
        try:
            self._make_request(f"live/{project_id}/{deploy_id}", method='DELETE', use_live=True)
            logger.info(f"Stopped live algorithm {deploy_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop live algorithm: {e}")
            return False

class QuantConnectDataManager:
    """
    QuantConnect data fetching integration
    """

    def __init__(self, qc_integration: QuantConnectIntegration):
        self.qc = qc_integration

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                          resolution: str = 'Daily') -> Any:
        """
        Fetch historical data from QuantConnect

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resolution: Data resolution (Tick, Second, Minute, Hour, Daily)

        Returns:
            DataFrame with historical data
        """
        try:
            # Lazy-import pandas so the main module doesn't require pandas at import time
            import pandas as pd  # local import

            # Note: QuantConnect data API requires specific authentication and
            # endpoint details. This implementation intentionally returns an
            # empty, well-formed DataFrame as a safe placeholder so callers can
            # continue to work without a hard dependency on QC.
            logger.warning("QuantConnect data fetching not fully implemented - using placeholder")
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            return pd.DataFrame(columns=cols)

        except Exception as e:
            logger.error(f"Failed to fetch QuantConnect data: {e}")
            try:
                import pandas as pd
                return pd.DataFrame()
            except Exception:
                # If pandas isn't available, return an empty list to avoid
                # crashing imports—callers should handle this case.
                return []

class QuantConnectAlgorithmManager:
    """
    Manage QuantConnect algorithm deployment and monitoring
    """

    def __init__(self, qc_integration: QuantConnectIntegration):
        self.qc = qc_integration

    def deploy_futures_algorithm(self, algorithm_code: str, algorithm_name: str = "AlgoTrendy Futures") -> Dict:
        """
        Deploy a futures trading algorithm to QuantConnect

        Args:
            algorithm_code: Python algorithm code
            algorithm_name: Name for the algorithm

        Returns:
            Deployment results dictionary
        """
        try:
            # Create project
            project_id = self.qc.create_project(algorithm_name)
            if not project_id:
                return {'success': False, 'error': 'Failed to create project'}

            # Update algorithm code
            success = self.qc.update_algorithm_code(project_id, 'main.py', algorithm_code)
            if not success:
                return {'success': False, 'error': 'Failed to update algorithm code'}

            # Compile algorithm (this would need to be implemented)
            # compile_id = self.qc.compile_algorithm(project_id)

            # For now, return project info
            return {
                'success': True,
                'project_id': project_id,
                'algorithm_name': algorithm_name,
                'message': f'Algorithm deployed to project {project_id}'
            }

        except Exception as e:
            logger.error(f"Failed to deploy futures algorithm: {e}")
            return {'success': False, 'error': str(e)}

def generate_qc_futures_algorithm(symbols: List[str], model_params: Dict) -> str:
    """
    Generate QuantConnect Python algorithm code for futures trading

    Args:
        symbols: List of futures symbols
        model_params: Model parameters dictionary

    Returns:
        QuantConnect algorithm code as string
    """
    algorithm_template = f'''
# QuantConnect Futures Algorithm - Generated by AlgoTrendy
# Algorithm for trading futures contracts

from AlgorithmImports import *

class AlgoTrendyFuturesAlgorithm(QCAlgorithm):

    def Initialize(self):
        """Initialize algorithm"""
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # Add futures contracts
        futures_symbols = {symbols}
        self.contracts = {{}}

        for symbol in futures_symbols:
            # Map common symbols to QC futures
            qc_symbol_map = {{
                'ES': Futures.Indices.SP500EMini,
                'NQ': Futures.Indices.NASDAQ100EMini,
                'RTY': Futures.Indices.Russell2000EMini,
                'CL': Futures.Energies.WTI,
                'GC': Futures.Metals.Gold
            }}

            if symbol in qc_symbol_map:
                future = self.AddFuture(qc_symbol_map[symbol])
                self.contracts[symbol] = future

        # Set up indicators and model parameters
        self.model_params = {model_params}

        # Risk management
        self.max_position_size = 0.1  # 10% of portfolio
        self.stop_loss_pct = 0.01     # 1% stop loss
        self.daily_loss_limit = 0.05  # 5% daily loss limit

        # Track daily P&L
        self.daily_start_value = self.Portfolio.TotalPortfolioValue
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(16, 0), self.ResetDailyPnL)

    def ResetDailyPnL(self):
        """Reset daily P&L tracking"""
        self.daily_start_value = self.Portfolio.TotalPortfolioValue

    def OnData(self, data):
        """Main algorithm logic"""
        for symbol, future in self.contracts.items():
            if not data.ContainsKey(future.Symbol):
                continue

            # Get current price
            current_price = data[future.Symbol].Close

            # Simple ML-based signal generation
            # (In production, load your trained model here)
            signal = self.GenerateSignal(symbol, current_price)

            # Execute trades based on signal
            self.ExecuteTrade(symbol, future, signal, current_price)

    def GenerateSignal(self, symbol: str, price: float) -> int:
        """
        Generate trading signal using simplified logic
        In production, replace with your trained ML model
        """
        # Placeholder logic - replace with actual model predictions
        if price > self.SMA(symbol, 20) and self.RSI(symbol, 14) < 70:
            return 1  # Buy
        elif price < self.SMA(symbol, 20) and self.RSI(symbol, 14) > 30:
            return -1  # Sell
        else:
            return 0  # Hold

    def ExecuteTrade(self, symbol: str, future, signal: int, price: float):
        """Execute trades with risk management"""
        # Check daily loss limit
        daily_pnl = (self.Portfolio.TotalPortfolioValue - self.daily_start_value) / self.daily_start_value
        if daily_pnl <= -self.daily_loss_limit:
            self.Log(f"Daily loss limit reached: {{daily_pnl:.2%}}")
            return

        # Calculate position size
        portfolio_value = self.Portfolio.TotalPortfolioValue
        max_position_value = portfolio_value * self.max_position_size

        # For futures, position size is in contracts
        # Simplified calculation - adjust based on your risk management
        contract_multiplier = 50 if 'ES' in symbol or 'RTY' in symbol else 20 if 'NQ' in symbol else 100
        contracts = int(max_position_value / (price * contract_multiplier))

        if contracts == 0:
            return

        # Execute signal
        if signal == 1:  # Buy
            if not self.Portfolio[future.Symbol].IsLong:
                self.SetHoldings(future.Symbol, self.max_position_size)
                self.Log(f"BUY {{contracts}} contracts of {{symbol}} @ ${{price:.2f}}")

        elif signal == -1:  # Sell
            if not self.Portfolio[future.Symbol].IsShort:
                self.SetHoldings(future.Symbol, -self.max_position_size)
                self.Log(f"SELL {{contracts}} contracts of {{symbol}} @ ${{price:.2f}}")

    def OnOrderEvent(self, orderEvent):
        """Handle order events"""
        self.Log(f"Order Event: {{orderEvent}}")

    def OnEndOfAlgorithm(self):
        """Algorithm completion"""
        self.Log(f"Algorithm completed. Final portfolio value: ${{self.Portfolio.TotalPortfolioValue:,.2f}}")
'''

    return algorithm_template

def setup_quantconnect_connection():
    """Interactive setup for QuantConnect connection"""
    print("🔗 QuantConnect Account Setup")
    print("=" * 40)

    # Check for existing credentials
    user_id = os.getenv('QC_USER_ID')
    api_token = os.getenv('QC_API_TOKEN')

    if user_id and api_token:
        print("✅ QuantConnect credentials found in environment")
        qc = QuantConnectIntegration(user_id, api_token)
        if qc.authenticate():
            print("✅ QuantConnect authentication successful!")
            return qc
        else:
            print("❌ QuantConnect authentication failed")

    print("\n📝 Please provide your QuantConnect credentials:")
    print("1. Go to https://www.quantconnect.com/account")
    print("2. Copy your User ID and API Token")

    user_id = input("Enter your QuantConnect User ID: ").strip()
    api_token = input("Enter your QuantConnect API Token: ").strip()

    if not user_id or not api_token:
        print("❌ Both User ID and API Token are required")
        return None

    # Test connection
    try:
        qc = QuantConnectIntegration(user_id, api_token)
        if qc.authenticate():
            print("✅ QuantConnect authentication successful!")

            # Save to environment (optional)
            save_creds = input("Save credentials to .env file? (y/n): ").lower().strip()
            if save_creds == 'y':
                with open('.env', 'a') as f:
                    f.write(f"\nQC_USER_ID={user_id}\n")
                    f.write(f"QC_API_TOKEN={api_token}\n")
                print("✅ Credentials saved to .env file")

            return qc
        else:
            print("❌ QuantConnect authentication failed")
            return None

    except Exception as e:
        print(f"❌ Error connecting to QuantConnect: {e}")
        return None

if __name__ == "__main__":
    # Demo usage
    qc = setup_quantconnect_connection()
    if qc:
        print("\n📊 QuantConnect Projects:")
        projects = qc.get_projects()
        for project in projects[:5]:  # Show first 5
            print(f"  - {project['name']} (ID: {project['projectId']})")

        print("\n✅ QuantConnect integration ready!")
    else:
        print("❌ Failed to connect to QuantConnect")
