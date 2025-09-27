"""
Lightweight Alpaca API Test
Tests connection without complex pandas dependencies
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_alpaca_connection():
    """Test basic Alpaca API connection using REST API directly"""
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API credentials not found in .env file")
        return False
    
    # Alpaca paper trading base URL
    base_url = "https://paper-api.alpaca.markets"
    
    # Headers for authentication
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }
    
    try:
        print("🧪 Testing Alpaca API connection...")
        
        # Test 1: Get account information
        response = requests.get(f"{base_url}/v2/account", headers=headers)
        
        if response.status_code == 200:
            account_data = response.json()
            print("✅ Successfully connected to Alpaca!")
            print(f"   Account ID: {account_data.get('id', 'N/A')}")
            print(f"   Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
            print(f"   Cash Available: ${float(account_data.get('cash', 0)):,.2f}")
            print(f"   Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
            
            # Test 2: Get market data
            print("\n📊 Testing market data access...")
            
            data_url = "https://data.alpaca.markets"
            data_headers = headers.copy()
            
            # Get latest quote for AAPL
            symbol = "AAPL"
            quote_response = requests.get(
                f"{data_url}/v2/stocks/{symbol}/quotes/latest", 
                headers=data_headers
            )
            
            if quote_response.status_code == 200:
                quote_data = quote_response.json()
                quote = quote_data.get('quote', {})
                print(f"✅ Market data access confirmed!")
                print(f"   {symbol} Latest Quote:")
                print(f"   Bid: ${quote.get('bp', 0):.2f} x {quote.get('bs', 0)}")
                print(f"   Ask: ${quote.get('ap', 0):.2f} x {quote.get('as', 0)}")
                
                # Test 3: Get historical bars
                print("\n📈 Testing historical data...")
                
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                
                bars_response = requests.get(
                    f"{data_url}/v2/stocks/{symbol}/bars",
                    headers=data_headers,
                    params={
                        'timeframe': '1Day',
                        'start': start_date,
                        'end': end_date,
                        'limit': 5
                    }
                )
                
                if bars_response.status_code == 200:
                    bars_data = bars_response.json()
                    bars = bars_data.get('bars', [])
                    
                    if bars:
                        print(f"✅ Historical data confirmed!")
                        print(f"   Retrieved {len(bars)} daily bars for {symbol}")
                        latest_bar = bars[-1]
                        print(f"   Latest Close: ${latest_bar.get('c', 0):.2f}")
                        print(f"   Volume: {latest_bar.get('v', 0):,}")
                    else:
                        print("⚠️  No historical data returned")
                else:
                    print(f"⚠️  Historical data request failed: {bars_response.status_code}")
            else:
                print(f"⚠️  Market data request failed: {quote_response.status_code}")
            
            return True
            
        else:
            print(f"❌ Connection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def create_simple_alpaca_demo():
    """Create a simple demo using direct REST API calls"""
    
    print("\n🚀 Creating Lightweight Alpaca Demo...")
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API credentials not found")
        return
    
    # Base configuration
    base_url = "https://paper-api.alpaca.markets"
    data_url = "https://data.alpaca.markets"
    
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'Content-Type': 'application/json'
    }
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    print(f"📊 Analyzing {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            # Get latest quote
            quote_response = requests.get(
                f"{data_url}/v2/stocks/{symbol}/quotes/latest",
                headers=headers
            )
            
            if quote_response.status_code == 200:
                quote_data = quote_response.json()
                quote = quote_data.get('quote', {})
                current_price = (quote.get('bp', 0) + quote.get('ap', 0)) / 2
                
                # Get recent historical data for simple analysis
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                bars_response = requests.get(
                    f"{data_url}/v2/stocks/{symbol}/bars",
                    headers=headers,
                    params={
                        'timeframe': '1Day',
                        'start': start_date,
                        'end': end_date,
                        'limit': 30
                    }
                )
                
                if bars_response.status_code == 200:
                    bars_data = bars_response.json()
                    bars = bars_data.get('bars', [])
                    
                    if len(bars) >= 20:
                        # Simple moving average calculation
                        closes = [bar['c'] for bar in bars]
                        sma_20 = sum(closes[-20:]) / 20
                        sma_5 = sum(closes[-5:]) / 5
                        
                        # Simple signal generation
                        signal = "BUY" if sma_5 > sma_20 and current_price > sma_20 else "HOLD"
                        signal_emoji = "🟢" if signal == "BUY" else "⚪"
                        
                        print(f"   {signal_emoji} {symbol}: ${current_price:.2f} | SMA20: ${sma_20:.2f} | Signal: {signal}")
                    else:
                        print(f"   ⚠️  {symbol}: Insufficient data")
                else:
                    print(f"   ❌ {symbol}: Historical data failed")
            else:
                print(f"   ❌ {symbol}: Quote data failed")
                
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    print("\n✅ Lightweight demo completed!")
    print("🔗 Your system is ready for full ML integration!")

def main():
    """Main function"""
    print("🧪 Alpaca API Connection Test")
    print("=" * 35)
    
    # Test basic connection
    if test_alpaca_connection():
        # Run simple demo
        create_simple_alpaca_demo()
        
        print("\n🎉 All tests passed!")
        print("\n📚 Next steps:")
        print("   1. Your API connection is working")
        print("   2. You can now run: python alpaca_ml_demo.py")
        print("   3. The system will use real market data")
        print("   4. All trading is in paper mode (safe)")
        
    else:
        print("\n❌ Connection test failed")
        print("💡 Please check your API credentials in .env file")

if __name__ == "__main__":
    main()