"""
Simple Alpaca Configuration Helper
Creates .env file without complex package dependencies
"""

import os
from pathlib import Path

def get_user_input():
    """Get API credentials from user"""
    print("🔧 Alpaca API Configuration")
    print("=" * 40)
    print()
    print("📋 Please provide your Alpaca API credentials:")
    print("   (Get these from: https://app.alpaca.markets/paper/dashboard/overview)")
    print()
    
    api_key = input("🔑 Enter your Alpaca API Key: ").strip()
    secret_key = input("🔐 Enter your Alpaca Secret Key: ").strip()
    
    return api_key, secret_key

def create_env_file(api_key, secret_key):
    """Create .env file with credentials"""
    env_content = f"""# Alpaca API Configuration
# Generated automatically

ALPACA_API_KEY={api_key}
ALPACA_SECRET_KEY={secret_key}

# Trading Configuration
PAPER_TRADING=true
MAX_POSITIONS=5
POSITION_SIZE_PCT=0.05

# Risk Management
MAX_DAILY_LOSS_PCT=0.02
MAX_POSITION_VALUE=10000

# Data Settings
DEFAULT_LOOKBACK_DAYS=252
UPDATE_INTERVAL_MINUTES=15

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
"""
    
    try:
        env_file = Path(".env")
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"\n✅ Created .env file: {env_file.absolute()}")
        print("🔒 Your API keys are now securely stored locally")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def main():
    """Main configuration function"""
    print("🚀 Simple Alpaca Configuration")
    print("=" * 35)
    
    # Check if .env already exists
    if Path(".env").exists():
        print("\n📁 Found existing .env file")
        choice = input("🔄 Do you want to update it? (y/N): ").strip().lower()
        if choice != 'y':
            print("ℹ️  Keeping existing configuration")
            return
    
    # Get credentials
    api_key, secret_key = get_user_input()
    
    if not api_key or not secret_key:
        print("\n❌ API credentials cannot be empty!")
        print("💡 Please get your keys from: https://alpaca.markets")
        return
    
    # Create .env file
    if create_env_file(api_key, secret_key):
        print("\n🎉 Configuration completed!")
        print("\n📚 Next steps:")
        print("   1. Run the demo: python alpaca_ml_demo.py")
        print("   2. Your system will now use real market data")
        print("   3. All trading is in PAPER MODE (no real money)")
        print("\n⚠️  Remember: This is paper trading - no real money at risk!")
    else:
        print("\n❌ Configuration failed")

if __name__ == "__main__":
    main()