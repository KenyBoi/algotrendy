"""
Simple test script to verify XGBoost installation and basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import xgboost as xgb
        print("✅ xgboost imported successfully")
        
        import yfinance as yf
        print("✅ yfinance imported successfully")
        
        import sklearn
        print("✅ scikit-learn imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
        
        import ta
        print("✅ ta (technical analysis) imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_fetch():
    """Test basic data fetching"""
    print("\nTesting data fetch...")
    
    try:
        import yfinance as yf
        
        # Fetch a small amount of data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d", interval="1d")
        
        if not data.empty:
            print(f"✅ Successfully fetched {len(data)} days of AAPL data")
            print(f"   Latest close price: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Data fetch error: {e}")
        return False

def test_xgboost():
    """Test XGBoost basic functionality"""
    print("\nTesting XGBoost...")
    
    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        print(f"✅ XGBoost model trained successfully")
        print(f"   Test accuracy: {accuracy:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ XGBoost test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AlgoTrendy System Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test data fetching
    if test_data_fetch():
        tests_passed += 1
    
    # Test XGBoost
    if test_xgboost():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Run: python example_usage.py")
        print("2. Or try: python main.py train --symbol AAPL")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)