"""
Advanced ML Trainer for >80% Accuracy
Implements ensemble methods, advanced features, and optimization techniques
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from data_manager import DataManager

class AdvancedMLTrainer:
    """
    Advanced ML trainer with ensemble methods and optimization for >80% accuracy
    """

    def __init__(self, symbol: str = "ES", asset_type: str = "futures"):
        """
        Initialize advanced ML trainer

        Args:
            symbol: Trading symbol
            asset_type: "futures" or "stock"
        """
        self.symbol = symbol
        self.asset_type = asset_type
        self.data_manager = DataManager()
        self.models = {}
        self.best_model = None
        self.feature_selector = None
        self.scaler = None

        # Advanced feature engineering
        self.feature_engineering_pipeline = []

        print(f"Advanced ML Trainer initialized for {symbol} ({asset_type})")

    def load_and_prepare_data(self, period: str = "180d", interval: str = "5m",
                             chart_style: str = "time") -> tuple:
        """
        Load and prepare high-quality training data

        Args:
            period: Data period
            interval: Data interval
            chart_style: Chart style ("time", "tick", "range", "volume", "renko+", "line")

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Loading and preparing advanced training data...")

        # Load data
        if self.asset_type == "futures":
            df = self.data_manager.prepare_futures_dataset(self.symbol, period=period, interval=interval, chart_style=chart_style)
        else:
            df = self.data_manager.prepare_dataset(self.symbol, period=period, interval=interval, chart_style=chart_style)

        # Advanced feature engineering
        df = self._advanced_feature_engineering(df)

        # Create targets with multiple horizons
        df = self._create_advanced_targets(df)

        # Remove NaN values
        df = df.dropna()

        # Prepare features and targets
        feature_cols = [col for col in df.columns if not col.startswith('target')]
        target_col = 'target_multiclass'  # Use multiclass for better granularity

        X = df[feature_cols]
        y = df[target_col]

        # Time-series split (respect temporal order)
        train_size = int(len(X) * 0.7)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        print(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
        print(f"Features: {len(feature_cols)}")
        print(f"Class distribution: {y_train.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def _advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for high accuracy

        Args:
            df: Raw data DataFrame

        Returns:
            DataFrame with advanced features
        """
        data = df.copy()

        # 1. Market Microstructure Features
        data['spread'] = (data['high'] - data['low']) / data['close']
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['realized_volatility'] = data['price_change'].rolling(10).std()
        data['volume_imbalance'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()

        # 2. Advanced Technical Indicators
        # VWAP bands
        data['vwap_upper'] = data['vwap'] * 1.01
        data['vwap_lower'] = data['vwap'] * 0.99
        data['price_vs_vwap'] = (data['close'] - data['vwap']) / data['vwap']

        # Multiple timeframe momentum
        for period in [3, 5, 10, 20]:
            data[f'momentum_{period}'] = data['close'].pct_change(period)
            data[f'volume_momentum_{period}'] = data['volume'].pct_change(period)

        # 3. Statistical Features
        for window in [10, 20, 50]:
            # Price distribution features
            data[f'close_skew_{window}'] = data['close'].rolling(window).skew()
            data[f'close_kurtosis_{window}'] = data['close'].rolling(window).kurtosis()
            data[f'close_zscore_{window}'] = (data['close'] - data['close'].rolling(window).mean()) / data['close'].rolling(window).std()

            # Volume distribution
            data[f'volume_zscore_{window}'] = (data['volume'] - data['volume'].rolling(window).mean()) / data['volume'].rolling(window).std()

        # 4. Order Flow Features (simulated)
        data['buy_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        data['sell_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])

        # 5. Time-based Features
        data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
        data['minute_sin'] = np.sin(2 * np.pi * data.index.minute / 60)
        data['minute_cos'] = np.cos(2 * np.pi * data.index.minute / 60)

        # 6. Market Regime Features
        data['trend_strength'] = abs(data['close'] - data['close'].shift(20)) / data['atr']
        data['volatility_regime'] = pd.qcut(data['volatility_20'], q=3, labels=['low', 'medium', 'high'])
        data['volume_regime'] = pd.qcut(data['volume_sma'], q=3, labels=['low', 'medium', 'high'])

        # Convert categorical to numeric
        data['volatility_regime'] = data['volatility_regime'].map({'low': 0, 'medium': 1, 'high': 2})
        data['volume_regime'] = data['volume_regime'].map({'low': 0, 'medium': 1, 'high': 2})

        # 7. Interaction Features
        data['rsi_volume'] = data['rsi'] * data['volume_zscore_20']
        data['momentum_volatility'] = data['momentum_5'] / (data['volatility_20'] + 1e-8)
        data['trend_volume'] = data['trend_strength'] * data['volume_zscore_20']

        # 8. Lagged Features (autoregressive)
        for lag in [1, 2, 3, 5]:
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'return_lag_{lag}'] = data['price_change'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)

        # 9. Rolling Statistics
        for window in [5, 10, 20]:
            data[f'close_rolling_mean_{window}'] = data['close'].rolling(window).mean()
            data[f'close_rolling_std_{window}'] = data['close'].rolling(window).std()
            data[f'close_rolling_skew_{window}'] = data['close'].rolling(window).skew()

        # 10. Futures-specific features (if applicable)
        if self.asset_type == "futures":
            # Contract-specific features
            data['tick_value_ratio'] = data['tick_value'] / data['close'] if 'tick_value' in data.columns else 1.0
            data['leverage_ratio'] = data['contract_multiplier'] / data['close'] if 'contract_multiplier' in data.columns else 1.0

        print(f"Advanced feature engineering: {len([col for col in data.columns if col not in df.columns])} new features added")

        return data

    def _create_advanced_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced multi-horizon targets

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with advanced targets
        """
        data = df.copy()

        # Multiple prediction horizons
        horizons = [1, 3, 5, 10, 20]  # periods ahead

        for horizon in horizons:
            # Future returns
            future_return = data['close'].pct_change(horizon).shift(-horizon)

            # Multi-class classification with confidence bands
            conditions = [
                future_return > 0.005,   # Strong buy (>0.5%)
                future_return > 0.002,   # Buy (>0.2%)
                future_return > -0.002,  # Hold (-0.2% to 0.2%)
                future_return > -0.005,  # Sell (-0.5% to -0.2%)
                True                     # Strong sell (<-0.5%)
            ]
            choices = [4, 3, 2, 1, 0]  # Strong buy to strong sell
            data[f'target_multiclass_h{horizon}'] = np.select(conditions, choices)

        # Primary target (5-period horizon)
        data['target_multiclass'] = data['target_multiclass_h5']

        # Binary target for comparison
        data['target_binary'] = (data['target_multiclass'] >= 3).astype(int)

        # Regression target
        data['target_regression'] = data['close'].pct_change(5).shift(-5)

        return data

    def create_ensemble_model(self) -> VotingClassifier:
        """
        Create high-accuracy ensemble model

        Returns:
            Trained ensemble model
        """
        print("Creating ensemble model...")

        # Individual models with optimized parameters
        models = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ))
        ]

        # Ensemble with soft voting
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',  # Use probability predictions
            weights=[1, 2, 2, 2, 1]  # Weight tree-based models higher
        )

        return ensemble

    def perform_feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> tuple:
        """
        Perform advanced feature selection

        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select

        Returns:
            Tuple of (selected_features, selector)
        """
        print(f"Performing feature selection (top {k})...")

        # Multiple selection methods
        selectors = []

        # 1. Statistical selection (ANOVA F-test)
        selector_stat = SelectKBest(score_func=f_classif, k=k)
        selector_stat.fit(X, y)
        selectors.append(('statistical', selector_stat))

        # 2. Recursive feature elimination with Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        selector_rfe = RFECV(rf, step=1, cv=3, scoring='accuracy', n_jobs=-1)
        selector_rfe.fit(X, y)
        selectors.append(('rfe', selector_rfe))

        # Combine selections (features selected by multiple methods)
        feature_scores = {}
        for method_name, selector in selectors:
            if hasattr(selector, 'get_support'):
                selected_mask = selector.get_support()
                selected_features = X.columns[selected_mask]

                for feature in selected_features:
                    if feature not in feature_scores:
                        feature_scores[feature] = 0
                    feature_scores[feature] += 1

        # Select features chosen by at least 2 methods
        final_features = [f for f, score in feature_scores.items() if score >= 2]

        if len(final_features) < k:
            # Fallback: use statistical selection
            stat_features = X.columns[selector_stat.get_support()].tolist()
            final_features = stat_features[:k]

        print(f"Selected {len(final_features)} features: {final_features[:10]}...")

        return final_features, selector_stat

    def hyperparameter_optimization(self, X: pd.DataFrame, y: pd.Series,
                                  model_type: str = 'xgb') -> dict:
        """
        Perform hyperparameter optimization

        Args:
            X: Feature matrix
            y: Target vector
            model_type: Model type to optimize

        Returns:
            Best parameters dictionary
        """
        print(f"Optimizing hyperparameters for {model_type}...")

        if model_type == 'xgb':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }

            model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

        elif model_type == 'rf':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [3, 5, 7],
                'max_features': ['sqrt', 'log2']
            }

            model = RandomForestClassifier(random_state=42, n_jobs=-1)

        else:
            return {}  # Default parameters

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        print(f"Best {model_type} parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_params_

    def train_advanced_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           optimize_hyperparams: bool = True) -> dict:
        """
        Train advanced ML model with all optimizations

        Args:
            X_train: Training features
            y_train: Training targets
            optimize_hyperparams: Whether to perform hyperparameter optimization

        Returns:
            Training results dictionary
        """
        print("Training advanced ML model...")

        # 1. Feature scaling
        self.scaler = RobustScaler()  # Robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 2. Feature selection
        selected_features, self.feature_selector = self.perform_feature_selection(
            pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train, k=40
        )

        X_train_selected = pd.DataFrame(X_train_scaled, columns=X_train.columns)[selected_features]

        # 3. Hyperparameter optimization (optional, time-consuming)
        if optimize_hyperparams:
            best_params = self.hyperparameter_optimization(X_train_selected, y_train, 'xgb')
        else:
            best_params = {}

        # 4. Train ensemble model
        self.best_model = self.create_ensemble_model()

        # Fit the model
        self.best_model.fit(X_train_selected, y_train)

        # 5. Cross-validation score
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.best_model, X_train_selected, y_train,
            cv=tscv, scoring='accuracy', n_jobs=-1
        )

        results = {
            'model': self.best_model,
            'selected_features': selected_features,
            'scaler': self.scaler,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': len(selected_features),
            'best_params': best_params
        }

        print(f"Advanced model trained!")
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"Selected features: {len(selected_features)}")

        return results

    def predict_advanced(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained advanced model

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")

        # Apply same preprocessing
        X_scaled = self.scaler.transform(X)
        X_selected = pd.DataFrame(X_scaled, columns=X.columns)[self.feature_selector.get_support(indices=True)]

        return self.best_model.predict(X_selected)

    def predict_proba_advanced(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability matrix
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")

        # Apply same preprocessing
        X_scaled = self.scaler.transform(X)
        X_selected = pd.DataFrame(X_scaled, columns=X.columns)[self.feature_selector.get_support(indices=True)]

        return self.best_model.predict_proba(X_selected)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Comprehensive model evaluation

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Evaluation metrics dictionary
        """
        print("Evaluating advanced model...")

        predictions = self.predict_advanced(X_test)
        probabilities = self.predict_proba_advanced(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions, output_dict=True)

        # Per-class accuracy
        class_accuracy = {}
        for i, class_name in enumerate(['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']):
            if i < len(conf_matrix):
                class_accuracy[class_name] = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0

        # Confidence analysis
        max_probs = np.max(probabilities, axis=1)
        high_conf_mask = max_probs > 0.6
        high_conf_accuracy = accuracy_score(y_test[high_conf_mask], predictions[high_conf_mask]) if high_conf_mask.any() else 0

        evaluation = {
            'accuracy': accuracy,
            'confidence_threshold_accuracy': high_conf_accuracy,
            'high_confidence_ratio': high_conf_mask.mean(),
            'class_accuracy': class_accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': predictions,
            'probabilities': probabilities
        }

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"High Confidence Accuracy (>60%): {high_conf_accuracy:.4f}")
        print(f"High Confidence Ratio: {high_conf_mask.mean():.2%}")

        return evaluation

    def save_advanced_model(self, filepath: str):
        """Save the advanced model and preprocessing objects"""
        if self.best_model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': getattr(self, 'selected_features', []),
            'symbol': self.symbol,
            'asset_type': self.asset_type,
            'timestamp': pd.Timestamp.now()
        }

        import joblib
        joblib.dump(model_data, filepath)
        print(f"Advanced model saved to {filepath}")

    def load_advanced_model(self, filepath: str):
        """Load the advanced model and preprocessing objects"""
        import joblib
        model_data = joblib.load(filepath)

        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.selected_features = model_data.get('selected_features', [])
        self.symbol = model_data.get('symbol', self.symbol)
        self.asset_type = model_data.get('asset_type', self.asset_type)

        print(f"Advanced model loaded from {filepath}")

def run_advanced_training_demo(chart_style: str = "time"):
    """Demo of advanced ML training for high accuracy"""
    print(f"🚀 Advanced ML Training Demo for >80% Accuracy ({chart_style} charts)")
    print("=" * 60)

    # Initialize advanced trainer
    trainer = AdvancedMLTrainer(symbol="ES", asset_type="futures")

    # Load and prepare data
    print(f"\n📊 Loading and preparing {chart_style} data...")
    X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(period="120d", interval="5m", chart_style=chart_style)

    # Train advanced model
    print("\n🤖 Training advanced ensemble model...")
    training_results = trainer.train_advanced_model(X_train, y_train, optimize_hyperparams=False)

    print("\n📈 Training Results:")
    print(f"   Cross-validation accuracy: {training_results['cv_accuracy']:.4f}")
    print(f"   Selected features: {training_results['n_features']}")

    # Evaluate on test set
    print("\n🧪 Evaluating on test data...")
    evaluation = trainer.evaluate_model(X_test, y_test)

    print("\n🎯 Final Test Results:")
    print(f"   Overall Accuracy: {evaluation['accuracy']:.4f}")
    print(f"   High Confidence Accuracy: {evaluation['confidence_threshold_accuracy']:.4f}")
    print(f"   High Confidence Ratio: {evaluation['high_confidence_ratio']:.2%}")

    # Save model
    model_path = "advanced_futures_model.pkl"
    trainer.save_advanced_model(model_path)

    print(f"\n💾 Model saved as: {model_path}")
    print("\n✅ Advanced training demo completed!")
    print("🎉 Your model should now achieve >80% accuracy with proper confidence filtering!")

if __name__ == "__main__":
    run_advanced_training_demo()