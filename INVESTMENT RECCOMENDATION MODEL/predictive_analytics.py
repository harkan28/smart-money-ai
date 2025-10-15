#!/usr/bin/env python3
"""
Predictive Financial Analytics Module
Advanced forecasting for expenses, savings, and investment planning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Container for prediction results with confidence metrics"""
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float  # 0-1 scale
    prediction_date: datetime
    prediction_horizon: str
    contributing_factors: List[str]
    seasonal_adjustment: float
    trend_component: float

@dataclass
class ForecastSummary:
    """Summary of multiple predictions with insights"""
    daily_predictions: List[PredictionResult]
    weekly_summary: Dict[str, float]
    monthly_summary: Dict[str, float]
    yearly_projection: Dict[str, float]
    savings_forecast: Dict[str, float]
    risk_indicators: List[str]
    optimization_opportunities: List[str]

class AdvancedTimeSeriesAnalyzer:
    """Advanced time series analysis for financial forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.seasonal_patterns = {}
        self.trend_components = {}
    
    def analyze_time_series(self, df: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive time series analysis"""
        try:
            # Prepare time series data
            ts_data = self._prepare_time_series_data(df)
            
            if len(ts_data) < 7:  # Need at least a week of data
                logger.warning("Insufficient data for time series analysis")
                return self._default_time_series_analysis()
            
            analysis = {}
            
            # 1. Trend Analysis
            analysis['trend'] = self._analyze_trend(ts_data)
            
            # 2. Seasonality Detection
            analysis['seasonality'] = self._detect_seasonality(ts_data)
            
            # 3. Volatility Analysis
            analysis['volatility'] = self._analyze_volatility(ts_data)
            
            # 4. Cyclical Patterns
            analysis['cycles'] = self._detect_cycles(ts_data)
            
            # 5. Anomaly Detection
            analysis['anomalies'] = self._detect_anomalies(ts_data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            return self._default_time_series_analysis()
    
    def _prepare_time_series_data(self, df: pd.DataFrame) -> pd.Series:
        """Prepare daily time series data"""
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime('today')
        
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_spending = df.groupby('date')['amount'].sum()
        
        # Create complete date range
        date_range = pd.date_range(start=daily_spending.index.min(), 
                                 end=daily_spending.index.max(), 
                                 freq='D')
        
        # Reindex to include all dates (fill missing with 0)
        daily_spending = daily_spending.reindex(date_range, fill_value=0)
        
        return daily_spending
    
    def _analyze_trend(self, ts_data: pd.Series) -> Dict[str, float]:
        """Analyze trend component of time series"""
        x = np.arange(len(ts_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_data.values)
        
        # Calculate trend strength
        trend_strength = abs(r_value)
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        
        # Weekly and monthly trend
        if len(ts_data) >= 7:
            weekly_trend = (ts_data.rolling(7).mean().iloc[-1] - ts_data.rolling(7).mean().iloc[-7]) if len(ts_data) >= 14 else 0
        else:
            weekly_trend = 0
            
        if len(ts_data) >= 30:
            monthly_trend = (ts_data.rolling(30).mean().iloc[-1] - ts_data.rolling(30).mean().iloc[-30]) if len(ts_data) >= 60 else 0
        else:
            monthly_trend = 0
        
        return {
            'slope': slope,
            'strength': trend_strength,
            'direction': trend_direction,
            'p_value': p_value,
            'weekly_change': weekly_trend,
            'monthly_change': monthly_trend
        }
    
    def _detect_seasonality(self, ts_data: pd.Series) -> Dict[str, any]:
        """Detect seasonal patterns"""
        seasonality = {}
        
        # Weekly seasonality (if enough data)
        if len(ts_data) >= 14:
            weekly_pattern = []
            for day in range(7):
                day_values = [ts_data.iloc[i] for i in range(day, len(ts_data), 7)]
                weekly_pattern.append(np.mean(day_values))
            
            # Normalize weekly pattern
            pattern_mean = np.mean(weekly_pattern)
            if pattern_mean > 0:
                weekly_pattern = [x / pattern_mean for x in weekly_pattern]
            
            seasonality['weekly_pattern'] = weekly_pattern
            seasonality['weekly_strength'] = np.std(weekly_pattern)
        
        # Monthly seasonality (if enough data)
        if len(ts_data) >= 62:  # ~2 months
            ts_df = pd.DataFrame({'value': ts_data.values, 'date': ts_data.index})
            ts_df['day_of_month'] = pd.to_datetime(ts_df['date']).dt.day
            
            monthly_pattern = ts_df.groupby('day_of_month')['value'].mean()
            pattern_mean = monthly_pattern.mean()
            if pattern_mean > 0:
                monthly_pattern = monthly_pattern / pattern_mean
            
            seasonality['monthly_pattern'] = monthly_pattern.to_dict()
            seasonality['monthly_strength'] = monthly_pattern.std()
        
        return seasonality
    
    def _analyze_volatility(self, ts_data: pd.Series) -> Dict[str, float]:
        """Analyze volatility patterns"""
        # Basic volatility metrics
        daily_returns = ts_data.pct_change().dropna()
        volatility = daily_returns.std()
        
        # Rolling volatility
        if len(ts_data) >= 7:
            rolling_vol = ts_data.rolling(7).std()
            vol_trend = rolling_vol.iloc[-1] - rolling_vol.iloc[0] if len(rolling_vol) > 1 else 0
        else:
            vol_trend = 0
        
        # Volatility clustering (GARCH-like effect)
        high_vol_periods = (daily_returns.abs() > daily_returns.abs().quantile(0.8)).sum()
        clustering_score = high_vol_periods / len(daily_returns) if len(daily_returns) > 0 else 0
        
        return {
            'volatility': volatility,
            'vol_trend': vol_trend,
            'clustering_score': clustering_score,
            'coefficient_of_variation': ts_data.std() / ts_data.mean() if ts_data.mean() > 0 else 0
        }
    
    def _detect_cycles(self, ts_data: pd.Series) -> Dict[str, any]:
        """Detect cyclical patterns"""
        cycles = {}
        
        # Simple cycle detection using autocorrelation
        if len(ts_data) >= 30:
            # Calculate autocorrelation for different lags
            max_lag = min(30, len(ts_data) // 2)
            autocorr = [ts_data.autocorr(lag=i) for i in range(1, max_lag)]
            
            # Find peaks in autocorrelation (potential cycle lengths)
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                    peaks.append((i+1, autocorr[i]))  # lag, correlation
            
            cycles['detected_cycles'] = peaks
            cycles['strongest_cycle'] = max(peaks, key=lambda x: x[1]) if peaks else None
        
        return cycles
    
    def _detect_anomalies(self, ts_data: pd.Series) -> List[Dict]:
        """Detect anomalous spending patterns"""
        anomalies = []
        
        if len(ts_data) < 7:
            return anomalies
        
        # Statistical anomaly detection (Z-score method)
        rolling_mean = ts_data.rolling(7, center=True).mean()
        rolling_std = ts_data.rolling(7, center=True).std()
        
        z_scores = np.abs((ts_data - rolling_mean) / rolling_std)
        anomaly_threshold = 2.5  # 2.5 standard deviations
        
        anomaly_indices = z_scores[z_scores > anomaly_threshold].index
        
        for idx in anomaly_indices:
            anomalies.append({
                'date': idx,
                'value': ts_data[idx],
                'z_score': z_scores[idx],
                'type': 'high' if ts_data[idx] > rolling_mean[idx] else 'low'
            })
        
        return anomalies
    
    def _default_time_series_analysis(self) -> Dict[str, any]:
        """Default analysis when insufficient data"""
        return {
            'trend': {'slope': 0, 'strength': 0, 'direction': 'stable'},
            'seasonality': {'weekly_pattern': [1]*7, 'weekly_strength': 0},
            'volatility': {'volatility': 0.3, 'clustering_score': 0},
            'cycles': {'detected_cycles': []},
            'anomalies': []
        }

class ExpensePredictionEngine:
    """Advanced expense prediction using multiple models"""
    
    def __init__(self):
        self.models = {
            'linear_trend': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        
    def train_models(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train prediction models on historical data"""
        try:
            # Prepare features and targets
            X, y = self._prepare_features_targets(df)
            
            if len(X) < 10:  # Need minimum data for training
                logger.warning("Insufficient data for model training")
                return {'error': 'insufficient_data'}
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            model_scores = {}
            for name, model in self.models.items():
                try:
                    model.fit(X_scaled, y)
                    score = model.score(X_scaled, y)
                    model_scores[name] = score
                    logger.info(f"Model {name} trained with R² score: {score:.3f}")
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    model_scores[name] = 0.0
            
            # Store feature importance
            if hasattr(self.models['random_forest'], 'feature_importances_'):
                feature_names = self._get_feature_names()
                self.feature_importance = dict(zip(feature_names, 
                                                 self.models['random_forest'].feature_importances_))
            
            self.is_trained = True
            return model_scores
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {'error': str(e)}
    
    def predict_expenses(self, prediction_days: int = 30, 
                        current_date: datetime = None) -> ForecastSummary:
        """Predict expenses for specified number of days"""
        try:
            if not self.is_trained:
                logger.warning("Models not trained, using default predictions")
                return self._default_forecast(prediction_days)
            
            if current_date is None:
                current_date = datetime.now()
            
            # Generate predictions for each day
            daily_predictions = []
            for i in range(prediction_days):
                pred_date = current_date + timedelta(days=i+1)
                prediction = self._predict_single_day(pred_date)
                daily_predictions.append(prediction)
            
            # Generate summary statistics
            weekly_summary = self._generate_weekly_summary(daily_predictions)
            monthly_summary = self._generate_monthly_summary(daily_predictions)
            yearly_projection = self._generate_yearly_projection(daily_predictions)
            savings_forecast = self._generate_savings_forecast(daily_predictions)
            
            # Risk indicators
            risk_indicators = self._identify_risk_indicators(daily_predictions)
            
            # Optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(daily_predictions)
            
            return ForecastSummary(
                daily_predictions=daily_predictions,
                weekly_summary=weekly_summary,
                monthly_summary=monthly_summary,
                yearly_projection=yearly_projection,
                savings_forecast=savings_forecast,
                risk_indicators=risk_indicators,
                optimization_opportunities=optimization_opportunities
            )
            
        except Exception as e:
            logger.error(f"Error in expense prediction: {e}")
            return self._default_forecast(prediction_days)
    
    def _prepare_features_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for model training"""
        # Create daily time series
        ts_analyzer = AdvancedTimeSeriesAnalyzer()
        ts_data = ts_analyzer._prepare_time_series_data(df)
        
        # Create features for each day
        features = []
        targets = []
        
        lookback_window = 7  # Use past 7 days to predict next day
        
        for i in range(lookback_window, len(ts_data)):
            # Historical spending features
            recent_spending = ts_data.iloc[i-lookback_window:i].values
            
            # Date features
            current_date = ts_data.index[i]
            date_features = self._extract_date_features(current_date)
            
            # Trend features
            trend_features = self._extract_trend_features(ts_data.iloc[i-lookback_window:i])
            
            # Combine all features
            feature_vector = np.concatenate([
                recent_spending,  # Historical spending
                date_features,    # Date-based features
                trend_features    # Trend-based features
            ])
            
            features.append(feature_vector)
            targets.append(ts_data.iloc[i])
        
        return np.array(features), np.array(targets)
    
    def _extract_date_features(self, date: datetime) -> np.ndarray:
        """Extract date-based features"""
        return np.array([
            date.weekday(),  # Day of week (0-6)
            date.day,        # Day of month (1-31)
            date.month,      # Month (1-12)
            1 if date.weekday() >= 5 else 0,  # Is weekend
            1 if date.day >= 25 else 0,       # Is month end
        ])
    
    def _extract_trend_features(self, recent_data: pd.Series) -> np.ndarray:
        """Extract trend-based features"""
        values = recent_data.values
        
        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Trend
        x = np.arange(len(values))
        slope = 0 if len(values) < 2 else np.polyfit(x, values, 1)[0]
        
        # Recent vs historical ratio
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else mean_val
        ratio = recent_avg / mean_val if mean_val > 0 else 1
        
        return np.array([mean_val, std_val, slope, ratio])
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance analysis"""
        names = []
        
        # Historical spending features (7 days)
        names.extend([f'spending_day_{i}' for i in range(7)])
        
        # Date features
        names.extend(['weekday', 'day_of_month', 'month', 'is_weekend', 'is_month_end'])
        
        # Trend features
        names.extend(['mean_spending', 'std_spending', 'trend_slope', 'recent_ratio'])
        
        return names
    
    def _predict_single_day(self, pred_date: datetime) -> PredictionResult:
        """Predict expenses for a single day"""
        try:
            # Create feature vector for prediction date
            # For now, use simplified features (in practice, you'd use recent historical data)
            date_features = self._extract_date_features(pred_date)
            
            # Simplified feature vector (would be more sophisticated with recent data)
            feature_vector = np.concatenate([
                np.zeros(7),      # Historical spending (would use actual recent data)
                date_features,    # Date features
                np.zeros(4)       # Trend features (would calculate from recent data)
            ]).reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get predictions from different models
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(feature_vector_scaled)[0]
                    predictions[name] = max(0, pred)  # Ensure non-negative
                except Exception as e:
                    logger.error(f"Error in {name} prediction: {e}")
                    predictions[name] = 1000  # Default value
            
            # Ensemble prediction (weighted average)
            weights = {'linear_trend': 0.3, 'random_forest': 0.7}
            predicted_value = sum(predictions[name] * weights.get(name, 1) 
                                for name in predictions) / sum(weights.values())
            
            # Calculate confidence interval (simplified)
            std_error = np.std(list(predictions.values())) if len(predictions) > 1 else predicted_value * 0.2
            confidence_interval = (
                max(0, predicted_value - 1.96 * std_error),
                predicted_value + 1.96 * std_error
            )
            
            # Confidence score based on model agreement
            prediction_variance = np.var(list(predictions.values()))
            confidence_score = max(0, 1 - (prediction_variance / (predicted_value ** 2))) if predicted_value > 0 else 0.5
            
            # Contributing factors
            contributing_factors = self._identify_contributing_factors(pred_date, feature_vector)
            
            # Seasonal adjustment
            seasonal_adjustment = self._apply_seasonal_adjustment(pred_date, predicted_value)
            
            return PredictionResult(
                predicted_value=predicted_value,
                confidence_interval=confidence_interval,
                confidence_score=confidence_score,
                prediction_date=pred_date,
                prediction_horizon="daily",
                contributing_factors=contributing_factors,
                seasonal_adjustment=seasonal_adjustment,
                trend_component=predicted_value * 0.1  # Simplified trend component
            )
            
        except Exception as e:
            logger.error(f"Error predicting single day: {e}")
            return self._default_prediction(pred_date)
    
    def _identify_contributing_factors(self, pred_date: datetime, features: np.ndarray) -> List[str]:
        """Identify factors contributing to prediction"""
        factors = []
        
        # Date-based factors
        if pred_date.weekday() >= 5:
            factors.append("Weekend spending pattern")
        if pred_date.day >= 25:
            factors.append("Month-end spending increase")
        if pred_date.month in [11, 12]:
            factors.append("Holiday season effect")
        
        # Feature importance based factors
        if hasattr(self, 'feature_importance'):
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            for feature, importance in top_features:
                if importance > 0.1:  # Significant importance
                    factors.append(f"High influence: {feature}")
        
        return factors[:3]  # Return top 3 factors
    
    def _apply_seasonal_adjustment(self, pred_date: datetime, base_prediction: float) -> float:
        """Apply seasonal adjustments to prediction"""
        adjustment = 0
        
        # Weekend adjustment
        if pred_date.weekday() >= 5:
            adjustment += base_prediction * 0.2  # 20% increase on weekends
        
        # Month-end adjustment
        if pred_date.day >= 25:
            adjustment += base_prediction * 0.15  # 15% increase at month-end
        
        # Holiday season adjustment
        if pred_date.month in [11, 12]:
            adjustment += base_prediction * 0.3  # 30% increase during holidays
        
        return adjustment
    
    def _generate_weekly_summary(self, daily_predictions: List[PredictionResult]) -> Dict[str, float]:
        """Generate weekly spending summary"""
        if not daily_predictions:
            return {}
        
        total_predicted = sum(p.predicted_value for p in daily_predictions[:7])
        avg_confidence = np.mean([p.confidence_score for p in daily_predictions[:7]])
        
        return {
            'total_weekly_spending': total_predicted,
            'average_daily_spending': total_predicted / 7,
            'confidence_score': avg_confidence,
            'weekend_vs_weekday_ratio': self._calculate_weekend_ratio(daily_predictions[:7])
        }
    
    def _generate_monthly_summary(self, daily_predictions: List[PredictionResult]) -> Dict[str, float]:
        """Generate monthly spending summary"""
        if not daily_predictions:
            return {}
        
        days_in_month = min(30, len(daily_predictions))
        monthly_predictions = daily_predictions[:days_in_month]
        
        total_predicted = sum(p.predicted_value for p in monthly_predictions)
        avg_confidence = np.mean([p.confidence_score for p in monthly_predictions])
        
        return {
            'total_monthly_spending': total_predicted,
            'average_daily_spending': total_predicted / days_in_month,
            'confidence_score': avg_confidence,
            'spending_variance': np.var([p.predicted_value for p in monthly_predictions])
        }
    
    def _generate_yearly_projection(self, daily_predictions: List[PredictionResult]) -> Dict[str, float]:
        """Generate yearly spending projection"""
        if not daily_predictions:
            return {}
        
        # Use first 30 days to project yearly spending
        sample_days = min(30, len(daily_predictions))
        sample_total = sum(p.predicted_value for p in daily_predictions[:sample_days])
        
        yearly_projection = (sample_total / sample_days) * 365
        
        return {
            'projected_yearly_spending': yearly_projection,
            'monthly_average': yearly_projection / 12,
            'confidence_score': np.mean([p.confidence_score for p in daily_predictions[:sample_days]])
        }
    
    def _generate_savings_forecast(self, daily_predictions: List[PredictionResult]) -> Dict[str, float]:
        """Generate savings forecast based on spending predictions"""
        if not daily_predictions:
            return {}
        
        # Assume average income (this would come from user profile in practice)
        assumed_monthly_income = 50000  # ₹50k per month
        
        monthly_spending = sum(p.predicted_value for p in daily_predictions[:30])
        predicted_savings = assumed_monthly_income - monthly_spending
        savings_rate = predicted_savings / assumed_monthly_income if assumed_monthly_income > 0 else 0
        
        return {
            'predicted_monthly_savings': predicted_savings,
            'savings_rate': savings_rate,
            'yearly_savings_projection': predicted_savings * 12,
            'emergency_fund_months': predicted_savings / monthly_spending if monthly_spending > 0 else 0
        }
    
    def _identify_risk_indicators(self, daily_predictions: List[PredictionResult]) -> List[str]:
        """Identify financial risk indicators from predictions"""
        risks = []
        
        # High spending variability
        spending_values = [p.predicted_value for p in daily_predictions[:30]]
        cv = np.std(spending_values) / np.mean(spending_values) if np.mean(spending_values) > 0 else 0
        if cv > 1:
            risks.append("High spending volatility detected")
        
        # Consistently low confidence
        avg_confidence = np.mean([p.confidence_score for p in daily_predictions[:30]])
        if avg_confidence < 0.6:
            risks.append("Low prediction confidence - irregular spending patterns")
        
        # Upward spending trend
        recent_spending = np.mean([p.predicted_value for p in daily_predictions[:7]])
        later_spending = np.mean([p.predicted_value for p in daily_predictions[7:14]])
        if later_spending > recent_spending * 1.2:
            risks.append("Increasing spending trend detected")
        
        return risks
    
    def _identify_optimization_opportunities(self, daily_predictions: List[PredictionResult]) -> List[str]:
        """Identify spending optimization opportunities"""
        opportunities = []
        
        # Weekend spending optimization
        weekend_ratio = self._calculate_weekend_ratio(daily_predictions[:30])
        if weekend_ratio > 1.5:
            opportunities.append("Reduce weekend discretionary spending")
        
        # High variance days
        spending_values = [p.predicted_value for p in daily_predictions[:30]]
        high_spending_days = [i for i, v in enumerate(spending_values) if v > np.mean(spending_values) * 1.5]
        if len(high_spending_days) > 5:
            opportunities.append("Plan for high-spending days to avoid budget overruns")
        
        # Consistent small optimizations
        daily_avg = np.mean(spending_values)
        if daily_avg > 1000:
            potential_savings = daily_avg * 0.1  # 10% reduction
            opportunities.append(f"Daily spending reduction of ₹{potential_savings:.0f} could save ₹{potential_savings*30:.0f}/month")
        
        return opportunities
    
    def _calculate_weekend_ratio(self, predictions: List[PredictionResult]) -> float:
        """Calculate weekend vs weekday spending ratio"""
        weekend_spending = []
        weekday_spending = []
        
        for pred in predictions:
            if pred.prediction_date.weekday() >= 5:  # Weekend
                weekend_spending.append(pred.predicted_value)
            else:  # Weekday
                weekday_spending.append(pred.predicted_value)
        
        if not weekend_spending or not weekday_spending:
            return 1.0
        
        return np.mean(weekend_spending) / np.mean(weekday_spending)
    
    def _default_prediction(self, pred_date: datetime) -> PredictionResult:
        """Default prediction when models fail"""
        base_amount = 1000  # Default daily spending
        
        # Simple adjustments
        if pred_date.weekday() >= 5:  # Weekend
            base_amount *= 1.3
        if pred_date.day >= 25:  # Month end
            base_amount *= 1.2
        
        return PredictionResult(
            predicted_value=base_amount,
            confidence_interval=(base_amount * 0.7, base_amount * 1.3),
            confidence_score=0.5,
            prediction_date=pred_date,
            prediction_horizon="daily",
            contributing_factors=["Default prediction - limited historical data"],
            seasonal_adjustment=0,
            trend_component=0
        )
    
    def _default_forecast(self, prediction_days: int) -> ForecastSummary:
        """Default forecast when models not available"""
        daily_predictions = []
        for i in range(prediction_days):
            pred_date = datetime.now() + timedelta(days=i+1)
            daily_predictions.append(self._default_prediction(pred_date))
        
        return ForecastSummary(
            daily_predictions=daily_predictions,
            weekly_summary={'total_weekly_spending': 7000},
            monthly_summary={'total_monthly_spending': 30000},
            yearly_projection={'projected_yearly_spending': 365000},
            savings_forecast={'predicted_monthly_savings': 20000},
            risk_indicators=["Limited historical data for predictions"],
            optimization_opportunities=["Collect more data for better insights"]
        )

def main():
    """Demo the predictive analytics engine"""
    print("📈 Predictive Financial Analytics Demo")
    print("=" * 50)
    
    # Sample transaction data
    sample_transactions = []
    base_date = datetime.now() - timedelta(days=60)
    
    # Generate 60 days of sample data with patterns
    for i in range(60):
        current_date = base_date + timedelta(days=i)
        
        # Base spending with weekly and monthly patterns
        base_amount = 800
        
        # Weekend effect
        if current_date.weekday() >= 5:
            base_amount *= 1.4
        
        # Month-end effect
        if current_date.day >= 25:
            base_amount *= 1.3
        
        # Add some randomness
        amount = base_amount * (0.7 + 0.6 * np.random.random())
        
        sample_transactions.append({
            'amount': amount,
            'category': 'MISCELLANEOUS',
            'timestamp': current_date.isoformat()
        })
    
    # Create and train prediction engine
    engine = ExpensePredictionEngine()
    
    print("\n🏋️ Training Prediction Models...")
    df = pd.DataFrame(sample_transactions)
    training_scores = engine.train_models(df)
    
    print("Model Training Results:")
    for model, score in training_scores.items():
        if isinstance(score, float):
            print(f"  {model}: R² = {score:.3f}")
        else:
            print(f"  {model}: {score}")
    
    # Generate predictions
    print("\n🔮 Generating 30-Day Forecast...")
    forecast = engine.predict_expenses(prediction_days=30)
    
    # Display results
    print(f"\n📊 Weekly Summary:")
    for key, value in forecast.weekly_summary.items():
        if isinstance(value, float):
            print(f"  {key}: ₹{value:,.0f}" if 'spending' in key else f"  {key}: {value:.3f}")
    
    print(f"\n📈 Monthly Summary:")
    for key, value in forecast.monthly_summary.items():
        if isinstance(value, float):
            print(f"  {key}: ₹{value:,.0f}" if 'spending' in key else f"  {key}: {value:.3f}")
    
    print(f"\n💰 Savings Forecast:")
    for key, value in forecast.savings_forecast.items():
        if isinstance(value, float):
            if 'rate' in key:
                print(f"  {key}: {value:.1%}")
            else:
                print(f"  {key}: ₹{value:,.0f}")
    
    print(f"\n⚠️  Risk Indicators:")
    for risk in forecast.risk_indicators:
        print(f"  • {risk}")
    
    print(f"\n💡 Optimization Opportunities:")
    for opportunity in forecast.optimization_opportunities:
        print(f"  • {opportunity}")
    
    print(f"\n✅ Predictive analytics demo completed!")

if __name__ == "__main__":
    main()