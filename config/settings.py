"""
Smart Money AI Configuration Management
=====================================

Centralized configuration system for all API keys, settings, and parameters.
Supports both environment variables and direct configuration.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class APIConfig:
    """API configuration container"""
    finnhub_api_key: Optional[str] = None
    breeze_app_key: Optional[str] = None
    breeze_secret_key: Optional[str] = None
    breeze_session_token: Optional[str] = None

@dataclass
class MLConfig:
    """Machine Learning model configuration"""
    model_save_path: str = "smart_money_ai/data/models"
    expense_categories: list = None
    confidence_threshold: float = 0.6
    training_data_size: int = 1000

    def __post_init__(self):
        if self.expense_categories is None:
            self.expense_categories = [
                'food_dining', 'transportation', 'entertainment', 
                'utilities', 'shopping', 'healthcare', 'education', 
                'investment', 'others'
            ]

@dataclass
class TradingConfig:
    """Trading automation configuration"""
    max_position_size: float = 0.05  # 5% of portfolio
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.06  # 6%
    max_daily_trades: int = 50
    max_daily_loss: float = 0.05  # 5%
    risk_tolerance: str = "moderate"

@dataclass
class SystemConfig:
    """System-wide configuration"""
    debug_mode: bool = False
    log_level: str = "INFO"
    cache_enabled: bool = True
    auto_save_models: bool = True
    backup_frequency: str = "daily"

class ConfigManager:
    """Centralized configuration manager for Smart Money AI"""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration manager"""
        self.config_file = config_file or os.path.join(os.path.dirname(__file__), "config.json")
        self.api_config = APIConfig()
        self.ml_config = MLConfig()
        self.trading_config = TradingConfig()
        self.system_config = SystemConfig()
        
        # Load configuration
        self._load_from_env()
        self._load_from_file()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API Keys
        self.api_config.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.api_config.breeze_app_key = os.getenv('BREEZE_APP_KEY')
        self.api_config.breeze_secret_key = os.getenv('BREEZE_SECRET_KEY')
        self.api_config.breeze_session_token = os.getenv('BREEZE_SESSION_TOKEN')
        
        # System settings
        self.system_config.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.system_config.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Trading settings
        max_pos = os.getenv('MAX_POSITION_SIZE')
        if max_pos:
            self.trading_config.max_position_size = float(max_pos)
        
        stop_loss = os.getenv('STOP_LOSS_PERCENTAGE')
        if stop_loss:
            self.trading_config.stop_loss_percentage = float(stop_loss)
    
    def _load_from_file(self):
        """Load configuration from JSON file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update API config
                if 'api' in config_data:
                    api_data = config_data['api']
                    for key, value in api_data.items():
                        if hasattr(self.api_config, key) and value:
                            setattr(self.api_config, key, value)
                
                # Update ML config  
                if 'ml' in config_data:
                    ml_data = config_data['ml']
                    for key, value in ml_data.items():
                        if hasattr(self.ml_config, key) and value is not None:
                            setattr(self.ml_config, key, value)
                
                # Update trading config
                if 'trading' in config_data:
                    trading_data = config_data['trading']
                    for key, value in trading_data.items():
                        if hasattr(self.trading_config, key) and value is not None:
                            setattr(self.trading_config, key, value)
                
                # Update system config
                if 'system' in config_data:
                    system_data = config_data['system']
                    for key, value in system_data.items():
                        if hasattr(self.system_config, key) and value is not None:
                            setattr(self.system_config, key, value)
                            
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            'api': {
                'finnhub_api_key': self.api_config.finnhub_api_key,
                'breeze_app_key': self.api_config.breeze_app_key,
                'breeze_secret_key': self.api_config.breeze_secret_key,
                'breeze_session_token': self.api_config.breeze_session_token
            },
            'ml': {
                'model_save_path': self.ml_config.model_save_path,
                'expense_categories': self.ml_config.expense_categories,
                'confidence_threshold': self.ml_config.confidence_threshold,
                'training_data_size': self.ml_config.training_data_size
            },
            'trading': {
                'max_position_size': self.trading_config.max_position_size,
                'stop_loss_percentage': self.trading_config.stop_loss_percentage,
                'take_profit_percentage': self.trading_config.take_profit_percentage,
                'max_daily_trades': self.trading_config.max_daily_trades,
                'max_daily_loss': self.trading_config.max_daily_loss,
                'risk_tolerance': self.trading_config.risk_tolerance
            },
            'system': {
                'debug_mode': self.system_config.debug_mode,
                'log_level': self.system_config.log_level,
                'cache_enabled': self.system_config.cache_enabled,
                'auto_save_models': self.system_config.auto_save_models,
                'backup_frequency': self.system_config.backup_frequency
            }
        }
        
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
    
    def update_api_keys(self, **kwargs):
        """Update API keys"""
        for key, value in kwargs.items():
            if hasattr(self.api_config, key) and value:
                setattr(self.api_config, key, value)
        
        # Save updated config
        self.save_config()
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'api': self.api_config,
            'ml': self.ml_config,
            'trading': self.trading_config,
            'system': self.system_config
        }
    
    def is_real_time_enabled(self) -> bool:
        """Check if real-time market data is enabled"""
        return self.api_config.finnhub_api_key is not None
    
    def is_trading_enabled(self) -> bool:
        """Check if automated trading is enabled"""
        return all([
            self.api_config.breeze_app_key,
            self.api_config.breeze_secret_key
        ])
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get system capabilities based on configuration"""
        return {
            'ml_models': True,  # Always available
            'real_time_data': self.is_real_time_enabled(),
            'automated_trading': self.is_trading_enabled(),
            'portfolio_monitoring': self.is_real_time_enabled(),
            'risk_management': self.is_trading_enabled()
        }

# Global configuration instance
config = ConfigManager()

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    return config

def setup_config(
    finnhub_api_key: str = None,
    breeze_app_key: str = None,
    breeze_secret_key: str = None,
    breeze_session_token: str = None,
    **kwargs
) -> ConfigManager:
    """Setup configuration with API keys"""
    global config
    
    # Update API keys
    config.update_api_keys(
        finnhub_api_key=finnhub_api_key,
        breeze_app_key=breeze_app_key,
        breeze_secret_key=breeze_secret_key,
        breeze_session_token=breeze_session_token
    )
    
    # Update other settings
    for key, value in kwargs.items():
        if hasattr(config.trading_config, key):
            setattr(config.trading_config, key, value)
        elif hasattr(config.ml_config, key):
            setattr(config.ml_config, key, value)
        elif hasattr(config.system_config, key):
            setattr(config.system_config, key, value)
    
    return config

# Example usage and setup functions
def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "api": {
            "finnhub_api_key": "your_finnhub_api_key_here",
            "breeze_app_key": "your_breeze_app_key_here", 
            "breeze_secret_key": "your_breeze_secret_key_here",
            "breeze_session_token": "your_session_token_here"
        },
        "ml": {
            "model_save_path": "smart_money_ai/data/models",
            "confidence_threshold": 0.6,
            "training_data_size": 1000
        },
        "trading": {
            "max_position_size": 0.05,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.06,
            "max_daily_trades": 50,
            "risk_tolerance": "moderate"
        },
        "system": {
            "debug_mode": False,
            "log_level": "INFO", 
            "cache_enabled": True,
            "auto_save_models": True
        }
    }
    
    config_file = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"✅ Sample configuration created at {config_file}")
    print("📝 Please update with your actual API keys")

if __name__ == "__main__":
    # Create sample config for first-time setup
    create_sample_config()
    
    # Test configuration
    config = get_config()
    print("🔧 Configuration Status:")
    print(f"Real-time data: {'✅' if config.is_real_time_enabled() else '❌'}")
    print(f"Automated trading: {'✅' if config.is_trading_enabled() else '❌'}")
    print(f"Capabilities: {config.get_capabilities()}")