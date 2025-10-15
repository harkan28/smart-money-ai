#!/usr/bin/env python3
"""
Configuration Manager for Smart Money AI
========================================

Handles all configuration settings and environment variables
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration management for Smart Money AI"""
    
    DEFAULT_CONFIG = {
        "ml_model_path": "models/expense_category_model.joblib",
        "feature_extractor_path": "models/feature_extractor.joblib",
        "data_storage_path": "data/",
        "log_level": "INFO",
        "sms_parsing_enabled": True,
        "ml_categorization_enabled": True,
        "investment_engine_enabled": True,
        "behavioral_analysis_enabled": True,
        "predictive_analytics_enabled": True,
        "api_rate_limit": 1000,
        "max_transactions_per_user": 10000,
        "backup_enabled": True,
        "backup_frequency_hours": 24,
        "encryption_enabled": False,
        "demo_mode": True
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_path = config_path or "config/settings.json"
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load configuration from file
        self.load_config()
        
        # Override with environment variables
        self.load_env_variables()
        
        logger.info("Configuration manager initialized")
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info("No config file found, using defaults")
                # Create default config file
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def load_env_variables(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "SMART_MONEY_LOG_LEVEL": "log_level",
            "SMART_MONEY_DEMO_MODE": "demo_mode",
            "SMART_MONEY_MODEL_PATH": "ml_model_path",
            "SMART_MONEY_DATA_PATH": "data_storage_path",
            "SMART_MONEY_ENCRYPTION": "encryption_enabled"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_key in ["demo_mode", "encryption_enabled", "backup_enabled"]:
                    self.config[config_key] = env_value.lower() in ('true', '1', 'yes')
                elif config_key in ["api_rate_limit", "max_transactions_per_user", "backup_frequency_hours"]:
                    self.config[config_key] = int(env_value)
                else:
                    self.config[config_key] = env_value
                
                logger.info(f"Config override from env: {config_key} = {self.config[config_key]}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
        logger.info(f"Config updated: {key} = {value}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(self.config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration settings"""
        return self.config.copy()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        logger.info("Configuration reset to defaults")
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check required paths exist or can be created
            required_paths = ["data_storage_path"]
            for path_key in required_paths:
                path = self.config.get(path_key)
                if path:
                    os.makedirs(path, exist_ok=True)
            
            # Validate numeric ranges
            if self.config.get("api_rate_limit", 0) <= 0:
                logger.warning("Invalid api_rate_limit, using default")
                self.config["api_rate_limit"] = self.DEFAULT_CONFIG["api_rate_limit"]
            
            if self.config.get("max_transactions_per_user", 0) <= 0:
                logger.warning("Invalid max_transactions_per_user, using default")
                self.config["max_transactions_per_user"] = self.DEFAULT_CONFIG["max_transactions_per_user"]
            
            logger.info("Configuration validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode"""
        return self.config.get("demo_mode", True)
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get ML model file paths"""
        return {
            "model_path": self.config.get("ml_model_path"),
            "feature_extractor_path": self.config.get("feature_extractor_path")
        }
    
    def get_storage_path(self) -> str:
        """Get data storage path"""
        return self.config.get("data_storage_path", "data/")


def main():
    """Demo function"""
    print("⚙️  Configuration Manager Demo")
    print("=" * 40)
    
    config = ConfigManager()
    
    print(f"Demo Mode: {config.is_demo_mode()}")
    print(f"ML Model Path: {config.get('ml_model_path')}")
    print(f"Data Storage: {config.get_storage_path()}")
    print(f"Log Level: {config.get('log_level')}")
    
    # Test configuration update
    config.set("test_setting", "test_value")
    print(f"Test Setting: {config.get('test_setting')}")
    
    # Validate configuration
    is_valid = config.validate_config()
    print(f"Configuration Valid: {is_valid}")


if __name__ == "__main__":
    main()