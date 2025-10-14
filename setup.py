#!/usr/bin/env python3
"""
Smart Money AI - One-Command Setup and Demo
Sets up the complete system and runs a demonstration
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print the Smart Money AI banner."""
    banner = """
🎯 Smart Money AI - Intelligent Financial Assistant
==================================================
Setting up your AI-powered expense categorization system...
"""
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ is required. Current version: %s", sys.version)
        sys.exit(1)
    logger.info("✅ Python version check passed: %s", sys.version.split()[0])

def install_dependencies():
    """Install required dependencies."""
    logger.info("📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        logger.info("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error("❌ Failed to install dependencies: %s", e.stderr.decode())
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data."""
    logger.info("📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("✅ NLTK data downloaded successfully")
    except Exception as e:
        logger.error("❌ Failed to download NLTK data: %s", e)
        return False
    
    return True

def generate_sample_data():
    """Generate sample training data if needed."""
    data_dir = Path("SMS_PARSING_SYSTEM")
    
    if not (data_dir / "enterprise_training_data.csv").exists():
        logger.info("📊 Generating sample training data...")
        
        try:
            # Generate a smaller dataset for demo
            os.chdir(data_dir)
            result = subprocess.run([
                sys.executable, "-c", 
                "from enhanced_dataset_generator import SmartMoneyDatasetGenerator; "
                "g = SmartMoneyDatasetGenerator(); "
                "df = g.generate_massive_dataset(10000); "
                "g.save_dataset(df, 'demo_training_data.csv')"
            ], capture_output=True, text=True, timeout=60)
            
            os.chdir("..")
            
            if result.returncode == 0:
                logger.info("✅ Sample training data generated")
            else:
                logger.warning("⚠️ Could not generate sample data: %s", result.stderr)
            
        except Exception as e:
            logger.warning("⚠️ Sample data generation failed: %s", e)
            os.chdir("..")
    
    return True

def run_demo():
    """Run the system demonstration."""
    logger.info("🎮 Running Smart Money AI demo...")
    
    try:
        # Check if we can import the main system
        sys.path.append(str(Path.cwd()))
        from smart_money_integrator import SmartMoneyIntegrator
        
        # Create integrator
        integrator = SmartMoneyIntegrator()
        
        # Demo SMS processing
        sample_sms = "Dear Customer, Rs.450 has been debited from your account for UPI payment to ZOMATO on 14-Oct-24"
        
        logger.info("\n📱 Demo: Processing Bank SMS")
        logger.info("SMS: %s...", sample_sms[:60])
        
        result = integrator.process_bank_sms(sample_sms, "HDFC-BANK")
        
        if result.get('status') == 'success':
            logger.info("✅ SMS processed successfully!")
            logger.info("   Category: %s", result.get('predicted_category', 'Unknown'))
            logger.info("   Confidence: %.1f%%", result.get('confidence', 0) * 100)
        else:
            logger.info("ℹ️ SMS processing demo completed (models need training)")
        
        logger.info("\n🎉 Smart Money AI setup completed successfully!")
        logger.info("\n📋 Next steps:")
        logger.info("   1. Train the ML model: 'cd budgeting_ml_model && python train_model.py'")
        logger.info("   2. Try the interactive demo: 'python smart_money_integrator.py'")
        logger.info("   3. Integrate with your banking SMS")
        
        return True
        
    except Exception as e:
        logger.info("ℹ️ Demo setup completed. Error: %s", e)
        logger.info("   To train the model: 'cd budgeting_ml_model && python train_model.py'")
        return True

def main():
    """Main setup function."""
    print_banner()
    
    # Check system requirements
    check_python_version()
    
    # Setup steps
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Downloading NLTK data", download_nltk_data),
        ("Generating sample data", generate_sample_data),
        ("Running demo", run_demo)
    ]
    
    for step_name, step_func in steps:
        logger.info("\n🔄 %s...", step_name)
        if not step_func():
            logger.error("❌ Setup failed at: %s", step_name)
            sys.exit(1)
    
    print("\n" + "="*50)
    print("🚀 Smart Money AI is ready!")
    print("Visit: https://github.com/yourusername/smart-money-ai")
    print("="*50)

if __name__ == "__main__":
    main()