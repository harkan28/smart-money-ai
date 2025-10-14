#!/usr/bin/env python3
"""
Smart Money ML Setup Script
Quick setup and demo for the expense categorization system
"""

import os
import sys
import subprocess
import time

def print_header():
    print("🎯 Smart Money ML - Quick Setup")
    print("=" * 40)
    print()

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True) 
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False
    
    return True

def train_model():
    """Train the ML model"""
    print("\n🤖 Training ML model...")
    print("   This will generate sample data and train the model")
    print("   Expected time: 2-3 minutes")
    
    try:
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Model trained successfully")
            return True
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training is taking longer than expected...")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def run_demo():
    """Run the demonstration"""
    print("\n🎮 Running demo...")
    
    try:
        subprocess.run([sys.executable, "demo.py"], check=True)
        print("✅ Demo completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False
    except FileNotFoundError:
        print("⚠️  Demo file not found, skipping...")
        return True
    
    return True

def main():
    """Main setup function"""
    print_header()
    
    # Check if we're in the right directory
    if not os.path.exists('requirements.txt'):
        print("❌ Please run this script from the budgeting_ml_model directory")
        sys.exit(1)
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Downloading NLTK data", download_nltk_data),
        ("Training model", train_model),
        ("Running demo", run_demo)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            print("   Please check the error messages above")
            sys.exit(1)
        time.sleep(0.5)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Check the models/ directory for trained models")
    print("   2. Run 'python demo.py' to see predictions")
    print("   3. Use the inference API in your applications")
    print("\n🚀 Happy expense categorizing!")

if __name__ == "__main__":
    main()