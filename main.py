#!/usr/bin/env python3
"""
Smart Money AI - Main Entry Point
Launch the complete financial intelligence system
"""

from smart_money_ai import SmartMoneyAI
import sys

def main():
    """Main entry point for Smart Money AI"""
    
    print("🎉 Welcome to Smart Money AI!")
    print("World-class financial intelligence system")
    print("=" * 50)
    
    # Initialize the system
    ai = SmartMoneyAI()
    
    # Show system statistics
    stats = ai.get_system_stats()
    print(f"📊 Loaded {stats.get('total_intelligence_profiles', '20,100+')} financial profiles")
    print("✅ System ready for intelligent financial analysis!")
    
    return ai

if __name__ == "__main__":
    main()
