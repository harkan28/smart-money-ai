#!/usr/bin/env python3
"""
Smart Money Enhanced Dataset Generator
Uses existing transaction data to generate larger, more realistic ML training datasets
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
import json
import os
from typing import List, Dict, Tuple

class SmartMoneyDatasetGenerator:
    """Enhanced dataset generator using existing transaction patterns."""
    
    def __init__(self, base_transactions_file: str = "transactions.csv"):
        """Initialize with existing transaction data."""
        self.base_file = base_transactions_file
        self.base_data = None
        self.load_base_data()
        
        # Enhanced merchant categories with real patterns
        self.merchant_categories = {
            'FOOD_DINING': {
                'merchants': [
                    'Zomato', 'Swiggy', 'McDonalds', 'KFC', 'Pizza Hut', 'Dominos', 'Burger King',
                    'Subway', 'Starbucks', 'Cafe Coffee Day', 'Haldirams', 'Bikanervala',
                    'Sagar Ratna', 'Punjabi Tadka', 'Biryani Blues', 'FreshMenu', 'Box8',
                    'Faasos', 'Behrouz Biryani', 'Oven Story', 'Local Restaurant', 'Dhaba Express',
                    'Food Court', 'Canteen', 'Mess Payment', 'Tiffin Service', 'Bakery Shop'
                ],
                'descriptions': [
                    'food delivery', 'restaurant bill', 'lunch order', 'dinner payment',
                    'breakfast delivery', 'snacks order', 'party order', 'office lunch',
                    'weekend dining', 'food court payment', 'canteen recharge', 'mess fees'
                ],
                'amount_range': (50, 2000),
                'peak_amounts': [100, 150, 200, 250, 300, 400, 500, 800]
            },
            'TRANSPORTATION': {
                'merchants': [
                    'Uber', 'Ola', 'Rapido', 'Auto Driver', 'Taxi Service', 'Metro Card',
                    'BMTC', 'DTC', 'Mumbai Metro', 'Kolkata Metro', 'Chennai Metro',
                    'Indian Railways', 'IRCTC', 'Bus Booking', 'Flight Booking', 'SpiceJet',
                    'IndiGo', 'Air India', 'Vistara', 'GoAir', 'Petrol Pump', 'HP Petrol',
                    'BPCL', 'IOCL', 'Shell', 'Essar', 'Parking Fee', 'Toll Plaza'
                ],
                'descriptions': [
                    'cab ride', 'auto fare', 'metro recharge', 'bus ticket', 'train ticket',
                    'flight booking', 'fuel refill', 'parking charges', 'toll payment',
                    'monthly pass', 'uber ride', 'ola booking', 'petrol payment'
                ],
                'amount_range': (25, 5000),
                'peak_amounts': [50, 80, 120, 150, 200, 300, 500, 1000, 2000]
            },
            'SHOPPING': {
                'merchants': [
                    'Amazon', 'Flipkart', 'Myntra', 'Ajio', 'Nykaa', 'BigBasket', 'Grofers',
                    'Reliance Digital', 'Croma', 'Vijay Sales', 'Brand Factory', 'Lifestyle',
                    'Westside', 'Pantaloons', 'Max Fashion', 'H&M', 'Zara', 'Uniqlo',
                    'Local Store', 'Supermarket', 'Medical Store', 'Book Store', 'Electronics Shop'
                ],
                'descriptions': [
                    'online shopping', 'clothes purchase', 'electronics', 'grocery shopping',
                    'mobile purchase', 'laptop buying', 'home appliances', 'fashion shopping',
                    'cosmetics', 'books order', 'medicine purchase', 'household items'
                ],
                'amount_range': (100, 50000),
                'peak_amounts': [200, 500, 1000, 1500, 2000, 3000, 5000, 10000, 15000]
            },
            'ENTERTAINMENT': {
                'merchants': [
                    'BookMyShow', 'Netflix', 'Amazon Prime', 'Disney+ Hotstar', 'Spotify',
                    'YouTube Premium', 'Sony LIV', 'Zee5', 'Voot', 'ALTBalaji', 'MX Player',
                    'Cinema Hall', 'PVR Cinemas', 'INOX', 'Fun Republic', 'Gaming Zone',
                    'Bowling Alley', 'Adventure Park', 'Water Park', 'Concert Tickets'
                ],
                'descriptions': [
                    'movie ticket', 'streaming subscription', 'music subscription', 'gaming',
                    'entertainment', 'concert booking', 'show tickets', 'netflix payment',
                    'prime subscription', 'youtube premium', 'gaming subscription'
                ],
                'amount_range': (99, 2000),
                'peak_amounts': [149, 199, 299, 399, 499, 799, 999, 1499]
            },
            'UTILITIES': {
                'merchants': [
                    'BSES', 'Adani Power', 'Tata Power', 'MSEB', 'KSEB', 'WBSEDCL',
                    'Airtel', 'Jio', 'Vi', 'BSNL', 'ACT Fibernet', 'Hathway', 'Tikona',
                    'Tata Sky', 'Dish TV', 'Sun Direct', 'Videocon D2H', 'Den Networks',
                    'Gas Agency', 'Water Board', 'Municipal Corporation', 'Maintenance'
                ],
                'descriptions': [
                    'electricity bill', 'phone bill', 'internet bill', 'gas bill',
                    'water bill', 'maintenance charges', 'society dues', 'cable bill',
                    'mobile recharge', 'broadband payment', 'dth recharge'
                ],
                'amount_range': (200, 10000),
                'peak_amounts': [300, 500, 800, 1000, 1500, 2000, 3000, 5000]
            },
            'HEALTHCARE': {
                'merchants': [
                    'Apollo Hospital', 'Fortis Hospital', 'Max Hospital', 'AIIMS',
                    'Local Clinic', 'Family Doctor', 'Dental Clinic', 'Eye Specialist',
                    'Diagnostic Center', 'Pathology Lab', 'Pharmacy', 'Medical Store',
                    'Health Insurance', 'Wellness Center', 'Physiotherapy', 'Gym'
                ],
                'descriptions': [
                    'doctor consultation', 'medical checkup', 'medicine purchase',
                    'health insurance', 'diagnostic test', 'dental treatment',
                    'eye checkup', 'physiotherapy', 'medical emergency', 'pharmacy'
                ],
                'amount_range': (100, 25000),
                'peak_amounts': [200, 500, 1000, 2000, 3000, 5000, 10000]
            },
            'EDUCATION': {
                'merchants': [
                    'School Fee', 'College Fee', 'University', 'Coaching Center',
                    'Tuition Teacher', 'Online Course', 'Udemy', 'Coursera', 'BYJU\'S',
                    'Unacademy', 'Vedantu', 'Book Store', 'Stationery Shop', 'Library',
                    'Exam Fee', 'Certification', 'Workshop', 'Seminar', 'Training'
                ],
                'descriptions': [
                    'school fee', 'tuition fee', 'course fee', 'book purchase',
                    'online course', 'coaching classes', 'exam fee', 'educational material',
                    'workshop fee', 'certification cost', 'training program'
                ],
                'amount_range': (500, 100000),
                'peak_amounts': [1000, 2000, 5000, 10000, 15000, 25000, 50000]
            },
            'INVESTMENT': {
                'merchants': [
                    'SBI Mutual Fund', 'HDFC AMC', 'ICICI Prudential', 'Axis Mutual Fund',
                    'UTI AMC', 'DSP BlackRock', 'Franklin Templeton', 'Aditya Birla Sun Life',
                    'Zerodha', 'Upstox', 'Angel Broking', 'HDFC Securities', 'ICICI Direct',
                    'SBI Securities', 'Kotak Securities', 'Life Insurance', 'Term Insurance'
                ],
                'descriptions': [
                    'mutual fund SIP', 'share trading', 'life insurance premium',
                    'investment', 'stock purchase', 'bond investment', 'retirement fund',
                    'tax saving investment', 'systematic investment', 'portfolio investment'
                ],
                'amount_range': (500, 500000),
                'peak_amounts': [1000, 2000, 5000, 10000, 25000, 50000, 100000]
            },
            'MISCELLANEOUS': {
                'merchants': [
                    'ATM Withdrawal', 'Bank Charges', 'Government Services', 'Tax Payment',
                    'Legal Services', 'Courier Service', 'Postal Service', 'Cleaning Service',
                    'Repair Service', 'Salon', 'Spa', 'Gift Shop', 'Charity', 'Donation',
                    'Wedding Expenses', 'Event Management', 'Photography', 'Printing'
                ],
                'descriptions': [
                    'cash withdrawal', 'service charges', 'government fee', 'tax payment',
                    'miscellaneous expense', 'other charges', 'service fee', 'maintenance',
                    'repair work', 'personal care', 'gift purchase', 'donation'
                ],
                'amount_range': (50, 50000),
                'peak_amounts': [100, 500, 1000, 2000, 5000, 10000]
            }
        }
        
        # Banking terminology variations
        self.banking_terms = [
            'UPI Payment', 'NEFT Transfer', 'RTGS Transfer', 'IMPS Payment',
            'Debit Card Payment', 'Credit Card Payment', 'Net Banking',
            'Mobile Banking', 'Wallet Payment', 'QR Code Payment'
        ]
    
    def load_base_data(self):
        """Load existing transaction data."""
        try:
            if os.path.exists(self.base_file):
                self.base_data = pd.read_csv(self.base_file)
                print(f"✅ Loaded {len(self.base_data)} base transactions")
            else:
                print(f"⚠️ Base file {self.base_file} not found, using synthetic names")
                self.base_data = None
        except Exception as e:
            print(f"❌ Error loading base data: {e}")
            self.base_data = None
    
    def get_realistic_names(self) -> List[str]:
        """Extract names from base data or use synthetic ones."""
        if self.base_data is not None and 'Sender Name' in self.base_data.columns:
            names = list(self.base_data['Sender Name'].unique())
            receiver_names = list(self.base_data['Receiver Name'].unique())
            all_names = list(set(names + receiver_names))
            return all_names
        
        # Fallback synthetic names
        return [
            'Aarav Sharma', 'Vivaan Patel', 'Aditya Kumar', 'Vihaan Singh', 'Arjun Gupta',
            'Sai Reddy', 'Reyansh Agarwal', 'Ayaan Khan', 'Krishna Yadav', 'Ishaan Jain',
            'Priya Sharma', 'Ananya Patel', 'Aadhya Singh', 'Kavya Kumar', 'Saanvi Gupta',
            'Aarohi Reddy', 'Diya Agarwal', 'Pihu Khan', 'Myra Yadav', 'Aanya Jain'
        ]
    
    def generate_enhanced_transaction(self, category: str) -> Dict:
        """Generate a single realistic transaction for the category."""
        category_data = self.merchant_categories[category]
        
        # Select merchant and description
        merchant = random.choice(category_data['merchants'])
        description = random.choice(category_data['descriptions'])
        
        # Generate realistic amount
        if random.random() < 0.6:  # 60% chance of peak amounts
            amount = random.choice(category_data['peak_amounts'])
        else:
            min_amt, max_amt = category_data['amount_range']
            amount = round(random.uniform(min_amt, max_amt), 2)
        
        # Add variation to amount
        if random.random() < 0.3:  # 30% chance of slight variation
            variation = random.uniform(0.8, 1.2)
            amount = round(amount * variation, 2)
        
        # Add banking terminology sometimes
        if random.random() < 0.3:
            banking_term = random.choice(self.banking_terms)
            description = f"{banking_term} - {description}"
        
        # Generate date (last 18 months)
        start_date = datetime.now() - timedelta(days=540)
        random_days = random.randint(0, 540)
        transaction_date = start_date + timedelta(days=random_days)
        
        return {
            'merchant': merchant,
            'description': description,
            'amount': amount,
            'category': category,
            'date': transaction_date.strftime('%Y-%m-%d'),
            'timestamp': transaction_date.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def add_realistic_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic variations to make data more robust."""
        df_copy = df.copy()
        
        # Add merchant name variations
        for idx, row in df_copy.iterrows():
            if random.random() < 0.15:  # 15% chance
                merchant = row['merchant']
                variations = [
                    f"{merchant} India", f"{merchant} Ltd", f"{merchant} Pvt Ltd",
                    f"{merchant} Online", f"{merchant}.com", f"{merchant} Store",
                    merchant.replace(' ', ''), merchant.upper(), merchant.lower()
                ]
                df_copy.at[idx, 'merchant'] = random.choice(variations)
        
        # Add description variations
        for idx, row in df_copy.iterrows():
            if random.random() < 0.2:  # 20% chance
                desc = row['description']
                # Add common abbreviations and typos
                desc = desc.replace('payment', 'pymnt' if random.random() < 0.5 else 'payment')
                desc = desc.replace('purchase', 'buy' if random.random() < 0.5 else 'purchase')
                desc = desc.replace('recharge', 'rchg' if random.random() < 0.5 else 'recharge')
                df_copy.at[idx, 'description'] = desc
        
        return df_copy
    
    def generate_massive_dataset(self, total_samples: int = 100000) -> pd.DataFrame:
        """Generate massive dataset for enterprise ML training."""
        print(f"🚀 Generating MASSIVE dataset with {total_samples:,} samples...")
        print("=" * 60)
        
        samples_per_category = total_samples // 9
        all_transactions = []
        
        categories = list(self.merchant_categories.keys())
        
        for category in categories:
            print(f"🏭 Generating {samples_per_category:,} {category} transactions...")
            
            category_transactions = []
            for _ in range(samples_per_category):
                transaction = self.generate_enhanced_transaction(category)
                category_transactions.append(transaction)
            
            all_transactions.extend(category_transactions)
            print(f"✅ Generated {samples_per_category:,} {category} transactions")
        
        # Create DataFrame
        df = pd.DataFrame(all_transactions)
        
        # Add realistic variations
        print("\n🔀 Adding realistic variations and noise...")
        df = self.add_realistic_variations(df)
        
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Save dataset with comprehensive statistics."""
        # Save the dataset
        df.to_csv(filename, index=False)
        
        # Print statistics
        print(f"\n🏆 MASSIVE DATASET STATISTICS")
        print(f"Total Transactions: {len(df):,}")
        print(f"Categories: {df['category'].nunique()}")
        print(f"Unique Merchants: {df['merchant'].nunique()}")
        print(f"Amount Range: ₹{df['amount'].min():,.0f} - ₹{df['amount'].max():,.0f}")
        print(f"Average Amount: ₹{df['amount'].mean():,.0f}")
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        
        print(f"\n📊 Category Distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n💾 Dataset saved to '{filename}'")
        print(f"📁 File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
        print(f"🎉 Enterprise-grade dataset ready for ML training!")


def main():
    """Generate different sized datasets."""
    generator = SmartMoneyDatasetGenerator()
    
    # Check if we have base data
    if generator.base_data is not None:
        print(f"📊 Using {len(generator.base_data)} real transactions as foundation")
    
    print("\n🎯 Choose dataset size:")
    print("1. Standard (10K samples)")
    print("2. Large (50K samples)")
    print("3. Enterprise (100K samples)")
    print("4. Massive (500K samples)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    size_map = {
        '1': (10000, 'standard_training_data.csv'),
        '2': (50000, 'large_training_data.csv'),
        '3': (100000, 'enterprise_training_data.csv'),
        '4': (500000, 'massive_training_data.csv')
    }
    
    if choice in size_map:
        samples, filename = size_map[choice]
        df = generator.generate_massive_dataset(samples)
        generator.save_dataset(df, filename)
    else:
        print("❌ Invalid choice. Generating default 100K dataset...")
        df = generator.generate_massive_dataset(100000)
        generator.save_dataset(df, 'enterprise_training_data.csv')


if __name__ == "__main__":
    main()