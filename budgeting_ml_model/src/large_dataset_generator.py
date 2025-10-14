"""
Enhanced Large Dataset Generator for Expense Categorization

This module generates a comprehensive, realistic dataset with thousands of transactions
across all categories to improve model performance and robustness.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

class LargeDatasetGenerator:
    """
    Generate large, diverse training datasets for expense categorization.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the dataset generator."""
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Comprehensive merchant and transaction patterns
        self.transaction_patterns = {
            'FOOD_DINING': {
                'merchants': [
                    'Zomato', 'Swiggy', 'Uber Eats', 'Foodpanda', 'McDonald\'s', 'KFC', 'Dominos', 
                    'Pizza Hut', 'Subway', 'Starbucks', 'Cafe Coffee Day', 'Barista', 'Costa Coffee',
                    'Burger King', 'Taco Bell', 'Dunkin Donuts', 'Baskin Robbins', 'Haldiram',
                    'Bikanervala', 'Sagar Ratna', 'Karim\'s', 'Punjabi Dhaba', 'South Indian Corner',
                    'Chinese Express', 'Thai Pavilion', 'Mainland China', 'Barbeque Nation',
                    'Buffet Palace', 'Fine Dine Restaurant', 'Local Restaurant', 'Street Food',
                    'Food Court', 'Hotel Restaurant', 'Club Restaurant', 'Rooftop Cafe'
                ],
                'descriptions': [
                    'food delivery', 'dinner order', 'lunch delivery', 'breakfast order', 'snacks delivery',
                    'pizza order', 'burger meal', 'biryani delivery', 'chinese food', 'south indian meal',
                    'north indian food', 'fast food', 'restaurant bill', 'dining out', 'takeaway order',
                    'coffee and snacks', 'evening tea', 'party order', 'family dinner', 'office lunch',
                    'weekend meal', 'celebration dinner', 'quick bite', 'late night food', 'healthy meal'
                ],
                'amount_range': (50, 2000),
                'amount_distribution': 'lognormal'
            },
            
            'TRANSPORTATION': {
                'merchants': [
                    'Uber', 'Ola', 'Meru Cabs', 'Quick Ride', 'Rapido', 'Auto Rickshaw',
                    'Indian Oil', 'Bharat Petroleum', 'Hindustan Petroleum', 'Shell', 'Reliance Petrol',
                    'Metro Card', 'BMTC', 'DTC', 'BEST', 'IRCTC', 'RedBus', 'MakeMyTrip Bus',
                    'Parking Meter', 'Mall Parking', 'Airport Parking', 'Toll Plaza', 'FASTag',
                    'Car Service', 'Bike Service', 'Tire Shop', 'Car Wash', 'Vehicle Insurance'
                ],
                'descriptions': [
                    'cab ride', 'taxi booking', 'auto fare', 'bike ride', 'airport transfer',
                    'petrol refill', 'diesel fuel', 'CNG refill', 'fuel expenses', 'car fuel',
                    'metro ticket', 'bus fare', 'train ticket', 'flight booking', 'travel expenses',
                    'parking fee', 'toll charges', 'vehicle maintenance', 'car service', 'bike repair',
                    'insurance premium', 'vehicle registration', 'driving license', 'traffic fine'
                ],
                'amount_range': (20, 5000),
                'amount_distribution': 'uniform'
            },
            
            'SHOPPING': {
                'merchants': [
                    'Amazon', 'Flipkart', 'Myntra', 'Ajio', 'Nykaa', 'Lenskart', 'FirstCry',
                    'BigBasket', 'Grofers', 'JioMart', 'Amazon Fresh', 'Spencer\'s', 'More',
                    'Reliance Digital', 'Croma', 'Vijay Sales', 'Chroma', 'Brand Factory',
                    'Westside', 'Lifestyle', 'Shoppers Stop', 'Central Mall', 'Phoenix Mall',
                    'Local Store', 'Supermarket', 'Departmental Store', 'Electronics Shop',
                    'Clothing Store', 'Shoe Store', 'Book Store', 'Gift Shop', 'Jewellery Store'
                ],
                'descriptions': [
                    'online shopping', 'mobile purchase', 'laptop buying', 'clothing shopping', 'grocery shopping',
                    'electronics purchase', 'home appliances', 'kitchen items', 'furniture buying', 'books purchase',
                    'gift shopping', 'cosmetics purchase', 'shoes buying', 'accessories shopping', 'toys purchase',
                    'sports equipment', 'health products', 'personal care', 'household items', 'stationery purchase',
                    'festival shopping', 'birthday gift', 'wedding shopping', 'baby products', 'pet supplies'
                ],
                'amount_range': (100, 50000),
                'amount_distribution': 'exponential'
            },
            
            'ENTERTAINMENT': {
                'merchants': [
                    'Netflix', 'Amazon Prime', 'Disney+ Hotstar', 'Zee5', 'Sony Liv', 'Voot',
                    'Spotify', 'YouTube Music', 'Gaana', 'JioSaavn', 'Apple Music',
                    'BookMyShow', 'Paytm Movies', 'PVR Cinemas', 'INOX', 'Cinepolis',
                    'Gaming Platform', 'Steam', 'PlayStation Store', 'Xbox Live', 'Google Play Games',
                    'Amusement Park', 'Water Park', 'Adventure Sports', 'Concert Tickets', 'Event Booking'
                ],
                'descriptions': [
                    'streaming subscription', 'movie subscription', 'music subscription', 'video streaming',
                    'movie tickets', 'cinema booking', 'concert tickets', 'event passes', 'show tickets',
                    'gaming subscription', 'game purchase', 'in-app purchase', 'entertainment package',
                    'amusement park', 'adventure activity', 'sports event', 'cultural event', 'comedy show',
                    'live performance', 'music festival', 'theater tickets', 'exhibition entry'
                ],
                'amount_range': (99, 3000),
                'amount_distribution': 'normal'
            },
            
            'UTILITIES': {
                'merchants': [
                    'Airtel', 'Jio', 'Vi', 'BSNL', 'Tata Sky', 'Dish TV', 'Videocon D2H',
                    'BSES', 'TATA Power', 'Adani Electricity', 'MSEB', 'KSEB', 'TNEB',
                    'Indane Gas', 'HP Gas', 'Bharatgas', 'Piped Gas', 'GAIL',
                    'Water Board', 'Municipal Corporation', 'Society Maintenance', 'Internet Provider',
                    'Broadband Bill', 'WiFi Charges', 'Landline Bill', 'Cable TV'
                ],
                'descriptions': [
                    'mobile recharge', 'phone bill', 'internet bill', 'broadband payment', 'wifi charges',
                    'electricity bill', 'power bill', 'current bill', 'energy charges', 'meter reading',
                    'gas cylinder', 'lpg refill', 'cooking gas', 'pipeline gas', 'gas connection',
                    'water bill', 'municipal tax', 'society charges', 'maintenance fee', 'utility payments',
                    'dth recharge', 'cable bill', 'satellite tv', 'landline charges'
                ],
                'amount_range': (100, 3000),
                'amount_distribution': 'normal'
            },
            
            'HEALTHCARE': {
                'merchants': [
                    'Apollo Hospital', 'Fortis Hospital', 'Max Healthcare', 'Manipal Hospital',
                    'Narayana Health', 'AIIMS', 'Government Hospital', 'Nursing Home',
                    'MedPlus', 'Apollo Pharmacy', '1mg', 'Netmeds', 'PharmEasy', 'Local Pharmacy',
                    'Practo', 'Lybrate', 'DocsApp', 'Portea Medical', 'HealthifyMe',
                    'Dental Clinic', 'Eye Clinic', 'Skin Clinic', 'Physiotherapy', 'Diagnostic Center'
                ],
                'descriptions': [
                    'doctor consultation', 'medical checkup', 'health screening', 'specialist visit',
                    'medicine purchase', 'pharmacy bill', 'prescription drugs', 'health supplements',
                    'hospital bill', 'medical treatment', 'surgery expenses', 'emergency treatment',
                    'dental treatment', 'eye checkup', 'blood test', 'x-ray scan', 'mri scan',
                    'vaccination', 'health insurance', 'medical premium', 'therapy session'
                ],
                'amount_range': (200, 10000),
                'amount_distribution': 'exponential'
            },
            
            'EDUCATION': {
                'merchants': [
                    'Byju\'s', 'Unacademy', 'Vedantu', 'WhiteHat Jr', 'Toppr', 'Khan Academy',
                    'Coursera', 'Udemy', 'edX', 'Skillshare', 'LinkedIn Learning', 'Pluralsight',
                    'School Fees', 'College Fees', 'University', 'Coaching Center', 'Tuition Classes',
                    'Book Store', 'Stationery Shop', 'Online Books', 'Educational Apps', 'Language Course'
                ],
                'descriptions': [
                    'online course', 'educational subscription', 'skill development', 'certification course',
                    'school fees', 'tuition fees', 'college admission', 'coaching classes', 'entrance exam prep',
                    'book purchase', 'study material', 'educational supplies', 'stationery items',
                    'language learning', 'professional course', 'workshop fees', 'seminar registration',
                    'library membership', 'research material', 'thesis printing', 'project expenses'
                ],
                'amount_range': (500, 25000),
                'amount_distribution': 'uniform'
            },
            
            'INVESTMENT': {
                'merchants': [
                    'Zerodha', 'Groww', 'Upstox', 'Angel Broking', 'ICICI Direct', '5paisa',
                    'SBI Mutual Fund', 'HDFC AMC', 'ICICI Prudential', 'Axis Mutual Fund',
                    'LIC Premium', 'HDFC Life', 'SBI Life', 'Max Life Insurance', 'Bajaj Allianz',
                    'PPF Account', 'NSC Investment', 'ELSS Fund', 'Gold Purchase', 'Real Estate'
                ],
                'descriptions': [
                    'stock trading', 'equity purchase', 'share buying', 'trading charges', 'brokerage fee',
                    'mutual fund sip', 'lump sum investment', 'portfolio investment', 'fund purchase',
                    'life insurance', 'term insurance', 'health insurance', 'car insurance', 'premium payment',
                    'fixed deposit', 'recurring deposit', 'savings plan', 'retirement fund', 'pension plan',
                    'gold investment', 'bond purchase', 'government securities', 'tax saving investment'
                ],
                'amount_range': (1000, 100000),
                'amount_distribution': 'exponential'
            },
            
            'MISCELLANEOUS': {
                'merchants': [
                    'ATM Withdrawal', 'Bank Charges', 'Government Office', 'Tax Payment', 'Legal Services',
                    'Post Office', 'Courier Service', 'Printing Shop', 'Photocopy Center', 'Document Service',
                    'Charity', 'Donation', 'Temple', 'Religious Institution', 'NGO Contribution',
                    'Gift Shop', 'Flower Shop', 'Event Management', 'Party Supplies', 'Decoration Service'
                ],
                'descriptions': [
                    'cash withdrawal', 'atm transaction', 'bank charges', 'service fee', 'processing charges',
                    'income tax', 'gst payment', 'property tax', 'government fee', 'license renewal',
                    'legal consultation', 'notary charges', 'registration fee', 'stamp duty', 'court fee',
                    'charity donation', 'religious donation', 'social cause', 'ngo contribution', 'temple offering',
                    'gift purchase', 'celebration expenses', 'party arrangement', 'event planning', 'miscellaneous expense'
                ],
                'amount_range': (50, 10000),
                'amount_distribution': 'uniform'
            }
        }
    
    def generate_transaction(self, category: str) -> Dict[str, any]:
        """Generate a single realistic transaction for the given category."""
        patterns = self.transaction_patterns[category]
        
        # Random merchant and description
        merchant = random.choice(patterns['merchants'])
        description = random.choice(patterns['descriptions'])
        
        # Generate amount based on distribution
        min_amt, max_amt = patterns['amount_range']
        distribution = patterns['amount_distribution']
        
        if distribution == 'normal':
            mean = (min_amt + max_amt) / 2
            std = (max_amt - min_amt) / 6
            amount = max(min_amt, min(max_amt, int(np.random.normal(mean, std))))
        elif distribution == 'lognormal':
            # For food - most transactions are small, few are large
            amount = int(np.random.lognormal(np.log(min_amt + 100), 0.8))
            amount = max(min_amt, min(max_amt, amount))
        elif distribution == 'exponential':
            # For shopping/investment - many small, few very large
            scale = (max_amt - min_amt) / 5
            amount = int(np.random.exponential(scale) + min_amt)
            amount = max(min_amt, min(max_amt, amount))
        else:  # uniform
            amount = random.randint(min_amt, max_amt)
        
        # Add some realistic variations to descriptions
        variations = [
            f"{description} payment", f"{description} bill", f"online {description}",
            f"monthly {description}", f"weekly {description}", f"daily {description}",
            f"{description} charges", f"{description} fee", f"{description} expense"
        ]
        
        if random.random() < 0.3:  # 30% chance of variation
            description = random.choice(variations)
        
        return {
            'merchant': merchant,
            'description': description,
            'amount': amount,
            'category': category
        }
    
    def generate_large_dataset(self, samples_per_category: int = 500) -> pd.DataFrame:
        """
        Generate a large, balanced dataset.
        
        Args:
            samples_per_category: Number of samples per category
            
        Returns:
            DataFrame with generated transactions
        """
        print(f"🏭 Generating large dataset with {samples_per_category} samples per category...")
        
        all_transactions = []
        categories = list(self.transaction_patterns.keys())
        
        for category in categories:
            print(f"   Generating {samples_per_category} {category} transactions...")
            for _ in range(samples_per_category):
                transaction = self.generate_transaction(category)
                all_transactions.append(transaction)
        
        # Shuffle the data
        random.shuffle(all_transactions)
        
        df = pd.DataFrame(all_transactions)
        
        print(f"✅ Generated {len(df)} total transactions")
        print(f"📊 Category distribution:")
        for category, count in df['category'].value_counts().items():
            print(f"   {category}: {count}")
        
        return df
    
    def generate_realistic_monthly_data(self, num_months: int = 12) -> pd.DataFrame:
        """
        Generate realistic monthly spending data with seasonal variations.
        
        Args:
            num_months: Number of months to generate
            
        Returns:
            DataFrame with time-based transaction data
        """
        print(f"📅 Generating {num_months} months of realistic transaction data...")
        
        all_transactions = []
        start_date = datetime.now() - timedelta(days=num_months * 30)
        
        # Monthly patterns (different spending in different months)
        monthly_multipliers = {
            'FOOD_DINING': [1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.2, 1.1, 1.0, 1.3, 1.4, 1.5],  # More in summer, festivals
            'SHOPPING': [1.2, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.1, 1.2, 1.5, 1.8, 2.0],      # Peak in Oct-Dec
            'ENTERTAINMENT': [1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.0, 1.0, 1.1, 1.2, 1.3], # More in summer, holidays
            'TRANSPORTATION': [1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.0, 1.0, 1.0, 1.1, 1.0, 1.1], # More in summer
            'UTILITIES': [1.2, 1.1, 1.0, 1.0, 1.1, 1.3, 1.3, 1.2, 1.0, 1.0, 1.0, 1.1],      # More in summer/winter
            'HEALTHCARE': [1.1, 1.1, 1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1],     # Seasonal variations
            'EDUCATION': [1.0, 1.0, 1.5, 1.2, 1.0, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],      # Peak in March, June
            'INVESTMENT': [1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2],      # March (tax), December
            'MISCELLANEOUS': [1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1]   # Slight variations
        }
        
        for month in range(num_months):
            month_date = start_date + timedelta(days=month * 30)
            month_index = month % 12
            
            # Generate transactions for this month
            for category in self.transaction_patterns.keys():
                # Base number of transactions per month per category
                base_transactions = random.randint(8, 25)  # 8-25 transactions per category per month
                
                # Apply seasonal multiplier
                multiplier = monthly_multipliers[category][month_index]
                num_transactions = int(base_transactions * multiplier)
                
                for _ in range(num_transactions):
                    transaction = self.generate_transaction(category)
                    
                    # Add date
                    day_offset = random.randint(0, 29)
                    transaction_date = month_date + timedelta(days=day_offset)
                    transaction['date'] = transaction_date.strftime('%Y-%m-%d')
                    
                    all_transactions.append(transaction)
        
        # Shuffle transactions
        random.shuffle(all_transactions)
        
        df = pd.DataFrame(all_transactions)
        
        print(f"✅ Generated {len(df)} transactions over {num_months} months")
        print(f"📊 Category distribution:")
        for category, count in df['category'].value_counts().items():
            print(f"   {category}: {count}")
        
        return df
    
    def add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic noise and variations to the dataset."""
        print("🔀 Adding realistic variations to the dataset...")
        
        df_noisy = df.copy()
        
        # Add typos and variations to merchant names (5% of data)
        noise_indices = random.sample(range(len(df_noisy)), int(0.05 * len(df_noisy)))
        
        for idx in noise_indices:
            merchant = df_noisy.loc[idx, 'merchant']
            
            # Common variations
            variations = [
                merchant.upper(),
                merchant.lower(),
                f"{merchant} India",
                f"{merchant} Pvt Ltd",
                f"{merchant}.com",
                f"{merchant} Store",
                f"{merchant} Online"
            ]
            
            # Small typos (character replacement)
            if len(merchant) > 3 and random.random() < 0.3:
                char_idx = random.randint(0, len(merchant) - 1)
                typo_chars = 'abcdefghijklmnopqrstuvwxyz'
                typo_merchant = list(merchant)
                typo_merchant[char_idx] = random.choice(typo_chars)
                variations.append(''.join(typo_merchant))
            
            df_noisy.loc[idx, 'merchant'] = random.choice(variations)
        
        # Add banking terms to descriptions (10% of data)
        banking_terms = ['UPI', 'NEFT', 'IMPS', 'Net Banking', 'Card Payment', 'Online Transfer']
        banking_indices = random.sample(range(len(df_noisy)), int(0.1 * len(df_noisy)))
        
        for idx in banking_indices:
            description = df_noisy.loc[idx, 'description']
            term = random.choice(banking_terms)
            df_noisy.loc[idx, 'description'] = f"{term} {description}"
        
        print("✅ Added realistic noise and variations")
        return df_noisy


def create_enhanced_training_data(samples_per_category: int = 1000, 
                                add_temporal_data: bool = True,
                                add_noise: bool = True) -> pd.DataFrame:
    """
    Create a comprehensive training dataset.
    
    Args:
        samples_per_category: Number of samples per category
        add_temporal_data: Whether to add time-based data
        add_noise: Whether to add realistic noise
        
    Returns:
        Large, diverse training dataset
    """
    print("🎯 Creating Enhanced Training Dataset")
    print("=" * 50)
    
    generator = LargeDatasetGenerator()
    
    # Generate main dataset
    main_df = generator.generate_large_dataset(samples_per_category)
    
    datasets = [main_df]
    
    # Add temporal data if requested
    if add_temporal_data:
        temporal_df = generator.generate_realistic_monthly_data(num_months=24)  # 2 years
        datasets.append(temporal_df)
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Add noise if requested
    if add_noise:
        combined_df = generator.add_realistic_noise(combined_df)
    
    # Final shuffle
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    print(f"\n🏆 FINAL DATASET STATISTICS")
    print(f"Total Transactions: {len(combined_df):,}")
    print(f"Categories: {combined_df['category'].nunique()}")
    print(f"Unique Merchants: {combined_df['merchant'].nunique()}")
    print(f"Amount Range: ₹{combined_df['amount'].min():,} - ₹{combined_df['amount'].max():,}")
    print(f"Average Amount: ₹{combined_df['amount'].mean():,.0f}")
    
    print(f"\n📊 Category Distribution:")
    for category, count in combined_df['category'].value_counts().items():
        percentage = (count / len(combined_df)) * 100
        print(f"   {category}: {count:,} ({percentage:.1f}%)")
    
    return combined_df


if __name__ == "__main__":
    # Generate enhanced dataset
    enhanced_df = create_enhanced_training_data(
        samples_per_category=800,  # 800 per category = 7,200 base transactions
        add_temporal_data=True,    # Add 2 years of realistic monthly data
        add_noise=True            # Add realistic variations
    )
    
    # Save the dataset
    enhanced_df.to_csv('data/enhanced_training_data.csv', index=False)
    print(f"\n💾 Dataset saved to 'data/enhanced_training_data.csv'")
    print(f"🎉 Enhanced dataset ready for training!")