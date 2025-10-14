"""
Ultra Enhanced Dataset Generator for Maximum ML Performance

This module creates the largest, most comprehensive training dataset possible
with extreme diversity, realistic patterns, and perfect category coverage.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

class UltraDatasetGenerator:
    """
    Generate ultra-comprehensive training datasets for maximum ML performance.
    """
    
    def __init__(self):
        """Initialize with extensive merchant and pattern data."""
        
        # Ultra-comprehensive merchant database
        self.merchants = {
            'FOOD_DINING': [
                # Restaurants & Cafes
                'Zomato', 'Swiggy', 'Dominos Pizza', 'Pizza Hut', 'KFC', 'McDonalds', 'Burger King',
                'Subway', 'Starbucks', 'Costa Coffee', 'Cafe Coffee Day', 'Barista', 'Dunkin Donuts',
                'Haldirams', 'Bikanervala', 'Sagar Ratna', 'Saravana Bhavan', 'Udupi Krishna',
                'Paradise Restaurant', 'Mainland China', 'Copper Chimney', 'TGI Fridays', 'Hard Rock Cafe',
                'Social', 'Imperfecto', 'The Beer Cafe', 'Cafe Mocha', 'Chaayos', 'Tea Point',
                # Food Delivery
                'Uber Eats', 'Food Panda', 'Box8', 'Faasos', 'Behrouz Biryani', 'Oven Story',
                'The Bowl Company', 'Lunch Box', 'Fresh Menu', 'Rebel Foods', 'EatFit', 'HealthyBytes',
                # Groceries & Markets
                'Big Basket', 'Grofers', 'Amazon Fresh', 'Flipkart Grocery', 'JioMart', 'Nature Basket',
                'Spencer Retail', 'More Supermarket', 'Reliance Fresh', 'Heritage Fresh', 'Food Bazaar',
                'DMart', 'Star Bazaar', 'Hypercity', 'Easyday', 'Nilgiris', 'Godrej Nature Basket',
                # Local vendors
                'Local Restaurant', 'Street Food', 'Dhaba', 'Canteen', 'Food Court', 'Bakery',
                'Sweet Shop', 'Juice Corner', 'Fruit Vendor', 'Vegetable Market', 'Grocery Store'
            ],
            
            'TRANSPORTATION': [
                # Ride sharing
                'Uber', 'Ola', 'Rapido', 'Bounce', 'Vogo', 'Yulu', 'Quick Ride', 'BlaBla Car',
                'Uber Auto', 'Ola Auto', 'Ola Bike', 'Meru Cabs', 'Taxi For Sure', 'Easy Cabs',
                # Fuel
                'Indian Oil', 'Bharat Petroleum', 'Hindustan Petroleum', 'Shell', 'Essar Oil',
                'Reliance Petrol', 'Total Oil', 'Nayara Energy', 'Petrol Pump', 'Diesel Station',
                # Public Transport
                'Delhi Metro', 'Mumbai Metro', 'Bangalore Metro', 'DMRC', 'BMRCL', 'Chennai Metro',
                'BEST Bus', 'DTC Bus', 'BMTC Bus', 'MTC Bus', 'KSRTC', 'MSRTC', 'Railway',
                'Indian Railways', 'IRCTC', 'Tatkal Booking', 'Platform Ticket', 'Auto Rickshaw',
                # Airlines
                'IndiGo', 'SpiceJet', 'Air India', 'Vistara', 'GoAir', 'AirAsia India', 'Jet Airways',
                # Parking & Tolls
                'Parking Fee', 'Toll Plaza', 'Highway Toll', 'Airport Parking', 'Mall Parking',
                'Vehicle Service', 'Car Wash', 'Tyre Shop', 'Battery Shop', 'Car Repair'
            ],
            
            'SHOPPING': [
                # E-commerce
                'Amazon', 'Flipkart', 'Myntra', 'Ajio', 'Nykaa', 'Jabong', 'Snapdeal', 'Paytm Mall',
                'Tata Cliq', 'ShopClues', 'Pepperfry', 'Urban Ladder', 'Limeroad', 'Koovs',
                'FirstCry', 'Hopscotch', 'Zivame', 'Clovia', 'Bewakoof', 'The Souled Store',
                # Fashion & Apparel
                'Zara', 'H&M', 'Uniqlo', 'Forever 21', 'Max Fashion', 'Pantaloons', 'Westside',
                'Lifestyle', 'Central', 'Shoppers Stop', 'Brand Factory', 'Fashion Big Bazaar',
                'Van Heusen', 'Allen Solly', 'Peter England', 'Louis Philippe', 'Arrow',
                # Electronics
                'Croma', 'Vijay Sales', 'Reliance Digital', 'Poorvika', 'Sangeetha Mobiles',
                'Apple Store', 'Samsung Store', 'OnePlus Store', 'Mi Store', 'Oppo Store',
                # Books & Stationery
                'Amazon Books', 'Flipkart Books', 'Crossword', 'Landmark', 'Oxford Bookstore',
                'Sapna Book House', 'Higginbothams', 'Odyssey', 'Starmark', 'WHSmith',
                # Local Shopping
                'Local Market', 'Street Shopping', 'Wholesale Market', 'Gift Shop', 'Souvenir Shop'
            ],
            
            'ENTERTAINMENT': [
                # Streaming Services
                'Netflix', 'Amazon Prime', 'Disney Hotstar', 'Sony Liv', 'Zee5', 'Voot', 'MX Player',
                'Eros Now', 'Alt Balaji', 'TVF Play', 'Hoichoi', 'Sun NXT', 'Jio Cinema',
                'Apple TV Plus', 'YouTube Premium', 'Spotify', 'Gaana', 'JioSaavn', 'Wynk Music',
                'Amazon Music', 'Apple Music', 'Google Play Music', 'Hungama Music',
                # Movies & Events
                'BookMyShow', 'Paytm Movies', 'TicketNew', 'Justickets', 'Insider.in', 'Explara',
                'PVR Cinemas', 'INOX', 'Cinepolis', 'Carnival Cinemas', 'SPI Cinemas', 'Fun Cinemas',
                # Gaming
                'Steam', 'Epic Games', 'PlayStation Store', 'Xbox Live', 'Google Play Games',
                'App Store Games', 'PUBG Mobile', 'Call of Duty', 'Free Fire', 'Clash of Clans',
                # Sports & Fitness
                'Cult Fit', 'Gold Gym', 'Fitness First', 'Anytime Fitness', 'Snap Fitness',
                'Yoga Classes', 'Zumba Classes', 'Swimming Pool', 'Sports Club', 'Cricket Academy',
                # Recreation
                'Amusement Park', 'Water Park', 'Gaming Zone', 'Bowling Alley', 'Mini Golf',
                'Adventure Sports', 'Paintball', 'Go Karting', 'Escape Room', 'Laser Tag'
            ],
            
            'UTILITIES': [
                # Telecom
                'Airtel', 'Jio', 'Vi', 'BSNL', 'Tata Docomo', 'Idea', 'Vodafone', 'ACT Fibernet',
                'Hathway', 'Tikona', 'Excitel', 'GTPL', 'DEN Networks', 'Siti Cable',
                # Electricity
                'BESCOM', 'KSEB', 'TNEB', 'MSEDCL', 'PSPCL', 'UPPCL', 'Adani Power', 'Tata Power',
                'Reliance Energy', 'BSES', 'NDPL', 'Torrent Power', 'Gujarat Urja', 'WBSEDCL',
                # Water & Gas
                'Indraprastha Gas', 'Mahanagar Gas', 'Gujarat Gas', 'Sabarmati Gas', 'Adani Gas',
                'Bharat Gas', 'HP Gas', 'Indian Oil LPG', 'Water Board', 'Municipal Corporation',
                # Banking & Finance
                'ICICI Bank', 'HDFC Bank', 'SBI', 'Axis Bank', 'Kotak Bank', 'Yes Bank', 'IndusInd',
                'Standard Chartered', 'Citibank', 'HSBC', 'DBS Bank', 'RBL Bank', 'Federal Bank',
                # DTH & Cable
                'Tata Sky', 'Dish TV', 'Videocon D2H', 'Sun Direct', 'Airtel Digital TV',
                'DEN Cable', 'Hathway Cable', 'GTPL Cable', 'Local Cable'
            ],
            
            'HEALTHCARE': [
                # Hospitals & Clinics
                'Apollo Hospital', 'Fortis Healthcare', 'Max Healthcare', 'Manipal Hospital',
                'Narayana Health', 'Columbia Asia', 'Global Hospitals', 'Continental Hospital',
                'AIIMS', 'Safdarjung Hospital', 'Sir Ganga Ram Hospital', 'BLK Hospital', 'Medanta',
                'Local Clinic', 'Family Doctor', 'Pediatrician', 'Dermatologist', 'Orthopedic',
                # Pharmacies
                'Apollo Pharmacy', 'MedPlus', 'Netmeds', '1mg', 'PharmEasy', 'Wellness Forever',
                'Guardian Pharmacy', 'Himalaya Store', 'Local Pharmacy', 'Medical Store',
                # Diagnostics
                'SRL Diagnostics', 'Dr Lal PathLabs', 'Metropolis Healthcare', 'Thyrocare',
                'Quest Diagnostics', 'Vijaya Diagnostics', 'Suburban Diagnostics', 'Core Diagnostics',
                # Health Services
                'Practo', 'Lybrate', 'DocsApp', 'mfine', 'Portea Medical', 'Nightingales',
                'Dental Clinic', 'Eye Clinic', 'Physiotherapy', 'Yoga Therapy', 'Ayurveda Center',
                # Insurance
                'Health Insurance', 'Mediclaim', 'Star Health', 'New India Assurance', 'Oriental Insurance'
            ],
            
            'EDUCATION': [
                # Schools & Colleges
                'School Fee', 'College Fee', 'University Fee', 'Coaching Classes', 'Tuition Fee',
                'IIT Delhi', 'IIM Bangalore', 'DU', 'JNU', 'Manipal University', 'VIT University',
                'Amity University', 'SRM University', 'Lovely Professional University', 'Christ University',
                # Online Learning
                'BYJU\'S', 'Unacademy', 'Vedantu', 'WhiteHat Jr', 'Toppr', 'Embibe', 'Meritnation',
                'Khan Academy', 'Coursera', 'Udemy', 'edX', 'Pluralsight', 'LinkedIn Learning',
                'Skillshare', 'MasterClass', 'Udacity', 'DataCamp', 'Codecademy', 'FreeCodeCamp',
                # Books & Materials
                'Study Material', 'Textbooks', 'Reference Books', 'Online Course', 'Certification',
                'Exam Fee', 'Application Fee', 'Library Fee', 'Lab Fee', 'Sports Fee',
                # Competitive Exams
                'JEE Coaching', 'NEET Coaching', 'CAT Coaching', 'UPSC Coaching', 'Bank PO Coaching',
                'GATE Coaching', 'GRE Coaching', 'TOEFL Coaching', 'IELTS Coaching', 'SAT Coaching'
            ],
            
            'INVESTMENT': [
                # Mutual Funds
                'SBI Mutual Fund', 'HDFC Mutual Fund', 'ICICI Prudential', 'Reliance Mutual Fund',
                'Axis Mutual Fund', 'Kotak Mutual Fund', 'DSP Mutual Fund', 'Franklin Templeton',
                'Aditya Birla Sun Life', 'UTI Mutual Fund', 'Nippon India MF', 'L&T Mutual Fund',
                # Banks & Fixed Deposits
                'Fixed Deposit', 'Recurring Deposit', 'PPF', 'NSC', 'ELSS', 'Tax Saving FD',
                'Senior Citizen FD', 'Corporate FD', 'Bank FD', 'Post Office', 'Sukanya Samriddhi',
                # Insurance
                'Life Insurance', 'Health Insurance', 'Term Insurance', 'ULIP', 'Endowment Plan',
                'LIC', 'HDFC Life', 'ICICI Prudential Life', 'SBI Life', 'Max Life', 'Bajaj Allianz',
                'Star Health', 'New India Assurance', 'Oriental Insurance', 'United India Insurance',
                # Trading & Stocks
                'Zerodha', 'Angel Broking', 'Upstox', 'Sharekhan', 'ICICI Direct', 'HDFC Securities',
                'Kotak Securities', 'Motilal Oswal', '5paisa', 'Groww', 'ETMoney', 'Paytm Money',
                # Gold & Commodities
                'Digital Gold', 'Gold ETF', 'Silver Investment', 'Commodity Trading', 'Gold Bonds',
                'Physical Gold', 'Jewelry Investment', 'Gold Scheme', 'Silver Coins', 'Precious Metals'
            ],
            
            'MISCELLANEOUS': [
                # ATM & Banking
                'ATM Withdrawal', 'Bank Charges', 'Service Charges', 'Account Maintenance',
                'Cheque Book', 'DD Charges', 'NEFT Charges', 'RTGS Charges', 'IMPS Charges',
                # Government & Legal
                'Income Tax', 'Property Tax', 'Vehicle Tax', 'Professional Tax', 'GST Payment',
                'Passport Fee', 'Visa Fee', 'License Fee', 'Registration Fee', 'Court Fee',
                'Legal Services', 'CA Fee', 'Lawyer Fee', 'Notary', 'Document Service',
                # Personal Services
                'Salon', 'Spa', 'Massage', 'Beauty Parlor', 'Barber Shop', 'Nail Art',
                'Dry Cleaning', 'Laundry', 'Tailoring', 'Shoe Repair', 'Watch Repair',
                # Charity & Donations
                'Donation', 'Charity', 'Temple Donation', 'NGO Contribution', 'Relief Fund',
                'Crowdfunding', 'Social Cause', 'Religious Donation', 'Educational Donation',
                # Miscellaneous
                'Pet Care', 'Veterinary', 'Plant Nursery', 'Gardening', 'Home Decoration',
                'Furniture', 'Appliance Repair', 'Maintenance', 'Security Deposit', 'Advance Payment'
            ]
        }
        
        # Enhanced description patterns
        self.description_patterns = {
            'FOOD_DINING': [
                'food delivery order', 'restaurant bill', 'online food order', 'dining out',
                'breakfast', 'lunch', 'dinner', 'snacks', 'beverages', 'coffee', 'tea',
                'pizza order', 'burger meal', 'biryani order', 'chinese food', 'italian cuisine',
                'grocery shopping', 'vegetables', 'fruits', 'dairy products', 'cooking ingredients',
                'bakery items', 'sweets', 'desserts', 'ice cream', 'juice', 'soft drinks',
                'home delivery', 'takeaway', 'dine in', 'buffet', 'party order', 'catering'
            ],
            'TRANSPORTATION': [
                'cab ride', 'auto ride', 'bus ticket', 'metro card recharge', 'flight booking',
                'train ticket', 'fuel refill', 'petrol', 'diesel', 'parking fee', 'toll charges',
                'bike ride', 'car rental', 'airport transfer', 'intercity travel', 'local transport',
                'ride sharing', 'vehicle maintenance', 'car service', 'tyre change', 'oil change',
                'insurance renewal', 'vehicle registration', 'pollution certificate', 'driving license'
            ],
            'SHOPPING': [
                'online shopping', 'clothing purchase', 'electronics', 'mobile phone', 'laptop',
                'home appliances', 'furniture', 'books', 'stationery', 'cosmetics', 'skincare',
                'fashion accessories', 'footwear', 'bags', 'watches', 'jewelry', 'gifts',
                'household items', 'kitchen utensils', 'home decor', 'gardening supplies',
                'sports equipment', 'gadgets', 'tools', 'toys', 'baby products', 'pet supplies'
            ],
            'ENTERTAINMENT': [
                'movie ticket', 'streaming subscription', 'music subscription', 'gaming',
                'concert ticket', 'sports event', 'theater show', 'amusement park',
                'adventure activity', 'hobby classes', 'art supplies', 'musical instrument',
                'fitness membership', 'yoga classes', 'dance classes', 'swimming pool',
                'weekend outing', 'vacation', 'travel booking', 'hotel booking', 'recreation'
            ],
            'UTILITIES': [
                'electricity bill', 'water bill', 'gas bill', 'internet bill', 'phone bill',
                'mobile recharge', 'DTH recharge', 'cable TV', 'broadband', 'landline',
                'postpaid bill', 'prepaid recharge', 'data pack', 'roaming charges',
                'service charges', 'installation charges', 'maintenance fee', 'renewal fee'
            ],
            'HEALTHCARE': [
                'doctor consultation', 'medical checkup', 'prescription medicine', 'surgery',
                'diagnostic test', 'blood test', 'x-ray', 'MRI scan', 'dental treatment',
                'eye checkup', 'vaccination', 'physiotherapy', 'health insurance premium',
                'medical emergency', 'ambulance charges', 'hospital admission', 'pharmacy'
            ],
            'EDUCATION': [
                'school fee', 'college fee', 'tuition fee', 'exam fee', 'course material',
                'online course', 'certification', 'skill development', 'language classes',
                'professional course', 'textbooks', 'study material', 'coaching classes',
                'entrance exam', 'competitive exam', 'library fee', 'lab fee', 'project work'
            ],
            'INVESTMENT': [
                'mutual fund SIP', 'life insurance premium', 'fixed deposit', 'PPF contribution',
                'share trading', 'gold purchase', 'bond investment', 'ELSS investment',
                'retirement planning', 'child education fund', 'emergency fund', 'wealth creation',
                'tax saving investment', 'dividend', 'capital gains', 'portfolio management'
            ],
            'MISCELLANEOUS': [
                'ATM withdrawal', 'bank charges', 'service tax', 'income tax', 'property tax',
                'legal fee', 'consultancy', 'professional service', 'government fee',
                'license renewal', 'registration charges', 'penalty', 'fine', 'donation',
                'charity', 'gift', 'personal loan EMI', 'credit card payment', 'miscellaneous'
            ]
        }
        
        # Realistic amount ranges for each category
        self.amount_ranges = {
            'FOOD_DINING': (30, 8000),      # Street food to fine dining
            'TRANSPORTATION': (20, 25000),   # Local auto to flight tickets
            'SHOPPING': (50, 75000),         # Small items to electronics
            'ENTERTAINMENT': (100, 15000),   # Movie tickets to event passes
            'UTILITIES': (200, 12000),       # Small bills to corporate bills
            'HEALTHCARE': (50, 50000),       # Medicine to surgery
            'EDUCATION': (500, 200000),      # Books to course fees
            'INVESTMENT': (500, 100000),     # Small SIP to major investments
            'MISCELLANEOUS': (10, 25000)     # ATM fees to taxes
        }
    
    def generate_mega_dataset(self, samples_per_category: int = 2000) -> pd.DataFrame:
        """
        Generate mega dataset with maximum diversity and realism.
        
        Args:
            samples_per_category: Number of samples per category (default 2000)
            
        Returns:
            DataFrame with comprehensive training data
        """
        print(f"🎯 Creating MEGA Training Dataset")
        print(f"📊 Target: {samples_per_category * 9:,} total transactions")
        print("=" * 60)
        
        all_transactions = []
        
        for category in self.merchants.keys():
            print(f"🏭 Generating {samples_per_category:,} {category} transactions...")
            
            category_transactions = []
            merchants = self.merchants[category]
            descriptions = self.description_patterns[category]
            min_amount, max_amount = self.amount_ranges[category]
            
            for i in range(samples_per_category):
                # Select merchant with weighted probability (some merchants more common)
                merchant = random.choices(
                    merchants, 
                    weights=[3 if 'Zomato' in m or 'Amazon' in m or 'Uber' in m else 1 for m in merchants]
                )[0]
                
                # Generate realistic description
                base_desc = random.choice(descriptions)
                
                # Add variations and banking terms
                variations = [
                    f"UPI {base_desc}",
                    f"NEFT {base_desc}",
                    f"card payment {base_desc}",
                    f"online {base_desc}",
                    f"mobile {base_desc}",
                    f"{base_desc} payment",
                    f"{base_desc} transaction",
                    base_desc,
                    base_desc.upper(),
                    base_desc.lower()
                ]
                
                description = random.choice(variations)
                
                # Generate realistic amounts with distribution
                if category == 'FOOD_DINING':
                    # More small amounts, fewer large ones
                    amount = np.random.exponential(300) + min_amount
                    amount = min(amount, max_amount)
                elif category == 'INVESTMENT':
                    # Bimodal distribution (small SIPs and large investments)
                    if random.random() < 0.7:
                        amount = random.randint(500, 10000)  # SIPs
                    else:
                        amount = random.randint(25000, max_amount)  # Large investments
                elif category == 'EDUCATION':
                    # Seasonal spikes (admission time)
                    if random.random() < 0.3:
                        amount = random.randint(50000, max_amount)  # Fees
                    else:
                        amount = random.randint(min_amount, 15000)  # Books, courses
                else:
                    # Normal distribution with realistic skew
                    mean_amount = (min_amount + max_amount) / 2
                    std_amount = (max_amount - min_amount) / 6
                    amount = np.random.normal(mean_amount, std_amount)
                    amount = max(min_amount, min(max_amount, amount))
                
                amount = round(amount)
                
                # Generate realistic dates (last 18 months)
                start_date = datetime.now() - timedelta(days=540)
                random_days = random.randint(0, 540)
                transaction_date = start_date + timedelta(days=random_days)
                
                # Add seasonal variations
                month = transaction_date.month
                if category == 'EDUCATION' and month in [4, 5, 6, 11]:  # Admission seasons
                    amount = int(amount * random.uniform(1.2, 2.0))
                elif category == 'SHOPPING' and month in [10, 11, 12]:  # Festival season
                    amount = int(amount * random.uniform(1.1, 1.5))
                elif category == 'ENTERTAINMENT' and month in [12, 1, 4, 5]:  # Holiday seasons
                    amount = int(amount * random.uniform(1.2, 1.8))
                
                transaction = {
                    'merchant': merchant,
                    'description': description,
                    'amount': amount,
                    'category': category,
                    'date': transaction_date.strftime('%Y-%m-%d')
                }
                
                category_transactions.append(transaction)
            
            all_transactions.extend(category_transactions)
            print(f"✅ Generated {len(category_transactions):,} {category} transactions")
        
        # Create DataFrame and add realistic noise
        df = pd.DataFrame(all_transactions)
        
        print(f"\n🔀 Adding realistic variations and noise...")
        df = self.add_ultra_realistic_variations(df)
        
        # Shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        print(f"\n🏆 MEGA DATASET STATISTICS")
        print(f"Total Transactions: {len(df):,}")
        print(f"Categories: {df['category'].nunique()}")
        print(f"Unique Merchants: {df['merchant'].nunique()}")
        print(f"Amount Range: ₹{df['amount'].min():,} - ₹{df['amount'].max():,}")
        print(f"Average Amount: ₹{df['amount'].mean():.0f}")
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        
        print(f"\n📊 Category Distribution:")
        for category, count in df['category'].value_counts().items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count:,} ({percentage:.1f}%)")
        
        return df
    
    def add_ultra_realistic_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ultra-realistic variations to make the dataset more robust."""
        
        # Add merchant name variations
        for idx, row in df.iterrows():
            if random.random() < 0.15:  # 15% chance of variation
                merchant = row['merchant']
                variations = [
                    f"{merchant} India",
                    f"{merchant} Ltd",
                    f"{merchant} Pvt Ltd",
                    f"{merchant} Online",
                    f"{merchant}.com",
                    f"{merchant} Store",
                    f"{merchant} Services",
                    merchant.replace(' ', ''),
                    merchant.upper(),
                    merchant.lower()
                ]
                df.at[idx, 'merchant'] = random.choice(variations)
        
        # Add banking terminology variations
        banking_terms = ['UPI', 'NEFT', 'RTGS', 'IMPS', 'Card', 'Wallet', 'Net Banking']
        for idx, row in df.iterrows():
            if random.random() < 0.2:  # 20% chance
                term = random.choice(banking_terms)
                df.at[idx, 'description'] = f"{term} {row['description']}"
        
        # Add realistic typos and abbreviations
        for idx, row in df.iterrows():
            if random.random() < 0.1:  # 10% chance of typos
                desc = row['description']
                # Common abbreviations
                desc = desc.replace('payment', 'pymnt')
                desc = desc.replace('transaction', 'txn')
                desc = desc.replace('recharge', 'rchg')
                desc = desc.replace('subscription', 'sub')
                df.at[idx, 'description'] = desc
        
        # Add amount variations (rounded amounts are common)
        for idx, row in df.iterrows():
            if random.random() < 0.3:  # 30% chance of round amounts
                amount = row['amount']
                # Round to nearest 10, 50, or 100
                round_to = random.choice([10, 50, 100])
                df.at[idx, 'amount'] = round(amount / round_to) * round_to
        
        return df


def main():
    """Generate the ultimate training dataset."""
    generator = UltraDatasetGenerator()
    
    # Generate mega dataset with 2000 samples per category (18,000 total)
    mega_df = generator.generate_mega_dataset(samples_per_category=2000)
    
    # Save the mega dataset
    output_file = 'data/mega_training_data.csv'
    mega_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Mega dataset saved to '{output_file}'")
    print(f"🎉 Ultra-comprehensive dataset ready for perfect ML training!")
    
    return mega_df


if __name__ == "__main__":
    main()