
import random
import datetime
from typing import List, Dict

class SyntheticDataGenerator:
    """Generate synthetic financial data for testing Smart Money AI"""
    
    def __init__(self):
        self.banks = [
            "HDFC", "SBI", "ICICI", "Axis", "Kotak", "IDFC", "PNB", 
            "BoB", "Canara", "Union", "Indian Bank", "Central Bank"
        ]
        
        self.merchants = [
            "AMAZON", "FLIPKART", "SWIGGY", "ZOMATO", "UBER", "OLA",
            "DMart", "BigBasket", "Myntra", "BookMyShow", "PVR",
            "Reliance", "Shell", "HP Petrol", "More Supermarket"
        ]
        
        self.categories = {
            "FOOD_DINING": ["SWIGGY", "ZOMATO", "KFC", "McDonald's"],
            "SHOPPING": ["AMAZON", "FLIPKART", "MYNTRA", "DMart"],
            "TRANSPORTATION": ["UBER", "OLA", "Shell", "HP Petrol"],
            "UTILITIES": ["Electricity", "Gas", "Water", "Internet"]
        }
    
    def generate_sms_transaction(self) -> str:
        """Generate realistic SMS transaction"""
        bank = random.choice(self.banks)
        amount = random.randint(50, 5000)
        merchant = random.choice(self.merchants)
        card_last4 = random.randint(1000, 9999)
        
        templates = [
            f"{bank}: Spent Rs.{amount} at {merchant} on Card xx{card_last4}",
            f"{bank} Alert: Rs {amount} debited from a/c xx1234 for {merchant}",
            f"{bank}: Payment of Rs.{amount} made to {merchant} via UPI",
        ]
        
        return random.choice(templates)
    
    def generate_expense_record(self) -> Dict:
        """Generate expense record with category"""
        category = random.choice(list(self.categories.keys()))
        merchant = random.choice(self.categories[category])
        
        return {
            "date": datetime.date.today() - datetime.timedelta(days=random.randint(1, 30)),
            "amount": random.randint(100, 2000),
            "merchant": merchant,
            "category": category,
            "payment_method": random.choice(["Card", "UPI", "Cash"])
        }

# Example usage
generator = SyntheticDataGenerator()
print("Sample SMS:", generator.generate_sms_transaction())
print("Sample Expense:", generator.generate_expense_record())
