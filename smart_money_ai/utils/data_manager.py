#!/usr/bin/env python3
"""
Data Manager for Smart Money AI
===============================

Handles data storage, retrieval, and management for all Smart Money AI components
"""

import os
import json
import sqlite3
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class DataManager:
    """Data management and persistence for Smart Money AI"""
    
    def __init__(self, storage_path: str = "data/"):
        """Initialize data manager"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Database file
        self.db_path = self.storage_path / "smart_money.db"
        
        # Initialize database
        self.init_database()
        
        logger.info(f"Data manager initialized with storage path: {storage_path}")
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE,
                        phone TEXT,
                        age INTEGER,
                        annual_income REAL,
                        risk_profile TEXT,
                        investment_timeline INTEGER,
                        financial_goals TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Transactions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transactions (
                        transaction_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        amount REAL NOT NULL,
                        merchant TEXT,
                        category TEXT,
                        transaction_type TEXT,
                        bank_name TEXT,
                        account_number TEXT,
                        timestamp TIMESTAMP,
                        confidence REAL,
                        balance REAL,
                        upi_id TEXT,
                        reference_number TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Insights table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS insights (
                        insight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        insight_type TEXT,
                        insight_data TEXT,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Settings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_user(self, user_data: Dict[str, Any]) -> bool:
        """Save user profile to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert financial_goals list to JSON string
                financial_goals_json = json.dumps(user_data.get('financial_goals', []))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO users 
                    (user_id, name, email, phone, age, annual_income, risk_profile, 
                     investment_timeline, financial_goals, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_data['user_id'],
                    user_data['name'],
                    user_data['email'],
                    user_data['phone'],
                    user_data['age'],
                    user_data['annual_income'],
                    user_data['risk_profile'],
                    user_data['investment_timeline'],
                    financial_goals_json,
                    datetime.now()
                ))
                
                conn.commit()
                logger.info(f"User {user_data['user_id']} saved successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error saving user: {e}")
            return False
    
    def load_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user profile from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
                row = cursor.fetchone()
                
                if row:
                    user_data = dict(row)
                    # Convert financial_goals JSON back to list
                    user_data['financial_goals'] = json.loads(user_data['financial_goals'])
                    logger.info(f"User {user_id} loaded successfully")
                    return user_data
                else:
                    logger.warning(f"User {user_id} not found")
                    return None
                    
        except Exception as e:
            logger.error(f"Error loading user: {e}")
            return None
    
    def save_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Save transaction to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO transactions 
                    (transaction_id, user_id, amount, merchant, category, transaction_type,
                     bank_name, account_number, timestamp, confidence, balance, upi_id, reference_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction_data['transaction_id'],
                    transaction_data['user_id'],
                    transaction_data['amount'],
                    transaction_data['merchant'],
                    transaction_data.get('category'),
                    transaction_data['transaction_type'],
                    transaction_data['bank_name'],
                    transaction_data['account_number'],
                    transaction_data['timestamp'],
                    transaction_data.get('confidence'),
                    transaction_data.get('balance'),
                    transaction_data.get('upi_id'),
                    transaction_data.get('reference_number')
                ))
                
                conn.commit()
                logger.info(f"Transaction {transaction_data['transaction_id']} saved")
                return True
                
        except Exception as e:
            logger.error(f"Error saving transaction: {e}")
            return False
    
    def load_user_transactions(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load transactions for a specific user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = 'SELECT * FROM transactions WHERE user_id = ? ORDER BY timestamp DESC'
                params = [user_id]
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                transactions = [dict(row) for row in rows]
                logger.info(f"Loaded {len(transactions)} transactions for user {user_id}")
                return transactions
                
        except Exception as e:
            logger.error(f"Error loading transactions: {e}")
            return []
    
    def save_insights(self, user_id: str, insight_type: str, insight_data: Dict[str, Any]) -> bool:
        """Save user insights to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                insight_json = json.dumps(insight_data, default=str)
                
                cursor.execute('''
                    INSERT INTO insights (user_id, insight_type, insight_data)
                    VALUES (?, ?, ?)
                ''', (user_id, insight_type, insight_json))
                
                conn.commit()
                logger.info(f"Insights saved for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving insights: {e}")
            return False
    
    def load_latest_insights(self, user_id: str, insight_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load latest insights for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if insight_type:
                    cursor.execute('''
                        SELECT * FROM insights 
                        WHERE user_id = ? AND insight_type = ?
                        ORDER BY generated_at DESC LIMIT 1
                    ''', (user_id, insight_type))
                else:
                    cursor.execute('''
                        SELECT * FROM insights 
                        WHERE user_id = ?
                        ORDER BY generated_at DESC LIMIT 1
                    ''', (user_id,))
                
                row = cursor.fetchone()
                
                if row:
                    insight_data = json.loads(row['insight_data'])
                    return insight_data
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error loading insights: {e}")
            return None
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a specific user"""
        try:
            # Load user profile
            user_data = self.load_user(user_id)
            if not user_data:
                return {}
            
            # Load transactions
            transactions = self.load_user_transactions(user_id)
            
            # Load insights
            insights = self.load_latest_insights(user_id)
            
            export_data = {
                "user_profile": user_data,
                "transactions": transactions,
                "insights": insights,
                "export_timestamp": datetime.now().isoformat(),
                "total_transactions": len(transactions)
            }
            
            logger.info(f"Data exported for user {user_id}")
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting user data: {e}")
            return {}
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Create database backup"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.storage_path / f"backup_smart_money_{timestamp}.db"
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count users
                cursor.execute('SELECT COUNT(*) FROM users')
                user_count = cursor.fetchone()[0]
                
                # Count transactions
                cursor.execute('SELECT COUNT(*) FROM transactions')
                transaction_count = cursor.fetchone()[0]
                
                # Count insights
                cursor.execute('SELECT COUNT(*) FROM insights')
                insight_count = cursor.fetchone()[0]
                
                # Database size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                stats = {
                    "total_users": user_count,
                    "total_transactions": transaction_count,
                    "total_insights": insight_count,
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / (1024 * 1024), 2),
                    "database_path": str(self.db_path)
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 365) -> bool:
        """Clean up old data (older than specified days)"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old insights
                cursor.execute('DELETE FROM insights WHERE generated_at < ?', (cutoff_date,))
                
                # Note: We don't delete transactions or users as they're valuable historical data
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days} days")
                return True
                
        except Exception as e:
            logger.error(f"Error cleaning up data: {e}")
            return False
    
    def save_to_file(self, filename: str, data: Any, format: str = "json") -> bool:
        """Save data to file in specified format"""
        try:
            file_path = self.storage_path / filename
            
            if format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format.lower() == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return False
    
    def load_from_file(self, filename: str, format: str = "json") -> Optional[Any]:
        """Load data from file"""
        try:
            file_path = self.storage_path / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return None
            
            if format.lower() == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif format.lower() == "pickle":
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data loaded from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading from file: {e}")
            return None


def main():
    """Demo function"""
    print("💾 Data Manager Demo")
    print("=" * 30)
    
    # Initialize data manager
    data_manager = DataManager("demo_data/")
    
    # Test user data
    user_data = {
        "user_id": "demo_001",
        "name": "Demo User",
        "email": "demo@example.com",
        "phone": "+91-9876543210",
        "age": 30,
        "annual_income": 600000,
        "risk_profile": "moderate",
        "investment_timeline": 15,
        "financial_goals": ["wealth_creation", "retirement"]
    }
    
    # Save user
    success = data_manager.save_user(user_data)
    print(f"User saved: {success}")
    
    # Load user
    loaded_user = data_manager.load_user("demo_001")
    print(f"User loaded: {loaded_user['name'] if loaded_user else 'Failed'}")
    
    # Database stats
    stats = data_manager.get_database_stats()
    print(f"Database stats: {stats}")


if __name__ == "__main__":
    main()