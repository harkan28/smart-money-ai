#!/usr/bin/env python3
"""
Performance Optimization System - Backend Core
===============================================

Backend performance optimization with:
- Intelligent caching strategies
- Batch processing for large datasets
- Comprehensive error logging and monitoring
- Memory optimization
- Query optimization
"""

import os
import json
import time
import logging
import sqlite3
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Optional Redis import for advanced caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Cache storage types"""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance measurement data"""
    operation: str
    duration: float
    timestamp: datetime
    memory_usage: float
    cache_hit: bool
    success: bool
    error_message: Optional[str] = None


@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    average_response_time: float
    last_updated: datetime


class InMemoryCache:
    """High-performance in-memory cache with TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self.stats = CacheStats(0, 0, 0, 0.0, 0.0, datetime.now())
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            self.stats.total_requests += 1
            
            # Check if key exists and not expired
            if key in self.cache:
                if key in self.expiry_times and self.expiry_times[key] < datetime.now():
                    # Expired, remove
                    self._remove_key(key)
                    self.stats.cache_misses += 1
                    return None
                
                # Valid cache hit
                self.access_times[key] = datetime.now()
                self.stats.cache_hits += 1
                self._update_hit_rate()
                return self.cache[key]
            
            # Cache miss
            self.stats.cache_misses += 1
            self._update_hit_rate()
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        with self._lock:
            try:
                # Check if cache is full and evict if necessary
                if len(self.cache) >= self.max_size and key not in self.cache:
                    self._evict_lru()
                
                # Set value
                self.cache[key] = value
                self.access_times[key] = datetime.now()
                
                # Set expiry
                ttl = ttl or self.default_ttl
                self.expiry_times[key] = datetime.now() + timedelta(seconds=ttl)
                
                return True
                
            except Exception as e:
                logger.error(f"Cache set error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self.cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.expiry_times.clear()
    
    def _remove_key(self, key: str):
        """Remove key and associated metadata"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def _update_hit_rate(self):
        """Update cache hit rate"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
        else:
            self.stats.hit_rate = 0.0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            self.stats.last_updated = datetime.now()
            return self.stats


class RedisCache:
    """Redis-based cache for distributed caching"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.stats = CacheStats(0, 0, 0, 0.0, 0.0, datetime.now())
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host, port=port, db=db, 
                    decode_responses=True, socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                self.available = True
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.available = False
                self.redis_client = None
        else:
            self.available = False
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.available:
            return None
        
        try:
            self.stats.total_requests += 1
            
            value = self.redis_client.get(key)
            if value is not None:
                self.stats.cache_hits += 1
                self._update_hit_rate()
                return json.loads(value)
            else:
                self.stats.cache_misses += 1
                self._update_hit_rate()
                return None
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.cache_misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self.available:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, default=str)
            return self.redis_client.setex(key, ttl, serialized_value)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not self.available:
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self):
        """Clear Redis cache"""
        if not self.available:
            return
        
        try:
            self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def _update_hit_rate(self):
        """Update cache hit rate"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        self.stats.last_updated = datetime.now()
        return self.stats


class FileCache:
    """File-based cache for persistent storage"""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.stats = CacheStats(0, 0, 0, 0.0, 0.0, datetime.now())
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key"""
        # Sanitize key for filename
        safe_key = "".join(c for c in key if c.isalnum() or c in ('-', '_'))
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        try:
            self.stats.total_requests += 1
            file_path = self._get_file_path(key)
            
            if not os.path.exists(file_path):
                self.stats.cache_misses += 1
                self._update_hit_rate()
                return None
            
            # Check if file is expired
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if datetime.now() - file_time > timedelta(seconds=self.default_ttl):
                os.remove(file_path)
                self.stats.cache_misses += 1
                self._update_hit_rate()
                return None
            
            # Read file
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.stats.cache_hits += 1
                self._update_hit_rate()
                return data['value']
                
        except Exception as e:
            logger.error(f"File cache get error: {e}")
            self.stats.cache_misses += 1
            self._update_hit_rate()
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache"""
        try:
            file_path = self._get_file_path(key)
            
            cache_data = {
                'value': value,
                'created_at': datetime.now().isoformat(),
                'ttl': ttl or self.default_ttl
            }
            
            with open(file_path, 'w') as f:
                json.dump(cache_data, f, default=str, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"File cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from file cache"""
        try:
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
            
        except Exception as e:
            logger.error(f"File cache delete error: {e}")
            return False
    
    def clear(self):
        """Clear file cache"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except Exception as e:
            logger.error(f"File cache clear error: {e}")
    
    def _update_hit_rate(self):
        """Update cache hit rate"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        self.stats.last_updated = datetime.now()
        return self.stats


class CacheManager:
    """Unified cache manager supporting multiple cache types"""
    
    def __init__(self, primary_cache: CacheType = CacheType.MEMORY,
                 fallback_cache: Optional[CacheType] = CacheType.FILE):
        self.caches = {}
        self.primary_cache_type = primary_cache
        self.fallback_cache_type = fallback_cache
        
        # Initialize caches
        self._init_caches()
    
    def _init_caches(self):
        """Initialize cache instances"""
        # Memory cache
        self.caches[CacheType.MEMORY] = InMemoryCache()
        
        # Redis cache
        if REDIS_AVAILABLE:
            self.caches[CacheType.REDIS] = RedisCache()
        
        # File cache
        self.caches[CacheType.FILE] = FileCache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value using cache hierarchy"""
        # Try primary cache first
        primary_cache = self.caches.get(self.primary_cache_type)
        if primary_cache:
            value = primary_cache.get(key)
            if value is not None:
                return value
        
        # Try fallback cache
        if self.fallback_cache_type and self.fallback_cache_type != self.primary_cache_type:
            fallback_cache = self.caches.get(self.fallback_cache_type)
            if fallback_cache:
                value = fallback_cache.get(key)
                if value is not None:
                    # Store in primary cache for faster access
                    if primary_cache:
                        primary_cache.set(key, value)
                    return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all configured caches"""
        success = True
        
        # Set in primary cache
        primary_cache = self.caches.get(self.primary_cache_type)
        if primary_cache:
            success &= primary_cache.set(key, value, ttl)
        
        # Set in fallback cache
        if self.fallback_cache_type and self.fallback_cache_type != self.primary_cache_type:
            fallback_cache = self.caches.get(self.fallback_cache_type)
            if fallback_cache:
                success &= fallback_cache.set(key, value, ttl)
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete key from all caches"""
        success = True
        
        for cache in self.caches.values():
            if cache:
                success &= cache.delete(key)
        
        return success
    
    def clear(self):
        """Clear all caches"""
        for cache in self.caches.values():
            if cache:
                cache.clear()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches"""
        stats = {}
        for cache_type, cache in self.caches.items():
            if cache:
                stats[cache_type.value] = cache.get_stats()
        return stats


def cache_result(ttl: int = 3600, cache_manager: Optional[CacheManager] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Use provided cache manager or create default
            cm = cache_manager or CacheManager()
            
            # Try to get from cache
            cached_result = cm.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cm.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


class BatchProcessor:
    """Efficient batch processing for large datasets"""
    
    def __init__(self, batch_size: int = 1000, parallel_workers: int = 4):
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
    
    def process_in_batches(self, data: List[Any], 
                          processor_func: Callable[[List[Any]], List[Any]],
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Any]:
        """Process large dataset in batches"""
        results = []
        total_items = len(data)
        
        try:
            for i in range(0, total_items, self.batch_size):
                batch = data[i:i + self.batch_size]
                
                # Process batch
                batch_results = processor_func(batch)
                results.extend(batch_results)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + len(batch), total_items)
                
                # Small delay to prevent overwhelming system
                time.sleep(0.001)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return results
    
    def process_transactions_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Optimized transaction processing"""
        processed = []
        
        for txn in transactions:
            try:
                # Add processing metadata
                txn['processed_at'] = datetime.now().isoformat()
                txn['batch_id'] = f"batch_{int(time.time())}"
                
                # Validate required fields
                if 'amount' in txn and 'merchant' in txn:
                    processed.append(txn)
                    
            except Exception as e:
                logger.error(f"Transaction processing error: {e}")
                continue
        
        return processed


class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics = deque(maxlen=max_metrics)
        self.operation_stats = defaultdict(list)
        self._lock = threading.RLock()
    
    def record_metric(self, operation: str, duration: float, 
                     memory_usage: float = 0, cache_hit: bool = False,
                     success: bool = True, error_message: Optional[str] = None):
        """Record performance metric"""
        with self._lock:
            metric = PerformanceMetric(
                operation=operation,
                duration=duration,
                timestamp=datetime.now(),
                memory_usage=memory_usage,
                cache_hit=cache_hit,
                success=success,
                error_message=error_message
            )
            
            self.metrics.append(metric)
            self.operation_stats[operation].append(duration)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            if not self.metrics:
                return {'total_operations': 0}
            
            total_operations = len(self.metrics)
            successful_operations = sum(1 for m in self.metrics if m.success)
            cache_hits = sum(1 for m in self.metrics if m.cache_hit)
            
            durations = [m.duration for m in self.metrics]
            avg_duration = sum(durations) / len(durations)
            
            # Operation-specific stats
            operation_summary = {}
            for operation, times in self.operation_stats.items():
                operation_summary[operation] = {
                    'count': len(times),
                    'avg_duration': sum(times) / len(times),
                    'min_duration': min(times),
                    'max_duration': max(times)
                }
            
            return {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'success_rate': successful_operations / total_operations * 100,
                'cache_hit_rate': cache_hits / total_operations * 100,
                'average_duration': avg_duration,
                'operation_breakdown': operation_summary,
                'last_updated': datetime.now().isoformat()
            }


def performance_monitor(monitor: Optional[PerformanceMonitor] = None):
    """Decorator for monitoring function performance"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation_name = func.__name__
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Record metric
                if monitor:
                    monitor.record_metric(
                        operation=operation_name,
                        duration=duration,
                        success=success,
                        error_message=error_message
                    )
        return wrapper
    return decorator


class ErrorLogger:
    """Comprehensive error logging system"""
    
    def __init__(self, log_file: str = "logs/smart_money_errors.log",
                 max_log_size: int = 10 * 1024 * 1024):  # 10MB
        self.log_file = log_file
        self.max_log_size = max_log_size
        self.error_counts = defaultdict(int)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger('smart_money_errors')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with rotation
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_error(self, error: Exception, context: Dict[str, Any], 
                  level: LogLevel = LogLevel.ERROR):
        """Log error with context"""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1
        
        log_data = {
            'error_type': error_type,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'count': self.error_counts[error_type]
        }
        
        # Log based on level
        if level == LogLevel.CRITICAL:
            self.logger.critical(json.dumps(log_data))
        elif level == LogLevel.ERROR:
            self.logger.error(json.dumps(log_data))
        elif level == LogLevel.WARNING:
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
        
        # Rotate log if too large
        self._rotate_log_if_needed()
    
    def _rotate_log_if_needed(self):
        """Rotate log file if it gets too large"""
        try:
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > self.max_log_size:
                backup_file = f"{self.log_file}.backup"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                os.rename(self.log_file, backup_file)
        except Exception as e:
            print(f"Log rotation error: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_types': dict(self.error_counts),
            'most_common_error': max(self.error_counts.keys(), key=self.error_counts.get) if self.error_counts else None,
            'last_updated': datetime.now().isoformat()
        }


class PerformanceOptimizationSystem:
    """Main performance optimization system"""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.batch_processor = BatchProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.error_logger = ErrorLogger()
    
    @performance_monitor()
    @cache_result(ttl=1800)  # 30 minutes cache
    def optimize_transaction_analysis(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Optimized transaction analysis with caching and batching"""
        try:
            if not transactions:
                return {'total_transactions': 0, 'categories': {}}
            
            # Process in batches for large datasets
            if len(transactions) > 1000:
                processed_transactions = self.batch_processor.process_in_batches(
                    transactions, 
                    self.batch_processor.process_transactions_batch
                )
            else:
                processed_transactions = transactions
            
            # Analyze transactions
            analysis = {
                'total_transactions': len(processed_transactions),
                'total_amount': sum(float(txn.get('amount', 0)) for txn in processed_transactions),
                'categories': self._analyze_categories(processed_transactions),
                'time_analysis': self._analyze_temporal_patterns(processed_transactions),
                'optimization_applied': True,
                'processed_at': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.error_logger.log_error(e, {
                'operation': 'optimize_transaction_analysis',
                'transaction_count': len(transactions)
            })
            raise
    
    def _analyze_categories(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Optimized category analysis"""
        categories = defaultdict(float)
        
        for txn in transactions:
            category = txn.get('category', 'MISCELLANEOUS')
            amount = float(txn.get('amount', 0))
            categories[category] += amount
        
        return dict(categories)
    
    def _analyze_temporal_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Optimized temporal pattern analysis"""
        daily_spending = defaultdict(float)
        
        for txn in transactions:
            try:
                timestamp = datetime.fromisoformat(txn.get('timestamp', '2024-01-01'))
                date_key = timestamp.date().isoformat()
                amount = float(txn.get('amount', 0))
                daily_spending[date_key] += amount
            except:
                continue
        
        amounts = list(daily_spending.values())
        return {
            'daily_average': sum(amounts) / len(amounts) if amounts else 0,
            'peak_day': max(daily_spending.keys(), key=daily_spending.get) if daily_spending else None,
            'spending_days': len(daily_spending)
        }
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        try:
            return {
                'cache_statistics': self.cache_manager.get_stats(),
                'performance_metrics': self.performance_monitor.get_performance_summary(),
                'error_summary': self.error_logger.get_error_summary(),
                'system_health': self._assess_system_health(),
                'optimization_recommendations': self._generate_optimization_recommendations(),
                'report_generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.error_logger.log_error(e, {'operation': 'get_system_performance_report'})
            return {'error': str(e)}
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        cache_stats = self.cache_manager.get_stats()
        perf_summary = self.performance_monitor.get_performance_summary()
        
        # Calculate health scores
        cache_health = 'good' if cache_stats.get('memory', CacheStats(0,0,0,0.5,0,datetime.now())).hit_rate > 0.7 else 'needs_improvement'
        performance_health = 'good' if perf_summary.get('success_rate', 0) > 95 else 'needs_improvement'
        
        return {
            'overall_health': 'good' if cache_health == 'good' and performance_health == 'good' else 'needs_improvement',
            'cache_health': cache_health,
            'performance_health': performance_health,
            'redis_available': REDIS_AVAILABLE,
            'uptime_status': 'operational'
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        cache_stats = self.cache_manager.get_stats()
        memory_cache_stats = cache_stats.get('memory')
        
        if memory_cache_stats and memory_cache_stats.hit_rate < 0.5:
            recommendations.append("Consider increasing cache TTL for better hit rates")
        
        if not REDIS_AVAILABLE:
            recommendations.append("Install Redis for improved distributed caching")
        
        perf_summary = self.performance_monitor.get_performance_summary()
        if perf_summary.get('average_duration', 0) > 1.0:
            recommendations.append("Consider database query optimization for faster response times")
        
        if not recommendations:
            recommendations.append("System is performing optimally")
        
        return recommendations


def main():
    """Demo the performance optimization system"""
    print("⚡ Performance Optimization System - Backend Demo")
    print("=" * 70)
    
    # Initialize system
    perf_system = PerformanceOptimizationSystem()
    
    # Generate sample data for testing
    sample_transactions = []
    for i in range(2500):  # Large dataset to test batching
        sample_transactions.append({
            'id': i,
            'amount': 100 + (i % 1000),
            'merchant': f'MERCHANT_{i % 50}',
            'category': ['FOOD_DINING', 'TRANSPORTATION', 'SHOPPING', 'UTILITIES'][i % 4],
            'timestamp': (datetime.now() - timedelta(days=i % 90)).isoformat()
        })
    
    print(f"\n📊 Processing {len(sample_transactions)} transactions...")
    print("-" * 60)
    
    # Test optimized analysis (will use caching and batching)
    start_time = time.time()
    
    # First run (cache miss)
    analysis1 = perf_system.optimize_transaction_analysis(sample_transactions)
    first_run_time = time.time() - start_time
    
    start_time = time.time()
    
    # Second run (cache hit)
    analysis2 = perf_system.optimize_transaction_analysis(sample_transactions)
    second_run_time = time.time() - start_time
    
    print(f"✅ Analysis completed!")
    print(f"   Total Transactions: {analysis1['total_transactions']:,}")
    print(f"   Total Amount: ₹{analysis1['total_amount']:,.2f}")
    print(f"   Categories: {len(analysis1['categories'])}")
    print(f"   First Run Time: {first_run_time:.3f}s")
    print(f"   Second Run Time: {second_run_time:.3f}s (cache hit)")
    print(f"   Performance Improvement: {((first_run_time - second_run_time) / first_run_time * 100):.1f}%")
    
    # Test cache statistics
    print(f"\n💾 Cache Performance:")
    print("-" * 30)
    cache_stats = perf_system.cache_manager.get_stats()
    
    for cache_type, stats in cache_stats.items():
        if stats.total_requests > 0:
            print(f"   {cache_type.title()} Cache:")
            print(f"     Hit Rate: {stats.hit_rate:.1%}")
            print(f"     Total Requests: {stats.total_requests}")
            print(f"     Cache Hits: {stats.cache_hits}")
    
    # Generate performance report
    print(f"\n📈 System Performance Report:")
    print("-" * 40)
    
    report = perf_system.get_system_performance_report()
    
    if 'system_health' in report:
        health = report['system_health']
        print(f"   Overall Health: {health['overall_health'].upper()}")
        print(f"   Cache Health: {health['cache_health']}")
        print(f"   Performance Health: {health['performance_health']}")
        print(f"   Redis Available: {health['redis_available']}")
    
    if 'performance_metrics' in report:
        metrics = report['performance_metrics']
        print(f"   Total Operations: {metrics.get('total_operations', 0)}")
        print(f"   Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"   Average Duration: {metrics.get('average_duration', 0):.3f}s")
    
    # Optimization recommendations
    if 'optimization_recommendations' in report:
        recommendations = report['optimization_recommendations']
        print(f"\n💡 Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    print(f"\n✅ Performance optimization system demo completed!")
    print(f"   System is optimized for production workloads")
    print(f"   Caching reduces response times by up to {((first_run_time - second_run_time) / first_run_time * 100):.0f}%")
    print(f"   Batch processing handles large datasets efficiently")


if __name__ == "__main__":
    main()