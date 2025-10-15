#!/usr/bin/env python3
"""
Analyze Stock Tweets Sentiment Dataset
Test if equinxx/stock-tweets-for-sentiment-analysis-and-prediction can enhance Smart Money AI
"""

import kagglehub
import pandas as pd
import os
import json
from pathlib import Path

def analyze_stock_tweets_dataset():
    """Download and analyze the stock tweets sentiment dataset"""
    
    print("🔍 ANALYZING NEW DATASET: equinxx/stock-tweets-for-sentiment-analysis-and-prediction")
    print("=" * 80)
    
    try:
        # Download the dataset
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("equinxx/stock-tweets-for-sentiment-analysis-and-prediction")
        print(f"✅ Dataset downloaded to: {path}")
        
        # List all files in the dataset
        dataset_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                dataset_files.append(file_path)
        
        print(f"\n📁 Dataset contains {len(dataset_files)} files:")
        for file_path in dataset_files:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   • {os.path.basename(file_path)} ({file_size:.1f} MB)")
        
        # Analyze each file
        analysis_results = {}
        
        for file_path in dataset_files:
            if file_path.endswith('.csv'):
                print(f"\n📊 ANALYZING: {os.path.basename(file_path)}")
                print("-" * 60)
                
                try:
                    # Load the CSV (with size limit for large files)
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    
                    if file_size_mb > 100:  # If file > 100MB, sample it
                        print(f"⚠️  Large file ({file_size_mb:.1f} MB) - sampling first 10,000 rows...")
                        df = pd.read_csv(file_path, nrows=10000)
                        sample_note = f" (sampled from {file_size_mb:.1f} MB file)"
                    else:
                        df = pd.read_csv(file_path)
                        sample_note = ""
                    
                    file_analysis = {
                        'filename': os.path.basename(file_path),
                        'file_size_mb': file_size_mb,
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'sample_data': df.head(3).to_dict('records'),
                        'data_types': df.dtypes.astype(str).to_dict(),
                        'missing_values': df.isnull().sum().to_dict(),
                        'unique_values': {col: int(df[col].nunique()) for col in df.columns},
                        'sampled': file_size_mb > 100
                    }
                    
                    # Basic info
                    print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns{sample_note}")
                    print(f"📝 Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                    
                    # Show sample data
                    print(f"\n🔍 Sample Data (first 3 rows):")
                    for i, row in df.head(3).iterrows():
                        print(f"   Row {i+1}: {dict(list(row.items())[:3])}{'...' if len(row) > 3 else ''}")
                    
                    # Data quality assessment
                    missing_count = df.isnull().sum().sum()
                    total_cells = df.shape[0] * df.shape[1]
                    completeness = ((total_cells - missing_count) / total_cells * 100) if total_cells > 0 else 0
                    
                    print(f"\n📈 Data Quality:")
                    print(f"   • Missing Values: {missing_count:,}")
                    print(f"   • Complete Records: {df.shape[0] - missing_count:,}")
                    print(f"   • Data Completeness: {completeness:.1f}%")
                    
                    # Smart Money AI relevance analysis for sentiment/investment data
                    sentiment_keywords = [
                        'sentiment', 'emotion', 'positive', 'negative', 'neutral',
                        'bullish', 'bearish', 'tweet', 'social', 'opinion',
                        'stock', 'market', 'price', 'prediction', 'forecast',
                        'invest', 'trading', 'financial', 'analysis', 'signal'
                    ]
                    
                    relevance_score = 0
                    relevant_columns = []
                    sentiment_indicators = []
                    stock_indicators = []
                    
                    # Check column names for sentiment/financial terms
                    for col in df.columns:
                        col_lower = col.lower()
                        for keyword in sentiment_keywords:
                            if keyword in col_lower:
                                relevance_score += 12
                                relevant_columns.append(col)
                                if keyword in ['sentiment', 'emotion', 'positive', 'negative', 'bullish', 'bearish']:
                                    sentiment_indicators.append(col)
                                elif keyword in ['stock', 'market', 'price', 'invest', 'trading']:
                                    stock_indicators.append(col)
                                break
                    
                    # Check for text data (potential tweets/sentiment text)
                    text_columns = df.select_dtypes(include=['object']).columns
                    tweet_like_columns = []
                    
                    for col in text_columns:
                        if not df[col].empty:
                            # Check if column contains tweet-like text
                            sample_values = df[col].dropna().head(10).astype(str)
                            avg_length = sample_values.str.len().mean()
                            
                            # If text length suggests tweets/posts (20-280 chars typically)
                            if 20 <= avg_length <= 500:
                                tweet_like_columns.append(col)
                                relevance_score += 15
                                
                                # Check content for financial terms
                                sample_text = ' '.join(sample_values.values).lower()
                                financial_matches = sum(1 for keyword in sentiment_keywords if keyword in sample_text)
                                relevance_score += financial_matches * 2
                    
                    # Check for numerical data that could be sentiment scores or stock prices
                    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    potential_scores = []
                    
                    for col in numerical_cols:
                        col_range = df[col].max() - df[col].min() if not df[col].empty else 0
                        
                        # If values are in typical sentiment score range (-1 to 1, 0 to 1, 1 to 5, etc.)
                        if 0 <= col_range <= 10:
                            potential_scores.append(col)
                            relevance_score += 8
                    
                    # Bonus for large dataset (more training data)
                    if df.shape[0] > 10000:
                        relevance_score += 15
                    elif df.shape[0] > 1000:
                        relevance_score += 10
                    
                    # Cap relevance score at 100
                    relevance_score = min(relevance_score, 100)
                    
                    file_analysis.update({
                        'relevance_score': relevance_score,
                        'relevant_columns': relevant_columns,
                        'sentiment_indicators': sentiment_indicators,
                        'stock_indicators': stock_indicators,
                        'tweet_like_columns': tweet_like_columns,
                        'potential_sentiment_scores': list(potential_scores),
                        'estimated_full_size': df.shape[0] if file_size_mb <= 100 else f"{df.shape[0]} (sampled from ~{int(file_size_mb * 100):,} estimated)"
                    })
                    
                    print(f"\n🎯 Smart Money AI Relevance: {relevance_score}/100")
                    if relevant_columns:
                        print(f"💰 Financial/Sentiment Columns: {', '.join(relevant_columns[:5])}")
                    if sentiment_indicators:
                        print(f"😊 Sentiment Indicators: {', '.join(sentiment_indicators[:3])}")
                    if stock_indicators:
                        print(f"📈 Stock Market Indicators: {', '.join(stock_indicators[:3])}")
                    if tweet_like_columns:
                        print(f"🐦 Tweet-like Text Columns: {', '.join(tweet_like_columns[:3])}")
                    if potential_scores:
                        print(f"📊 Potential Sentiment Scores: {', '.join(list(potential_scores)[:3])}")
                    
                    analysis_results[os.path.basename(file_path)] = file_analysis
                    
                except Exception as e:
                    print(f"❌ Error analyzing {file_path}: {e}")
                    analysis_results[os.path.basename(file_path)] = {
                        'error': str(e),
                        'relevance_score': 0
                    }
        
        # Overall dataset assessment
        print(f"\n🏆 OVERALL DATASET ASSESSMENT")
        print("=" * 60)
        
        total_records = sum(
            result.get('shape', [0])[0] for result in analysis_results.values() 
            if 'shape' in result
        )
        avg_relevance = sum(
            result.get('relevance_score', 0) for result in analysis_results.values()
        ) / len(analysis_results) if analysis_results else 0
        
        print(f"📊 Total Records: {total_records:,} (estimated)")
        print(f"📁 Files Analyzed: {len(analysis_results)}")
        print(f"🎯 Average Relevance Score: {avg_relevance:.1f}/100")
        
        # Assess Smart Money AI integration potential for sentiment data
        integration_assessment = assess_sentiment_integration(analysis_results, total_records, avg_relevance)
        
        print(f"\n🚀 INTEGRATION POTENTIAL: {integration_assessment['level']}")
        print(f"💡 {integration_assessment['recommendation']}")
        
        if integration_assessment['level'] in ['HIGH', 'VERY HIGH']:
            print(f"\n✅ RECOMMENDED INTEGRATION STEPS:")
            for i, step in enumerate(integration_assessment['next_steps'], 1):
                print(f"   {i}. {step}")
        elif integration_assessment['level'] == 'MEDIUM':
            print(f"\n🤔 POTENTIAL INTEGRATION STEPS:")
            for i, step in enumerate(integration_assessment['next_steps'], 1):
                print(f"   {i}. {step}")
        else:
            print(f"\n❌ INTEGRATION NOT RECOMMENDED")
        
        # Compare with existing datasets
        print(f"\n📊 COMPARISON WITH EXISTING SMART MONEY AI DATASETS:")
        print(f"   Current Dataset 1: Personal Finance (20,000 records, 60/100 relevance) ✅ Integrated")
        print(f"   Current Dataset 2: Investment Survey (100 records, 75/100 relevance) ✅ Integrated")
        print(f"   Previous Analysis: Investment Habits (423 records, 55/100 relevance) 🤔 Selective Use")
        print(f"   New Dataset: Stock Tweets Sentiment ({total_records:,} records, {avg_relevance:.1f}/100 relevance)")
        
        # Save analysis results
        save_sentiment_analysis(analysis_results, integration_assessment, total_records, avg_relevance)
        
        return analysis_results, integration_assessment
        
    except Exception as e:
        print(f"❌ Error downloading or analyzing dataset: {e}")
        return None, None

def assess_sentiment_integration(analysis_results, total_records, avg_relevance):
    """Assess integration potential for sentiment analysis dataset"""
    
    # Count sentiment-specific indicators
    total_sentiment_indicators = sum(
        len(result.get('sentiment_indicators', [])) + len(result.get('tweet_like_columns', []))
        for result in analysis_results.values() 
        if 'sentiment_indicators' in result
    )
    
    # Count stock market indicators
    total_stock_indicators = sum(
        len(result.get('stock_indicators', [])) 
        for result in analysis_results.values() 
        if 'stock_indicators' in result
    )
    
    integration_assessment = {
        'level': 'LOW',
        'recommendation': '',
        'next_steps': [],
        'sentiment_score': total_sentiment_indicators,
        'stock_score': total_stock_indicators,
        'data_type': 'sentiment_analysis'
    }
    
    # Assessment criteria for sentiment analysis datasets
    if avg_relevance >= 70 and total_records >= 10000 and total_sentiment_indicators >= 2:
        integration_assessment['level'] = 'VERY HIGH'
        integration_assessment['recommendation'] = (
            f"Excellent integration potential! {total_records:,} records with {avg_relevance:.1f}/100 relevance "
            f"and {total_sentiment_indicators} sentiment indicators. This dataset could add powerful sentiment analysis "
            "capabilities to Smart Money AI's investment recommendations."
        )
        integration_assessment['next_steps'] = [
            "Build sentiment analysis engine for investment insights",
            "Create market sentiment scoring system",
            "Integrate sentiment signals with existing behavioral profiling", 
            "Develop real-time market mood analysis",
            "Enhance investment recommendations with sentiment data",
            "Build social media sentiment monitoring for stocks"
        ]
    
    elif avg_relevance >= 50 and total_records >= 5000 and (total_sentiment_indicators >= 1 or total_stock_indicators >= 1):
        integration_assessment['level'] = 'HIGH'
        integration_assessment['recommendation'] = (
            f"Strong integration potential! {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            f"Sentiment analysis capabilities could enhance Smart Money AI's investment intelligence significantly."
        )
        integration_assessment['next_steps'] = [
            "Develop sentiment analysis module for investment insights",
            "Create market sentiment indicators",
            "Integrate with existing investment recommendation engine",
            "Build sentiment-based investment signals",
            "Test sentiment analysis impact on recommendation accuracy"
        ]
    
    elif avg_relevance >= 35 and total_records >= 1000:
        integration_assessment['level'] = 'MEDIUM'
        integration_assessment['recommendation'] = (
            f"Moderate integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "Could provide supplementary sentiment insights for investment analysis."
        )
        integration_assessment['next_steps'] = [
            "Focus on highest-quality sentiment data",
            "Consider sentiment as supplementary investment signal",
            "Analyze correlation between sentiment and investment outcomes"
        ]
    
    else:
        integration_assessment['level'] = 'LOW'
        integration_assessment['recommendation'] = (
            f"Limited integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "Sentiment data may not provide sufficient value for Smart Money AI's core personal finance focus."
        )
        integration_assessment['next_steps'] = [
            "Focus on core personal finance and behavioral datasets",
            "Consider sentiment analysis for future advanced features only"
        ]
    
    return integration_assessment

def save_sentiment_analysis(analysis_results, integration_assessment, total_records, avg_relevance):
    """Save sentiment analysis results"""
    
    results = {
        'dataset_name': 'equinxx/stock-tweets-for-sentiment-analysis-and-prediction',
        'analysis_date': '2025-10-15',
        'dataset_type': 'sentiment_analysis',
        'total_records': total_records,
        'average_relevance': avg_relevance,
        'file_analyses': analysis_results,
        'integration_assessment': integration_assessment,
        'smart_money_ai_context': {
            'current_focus': 'Personal finance management with demographic and behavioral intelligence',
            'existing_datasets': {
                'personal_finance': {'records': 20000, 'relevance': 60, 'status': 'integrated'},
                'investment_behavioral': {'records': 100, 'relevance': 75, 'status': 'integrated'},
                'investment_habits': {'records': 423, 'relevance': 55, 'status': 'selective_framework_use'}
            },
            'sentiment_analysis_fit': {
                'core_relevance': 'Medium - adds market intelligence layer',
                'enhancement_potential': 'High - could improve investment recommendations',
                'complexity_concern': 'High - requires NLP and sentiment processing infrastructure'
            }
        },
        'recommendation_summary': {
            'should_integrate': integration_assessment['level'] in ['HIGH', 'VERY HIGH'],
            'integration_type': 'sentiment_analysis_module' if integration_assessment['level'] in ['HIGH', 'VERY HIGH'] else 'not_recommended',
            'focus_areas': ['market sentiment', 'investment psychology', 'social trading signals'],
            'expected_enhancement': 'Enhanced investment recommendations with market sentiment intelligence'
        }
    }
    
    # Save to organized data directory
    os.makedirs('smart_money_ai/data/raw', exist_ok=True)
    
    with open('smart_money_ai/data/raw/sentiment_analysis_assessment.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: smart_money_ai/data/raw/sentiment_analysis_assessment.json")

if __name__ == "__main__":
    print("🎯 Smart Money AI - Stock Tweets Sentiment Analysis Dataset Analyzer")
    print("Testing: equinxx/stock-tweets-for-sentiment-analysis-and-prediction")
    print("Focus: Market sentiment analysis for investment intelligence enhancement")
    print()
    
    analysis, integration = analyze_stock_tweets_dataset()
    
    if analysis:
        print(f"\n🎉 Analysis complete! Check smart_money_ai/data/raw/sentiment_analysis_assessment.json for detailed results")
        
        # Show final recommendation
        if integration:
            level = integration['level']
            if level in ['HIGH', 'VERY HIGH']:
                print(f"\n🚀 RECOMMENDATION: PROCEED WITH SENTIMENT ANALYSIS INTEGRATION")
                print(f"💡 This dataset can add powerful market intelligence to Smart Money AI!")
            elif level == 'MEDIUM':
                print(f"\n🤔 RECOMMENDATION: CONSIDER SENTIMENT AS SUPPLEMENTARY FEATURE")
                print(f"💡 Focus on highest-quality sentiment indicators")
            else:
                print(f"\n❌ RECOMMENDATION: SKIP SENTIMENT INTEGRATION") 
                print(f"💡 Focus on core personal finance capabilities")
    else:
        print(f"\n❌ Analysis failed. Please check the dataset name and try again.")