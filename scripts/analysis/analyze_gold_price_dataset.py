#!/usr/bin/env python3
"""
Analyze Gold Price Prediction Dataset
Test if sid321axn/gold-price-prediction-dataset can enhance Smart Money AI's investment intelligence
"""

import kagglehub
import pandas as pd
import os
import json
from pathlib import Path

def analyze_gold_price_dataset():
    """Download and analyze the gold price prediction dataset"""
    
    print("🔍 ANALYZING NEW DATASET: sid321axn/gold-price-prediction-dataset")
    print("=" * 75)
    
    try:
        # Download the dataset
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("sid321axn/gold-price-prediction-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        
        # List all files in the dataset
        dataset_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                dataset_files.append(file_path)
        
        print(f"\n📁 Dataset contains {len(dataset_files)} files:")
        for file_path in dataset_files:
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   • {os.path.basename(file_path)} ({file_size:.1f} KB)")
        
        # Analyze each file
        analysis_results = {}
        
        for file_path in dataset_files:
            if file_path.endswith('.csv'):
                print(f"\n📊 ANALYZING: {os.path.basename(file_path)}")
                print("-" * 60)
                
                try:
                    # Load the CSV
                    df = pd.read_csv(file_path)
                    
                    file_analysis = {
                        'filename': os.path.basename(file_path),
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'sample_data': df.head(5).to_dict('records'),
                        'data_types': df.dtypes.astype(str).to_dict(),
                        'missing_values': df.isnull().sum().to_dict(),
                        'unique_values': {col: int(df[col].nunique()) for col in df.columns}
                    }
                    
                    # Basic info
                    print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    print(f"📝 Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                    
                    # Show sample data
                    print(f"\n🔍 Sample Data (first 5 rows):")
                    for i, row in df.head(3).iterrows():
                        print(f"   Row {i+1}: {dict(list(row.items())[:4])}{'...' if len(row) > 4 else ''}")
                    
                    # Data quality assessment
                    missing_count = df.isnull().sum().sum()
                    total_cells = df.shape[0] * df.shape[1]
                    completeness = ((total_cells - missing_count) / total_cells * 100) if total_cells > 0 else 0
                    
                    print(f"\n📈 Data Quality:")
                    print(f"   • Missing Values: {missing_count:,}")
                    print(f"   • Complete Records: {df.shape[0] - (missing_count // df.shape[1]):,}")
                    print(f"   • Data Completeness: {completeness:.1f}%")
                    
                    # Gold investment relevance analysis
                    gold_keywords = [
                        'gold', 'price', 'precious', 'metal', 'commodity',
                        'investment', 'portfolio', 'hedge', 'inflation',
                        'date', 'time', 'return', 'volatility', 'market'
                    ]
                    
                    relevance_score = 0
                    relevant_columns = []
                    price_indicators = []
                    time_indicators = []
                    
                    # Check column names for gold/investment terms
                    for col in df.columns:
                        col_lower = col.lower()
                        for keyword in gold_keywords:
                            if keyword in col_lower:
                                relevance_score += 12
                                relevant_columns.append(col)
                                if keyword in ['price', 'return', 'volatility']:
                                    price_indicators.append(col)
                                elif keyword in ['date', 'time']:
                                    time_indicators.append(col)
                                break
                    
                    # Check for numerical price data
                    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    potential_prices = []
                    
                    for col in numerical_cols:
                        if not df[col].empty:
                            # Check if values could be gold prices (typically $1000-$3000 range)
                            col_min, col_max = df[col].min(), df[col].max()
                            col_mean = df[col].mean()
                            
                            # Gold price characteristics
                            if 500 <= col_mean <= 5000 and col_max > col_min:
                                potential_prices.append(col)
                                relevance_score += 15
                            elif col_min >= 0 and col_max <= 100:  # Could be percentage returns
                                relevance_score += 8
                    
                    # Check for date/time columns
                    date_columns = df.select_dtypes(include=['object']).columns
                    for col in date_columns:
                        if not df[col].empty:
                            # Try to parse as date
                            try:
                                pd.to_datetime(df[col].head(10))
                                time_indicators.append(col)
                                relevance_score += 10
                            except:
                                pass
                    
                    # Bonus for time series data (essential for price prediction)
                    if len(time_indicators) >= 1 and len(potential_prices) >= 1:
                        relevance_score += 20
                    
                    # Bonus for sufficient historical data
                    if df.shape[0] > 1000:
                        relevance_score += 15
                    elif df.shape[0] > 500:
                        relevance_score += 10
                    
                    # Cap relevance score at 100
                    relevance_score = min(relevance_score, 100)
                    
                    file_analysis.update({
                        'relevance_score': relevance_score,
                        'relevant_columns': relevant_columns,
                        'price_indicators': price_indicators,
                        'time_indicators': time_indicators,
                        'potential_price_columns': list(potential_prices),
                        'has_time_series': len(time_indicators) >= 1 and len(potential_prices) >= 1
                    })
                    
                    print(f"\n🎯 Smart Money AI Relevance: {relevance_score}/100")
                    if relevant_columns:
                        print(f"💰 Gold/Investment Columns: {', '.join(relevant_columns[:5])}")
                    if price_indicators:
                        print(f"💎 Price Data Columns: {', '.join(price_indicators[:3])}")
                    if time_indicators:
                        print(f"📅 Time Series Columns: {', '.join(time_indicators[:3])}")
                    if potential_prices:
                        print(f"📊 Potential Gold Price Data: {', '.join(list(potential_prices)[:3])}")
                    
                    # Analyze date range if time series detected
                    if time_indicators and not df.empty:
                        time_col = time_indicators[0]
                        try:
                            dates = pd.to_datetime(df[time_col])
                            date_range = f"{dates.min().date()} to {dates.max().date()}"
                            years_coverage = (dates.max() - dates.min()).days / 365.25
                            print(f"📅 Date Coverage: {date_range} ({years_coverage:.1f} years)")
                            file_analysis['date_range'] = date_range
                            file_analysis['years_coverage'] = years_coverage
                        except:
                            pass
                    
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
        
        has_time_series = any(
            result.get('has_time_series', False) for result in analysis_results.values()
        )
        
        print(f"📊 Total Records: {total_records:,}")
        print(f"📁 Files Analyzed: {len(analysis_results)}")
        print(f"🎯 Average Relevance Score: {avg_relevance:.1f}/100")
        print(f"📈 Time Series Data: {'✅ Yes' if has_time_series else '❌ No'}")
        
        # Assess Smart Money AI integration potential for gold price data
        integration_assessment = assess_gold_price_integration(analysis_results, total_records, avg_relevance)
        
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
        print(f"   Previous Analysis: Investment Habits (423 records, 55/100 relevance) 🤔 Framework Use")
        print(f"   Previous Analysis: Stock Sentiment (87K records, 47.5/100 relevance) ❌ Deferred")
        print(f"   New Dataset: Gold Price Prediction ({total_records:,} records, {avg_relevance:.1f}/100 relevance)")
        
        # Save analysis results
        save_gold_price_analysis(analysis_results, integration_assessment, total_records, avg_relevance)
        
        return analysis_results, integration_assessment
        
    except Exception as e:
        print(f"❌ Error downloading or analyzing dataset: {e}")
        return None, None

def assess_gold_price_integration(analysis_results, total_records, avg_relevance):
    """Assess integration potential for gold price prediction dataset"""
    
    # Count gold-specific indicators
    total_price_indicators = sum(
        len(result.get('price_indicators', [])) + len(result.get('potential_price_columns', []))
        for result in analysis_results.values() 
        if 'price_indicators' in result
    )
    
    # Check for time series capability
    has_time_series = any(
        result.get('has_time_series', False) for result in analysis_results.values()
    )
    
    # Check data coverage
    max_years_coverage = max(
        (result.get('years_coverage', 0) for result in analysis_results.values()),
        default=0
    )
    
    integration_assessment = {
        'level': 'LOW',
        'recommendation': '',
        'next_steps': [],
        'price_score': total_price_indicators,
        'time_series': has_time_series,
        'years_coverage': max_years_coverage,
        'data_type': 'gold_price_prediction'
    }
    
    # Assessment criteria for gold price prediction datasets
    if avg_relevance >= 70 and total_records >= 1000 and has_time_series and max_years_coverage >= 3:
        integration_assessment['level'] = 'VERY HIGH'
        integration_assessment['recommendation'] = (
            f"Excellent integration potential! {total_records:,} records with {avg_relevance:.1f}/100 relevance, "
            f"{max_years_coverage:.1f} years of time series data. This dataset could add sophisticated gold price "
            "prediction capabilities to Smart Money AI's investment recommendations."
        )
        integration_assessment['next_steps'] = [
            "Build gold price prediction model using time series analysis",
            "Create gold investment timing recommendations",
            "Integrate gold price forecasts with portfolio allocation advice", 
            "Develop inflation hedge recommendations using gold predictions",
            "Add gold investment insights to behavioral investment profiling",
            "Create gold vs other asset allocation optimization"
        ]
    
    elif avg_relevance >= 55 and total_records >= 500 and has_time_series and max_years_coverage >= 1:
        integration_assessment['level'] = 'HIGH'
        integration_assessment['recommendation'] = (
            f"Strong integration potential! {total_records:,} records with {avg_relevance:.1f}/100 relevance "
            f"and {max_years_coverage:.1f} years of data. Gold price prediction could enhance Smart Money AI's "
            "investment intelligence for precious metals recommendations."
        )
        integration_assessment['next_steps'] = [
            "Develop basic gold price prediction capabilities",
            "Add gold investment recommendations to portfolio advice",
            "Create inflation protection guidance using gold data",
            "Integrate gold insights with existing investment engine",
            "Test gold prediction accuracy and user value"
        ]
    
    elif avg_relevance >= 40 and total_records >= 200:
        integration_assessment['level'] = 'MEDIUM'
        integration_assessment['recommendation'] = (
            f"Moderate integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "Could provide supplementary gold investment insights for portfolio diversification."
        )
        integration_assessment['next_steps'] = [
            "Focus on basic gold investment guidance",
            "Consider gold as alternative investment option in recommendations",
            "Use historical gold data for portfolio diversification advice"
        ]
    
    else:
        integration_assessment['level'] = 'LOW'
        integration_assessment['recommendation'] = (
            f"Limited integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "Gold price data may not provide sufficient value for Smart Money AI's core personal finance focus."
        )
        integration_assessment['next_steps'] = [
            "Focus on core personal finance and behavioral investment capabilities",
            "Consider gold predictions for future advanced investment features only"
        ]
    
    return integration_assessment

def save_gold_price_analysis(analysis_results, integration_assessment, total_records, avg_relevance):
    """Save gold price analysis results"""
    
    results = {
        'dataset_name': 'sid321axn/gold-price-prediction-dataset',
        'analysis_date': '2025-10-15',
        'dataset_type': 'gold_price_prediction',
        'total_records': total_records,
        'average_relevance': avg_relevance,
        'file_analyses': analysis_results,
        'integration_assessment': integration_assessment,
        'smart_money_ai_context': {
            'current_focus': 'Personal finance management with behavioral investment intelligence',
            'existing_datasets': {
                'personal_finance': {'records': 20000, 'relevance': 60, 'status': 'integrated'},
                'investment_behavioral': {'records': 100, 'relevance': 75, 'status': 'integrated'},
                'investment_habits': {'records': 423, 'relevance': 55, 'status': 'selective_framework_use'},
                'stock_sentiment': {'records': 87093, 'relevance': 47.5, 'status': 'deferred_for_future'}
            },
            'gold_investment_fit': {
                'core_relevance': 'Medium - adds alternative investment intelligence',
                'enhancement_potential': 'High - could improve portfolio diversification advice',
                'complexity_concern': 'Medium - requires time series prediction infrastructure',
                'user_relevance': 'Medium - gold investment is common in Indian market'
            }
        },
        'recommendation_summary': {
            'should_integrate': integration_assessment['level'] in ['HIGH', 'VERY HIGH'],
            'integration_type': 'gold_prediction_module' if integration_assessment['level'] in ['HIGH', 'VERY HIGH'] else 'consider_future',
            'focus_areas': ['gold price prediction', 'inflation hedging', 'portfolio diversification', 'precious metals allocation'],
            'expected_enhancement': 'Enhanced investment recommendations with gold price intelligence and inflation protection guidance'
        }
    }
    
    # Save to organized data directory
    os.makedirs('smart_money_ai/data/raw', exist_ok=True)
    
    with open('smart_money_ai/data/raw/gold_price_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: smart_money_ai/data/raw/gold_price_analysis.json")

if __name__ == "__main__":
    print("🎯 Smart Money AI - Gold Price Prediction Dataset Analyzer")
    print("Testing: sid321axn/gold-price-prediction-dataset")
    print("Focus: Gold price prediction for investment intelligence enhancement")
    print("Context: Gold investment is popular in Indian market for inflation protection")
    print()
    
    analysis, integration = analyze_gold_price_dataset()
    
    if analysis:
        print(f"\n🎉 Analysis complete! Check smart_money_ai/data/raw/gold_price_analysis.json for detailed results")
        
        # Show final recommendation
        if integration:
            level = integration['level']
            if level in ['HIGH', 'VERY HIGH']:
                print(f"\n🚀 RECOMMENDATION: PROCEED WITH GOLD PRICE PREDICTION INTEGRATION")
                print(f"💡 This dataset can add valuable gold investment intelligence to Smart Money AI!")
                print(f"🇮🇳 Especially relevant for Indian market where gold investment is culturally significant!")
            elif level == 'MEDIUM':
                print(f"\n🤔 RECOMMENDATION: CONSIDER GOLD AS SUPPLEMENTARY INVESTMENT OPTION")
                print(f"💡 Focus on basic gold investment guidance and portfolio diversification")
            else:
                print(f"\n❌ RECOMMENDATION: SKIP GOLD PRICE INTEGRATION") 
                print(f"💡 Focus on core personal finance capabilities")
    else:
        print(f"\n❌ Analysis failed. Please check the dataset name and try again.")