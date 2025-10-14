# 🎯 Smart Money - ML Expense Categorization

**Lightweight Machine Learning system for automatic expense categorization**

A clean, production-ready machine learning system that automatically categorizes financial transactions into budget categories using advanced NLP and ML techniques.

## 🌟 Features

- **Automatic Expense Categorization**: Uses advanced NLP and ML to categorize transactions
- **High Accuracy**: >90% accuracy on common transaction types  
- **Real-time Predictions**: Fast inference for real-time transaction processing
- **Confidence Scores**: Each prediction includes confidence levels
- **Incremental Learning**: Improves from user feedback
- **Production Ready**: Clean, modular codebase

## 📋 Supported Categories

- **FOOD_DINING**: Restaurants, food delivery, cafes, groceries
- **TRANSPORTATION**: Cabs, fuel, public transport, parking  
- **SHOPPING**: Online/offline retail, clothing, electronics
- **ENTERTAINMENT**: Streaming services, movies, games
- **UTILITIES**: Phone bills, electricity, internet, gas
- **HEALTHCARE**: Medical consultations, medicines, hospitals
- **EDUCATION**: Courses, books, tuition, certifications
- **INVESTMENT**: Stocks, mutual funds, FDs, insurance
- **MISCELLANEOUS**: ATM withdrawals, taxes, other expenses

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Train the Model

```bash
# Train the ML model with generated data
python train_model.py
```

### 3. Run Demo

```bash
# See the model in action
python demo.py
```



## �️ Project Structure

```
budgeting_ml_model/
├── src/
│   ├── preprocessor.py      # Data cleaning and preprocessing
│   ├── feature_extractor.py # TF-IDF vectorization and feature engineering
│   ├── model.py            # ML model training and evaluation
│   ├── inference.py        # Prediction pipeline
│   └── incremental_learning.py # User feedback system
├── models/                 # Saved trained models (generated after training)
├── data/                   # Training data (generated during training)
├── tests/                  # Test suite
├── train_model.py          # Main training script
├── demo.py                 # Demonstration script
└── requirements.txt        # Dependencies
```

## 🎮 Usage Examples

### Basic Prediction

```python
from src.inference import ExpenseCategorizer

# Initialize categorizer (after training)
categorizer = ExpenseCategorizer(
    model_path="models/expense_category_model.joblib",
    feature_extractor_path="models/feature_extractor.joblib"
)

# Categorize expense
result = categorizer.categorize_expense(
    merchant="Zomato",
    description="Food delivery order",
    amount=450
)

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Incremental Learning

```python
from src.incremental_learning import IncrementalLearner

# Initialize learner
learner = IncrementalLearner()

# Record user feedback
learner.record_prediction(
    merchant="Zomato", 
    description="Food order", 
    amount=450,
    predicted_category="FOOD_DINING",
    confidence=0.68,
    user_confirmed=True
)

# Trigger learning when ready
if learner.should_retrain():
    learner.trigger_incremental_learning()
```

## 🔧 Technical Details

### Preprocessing Pipeline
- Text cleaning and normalization
- Banking term removal (UPI, INR, etc.)
- Merchant name standardization  
- Keyword extraction using NLP
- Feature engineering for amounts and metadata

### Feature Extraction
- **TF-IDF Vectorization**: Convert text to numerical features
- **N-gram Analysis**: Capture phrase patterns (unigrams + bigrams)
- **Numerical Features**: Amount, length, keyword counts
- **Boolean Features**: Category-specific keyword presence
- **Categorical Features**: Amount ranges, merchant types
### Machine Learning Models
- **Random Forest**: Default choice for reliability and interpretability
- **XGBoost**: High performance gradient boosting  
- **LightGBM**: Fast and efficient for large datasets

### Model Selection
- Automatic hyperparameter tuning using GridSearchCV
- Cross-validation with stratified k-fold
- Model comparison based on weighted F1-score
- Class imbalance handling with balanced weights

## 📊 Expected Performance

- **Accuracy**: >90% on well-formed transaction data
- **F1-Score**: >88% across all categories
- **High-confidence predictions**: >95% accuracy
- **Real-time Processing**: <1 second per transaction

## 🔮 Usage Scenarios

### Banking Integration
```python
# Process bank SMS notifications
sms_text = "UPI Payment to ZOMATO for Rs.450"
result = categorizer.categorize_expense("ZOMATO", sms_text, 450)
```

### Batch Processing
```python
# Process multiple transactions
transactions = [
    {"merchant": "NETFLIX.COM", "description": "Subscription", "amount": 650},
    {"merchant": "SHELL PETROL", "description": "Fuel", "amount": 3000}
]
results = categorizer.predict_batch_transactions(transactions)
```

## 🛠️ Customization

### Adding New Categories
1. Update `preprocessor.py` categories list
2. Add category-specific keywords  
3. Include sample data for the new category
4. Retrain the model

### Improving Accuracy
1. **More Training Data**: Add your historical transactions
2. **Domain-Specific Terms**: Update merchant mappings
3. **Feature Engineering**: Add new features based on your data
4. **Model Tuning**: Adjust hyperparameters for your use case

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules  
python tests/test_model.py
```

## 🎯 Getting Started

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Train Model**: `python train_model.py`
3. **Run Demo**: `python demo.py`
4. **Integrate**: Use the inference API in your application

---

**Built with ❤️ for Smart Money - Making Personal Finance Intelligent**