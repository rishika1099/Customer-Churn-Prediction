# 🍔 Customer Polarity Classification in Online Food Delivery

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project analyzing customer purchase decisions in the online food delivery industry using multiple classification algorithms.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models Implemented](#-models-implemented)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Technical Details](#-technical-details)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#-authors)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project predicts whether customers will make repeat purchases on online food delivery platforms by analyzing demographic data, ordering patterns, and service quality factors. The analysis helps food delivery businesses:

- 🎯 **Identify high-value customers** likely to order again
- 📊 **Understand key factors** influencing purchase decisions
- 🔍 **Optimize marketing strategies** based on customer segments
- 💡 **Improve service quality** in areas that matter most

### Business Impact

- **Customer Retention**: Predict and prevent customer churn
- **Targeted Marketing**: Focus resources on high-probability repeat customers
- **Service Optimization**: Identify and fix pain points in the ordering process
- **Revenue Growth**: Increase repeat orders through data-driven insights

---

## 📊 Dataset

### Source
- **Platform**: Kaggle
- **Region**: Bangalore, India
- **Size**: 388 customer records
- **Features**: 55 variables (reduced to 50 after preprocessing)

### Data Characteristics

**Demographics** (10 variables):
- Age, Gender, Marital Status
- Occupation, Income, Education
- Location (Latitude, Longitude, Pin Code)

**Service Quality Indicators** (25 variables):
- Delivery time, Order accuracy
- Food quality, Tracking system
- Customer support quality
- Payment convenience

**Ordering Preferences** (10 variables):
- Preferred medium (App, Web, Call)
- Meal types, Cuisine preferences
- Ordering frequency

**Rating Factors** (8 variables):
- Restaurant ratings
- Delivery person ratings
- Overall experience ratings

**Target Variable**:
- `Output`: Binary (Yes/No) - Will customer order again?

### Data Split
- **Training Set**: 80% (310 samples)
- **Test Set**: 20% (78 samples)

---

## 🤖 Models Implemented

### 1. **Logistic Regression**
- **Purpose**: Baseline linear model
- **Accuracy**: ~91%
- **Strengths**: Fast, interpretable, good for binary classification

### 2. **Naive Bayes**
- **Type**: Gaussian Naive Bayes
- **Accuracy**: ~36%
- **Note**: Struggles with feature dependencies

### 3. **K-Nearest Neighbors (sklearn)**
- **K value**: 8
- **Accuracy**: ~95%
- **Distance Metric**: Euclidean

### 4. **Enhanced KNN** ⭐ (Best Performer)
- **Type**: Inverse Distance Weighted KNN
- **Accuracy**: **96.15%**
- **Key Features**:
  - Feature scaling with StandardScaler
  - Squared inverse distance weighting (1/d²)
  - Grid search optimization
  - Custom implementation with weighted voting
- **Why Best**: Combines distance-based learning with proper feature normalization

### 5. **Support Vector Machine**
- **Kernel**: RBF
- **Accuracy**: ~94%
- **C Parameter**: 5 (from grid search)

### 6. **Artificial Neural Network**
- **Architecture**: Deep neural network
- **Accuracy**: ~88%
- **Layers**: 128 → 64 → 32 → 16 → 1
- **Regularization**: Batch normalization, Dropout
- **Optimization**: Adam, Early stopping, Learning rate scheduling

---

## ✨ Key Features

### 🔥 Advanced Optimizations

1. **Feature Scaling**
   - StandardScaler normalization
   - Critical for distance-based algorithms
   - Improved KNN accuracy by 10-12%

2. **Enhanced KNN Implementation**
   - Custom weighted voting algorithm
   - Inverse distance weighting: closer neighbors have more influence
   - Grid search for optimal hyperparameters

3. **Deep Learning Architecture**
   - Batch normalization for stable training
   - Dropout layers to prevent overfitting
   - Early stopping and learning rate scheduling
   - Multiple hidden layers for complex pattern recognition

4. **Comprehensive Evaluation**
   - Confusion matrices for all models
   - Precision, Recall, F1-Score metrics
   - Side-by-side model comparison
   - Detailed classification reports

### 📈 Data Visualizations

- **Correlation Heatmap**: Feature relationships
- **Nested Pie Charts**: Demographic distributions vs Output
- **Geographical Maps**: Customer distribution across Bangalore
- **Feature Analysis**: Key factors influencing decisions
- **Performance Comparison**: Bar charts of model accuracies

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or Google Colab

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/customer-polarity-classification.git
cd customer-polarity-classification
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Download from [Kaggle](https://www.kaggle.com/benroshan/online-food-delivery-preferencesbangalore-region)
2. Place `onlinedeliverydata.xls` in the project root directory

---

## 💻 Usage

### Quick Start

#### Option 1: Jupyter Notebook
```bash
jupyter notebook Customer_Polarity_Classification.ipynb
```

#### Option 2: Google Colab
1. Upload notebook to Google Drive
2. Open with Google Colab
3. Upload dataset when prompted
4. Run all cells: `Runtime` → `Run all`

### Running the Analysis

1. **Load and Explore Data**
   ```python
   data = pd.read_csv('onlinedeliverydata.xls')
   print(data.head())
   print(data.info())
   ```

2. **Preprocess Data**
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features
   - Split train/test sets

3. **Train Models**
   ```python
   # Example: Enhanced KNN
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # Train and evaluate
   predictions = enhanced_knn.predict(X_test_scaled)
   ```

4. **Evaluate Performance**
   - View confusion matrices
   - Check classification reports
   - Compare model accuracies

### Customization

**Change KNN parameters:**
```python
k_values = [5, 7, 8, 10, 12, 15]
weighting_schemes = ['inverse', 'squared']
# Grid search will find best combination
```

**Modify ANN architecture:**
```python
model = keras.Sequential([
    layers.Dense(256, activation='relu'),  # Change layer sizes
    layers.BatchNormalization(),
    layers.Dropout(0.4),  # Adjust dropout rate
    # Add more layers as needed
])
```

---

## 📁 Project Structure

```
customer-polarity-classification/
│
├── Customer_Polarity_Classification.ipynb   # Main Jupyter notebook
├── README.md                                 # This file
├── requirements.txt                          # Python dependencies
├── LICENSE                                   # MIT License
│
├── data/
│   └── onlinedeliverydata.xls               # Dataset
│
├── models/                                   # Saved model files (optional)
│   ├── enhanced_knn_model.pkl
│   ├── svm_model.pkl
│   └── ann_model.h5
│
├── visualizations/                           # Generated plots
│   ├── correlation_heatmap.png
│   ├── nested_pie_charts.png
│   ├── customer_distribution_map.html
│   └── model_comparison.png
│
└── docs/                                     # Additional documentation
    ├── METHODOLOGY.md
    ├── RESULTS.md
    └── API_REFERENCE.md
```

---

## 🏆 Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Enhanced KNN** ⭐ | **96.15%** | **0.96** | **0.98** | **0.97** | ~3 sec |
| KNN (sklearn) | 94.87% | 0.96 | 0.98 | 0.97 | <1 sec |
| SVM | 93.59% | 0.94 | 0.98 | 0.96 | ~2 sec |
| Logistic Regression | 91.03% | 0.94 | 0.98 | 0.96 | <1 sec |
| ANN | 88.46% | 0.93 | 0.95 | 0.94 | ~30 sec |
| Naive Bayes | 35.90% | 0.89 | 0.67 | 0.76 | <1 sec |

### Key Insights

✅ **Best Model**: Enhanced KNN with feature scaling (96.15%)
- Feature scaling improved accuracy by 10-12%
- Inverse distance weighting provides better predictions
- Optimal k=8-12 neighbors

📊 **Important Features**:
1. **Self Cooking** (correlation: 0.31)
2. **Age** (correlation: -0.28)
3. **Occupation** (correlation: 0.28)
4. **Marital Status** (correlation: 0.28)
5. **Maximum Wait Time** (correlation: 0.27)

💡 **Customer Patterns**:
- Younger customers (18-30) order more frequently
- Employed professionals and students are primary users
- Web browser is the most popular ordering medium
- Dinner is the most common meal time
- Quick delivery (<20 min) significantly impacts repeat orders

---

## 📊 Visualizations

### 1. Correlation Heatmap
Shows relationships between features and target variable. Key correlations identified for feature selection.

### 2. Nested Pie Charts
Displays demographic distributions with inner (No) and outer (Yes) rings showing purchase intent:
- Gender distribution
- Marital status
- Occupation types
- Income levels
- Preferred ordering mediums
- Meal preferences

### 3. Geographical Distribution Map
Interactive map showing customer locations across Bangalore with:
- Customer density by pin code
- Clickable markers with details
- Service coverage visualization

### 4. Model Comparison Chart
Bar chart comparing all model accuracies with:
- 95% target line
- Color-coded bars (green for top performers)
- Accuracy percentages labeled on bars

---

## 🔧 Technical Details

### Data Preprocessing Pipeline

1. **Missing Value Handling**
   - No missing values found in dataset
   - Ready for immediate processing

2. **Feature Engineering**
   - Label encoding for categorical variables
   - StandardScaler for numerical features
   - Feature selection based on correlation (threshold: 0.4)

3. **Dimensionality Reduction**
   - Dropped irrelevant columns (latitude, longitude, pin code for modeling)
   - Kept Educational Qualifications initially, removed before final models
   - Final feature count: 49 features

### Enhanced KNN Algorithm

```python
def enhanced_knn_predict(X_train, X_test, y_train, k):
    predictions = []
    for test_point in X_test:
        # Calculate distances
        distances = [euclidean_distance(test_point, train_point) 
                    for train_point in X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        
        # Inverse distance weighting (squared)
        weights = [1 / (distances[i]**2 + 1e-5) for i in k_indices]
        
        # Weighted voting
        weighted_votes = {}
        for idx, weight in zip(k_indices, weights):
            label = y_train[idx]
            weighted_votes[label] = weighted_votes.get(label, 0) + weight
        
        # Predict class with highest weighted vote
        prediction = max(weighted_votes, key=weighted_votes.get)
        predictions.append(prediction)
    
    return predictions
```

### Deep Learning Architecture

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_dim=n_features),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(1, activation='sigmoid')
])
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Epochs: 100 (with early stopping)
- Batch Size: 32
- Validation Split: 20%

**Callbacks:**
- EarlyStopping (patience=15, restore best weights)
- ReduceLROnPlateau (patience=5, factor=0.5)

---

## 📦 Requirements

### Core Libraries
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tensorflow>=2.6.0
```

### Additional Libraries
```
folium>=0.12.0          # For geographical maps
jupyter>=1.0.0          # For notebook interface
ipykernel>=6.0.0       # Jupyter kernel
```
