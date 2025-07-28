# 🐷 Sow Feed Intake Trajectory Clustering and Forecasting System

A comprehensive Python implementation for precision livestock farming that combines **offline trajectory clustering** with **online forecasting** to predict sow feed intake patterns and optimize feeding strategies.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Detailed Workflow](#detailed-workflow)
- [Visualization Examples](#visualization-examples)
- [Performance Metrics](#performance-metrics)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🎯 Overview

This system implements a novel approach to sow feed intake prediction based on research in precision livestock farming. It identifies distinct feeding trajectory patterns through unsupervised clustering and uses these patterns to forecast future intake needs in real-time.

### The Problem
- Sow feed intake varies significantly during lactation
- Manual feeding adjustments are labor-intensive and imprecise
- Need for automated, data-driven feeding strategies

### The Solution
- **Offline Learning**: Identify feeding trajectory clusters from historical data
- **Online Forecasting**: Classify new sows early and predict their feeding needs
- **Precision Feeding**: Enable automated feeding systems with accurate predictions

## ✨ Key Features

### 🔍 Offline Learning Procedure
- **Time-series Clustering**: DTW-based clustering similar to kShape algorithm
- **Multiple Cluster Validation**: Silhouette and Calinski-Harabasz scores
- **Parity-aware Analysis**: Considers different sow parities (P1, P2, P3+)
- **Environmental Integration**: Temperature, humidity, and heat stress factors

### 🤖 Online Forecasting Procedure
- **Early Classification**: Predict trajectory cluster from days 2-5 data
- **Intake Forecasting**: Cluster-specific models for remaining lactation intake
- **Real-time Processing**: Minimal computational requirements for farm deployment
- **Uncertainty Quantification**: Probability estimates for cluster membership

### 📊 Advanced Analytics
- **Comprehensive Visualizations**: Trajectory plots, cluster analysis, performance metrics
- **Feature Importance Analysis**: Identify key predictive factors
- **Cross-validation**: Robust model evaluation and selection
- **Performance Monitoring**: Track prediction accuracy over time

## 🛠 Installation

### Prerequisites
- Python 3.7 or higher
- Minimum 4GB RAM (8GB recommended for large datasets)

### Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tslearn
```

### Optional Packages (for enhanced visualizations)
```bash
pip install plotly kaleido jupyter
```

### Installation from Source
```bash
git clone https://github.com/your-repo/sow-feed-forecasting.git
cd sow-feed-forecasting
pip install -r requirements.txt
```

## 📊 Data Requirements

### Required Columns
Your CSV file must contain the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `sow_id` | int/str | Unique sow identifier | 183553 |
| `lactation_day` | int | Day of lactation (2-22) | 5 |
| `daily_feed_intake` | float | Feed intake in kg | 4.63 |
| `Parity` | str | Sow parity group | P1, P2, P3+ |
| `temperature` | float | Daily temperature (°C) | 25.08 |
| `dew_point` | float | Dew point temperature (°C) | 17.79 |
| `Hours_T2M_GT_24` | int | Hours above 24°C | 6 |

### Data Quality Requirements
- **Minimum Records**: At least 200 sows for reliable clustering
- **Lactation Length**: Minimum 15 days per sow
- **Missing Data**: <10% missing values per sow trajectory
- **Time Range**: Consistent daily measurements

### Sample Data Format
```csv
sow_id,lactation_day,daily_feed_intake,Parity,temperature,dew_point,Hours_T2M_GT_24
183553,2,2.9,P1,25.08,17.79,6
183553,3,3.17,P1,28.42,22.85,9
183553,4,4.63,P1,25.34,21.69,9
```

## 🚀 Quick Start

### Basic Usage
```python
import pandas as pd
from sow_forecasting import SowFeedIntakeForecaster

# 1. Load your data
df = pd.read_csv('your_sow_data.csv')

# 2. Initialize the system
forecaster = SowFeedIntakeForecaster()

# 3. Prepare data
trajectory_matrix, sow_metadata = forecaster.load_and_prepare_data(df)

# 4. Perform offline clustering
cluster_assignments, cluster_scores = forecaster.offline_trajectory_clustering(trajectory_matrix)

# 5. Analyze clusters
forecaster.analyze_cluster_characteristics(trajectory_matrix, cluster_assignments)

# 6. Train early classifier
forecaster.train_early_classifier(df, cluster_assignments)

# 7. Train forecasting models
forecaster.forecast_remaining_intake(df, cluster_assignments)
```

### Predict for New Sow
```python
# Example: Predict for a new sow with early data
new_sow_data = {
    'feed_intake': [3.2, 3.8, 4.1, 4.5],  # Days 2-5
    'temperature': [25.1, 26.3, 24.8, 27.2],
    'dew_point': [18.5, 19.1, 17.8, 20.1],
    'hours_gt24': [6, 8, 5, 9],
    'parity': 'P1'
}

prediction = forecaster.predict_new_sow(new_sow_data)
print(f"Predicted cluster: {prediction['predicted_cluster']}")
print(f"Confidence: {max(prediction['cluster_probabilities'].values()):.2f}")
```

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA PROCESSING                      │
│  • Data validation and cleaning                            │
│  • Missing value imputation                                │  
│  • Environmental data integration                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               OFFLINE LEARNING PROCEDURE                    │
│                                                            │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │ Time-series      │  │ Cluster        │  │ Trajectory  │ │
│  │ Clustering       │  │ Validation     │  │ Analysis    │ │
│  │ (DTW-based)      │  │ (Silhouette,   │  │ & Profiling │ │
│  │                  │  │  Calinski-H)   │  │             │ │
│  └──────────────────┘  └────────────────┘  └─────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                ONLINE FORECASTING PROCEDURE                 │
│                                                            │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │ Early Cluster    │  │ Intake         │  │ Real-time   │ │
│  │ Classification   │  │ Forecasting    │  │ Prediction  │ │
│  │ (Days 2-5)       │  │ (Cluster-based)│  │ Pipeline    │ │
│  └──────────────────┘  └────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📈 Detailed Workflow

### Phase 1: Data Preparation
1. **Data Loading**: Read CSV file and validate required columns
2. **Quality Control**: Remove incomplete records and outliers
3. **Trajectory Matrix**: Create sow × lactation_day pivot table
4. **Metadata Extraction**: Calculate environmental averages per sow

### Phase 2: Offline Learning
1. **Time Series Preprocessing**: Scale and normalize trajectories
2. **Clustering Algorithm**: Apply DTW-based TimeSeriesKMeans
3. **Optimal Cluster Selection**: Evaluate 2-4 clusters using multiple metrics
4. **Cluster Characterization**: Analyze feeding patterns and sow characteristics

### Phase 3: Online Forecasting
1. **Feature Engineering**: Extract predictive features from early lactation
2. **Early Classification**: Train Random Forest for cluster prediction
3. **Intake Modeling**: Develop cluster-specific forecasting models
4. **Validation**: Cross-validate all models for reliability

### Phase 4: Deployment
1. **Real-time Classification**: Assign new sows to trajectory clusters
2. **Intake Prediction**: Forecast remaining lactation needs
3. **Confidence Assessment**: Provide uncertainty estimates
4. **Feeding Recommendations**: Generate actionable insights

## 📊 Visualization Examples

The system generates multiple types of visualizations to help understand the data and model performance:

### 1. Trajectory Cluster Visualization
```python
# Automatic generation during cluster analysis
forecaster.analyze_cluster_characteristics(trajectory_matrix, cluster_assignments)
```
**Shows**: 
- Individual sow trajectories (light lines)
- Cluster mean trajectories (bold lines)
- Feeding pattern differences between clusters

### 2. Cluster Statistics Dashboard
```python
# Generated automatically with cluster analysis
print("Cluster Statistics:")
for cluster in range(n_clusters):
    print(f"Cluster {cluster}:")
    print(f"  - Size: {cluster_sizes[cluster]} sows")
    print(f"  - Peak intake: {peak_intakes[cluster]:.2f} kg")
    print(f"  - Parity distribution: {parity_dist[cluster]}")
```

### 3. Model Performance Visualization
```python
# Classification performance
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix for early classifier
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Early Cluster Classification Accuracy')
plt.ylabel('True Cluster')
plt.xlabel('Predicted Cluster')
plt.show()
```

### 4. Feature Importance Analysis
```python
# Automatically generated during classifier training
print("Top Predictive Features:")
for feature, importance in feature_importance.items():
    print(f"  {feature}: {importance:.3f}")
```

### 5. Forecasting Performance
```python
# RMSE by cluster and forecast horizon
plt.figure(figsize=(12, 8))
for cluster in clusters:
    plt.plot(forecast_days, rmse_by_cluster[cluster], 
             label=f'Cluster {cluster}', marker='o')
plt.xlabel('Forecast Horizon (days)')
plt.ylabel('RMSE (kg)')
plt.title('Forecasting Accuracy by Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 📏 Performance Metrics

### Clustering Quality
- **Silhouette Score**: Measures cluster separation (higher = better)
  - Excellent: > 0.7
  - Good: 0.5 - 0.7
  - Acceptable: 0.25 - 0.5

- **Calinski-Harabasz Score**: Measures cluster compactness (higher = better)
  - Rule of thumb: > 100 for good clustering

### Classification Performance
- **Accuracy**: Percentage of correct cluster predictions
  - Target: > 80% for practical deployment
- **Precision/Recall**: Per-cluster performance metrics
- **Cross-validation Score**: Generalization capability

### Forecasting Performance
- **Mean Error (ME)**: Average prediction bias
  - Target: Close to 0 (unbiased predictions)
- **Root Mean Square Error (RMSE)**: Prediction accuracy
  - Target: < 1.5 kg for practical applications
- **Mean Absolute Percentage Error (MAPE)**: Relative accuracy

### Example Performance Report
```
CLUSTERING RESULTS:
✅ Optimal clusters: 2
✅ Silhouette score: 0.68 (Good)
✅ Calinski-Harabasz: 245.3 (Excellent)

CLASSIFICATION RESULTS:
✅ Early classifier accuracy: 0.847
✅ Cross-validation score: 0.823 ± 0.045

FORECASTING RESULTS:
✅ Mean Error: -0.08 kg/d (minimal bias)
✅ RMSE: 1.06 kg/d (excellent accuracy)
✅ MAPE: 12.3% (good relative accuracy)
```

## 🎛 Customization

### Adjusting Clustering Parameters
```python
# Try different numbers of clusters
cluster_assignments, scores = forecaster.offline_trajectory_clustering(
    trajectory_matrix, 
    n_clusters_range=[2, 3, 4, 5]
)

# Modify clustering algorithm parameters
forecaster.cluster_model = TimeSeriesKMeans(
    n_clusters=3,
    metric="dtw",
    max_iter=100,  # Increase for better convergence
    random_state=42
)
```

### Customizing Early Classification
```python
# Use different early days
early_classifier = forecaster.train_early_classifier(
    df, cluster_assignments, 
    early_days=[1, 2, 3, 4]  # Include day 1
)

# Try different classifiers
from sklearn.ensemble import GradientBoostingClassifier
forecaster.early_classifier = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)
```

### Adding Environmental Features
```python
# Include additional environmental variables
def extract_features(sow_data):
    features = [
        # Existing features...
        sow_data['humidity'].mean(),      # Add humidity
        sow_data['wind_speed'].mean(),    # Add wind speed
        sow_data['barometric_pressure'].mean()  # Add pressure
    ]
    return features
```

### Modifying Forecasting Horizon
```python
# Forecast from different day
forecasting_models = forecaster.forecast_remaining_intake(
    df, cluster_assignments, 
    forecast_day=7  # Earlier forecasting
)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Not enough data for clustering"
**Problem**: Dataset too small or too many missing values
**Solution**:
```python
# Check data quality
print(f"Total sows: {df['sow_id'].nunique()}")
print(f"Average days per sow: {df.groupby('sow_id').size().mean()}")

# Reduce minimum days requirement
valid_sows = trajectory_matrix.dropna(thresh=10, axis=0)  # Reduce from 15 to 10
```

#### 2. "Poor clustering quality (low silhouette score)"
**Problem**: Data doesn't naturally cluster well
**Solutions**:
```python
# Try different preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(trajectory_matrix.fillna(0))

# Try different distance metrics
cluster_model = TimeSeriesKMeans(n_clusters=2, metric="euclidean")
```

#### 3. "Low classification accuracy"
**Problem**: Early days don't predict cluster well
**Solutions**:
```python
# Use more early days
early_days = [1, 2, 3, 4, 5, 6]

# Add more features
# Include environmental variability, not just means
features.extend([
    early_data['temperature'].std(),  # Temperature variability
    early_data['dew_point'].max(),    # Peak humidity
])
```

#### 4. "High forecasting error"
**Problem**: Poor prediction accuracy
**Solutions**:
```python
# Use ensemble methods
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

ensemble = VotingRegressor([
    ('rf', RandomForestRegressor()),
    ('lr', LinearRegression())
])

# Include more historical context
# Use more days of history for forecasting
historical_days = 14  # Instead of 10
```

### Performance Optimization

#### For Large Datasets (>10,000 sows)
```python
# Use sampling for initial clustering
sample_size = 2000
sample_sows = trajectory_matrix.sample(n=sample_size, random_state=42)
cluster_assignments, _ = forecaster.offline_trajectory_clustering(sample_sows)

# Then assign all sows to clusters
all_assignments = forecaster.assign_to_existing_clusters(trajectory_matrix)
```

#### For Real-time Deployment
```python
# Precompute cluster centers
forecaster.save_model('cluster_model.pkl')

# Fast prediction pipeline
def fast_predict(early_data):
    features = extract_features_fast(early_data)
    cluster = forecaster.early_classifier.predict([features])[0]
    return cluster
```

## 📚 References

1. **Original Research**: Precision livestock farming approach to feed intake forecasting
2. **Time Series Clustering**: 
   - Paparrizos, J., & Gravano, L. (2015). k-Shape: Efficient and accurate clustering of time series. ACM SIGMOD Record.
   - Petitjean, F., et al. (2011). A global averaging method for dynamic time warping. Pattern recognition.

3. **Sow Lactation Research**:
   - Lopez, S., et al. (2000). A generalized Michaelis-Menten equation for the analysis of growth. Journal of Animal Science.
   - Young, M. G., et al. (2004). Comparison of three methods of feeding sows in gestation and the subsequent effects on lactation performance. Journal of Animal Science.

4. **Machine Learning Methods**:
   - Breiman, L. (2001). Random forests. Machine learning.
   - Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics.

---
