# Sow Feed Intake Prediction Pipeline - Detailed README

## Overview

This R script creates a comprehensive machine learning pipeline to predict daily feed intake for sows (female pigs) during lactation. The system compares three different modeling approaches to find the best method for predicting how much feed a sow will consume on any given day.

## What This Code Does

**Main Purpose**: Predict how much feed sows will eat tomorrow based on their current conditions and historical data.

**Business Value**: Helps farmers optimize feed management, reduce waste, and ensure proper nutrition for lactating sows.

## Required Data Structure

Your input CSV file (`sow_data.csv`) should contain these columns:

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| `sow_id` | Unique identifier for each sow | 1, 2, 3, etc. |
| `parity` | Number of times the sow has given birth | 1, 2, "P3+" |
| `lactation_day` | Day number in current lactation period | 1-22 |
| `daily_feed_intake` | Amount of feed consumed (kg/day) | 5.2, 6.8, 4.9 |
| `temperature` | Daily temperature (°C) | 18.5, 22.3, 15.7 |
| `dew_point` | Dew point temperature (°C) | 12.1, 15.8, 8.9 |
| `Hours_T2M_GT_24` | Hours when temperature > 24°C | 0, 3, 8 |

## Code Structure Breakdown

### 1. Library Loading and Setup
```r
library(tidyverse)    # Data manipulation and visualization
library(randomForest) # Random Forest machine learning
library(caret)        # Machine learning toolkit
library(xgboost)      # Gradient boosting algorithm
# ... other libraries
```

**What it does**: Loads all the necessary tools for data processing, machine learning, and visualization.

### 2. Data Preparation (`prepare_data` function)

**Purpose**: Transforms raw data into features that machine learning models can use effectively.

**Key Transformations**:

1. **Polynomial Terms**: Creates mathematical curves (t1, t2, t3) to capture how feed intake changes over lactation days
   - t1: Linear trend (straight line)
   - t2: Quadratic trend (curved)
   - t3: Cubic trend (more complex curve)

2. **Lagged Variables**: Yesterday's values become today's predictors
   - `prev_feed_intake`: How much the sow ate yesterday
   - `prev_temperature`: Yesterday's temperature
   - This helps because past conditions influence current behavior

3. **Moving Averages**: 3-day averages to smooth out daily variations
   - `ma3_feed`: Average feed intake over last 3 days
   - `ma3_temp`: Average temperature over last 3 days

4. **Categorical Features**:
   - **Lactation Phase**: "early" (days 1-7), "mid" (days 8-14), "late" (days 15+)
   - **Parity Group**: "first", "second", "third or higher" 

**Why These Features Matter**:
- Lactation day patterns: Feed intake typically increases early, peaks mid-lactation, then decreases
- Weather effects: Hot weather reduces FI
- Individual sow patterns: Each sow has unique eating habits
- Parity effects: First-time sows eat differently than older sows

### 3. Three Modeling Approaches

#### A. Polynomial Model (Traditional Statistical Approach)
```r
fit_polynomial_model <- function(train_data) {
  model <- lmer(daily_feed_intake ~ t1 + t2 + t3 + dew_scaled + 
                t1:dew_scaled + t2:dew_scaled + t3:dew_scaled + 
                (1 + t1 + t2|sow_id), data = train_data)
}
```

**What it does**: 
- Uses mathematical curves to model feed intake over time
- Accounts for weather effects (dew point interactions)
- Includes "random effects" - each sow can have her own unique pattern
- It handles repeated measurements 

**Strengths**: Interpretable, handles individual sow differences well
**Weaknesses**: May miss complex non-linear patterns

#### B. Random Forest Model
```r
fit_random_forest <- function(train_data, features) {
  rf_model <- randomForest(daily_feed_intake ~ ., 
                          data = train_clean[, feature_cols],
                          ntree = 500, ...)
}
```

**What it does**:
- Creates 500 "decision trees" that each make predictions
- Each tree uses random subsets of data and features
- Final prediction is the average of all trees
- Can automatically detect complex patterns and interactions

**How it works**: 
1. Tree 1 might say: "If it's day 10 and temp > 20°C, predict 5.5kg"
2. Tree 2 might say: "If it's a first-time mother on day 10, predict 4.8kg"
3. Average all 500 tree predictions for final answer - this value was chosen based on this dataset

**Strengths**: Handles complex patterns, resistant to overfitting
**Weaknesses**: Less interpretable, can't extrapolate beyond training data

#### C. XGBoost Model
```r
fit_xgboost <- function(train_data, features) {
  xgb_model <- xgboost(data = train_matrix,
                      label = train_for_xgb$daily_feed_intake,
                      nrounds = 100, ...)
}
```

**What it does**:
- Builds models sequentially - each new model corrects previous mistakes
- Starts with simple prediction, then adds complexity iteratively
- Very powerful for prediction competitions

**How it works**:
1. Model 1: Makes basic predictions
2. Model 2: Focuses on correcting Model 1's biggest errors
3. Model 3: Corrects remaining errors from Models 1+2
4. Continues for 100 rounds
5. Final prediction combines all models

**Strengths**: Often highest accuracy, handles missing data well
**Weaknesses**: Can overfit, requires careful tuning

### 4. Model Evaluation (`evaluate_model` function)

**Metrics Used**:

1. **RMSE (Root Mean Square Error)**: 
   - Average prediction error in same units as target (kg/day)
   - Lower is better
   - Penalizes large errors more heavily

2. **MAE (Mean Absolute Error)**:
   - Average absolute difference between predicted and actual
   - More interpretable than RMSE
   - Lower is better

3. **R² (R-squared)**:
   - Proportion of variance explained (0-1 scale)
   - Higher is better (1 = perfect predictions)

**Example Interpretation**:
- RMSE = 0.5 means predictions are typically within ±0.5 kg/day
- R² = 0.85 means the model explains 85% of the variation in feed intake

### 5. Main Pipeline (`run_feed_intake_pipeline` function)

**Step-by-Step Process**:

1. **Data Validation**: Checks if required columns exist
2. **Data Preparation**: Applies all feature engineering
3. **Train/Test Split**: 80% for training, 20% for testing
4. **Model Training**: Fits all three model types
5. **Model Evaluation**: Tests performance on unseen data
6. **Comparison**: Ranks models by performance metrics

**Error Handling**: If any model fails, the pipeline continues with successful models.

### 6. Prediction Function (`predict_tomorrow_intake`)

**Purpose**: Use the best-performing model to predict next-day feed intake.

**Process**:
1. Identifies the model with lowest RMSE (best accuracy)
2. Prepares current sow data with same feature engineering
3. Makes predictions for next day
4. Returns predictions with model used

### 7. Visualization (`create_prediction_plots`)

**Creates Four Types of Plots**:

1. **Model Comparison**: Bar charts comparing RMSE, MAE, and R² across models
2. **Actual vs Predicted Scatter Plots**: One for each model type
   - Points close to diagonal line = good predictions
   - Scattered points = poor predictions

## How to Use This Code

### Basic Usage:
```r
# 1. Load your data
sow_data <- read.csv("your_sow_data.csv")

# 2. Run the complete pipeline
results <- run_feed_intake_pipeline(sow_data)

# 3. View model comparison
print(results$comparison)

# 4. Make tomorrow's predictions
current_data <- sow_data %>% 
  group_by(sow_id) %>% 
  slice_max(lactation_day, n = 1)

tomorrow_predictions <- predict_tomorrow_intake(results, current_data)
print(tomorrow_predictions)

# 5. Create visualizations
plots <- create_prediction_plots(results)
plots$comparison  # View model comparison
```

### Expected Output:

**Model Comparison Table**:
```
        Model      RMSE       MAE R_squared
1  Polynomial 0.421833 0.3156045 0.8234567
2 Random Forest 0.398765 0.2987654 0.8456789
3     XGBoost 0.387432 0.2876543 0.8567890
```

**Tomorrow's Predictions**:
```
  sow_id current_day predicted_next_day_intake  model_used
1      1          15                  5.234567     XGBoost
2      2          12                  6.123456     XGBoost
3      3          18                  4.567890     XGBoost
```

## Key Features

### Robustness Features:
- **Missing Data Handling**: Automatically handles missing values
- **Error Recovery**: Continues if individual models fail
- **Data Validation**: Checks for required columns before processing
- **Feature Flexibility**: Uses available features, warns about missing ones

### Advanced Features:
- **Individual Sow Effects**: Polynomial model accounts for sow-specific patterns
- **Temporal Features**: Captures trends over lactation period
- **Weather Integration**: Includes temperature and humidity effects
- **Lag Features**: Uses historical values as predictors

## Common Issues and Solutions

### 1. "Missing required columns" Error
**Problem**: Your CSV doesn't have the expected column names
**Solution**: Check your column names match exactly (case-sensitive)

### 2. "Insufficient data for training" Error
**Problem**: Too few complete observations
**Solution**: Ensure you have at least 20 rows with non-missing values

### 3. All Models Fail
**Problem**: Data quality issues or missing critical features
**Solution**: Check data types, ensure numeric columns are actually numeric

### 4. Poor Model Performance
**Problem**: R² < 0.5, high RMSE
**Solutions**: 
- Check for outliers in feed intake data
- Ensure lactation_day and other predictors are reasonable
- Consider adding more relevant features

## Model Selection Guidance

**Choose Polynomial Model When**:
- You need interpretable results
- You have relatively simple patterns
- You want to understand individual sow effects

**Choose Random Forest When**:
- You have complex, non-linear patterns
- You want robust predictions
- Interpretability is less important

**Choose XGBoost When**:
- Maximum prediction accuracy is needed
- You have large datasets
- You can invest time in parameter tuning
