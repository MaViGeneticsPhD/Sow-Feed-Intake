# Sow Feed Intake Prediction Pipeline - README

## Required Data Structure

Input CSV file (`sow_data.csv`) that should contain these columns:

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
# ... other libraries
```
### 2. Data Preparation (`prepare_data` function)

**Purpose**: Transforms raw data into features that machine learning models can use effectively.

1. **Polynomial Terms**: Creates mathematical curves (t1, t2, t3) to capture how feed intake changes over lactation days

2. **Lagged Variables**: Yesterday's values become today's predictors
   - `prev_feed_intake`
   - `prev_temperature`
   
3. **Moving Averages**: 3-day averages to smooth out daily variations
   - `ma3_feed`: Average feed intake over last 3 days
   - `ma3_temp`: Average temperature over last 3 days

4. **Categorical Features**:
   - **Lactation Phase**: "early" (days 1-7), "mid" (days 8-14), "late" (days 15+)
   - **Parity Group**: "first", "second", "third or higher" 

### 3. Modeling Approaches

#### A. Polynomial Model 
      (with 3 random effects and dew point as cov (interacting with lactation day)
```r
fit_polynomial_model <- function(train_data) {
  model <- lmer(daily_feed_intake ~ t1 + t2 + t3 + dew_scaled + 
                t1:dew_scaled + t2:dew_scaled + t3:dew_scaled + 
                (1 + t1 + t2|sow_id), data = train_data)
}
```
**What it does**: 
- model feed intake over time (up to 22d)
- Accounts for environmental effects (dew point interactions)
- Includes "random effects" - each sow can have her own unique pattern

#### B. Random Forest Model
```r
fit_random_forest <- function(train_data, features) {
  rf_model <- randomForest(daily_feed_intake ~ ., 
                          data = train_clean[, feature_cols],
                          ntree = 500, ...)
}
```

**What it does**:
- Creates 500 "decision trees" that each make predictions # for future: test the optimum value of decision treess
- Each tree uses random subsets of data and features and final prediction is the average of all trees

**Strengths**: Handles complex patterns, resistant to overfitting
**Weaknesses**: Less interpretable, can't extrapolate beyond training data...

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
   - Higher is better 

### 5. Main Pipeline (`run_feed_intake_pipeline` function)

**Step-by-Step Process**:

1. **Data Validation**: Checks if required columns exist
2. **Data Preparation**: Applies all feature engineering
3. **Train/Test Split**: 80% for training, 20% for testing
4. **Model Training**: Fits all model types
5. **Model Evaluation**: Tests performance on unseen data
6. **Comparison**: Ranks models by performance metrics

**Error Handling**: If any model fails, the pipeline continues with successful models.

### 6. Prediction Function (`predict_tomorrow_intake`) - IT'S NOT WORKING YET!!!!

**Purpose**: Use the best-performing model to predict next-day feed intake.

**Process**:
1. Identifies the model with lowest RMSE (best accuracy)
2. Prepares current sow data with same feature engineering
3. Makes predictions for next day
4. Returns predictions with model used

### 7. Visualization (`create_prediction_plots`)

1. **Model Comparison**: Bar charts comparing RMSE, MAE, and R² across models
2. **Actual vs Predicted Scatter Plots**: One for each model type


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


