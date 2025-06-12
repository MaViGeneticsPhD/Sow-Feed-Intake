# Sow Feed Intake Prediction Pipeline
# Comparing Polynomial Model vs Machine Learning Models

# Load required libraries
library(tidyverse)
library(randomForest)
library(caret)
library(xgboost)
library(ranger)
library(Metrics)
library(corrplot)
library(gridExtra)
library(zoo)  # for rollmean function

# Set seed for reproducibility
set.seed(123)

# 1. DATA PREPARATION ----

sow_data <- read.csv("sow_data.csv")

# Example data structure:
# sow_data columns: sow_id, parity, lactation_day, daily_feed_intake, 
# temperature, dew_point, DGH24

check_data_columns <- function(data) {
  cat("Available columns in your data:\n")
  print(colnames(data))
  cat("\nFirst few rows:\n")
  print(head(data))
  cat("\nData structure:\n")
  print(str(data))
}

prepare_data <- function(data) {
  data <- data %>%
    arrange(sow_id, lactation_day)
  
  # Create polynomial terms (orthogonal)
  poly_terms <- poly(data$lactation_day, degree = 3, raw = FALSE)
  data$t1 <- poly_terms[, 1]
  data$t2 <- poly_terms[, 2]
  data$t3 <- poly_terms[, 3]
  
  # Scale dew point
  data$dew_scaled <- scale(data$dew_point)[, 1]
  
  # Create lagged and smoothed variables, and categorized factors
  data <- data %>%
    mutate(
      DGH24 = Hours_T2M_GT_24,
      
      # Lagged features
      prev_feed_intake = lag(daily_feed_intake, 1),
      prev_temperature = lag(temperature, 1),
      prev_dew_point = lag(dew_point, 1),
      prev_DGH24 = lag(DGH24, 1),
      
      # Moving averages (3-day window)
      ma3_feed = zoo::rollmean(daily_feed_intake, k = 3, fill = NA, align = "right"),
      ma3_temp = zoo::rollmean(temperature, k = 3, fill = NA, align = "right"),
      
      # Lactation phases
      lactation_phase = case_when(
        lactation_day <= 7 ~ "early",
        lactation_day <= 14 ~ "mid",
        TRUE ~ "late"
      ),
      
      # Parity group 
      parity_group = case_when(
        Parity == 1 ~ "first",
        Parity == 2 ~ "second",
        Parity == "P3+" ~ "third or higher",
        TRUE ~ "unknown"
      )
    ) %>% filter(lactation_day <= 22) %>%
    ungroup()
  
  return(data)
}


# 2. POLYNOMIAL MODEL (Your current best model) ----
fit_polynomial_model <- function(train_data) {
  model <- lmer(daily_feed_intake ~ t1 + t2 + t3 + dew_scaled + 
                  t1:dew_scaled + t2:dew_scaled + t3:dew_scaled + (1 + t1 + t2|sow_id), 
                data = train_data)
  return(model)
}

# 3. MACHINE LEARNING MODELS ----

# Random Forest
fit_random_forest <- function(train_data, features) {
  # Check which features actually exist in the data
  available_features <- features[features %in% colnames(train_data)]
  missing_features <- features[!features %in% colnames(train_data)]
  
  if(length(missing_features) > 0) {
    cat("Warning: These features are not available in the data:", paste(missing_features, collapse = ", "), "\n")
  }
  
  cat("Using features:", paste(available_features, collapse = ", "), "\n")
  
  if(length(available_features) == 0) {
    stop("No valid features found for Random Forest model")
  }
  
  # Remove rows with NA values in target and available features
  feature_cols <- c("daily_feed_intake", available_features)
  train_clean <- train_data[complete.cases(train_data[, feature_cols]), ]
  
  cat("Training data: ", nrow(train_clean), "complete cases out of", nrow(train_data), "total\n")
  
  if(nrow(train_clean) < 10) {
    stop("Insufficient complete cases for training Random Forest")
  }
  
  # Convert character/factor variables
  for(col in available_features) {
    if(is.character(train_clean[[col]])) {
      train_clean[[col]] <- as.factor(train_clean[[col]])
    }
  }
  
  rf_model <- randomForest(
    daily_feed_intake ~ ., 
    data = train_clean[, feature_cols],
    ntree = 500,
    mtry = max(1, floor(sqrt(length(available_features)))),
    importance = TRUE,
    nodesize = 5
  )
  
  return(list(model = rf_model, features_used = available_features))
}

# XGBoost
fit_xgboost <- function(train_data, features) {
  # Check which features actually exist in the data
  available_features <- features[features %in% colnames(train_data)]
  missing_features <- features[!features %in% colnames(train_data)]
  
  if(length(missing_features) > 0) {
    cat("Warning: These features are not available in the data:", paste(missing_features, collapse = ", "), "\n")
  }
  
  if(length(available_features) == 0) {
    stop("No valid features found for XGBoost model")
  }
  
  feature_cols <- c("daily_feed_intake", available_features)
  train_clean <- train_data[complete.cases(train_data[, feature_cols]), ]
  
  cat("XGBoost training data: ", nrow(train_clean), "complete cases\n")
  
  if(nrow(train_clean) < 10) {
    stop("Insufficient complete cases for training XGBoost")
  }
  
  # Convert character variables to factors then to numeric
  train_for_xgb <- train_clean[, feature_cols]
  for(col in available_features) {
    if(is.character(train_for_xgb[[col]]) || is.factor(train_for_xgb[[col]])) {
      train_for_xgb[[col]] <- as.numeric(as.factor(train_for_xgb[[col]]))
    }
  }
  
  # Prepare data for XGBoost
  train_matrix <- as.matrix(train_for_xgb[, available_features])
  
  xgb_model <- xgboost(
    data = train_matrix,
    label = train_for_xgb$daily_feed_intake,
    nrounds = 100,
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    verbose = 0
  )
  
  return(list(model = xgb_model, features_used = available_features))
}

# 4. MODEL EVALUATION ----
evaluate_model <- function(model, test_data, model_type, features_used = NULL) {
  if (model_type == "polynomial") {
    predictions <- predict(model, test_data)
    actual_values <- test_data$daily_feed_intake
  } else if (model_type == "rf") {
    feature_cols <- c("daily_feed_intake", features_used)
    test_clean <- test_data[complete.cases(test_data[, feature_cols]), ]
    
    if(nrow(test_clean) == 0) {
      return(list(rmse = NA, mae = NA, r2 = NA, predictions = NA, actual = NA))
    }
    
    # Convert character variables to factors
    for(col in features_used) {
      if(is.character(test_clean[[col]])) {
        test_clean[[col]] <- as.factor(test_clean[[col]])
      }
    }
    
    predictions <- predict(model$model, test_clean[, features_used])
    actual_values <- test_clean$daily_feed_intake
  } else if (model_type == "xgb") {
    feature_cols <- c("daily_feed_intake", features_used)
    test_clean <- test_data[complete.cases(test_data[, feature_cols]), ]
    
    if(nrow(test_clean) == 0) {
      return(list(rmse = NA, mae = NA, r2 = NA, predictions = NA, actual = NA))
    }
    
    # Convert character variables to numeric
    test_for_xgb <- test_clean[, feature_cols]
    for(col in features_used) {
      if(is.character(test_for_xgb[[col]]) || is.factor(test_for_xgb[[col]])) {
        test_for_xgb[[col]] <- as.numeric(as.factor(test_for_xgb[[col]]))
      }
    }
    
    test_matrix <- as.matrix(test_for_xgb[, features_used])
    predictions <- predict(model$model, test_matrix)
    actual_values <- test_for_xgb$daily_feed_intake
  }
  
  # Calculate metrics
  valid_indices <- !is.na(predictions) & !is.na(actual_values)
  
  if(sum(valid_indices) == 0) {
    return(list(rmse = NA, mae = NA, r2 = NA, predictions = NA, actual = NA))
  }
  
  predictions <- predictions[valid_indices]
  actual_values <- actual_values[valid_indices]
  
  rmse <- sqrt(mean((actual_values - predictions)^2))
  mae <- mean(abs(actual_values - predictions))
  r2 <- cor(actual_values, predictions)^2
  
  return(list(
    rmse = rmse,
    mae = mae,
    r2 = r2,
    predictions = predictions,
    actual = actual_values
  ))
}

# 5. NEXT-DAY PREDICTION FUNCTION ----
predict_next_day <- function(model, current_data, model_type, features = NULL) {
  if (model_type == "polynomial") {
    predictions <- predict(model, current_data)
  } else if (model_type == "rf") {
    predictions <- predict(model, current_data[, features])
  } else if (model_type == "xgb") {
    pred_matrix <- model.matrix(~ . - 1, data = current_data[, features])
    predictions <- predict(model$model, pred_matrix)
  }
  
  return(predictions)
}

# 6. MAIN PIPELINE FUNCTION ----
run_feed_intake_pipeline <- function(sow_data) {
  
  # Check data first
  check_data_columns(sow_data)
  
  # Prepare data
  cat("\nPreparing data...\n")
  prepared_data <- prepare_data(sow_data)
  
  # Define features for ML models based on what's available
  base_features <- c("Parity", "lactation_day", "t1", "t2", "t3", "dew_point")
  
  # Add optional features if they exist
  optional_features <- c("temperature", "DGH24", "prev_feed_intake", "prev_temperature", 
                         "prev_dew_point", "prev_DGH24", "ma3_feed", "ma3_temp", 
                         "lactation_phase", "parity_group")
  
  available_optional <- optional_features[optional_features %in% colnames(prepared_data)]
  ml_features <- c(base_features, available_optional)
  
  # Check if we have the minimum required features
  required_features <- c("Parity", "lactation_day", "dew_point", "daily_feed_intake")
  missing_required <- required_features[!required_features %in% colnames(prepared_data)]
  
  if(length(missing_required) > 0) {
    stop(paste("Missing required columns:", paste(missing_required, collapse = ", ")))
  }
  
  cat("Final ML features to use:", paste(ml_features, collapse = ", "), "\n")
  
  # Remove rows with missing target variable
  prepared_data <- prepared_data[!is.na(prepared_data$daily_feed_intake), ]
  
  if(nrow(prepared_data) < 20) {
    stop("Insufficient data for training (need at least 20 complete observations)")
  }
  
  # Split data (80% train, 20% test)
  set.seed(123)
  train_indices <- sample(nrow(prepared_data), 0.8 * nrow(prepared_data))
  train_data <- prepared_data[train_indices, ]
  test_data <- prepared_data[-train_indices, ]
  
  cat("Training data size:", nrow(train_data), "\n")
  cat("Test data size:", nrow(test_data), "\n")
  
  # Fit models
  cat("\nFitting polynomial model...\n")
  tryCatch({
    poly_model <- fit_polynomial_model(train_data)
    poly_success <- TRUE
  }, error = function(e) {
    cat("Polynomial model failed:", e$message, "\n")
    poly_model <<- NULL
    poly_success <<- FALSE
  })
  
  cat("Fitting Random Forest model...\n")
  tryCatch({
    rf_model <- fit_random_forest(train_data, ml_features)
    rf_success <- TRUE
  }, error = function(e) {
    cat("Random Forest model failed:", e$message, "\n")
    rf_model <<- NULL
    rf_success <<- FALSE
  })
  
  cat("Fitting XGBoost model...\n")
  tryCatch({
    xgb_model <- fit_xgboost(train_data, ml_features)
    xgb_success <- TRUE
  }, error = function(e) {
    cat("XGBoost model failed:", e$message, "\n")
    xgb_model <<- NULL
    xgb_success <<- FALSE
  })
  
  # Evaluate models
  cat("\nEvaluating models...\n")
  
  results_list <- list()
  comparison_data <- data.frame()
  
  if(exists("poly_success") && poly_success) {
    poly_results <- evaluate_model(poly_model, test_data, "polynomial")
    results_list$polynomial <- poly_results
    comparison_data <- rbind(comparison_data, 
                             data.frame(Model = "Polynomial", 
                                        RMSE = poly_results$rmse,
                                        MAE = poly_results$mae,
                                        R_squared = poly_results$r2))
  }
  
  if(exists("rf_success") && rf_success) {
    rf_results <- evaluate_model(rf_model, test_data, "rf", rf_model$features_used)
    results_list$random_forest <- rf_results
    comparison_data <- rbind(comparison_data, 
                             data.frame(Model = "Random Forest", 
                                        RMSE = rf_results$rmse,
                                        MAE = rf_results$mae,
                                        R_squared = rf_results$r2))
  }
  
  if(exists("xgb_success") && xgb_success) {
    xgb_results <- evaluate_model(xgb_model, test_data, "xgb", xgb_model$features_used)
    results_list$xgboost <- xgb_results
    comparison_data <- rbind(comparison_data, 
                             data.frame(Model = "XGBoost", 
                                        RMSE = xgb_results$rmse,
                                        MAE = xgb_results$mae,
                                        R_squared = xgb_results$r2))
  }
  
  if(nrow(comparison_data) == 0) {
    stop("All models failed to train successfully")
  }
  
  cat("\nModel Comparison Results:\n")
  print(comparison_data)
  
  
  # Create models list
  models_list <- list()
  if(exists("poly_success") && poly_success) models_list$polynomial <- poly_model
  if(exists("rf_success") && rf_success) models_list$random_forest <- rf_model
  if(exists("xgb_success") && xgb_success) models_list$xgboost <- xgb_model
  
  # Return results
  return(list(
    models = models_list,
    results = results_list,
    comparison = comparison_data,
    ml_features = ml_features,
    prepared_data = prepared_data
  ))
}

# 7. NEXT-DAY PREDICTION PIPELINE ----
predict_tomorrow_intake <- function(pipeline_results, current_sow_data) {
  
  # Prepare current data
  current_prepared <- prepare_data(current_sow_data)
  
  # Get the best performing model
  best_model_idx <- which.min(pipeline_results$comparison$RMSE)
  best_model_name <- pipeline_results$comparison$Model[best_model_idx]
  
  cat(paste("Using", best_model_name, "for prediction (lowest RMSE)\n"))
  
  # Make predictions with all models
  if (best_model_name == "Polynomial") {
    prediction <- predict_next_day(pipeline_results$models$polynomial, 
                                   current_prepared, "polynomial")
  } else if (best_model_name == "Random Forest") {
    prediction <- predict_next_day(pipeline_results$models$random_forest, 
                                   current_prepared, "rf", 
                                   pipeline_results$ml_features)
  } else {
    prediction <- predict_next_day(pipeline_results$models$xgboost, 
                                   current_prepared, "xgb", 
                                   pipeline_results$ml_features)
  }
  
  return(data.frame(
    sow_id = current_prepared$sow_id,
    current_day = current_prepared$lactation_day,
    predicted_next_day_intake = prediction,
    model_used = best_model_name
  ))
}

# # Run the pipeline
results <- run_feed_intake_pipeline(sow_data)

# # For next-day predictions, prepare current data
current_data <- sow_data %>% 
   group_by(sow_id) %>% 
   slice_max(lactation_day, n = 1) %>%  # Get latest data for each sow
   ungroup()
# 
# # Predict tomorrow's intake
tomorrow_predictions <- predict_tomorrow_intake(results, current_data)
print(tomorrow_predictions)

# 9. VISUALIZATION FUNCTION ----
create_prediction_plots <- function(pipeline_results) {
  
  # Model comparison plot
  comp_plot <- pipeline_results$comparison %>%
    pivot_longer(cols = c(RMSE, MAE, R_squared), names_to = "Metric", values_to = "Value") %>%
    ggplot(aes(x = Model, y = Value, fill = Model)) +
    geom_col() +
    facet_wrap(~Metric, scales = "free_y") +
    theme_minimal() +
    labs(title = "Model Performance Comparison") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  plot_list <- list(comparison = comp_plot)
  
  # Actual vs Predicted plots for each available model
  # Random Forest plot
  if("random_forest" %in% names(pipeline_results$results) && 
     !is.na(pipeline_results$results$random_forest$actual[1])) {
    rf_plot <- data.frame(
      Actual = pipeline_results$results$random_forest$actual,
      Predicted = pipeline_results$results$random_forest$predictions
    ) %>%
      ggplot(aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.6, color = "blue") +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      labs(title = "Random Forest: Actual vs Predicted",
           x = "Actual Feed Intake",
           y = "Predicted Feed Intake") +
      theme_minimal()
    
    plot_list$rf_pred <- rf_plot
  }
  
  # XGBoost plot
  if("xgboost" %in% names(pipeline_results$results) && 
     !is.na(pipeline_results$results$xgboost$actual[1])) {
    xgb_plot <- data.frame(
      Actual = pipeline_results$results$xgboost$actual,
      Predicted = pipeline_results$results$xgboost$predictions
    ) %>%
      ggplot(aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.6, color = "green") +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      labs(title = "XGBoost: Actual vs Predicted",
           x = "Actual Feed Intake",
           y = "Predicted Feed Intake") +
      theme_minimal()
    
    plot_list$xgb_pred <- xgb_plot
  }
  
  # Polynomial plot
  if("polynomial" %in% names(pipeline_results$results) && 
     !is.na(pipeline_results$results$polynomial$actual[1])) {
    poly_plot <- data.frame(
      Actual = pipeline_results$results$polynomial$actual,
      Predicted = pipeline_results$results$polynomial$predictions
    ) %>%
      ggplot(aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.6, color = "purple") +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      labs(title = "Polynomial: Actual vs Predicted",
           x = "Actual Feed Intake",
           y = "Predicted Feed Intake") +
      theme_minimal()
    
    plot_list$poly_pred <- poly_plot
  }
  
  return(plot_list)
}

# After running your pipeline
#results <- run_feed_intake_pipeline(sow_data)

# Create all plots
plots <- create_prediction_plots(results)

# View individual plots
plots$comparison      # Model comparison plot
plots$rf_pred        # Random Forest actual vs predicted
plots$xgb_pred       # XGBoost actual vs predicted  
plots$poly_pred      # Polynomial actual vs predicted

# display all prediction plots together
grid.arrange(plots$rf_pred, plots$xgb_pred, plots$poly_pred, ncol = 2)


