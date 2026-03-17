# ==============================================================================
# PROJECT 5: Real Estate Investment ROI Optimizer
# GOAL: Predict Property Value & Identify Key ROI Drivers
# MODELS: OLS Linear Regression, Ridge Regression, Lasso Regression
# ==============================================================================

# ------------------------------------------------------------------------------
# PHASE 1: Setup & Data Acquisition
# ------------------------------------------------------------------------------
options(scipen = 999) # Disable scientific notation for price readability
if(!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, glmnet, corrplot, scales, broom)


df_raw <- read_csv("kc_house_data.csv")

# ------------------------------------------------------------------------------
# PHASE 2: Data Cleaning & Feature Engineering
# ------------------------------------------------------------------------------
# We transform raw data into "Investment Features"
df <- df_raw %>%
  select(-id, -date, -zipcode) %>% 
  mutate(
    house_age = 2026 - yr_built,
    is_renovated = ifelse(yr_renovated > 0, 1, 0),
    total_rooms = bathrooms + bedrooms
  ) %>%
  select(-yr_built, -yr_renovated) %>%
  drop_na()

# Split into Training (80%) and Testing (20%)
train_index <- createDataPartition(df$price, p = 0.8, list = FALSE)
train_df <- df[train_index, ]
test_df  <- df[-train_index, ]

# ------------------------------------------------------------------------------
# PHASE 3: Prepare Matrices for Regularization (glmnet)
# ------------------------------------------------------------------------------
# glmnet requires matrix input, unlike standard lm()
train_x <- as.matrix(train_df %>% select(-price))
train_y <- train_df$price
test_x  <- as.matrix(test_df %>% select(-price))
test_y  <- test_df$price

# ------------------------------------------------------------------------------
# PHASE 4: Model Training (The Comparison)
# ------------------------------------------------------------------------------

# 1. Linear Regression (The Baseline)
fit_lm <- lm(price ~ ., data = train_df)

# 2. Ridge Regression (Alpha = 0) - Handles Multicollinearity
# We use cross-validation to find the optimal penalty (lambda)
cv_ridge <- cv.glmnet(train_x, train_y, alpha = 0)
fit_ridge <- glmnet(train_x, train_y, alpha = 0, lambda = cv_ridge$lambda.min)

# 3. Lasso Regression (Alpha = 1) - The ROI Filter (Feature Selection)
cv_lasso <- cv.glmnet(train_x, train_y, alpha = 1)
fit_lasso <- glmnet(train_x, train_y, alpha = 1, lambda = cv_lasso$lambda.min)

# ------------------------------------------------------------------------------
# PHASE 5: Evaluation & Comparison
# ------------------------------------------------------------------------------
# Function to calculate metrics
eval_regression <- function(actual, predicted, model_name, data_x) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae  <- mean(abs(actual - predicted))
  r2   <- cor(actual, predicted)^2
  
  # Adjusted R-squared
  n <- length(actual)
  p <- ncol(data_x)
  adj_r2 <- 1 - ((1 - r2) * (n - 1) / (n - p - 1))
  
  return(data.frame(
    Model = model_name, 
    RMSE = round(rmse, 2), 
    MAE = round(mae, 2),
    Adj_R2 = round(adj_r2, 4)
  ))
}

# Generate Predictions
pred_lm    <- predict(fit_lm, test_df)
pred_ridge <- as.vector(predict(fit_ridge, test_x))
pred_lasso <- as.vector(predict(fit_lasso, test_x))

comparison_df <- rbind(
  eval_regression(test_y, pred_lm, "OLS Linear", test_x),
  eval_regression(test_y, pred_ridge, "Ridge", test_x),
  eval_regression(test_y, pred_lasso, "Lasso", test_x)
)

print(comparison_df)

# ------------------------------------------------------------------------------
# PHASE 6: The "Client Insight" (Lasso Coefficients)
# ------------------------------------------------------------------------------
# This shows the client which features actually drive ROI
lasso_final_coefs <- as.matrix(coef(fit_lasso))
lasso_final_coefs <- data.frame(
  Feature = rownames(lasso_final_coefs),
  Coefficient = as.numeric(lasso_final_coefs)
) %>% filter(Feature != "(Intercept)" & Coefficient != 0) %>%
  arrange(desc(abs(Coefficient)))

print("Top Drivers identified by Lasso:")
print(lasso_final_coefs)

# ------------------------------------------------------------------------------
# PHASE 7: Visualizing Predictions
# ------------------------------------------------------------------------------
ggplot(data.frame(Actual = test_y, Predicted = pred_lasso), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.3, color = "#2c3e50") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  scale_x_continuous(labels = label_dollar()) +
  scale_y_continuous(labels = label_dollar()) +
  labs(title = "Lasso Regression: Actual vs Predicted Property Values",
       subtitle = "Visualizing accuracy for Wealth Management ROI Analysis") +
  theme_minimal()
