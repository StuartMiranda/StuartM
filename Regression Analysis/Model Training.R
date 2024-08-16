library(glmnet)
library(splines)
library(mgcv)
library(rpart)
library(randomForest)
library(gt)

# variable selection - best score for all of them

ozone<- read.table("https://hastie.su.domains/ElemStatLearn/datasets/ozone.data", header = TRUE)

# Define a function to calculate the GCV score
gcv_score <- function(y, yhat, p) {
  n <- length(y)
  mean((y - yhat)^2 / (1 - p/n)^2)
}

# Define a function to calculate the a.djusted R-squared value
adj_r_squared <- function(model, X, y) {
  yhat <- predict(model, newx = X)
  SST <- sum((y - mean(y))^2)
  SSE <- sum((y - yhat)^2)
  p <- ncol(X)  # Number of predictors
  n <- length(y)  # Number of observations
  1 - (SSE / (n - p - 1)) / (SST / (n - 1))
}

# Prepare the data
X <- model.matrix(ozone ~ radiation + temperature + wind, data = ozone)[,-1]
y <- ozone$ozone

# Linear Model
lm_model <- lm(ozone ~ radiation + temperature + wind, data = ozone)
lm_predictions <- predict(lm_model)
lm_mse <- mean((y - lm_predictions)^2)
lm_gcv <- gcv_score(y, lm_predictions, ncol(X) + 1)  # +1 for intercept
lm_adj_r2 <- summary(lm_model)$adj.r.squared

# Ridge Regression
ridge_model <- cv.glmnet(X, y, alpha = 0)
lambda_ridge <- ridge_model$lambda.min
ridge_predictions <- predict(ridge_model, newx = X, s = lambda_ridge)
ridge_mse <- mean((y - ridge_predictions)^2)
ridge_gcv <- gcv_score(y, ridge_predictions, ncol(X))  # No intercept
ridge_adj_r2 <- adj_r_squared(ridge_model, X, y)

# Lasso Regression
lasso_model <- cv.glmnet(X, y, alpha = 1)
lambda_lasso <- lasso_model$lambda.min
lasso_predictions <- predict(lasso_model, newx = X, s = lambda_lasso)
lasso_mse <- mean((y - lasso_predictions)^2)
lasso_gcv <- gcv_score(y, lasso_predictions, ncol(X))  # No intercept
lasso_adj_r2 <- adj_r_squared(lasso_model, X, y)

# Spline Model
spline_model <- lm(ozone ~ ns(radiation, df = 4) + ns(temperature, df = 4) + ns(wind, df = 4), data = ozone)
spline_predictions <- predict(spline_model)
spline_mse <- mean((y - spline_predictions)^2)
spline_gcv <- gcv_score(y, spline_predictions, ncol(model.matrix(spline_model)))  # Including spline basis functions
spline_adj_r2 <- summary(spline_model)$adj.r.squared

# Additive Model
gam_model <- gam(ozone ~ s(radiation) + s(temperature) + s(wind), data = ozone)
gam_predictions <- predict(gam_model)
gam_mse <- mean((y - gam_predictions)^2)
gam_gcv <- gcv_score(y, gam_predictions, ncol(X))  # Approximation
gam_adj_r2 <- summary(gam_model)$r.sq  # Note: GAM models don't have an adjusted R-squared directly available

# Decision Tree Model
tree_model <- rpart(ozone ~ radiation + temperature + wind, data = ozone)
tree_predictions <- predict(tree_model)
tree_mse <- mean((y - tree_predictions)^2)

tree_errors <- c()
for (i in 1:111){
  tree_cv_model <- rpart(ozone ~ radiation + temperature + wind, data = ozone[-i,])
  tree_errors[i] = (predict(tree_cv_model,newdata = ozone[i,]) - ozone[i,1])^2
}
tree_gcv <- mean(tree_errors)

tree_adj_r2 <- 1 - (mean((y - tree_predictions)^2) / var(y))


# Random Forest Model
rf_model <- randomForest(ozone ~ radiation + temperature + wind, data = ozone, ntree = 500)
rf_predictions <- predict(rf_model, newdata = ozone)
rf_mse <- mean((y - rf_predictions)^2)

rf_errors = c()
for (i in 1:111){
  rf_cv_model <- randomForest(ozone ~ radiation + temperature + wind, data = ozone[-i,], ntree = 500)
  rf_errors[i] = (predict(rf_cv_model,newdata = ozone[i,]) - ozone[i,1])^2
}
rf_gcv <- mean(rf_errors)

rf_adj_r2 <- 1 - (mean((y - rf_predictions)^2) / var(y))  # Approximation of Adjusted R-squared

# Random Forest Model
rf_model2 <- randomForest(ozone ~ radiation + temperature + wind, data = ozone, ntree = 50)
rf_predictions2 <- predict(rf_model2, newdata = ozone)
rf_mse2 <- mean((y - rf_predictions2)^2)

rf_errors2 = c()
for (i in 1:111){
  rf_cv_model2 <- randomForest(ozone ~ radiation + temperature + wind, data = ozone[-i,], ntree = 50)
  rf_errors2[i] = (predict(rf_cv_model2,newdata = ozone[i,]) - ozone[i,1])^2
}
rf_gcv2 <- mean(rf_errors2)

rf_adj_r2_2 <- 1 - (mean((y - rf_predictions2)^2) / var(y))  # Approximation of Adjusted R-squared

# Random Forest Model
rf_model3 <- randomForest(ozone ~ radiation + temperature + wind, data = ozone, ntree = 5)
rf_predictions3 <- predict(rf_model3, newdata = ozone)
rf_mse3 <- mean((y - rf_predictions3)^2)

rf_errors3 = c()
for (i in 1:111){
  rf_cv_model3 <- randomForest(ozone ~ radiation + temperature + wind, data = ozone[-i,], ntree = 5)
  rf_errors3[i] = (predict(rf_cv_model3,newdata = ozone[i,]) - ozone[i,1])^2
}
rf_gcv3 <- mean(rf_errors3)

rf_adj_r2_3 <- 1 - (mean((y - rf_predictions3)^2) / var(y))  # Approximation of Adjusted R-squared

# Compile the results
results <- data.frame(
  Model = c("Linear", "Ridge Regression", "Lasso Regression", "Spline", "Additive", 
            "Tree", "Random Forest (n=5)", "Random Forest (n=50)", "Random Forest (n=500)"
            ),
  Prediction_Error = c(lm_mse, ridge_mse, lasso_mse, spline_mse, gam_mse, tree_mse, rf_mse3, rf_mse2, rf_mse),
  GCV = c(lm_gcv, ridge_gcv, lasso_gcv, spline_gcv, gam_gcv, tree_gcv, rf_gcv3, rf_gcv2, rf_gcv),
  Adjusted_R_Squared = c(lm_adj_r2, ridge_adj_r2, lasso_adj_r2, spline_adj_r2, gam_adj_r2, 
                         tree_adj_r2, rf_adj_r2_3, rf_adj_r2_2, rf_adj_r2)
)

print(results)


results %>% gt() %>%
  tab_header(title = "Model Comparisons") %>%
  tab_style(style = list(cell_fill(color = "#b2f7ef"),
                         cell_text(weight = "bold")),
            locations = cells_body(columns = Model, rows = c(5,7)))%>%
  tab_style(
    style = list(cell_fill(color = "#ffefb5"),
                 cell_text(weight = "bold")), 
    locations = cells_body(columns = GCV, rows = c(5,7)))

        
        
# plot(gam_model, rug = TRUE, main= "Ozone Additive Model")
# 
# plot(residuals(rf_model) ~ fitted(rf_model),
#      xlab = "Fitted", ylab = "Residuals", main = "Resids vs Fitted Random Forest")
# 
# par(mfrow=c(1,1))
# 
# library(dplyr)
# library(lubridate)
# library(forecast)
# library(MASS)
# library(ggplot2)

# rf_model <- boxcox(rf_model, main = "Boxcox Lambda Values")
# 
# gam_model <- boxcox(gam_model, main = "Boxcox Lambda Values")
# 
# # Make predictions
# fitted_values <- predict(rf_model, newdata = ozone)
# 
# # Calculate residuals
# residuals <- ozone$ozone - fitted_values
# 
# # Create a data frame for plotting
# plot_data <- data.frame(Fitted = fitted_values, Residuals = residuals)
# 
# plot(plot_data,
#      xlab = "Fitted", ylab = "Residuals", main = "Resids vs Fit Random Forest")
# 
# importance(rf_model)
# 
# par(cex.lab = 1.5)
# varImpPlot(rf_model, main = "Feature Importance", cex = 1.5)
# par(cex.lab = 1)
# 
# 
# importance_data <- importance(rf_model)
# importance_df <- data.frame(Feature = rownames(importance_data), Importance = importance_data[, 1])
# 
# # Create a ggplot2 plot with customized y-axis labels
# ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
#   geom_bar(stat = "identity") +
#   coord_flip() +
#   labs(x = "Feature", y = "Mean Decrease Gini") +
#   theme(axis.title.x = element_text(size = 14), # X-axis title size
#         axis.title.y = element_text(size = 14), # Y-axis title size
#         axis.text.x = element_text(size = 12),  # X-axis text size
#         axis.text.y = element_text(size = 16))  # Y-axis text size (labels)
# 
# par(mfrow=c(1,3))
# plot(gam_model, rug = TRUE, main = "Ozone Additive Model", main.cex = 1.5, cex=1.5)
