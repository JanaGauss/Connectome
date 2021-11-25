# R functions for models

#' elastic net model 
#'
#' calculates elastic net model for several alpha values and chooses best model according to some metric
#' @param test test data
#' @param train training data
#' @param vars which variables should be used? If null, columns containing connectivity data are automatically extracted
#' @param alpha alpha values for glmnet. 0 = Lasso, 1 = Ridge
#' @param y_0 which levels of data$prmdiag use as 0
#' @param y_1 which levels of data$prmdiag use as 1
#' @param data_source which dataset is it? (in case there have to be made adjustments)
#' @param metric which metric is used to find the best lambda? !!! ToDo: add more metrics, think about advantages/disadvantages
#' @param ... Additional parameters for glmnet function, e.g. nlambda
#' @import dplyr glmnet checkmate
#' @export
el_net <- function(test, train, vars = NULL, alpha = seq(0, 1, by = 0.1), y_0, y_1, 
                   data_source = "DELCODE", metric = "accuracy", ...){


  checkmate::assert_data_frame(test)
  checkmate::assert_data_frame(train)
  checkmate::assert_character(vars, null.ok = TRUE)
  checkmate::assert_numeric(alpha, lower = 0, upper = 1)
  checkmate::assert_true(all(y_0 %in% levels(train$prmdiag)))
  checkmate::assert_true(all(y_1 %in% levels(train$prmdiag)))
  checkmate::assert_choice(metric, choices = c("accuracy")) # ToDo: add more metrics
  checkmate::assert_choice(data_source, choices = c("DELCODE"))
  
  if(!is.null(vars)){
    checkmate::assert_true(vars %in% colnames(train))
  } else{
    # extract all columns with connectivity data automatically
    vars <- names_conn(colnames(train))
  }

  # prepare data
  test <- prep_y(test, y_0, y_1)
  train <- prep_y(train, y_0, y_1)
  
  data_list <- list(test, train)
  names(data_list) <- c("test", "train")

  # calculate elastic net for all alpha values
  results <- list()
  metric_values <- rep(0, length(alpha))
  for(a in 1:length(alpha)){
    results[[a]] <- calc_el_net(train, test, vars, alpha = alpha[a], metric, ...)
    metric_values[a] <- results[[a]][["metric_value"]]
  }
  
  model_best <- results[[which(metric_values == max(metric_values))[1]]]
  
  res <- list(results, metric_values, data_list, model_best)
  names(res) <- c("results_models", "metric_values", "data_list", "model_best")
  return(res)
  
  
}


#' elastic net model for one alpha
#'
#' calculates elastic net model for one alpha value and chooses best model according to some metric (returns list with information)
#' @param data_train training data
#' @param data_test test data
#' @param vars which variables should be used? If null, columns containing connectivity data are automatically extracted
#' @param alpha alpha value for glmnet. 0 = Lasso, 1 = Ridge
#' @param metric which metric is used to find the best lambda? !!! ToDo: add more metrics, think about advantages/disadvantages
#' @param ... Additional parameters for glmnet function, e.g. nlambda
#' @import dplyr glmnet checkmate
#' @export
calc_el_net <- function(data_train, data_test, vars, alpha = 0, metric = "accuracy", ...){
  
  checkmate::assert_data_frame(data_train)
  checkmate::assert_data_frame(data_test)
  checkmate::assert_character(vars)
  checkmate::assert_number(alpha, lower = 0, upper = 1)
  checkmate::assert_choice(metric, choices = c("accuracy")) # ToDo: add more metrics
  
  model <- glmnet(x = data_train[, vars], y = data_train$y, family = "binomial", alpha = alpha, ...)
  
  pred <- predict(model, newx = as.matrix(data_test[, vars]), type = "response") # predictions on test data for all lambda
  
  if(metric == "accuracy"){
    acc <- calc_acc_elnet(pred, data_test$y)
    ind_best_lambda <- which(acc == max(acc))[1] # [1] -> take only one lambda if there several best ones
    
    value_metric <- max(acc)
    
  } else{
    # ToDo
  }
  
  # return information of best model
  result <- list()
  result[["model"]] <- model
  result[["alpha"]] <- alpha
  result[["a0"]] <- model$a0[ind_best_lambda] # Intercept
  result[["beta"]] <- model$beta[, ind_best_lambda]
  result[["ind_lambda"]] <- ind_best_lambda
  result[["lambda"]] <- model$lambda[ind_best_lambda]
  result[["metric_value"]] <- value_metric
  
  return(result)

}



#' calculate accuracy for one elastic net model
#'
#' calculates accuracy for one elastic net model for several lambda values, returns accuracy for every lambda
#' @param pred matrix with predictions
#' @param y real y of predicitons
#' @import dplyr caret checkmate
#' @export
calc_acc_elnet <- function(pred, y){
  
  checkmate::assert_matrix(pred)
  checkmate::assert_numeric(y)
  checkmate::assert_true(nrow(pred) == length(y))
  
  pred_01 <- matrix(as.integer(pred >= 0.5), byrow = FALSE, nrow = nrow(pred))
  y <- factor(y)
  
  acc <- rep(0, ncol(pred))
  for(i in 1:ncol(pred)){ # calculate accuracy for each lambda value (=each column of prediction matrix)
    acc[i] <- confusionMatrix(factor(pred_01[, i], levels = c("0", "1")), y)$overall["Accuracy"]
  }
  
  return(acc)
}


#' get confusion matrix for best elastic net model
#'
#' @param elnet_result result of el_net function
#' @import dplyr caret checkmate
#' @export
get_confMatrix_elnet <- function(elnet_result){
  
  pred <- predict(elnet_result$model_best$model,
                  newx = as.matrix(elnet_result$data_list$test[, names(elnet_result$model_best$beta)]), 
                  type = "response", s = elnet_result$model_best$lambda)
  
  x <- confusionMatrix(data = factor(elnet_result$data_list$test$y), 
                       reference = factor(as.integer(pred>0.5), levels = c("0", "1")),
                       positive = "1")
  
  return(x)
}
