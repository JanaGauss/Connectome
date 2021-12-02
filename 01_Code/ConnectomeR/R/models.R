# R functions for models

#' elastic net model 
#'
#' calculates elastic net model for several alpha values
#' @param test test data
#' @param train training data
#' @param vars which variables should be used? If null, columns containing connectivity data are automatically extracted
#' @param alpha alpha values for glmnet. 0 = Lasso, 1 = Ridge
#' @param y_0 which levels of data$prmdiag use as 0
#' @param y_1 which levels of data$prmdiag use as 1
#' @param data_source which dataset is it? (in case there have to be made adjustments)
#' @param ... Additional parameters for glmnet function, e.g. nlambda
#' @import dplyr glmnet checkmate
#' @export
el_net <- function(test, train, vars = NULL, alpha = seq(0, 1, by = 0.1), y_0, y_1, 
                   data_source = "DELCODE", ...){


  checkmate::assert_data_frame(test)
  checkmate::assert_data_frame(train)
  checkmate::assert_character(vars, null.ok = TRUE)
  checkmate::assert_numeric(alpha, lower = 0, upper = 1)
  checkmate::assert_true(all(y_0 %in% levels(train$prmdiag)))
  checkmate::assert_true(all(y_1 %in% levels(train$prmdiag)))
  checkmate::assert_choice(data_source, choices = c("DELCODE"))
  
  if(!is.null(vars)){
    checkmate::assert_true(all(vars %in% colnames(train)))
  } else{
    # extract all columns with connectivity data automatically
    vars <- names_conn(colnames(train))
  }

  # prepare data
  test <- prep_y(test, y_0, y_1)
  train <- prep_y(train, y_0, y_1)
  
  
  test <- test %>% select(y, vars_model)
  train <- train %>% select(y, vars_model)
  
  test <- test[complete.cases(test), ]
  train <- train[complete.cases(train), ]
  
  
  data_list <- list(test, train)
  names(data_list) <- c("test", "train")

  # calculate elastic net for all alpha values
  results <- list()
  for(a in 1:length(alpha)){
    results[[a]] <- calc_el_net(train, test, vars, alpha = alpha[a], ...)
  }
  
  
  res <- list(results, data_list)
  names(res) <- c("results_models", "data_list")
  return(res)
  
  
}


#' elastic net model for one alpha
#'
#' calculates elastic net model for one alpha value (returns list with information and evaluation)
#' @param data_train training data
#' @param data_test test data
#' @param vars which variables should be used? If null, columns containing connectivity data are automatically extracted
#' @param alpha alpha value for glmnet. 0 = Lasso, 1 = Ridge
#' @param ... Additional parameters for glmnet function, e.g. nlambda
#' @import dplyr glmnet checkmate
#' @export
calc_el_net <- function(data_train, data_test, vars, alpha = 0, ...){
  
  checkmate::assert_data_frame(data_train)
  checkmate::assert_data_frame(data_test)
  checkmate::assert_character(vars)
  checkmate::assert_number(alpha, lower = 0, upper = 1)
  
  model <- glmnet(x = data_train[, vars], y = data_train$y, family = "binomial", alpha = alpha, ...)
  
  new_x <- as.matrix(data_test[, vars])
  class(new_x) <- "numeric"
  pred <- predict(model, newx = new_x, type = "response") # predictions on test data for all lambda
  
  eval <- evaluation_elnet(pred, data_test$y)
  
  # return information of best model
  result <- list()
  result[["model"]] <- model
  result[["alpha"]] <- alpha
  result[["evaluation"]] <- eval
  
  return(result)

}



#' evaluation for one elastic net model
#'
#' calculates accuracy, auc for one elastic net model for several lambda values
#' @param pred matrix with predictions
#' @param y real y of predicitons
#' @import dplyr caret checkmate pROC
#' @export
evaluation_elnet <- function(pred, y){
  
  checkmate::assert_matrix(pred)
  checkmate::assert_numeric(y)
  checkmate::assert_true(nrow(pred) == length(y))
  
  pred_01 <- matrix(as.integer(pred >= 0.5), byrow = FALSE, nrow = nrow(pred))
  y <- factor(y)
  
  acc <- auc <- precision <- recall <- F1 <- rep(0, ncol(pred))
  
  for(i in 1:ncol(pred)){ # calculate metrics for each lambda value (=each column of prediction matrix)
    acc[i] <- confusionMatrix(data = factor(pred_01[, i], levels = c("0", "1")), reference = y)$overall["Accuracy"]
    auc[i] <- roc(response = y, predictor = pred[, i], quiet = TRUE)$auc
    precision[i] <- precision(factor(pred_01[, i], levels = c("0", "1")), y, relevant = c("1"))
    recall[i] <- recall(factor(pred_01[, i], levels = c("0", "1")), y, relevant = c("1"))
    F1[i] <- F_meas(factor(pred_01[, i], levels = c("0", "1")), y, relevant = c("1"))
  }
  
  metrics <- list(acc, auc, precision, recall, F1)
  names(metrics) <- c("accuracy", "auc", "precision", "recall", "F1")
  
  return(metrics)
}


#' get result table based on metric for el_net result
#'
#' @param elnet_result result of el_net function
#' @import dplyr 
#' @export
result_table_elnet <- function(elnet_result){
  
  result <- list()
  
  for(i in c("accuracy", "auc", "precision", "recall", "F1")){
    dat_best <- data.frame(value = NA, ind_lambda = NA, alpha = NA)
    for(j in 1:length(elnet_result$results_models)){
      max_metric_j <- max(elnet_result$results_models[[j]]$evaluation[[i]])
      ind_best_j <- which(elnet_result$results_models[[j]]$evaluation[[i]] == max_metric_j)[1] # [1] -> take only one lambda if there several best ones
      alpha_j <- elnet_result$results_models[[j]]$alpha
      
      dat_best <- rbind(dat_best, c(max_metric_j, ind_best_j, alpha_j))
    }
    
    result[[length(result) + 1]] <- dat_best[-1, ]
  }
  
  
  names(result) <- c("accuracy", "auc", "precision", "recall", "F1")
  return(result)
}


#' get confusion matrix for elastic net model
#'
#' @param elnet_result result of el_net function
#' @param ind_alpha index of alpha that should be used
#' @param ind index of lambda value that should be used
#' @import dplyr caret checkmate
#' @export
get_confMatrix_elnet <- function(elnet_result, ind_alpha, ind_lambda){
  
  print(paste("alpha: ", elnet_result$results_models[[ind_alpha]]$alpha))
  print(paste("lambda: ", elnet_result$results_models[[ind_alpha]]$model$lambda[ind_lambda]))
  
  new_x <- as.matrix(elnet_result$data_list$test[, dimnames(elnet_result$results_models[[ind_alpha]]$model$beta)[[1]]])
  class(new_x) <- "numeric"
  pred <- predict(elnet_result$results_models[[ind_alpha]]$model, newx = new_x, type = "response") 
  
  x <- confusionMatrix(reference = factor(elnet_result$data_list$test$y), 
                       data = factor(as.integer(pred[, ind_lambda]>0.5), levels = c("0", "1")),
                       positive = "1")
  
  return(x)
}
