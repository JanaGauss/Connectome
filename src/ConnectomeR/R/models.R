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
#' @param target_diag diagnosis as target variable (logistic regression) or linear model for MEM-score
#' @param option standard: standard regression model, interactions: all two way interactions, quadratic: all quadratic terms (x1^2, x2^2, but no interactions), abs: absolute value of variables, squ: squared value of variables
#' @param ... Additional parameters for glmnet function, e.g. nlambda
#' @import dplyr glmnet checkmate
#' @export
el_net <- function(test, train, vars = NULL, alpha = seq(0, 1, by = 0.1), y_0 = NULL, y_1 = NULL, 
                   data_source = "DELCODE", target_diag = TRUE, option = "standard", ...){


  checkmate::assert_data_frame(test)
  checkmate::assert_data_frame(train)
  checkmate::assert_character(vars, null.ok = TRUE)
  checkmate::assert_numeric(alpha, lower = 0, upper = 1)
  checkmate::assert_choice(data_source, choices = c("DELCODE"))
  checkmate::assert_choice(option, choices = c("standard", "interactions", "quadratic", "abs", "squ"))
  
  if(target_diag == TRUE){
    checkmate::assert_true(!is.null(y_0))
    checkmate::assert_true(!is.null(y_1))
    checkmate::assert_true(all(y_0 %in% levels(train$prmdiag)))
    checkmate::assert_true(all(y_1 %in% levels(train$prmdiag)))
  }
  
  
  if(!is.null(vars)){
    checkmate::assert_true(all(vars %in% colnames(train)))
  } else{
    # extract all columns with connectivity data automatically
    vars <- names_conn(colnames(train))
  }

  # prepare data
  if(target_diag == TRUE){
    test <- prep_y(test, y_0, y_1)
    train <- prep_y(train, y_0, y_1)
  } else{
    test$y <- test$MEM_score
    train$y <- train$MEM_score
  }
  
  
  # select only relevant variables
  test <- test %>% select(y, vars)
  train <- train %>% select(y, vars)
  
  # only complete cases
  test <- test[complete.cases(test), ]
  train <- train[complete.cases(train), ]
  
  # store data 
  data_list <- list(test, train)
  names(data_list) <- c("test", "train")
  
  
  # calculate elastic net for all alpha values
  results <- list()
  for(a in 1:length(alpha)){
    print(paste0("Calculation for alpha = ", alpha[a]))
    results[[a]] <- calc_el_net(train, test, alpha = alpha[a], target_diag, option, ...)
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
#' @param alpha alpha value for glmnet. 0 = Lasso, 1 = Ridge
#' @param target_diag diagnosis as target variable (logistic regression) or linear model for MEM-score
#' @param option standard regression model, all two way interactions, ..., see el_net function
#' @param ... Additional parameters for glmnet function, e.g. nlambda
#' @import dplyr glmnet checkmate
#' @export
calc_el_net <- function(data_train, data_test, alpha = 0, target_diag, option, ...){
  
  checkmate::assert_data_frame(data_train)
  checkmate::assert_data_frame(data_test)
  checkmate::assert_number(alpha, lower = 0, upper = 1)
  
  
  if(target_diag == TRUE){
    fam <- "binomial"
  } else{
    fam <- "gaussian"
  }
  
  if(option == "standard"){
    
    model <- glmnet(x = select(data_train, -y), y= data_train$y, family = fam, alpha = alpha, ...)
    
    # model matrix for predicitons based on test data
    new_x <- as.matrix(select(data_test, -y))
    
  } else if(option == "interactions"){
    
    # include all two way interactions in model matrix
    model <- glmnet(x = model.matrix(~.^2, data = select(data_train, -y)), y= data_train$y, family = fam, alpha = alpha, ...)
    
    # model matrix for predicitons based on test data
    new_x <- model.matrix(~.^2, data = select(data_test, -y))
    
    } else if(option == "quadratic") {
      
      # include all quadratic terms in model matrix
      squared_terms <- data_train %>% 
        select(-y) %>% 
        select_if(is.numeric) %>%
        mutate_all(function(x) x^2)
      dat_model <- cbind(select(data_train, -y), squared_terms)
      
      model <- glmnet(x = dat_model, y= data_train$y, family = fam, alpha = alpha, ...)
      
      
      # model matrix for predicitons based on test data
      squared_terms_new <- data_test %>% 
        select(-y) %>%  
        select_if(is.numeric) %>%
        mutate_all(function(x) x^2)
      dat_model_new <- cbind(select(data_test, -y), squared_terms_new)
      new_x <- as.matrix(dat_model_new)
      
    } else if(option == "abs"){
        
      # include only absolute values
      abs_terms <- data_train %>% 
        select(-y) %>% 
        select_if(is.numeric) %>%
        mutate_all(function(x) abs(x))
      dat_model <- cbind(select(data_train, -y), abs_terms)
      
      model <- glmnet(x = dat_model, y= data_train$y, family = fam, alpha = alpha, ...)
      
      
      # model matrix for predicitons based on test data
      abs_terms_new <- data_test %>% 
        select(-y) %>%
        select_if(is.numeric) %>%
        mutate_all(function(x) abs(x))
      dat_model_new <- cbind(select(data_test, -y), abs_terms_new)
      new_x <- as.matrix(dat_model_new)
      
    } else if(option == "squ"){
      
      # use squared value of (numeric) variables
      dat_model <- data_train %>% 
        select(-y) %>% 
        mutate_if(is.numeric, function(x) x^2)
      
      model <- glmnet(x = dat_model, y= data_train$y, family = fam, alpha = alpha, ...)
      
      
      # model matrix for predicitons based on test data
      dat_model_new <- data_test %>% 
        select(-y) %>%  
        mutate_if(is.numeric, function(x) x^2)
      new_x <- as.matrix(dat_model_new)
      }
   
  class(new_x) <- "numeric"
  pred <- predict(model, newx = new_x, type = "response") # predictions on test data for all lambda
  
  eval <- evaluation_elnet(pred, data_test$y)
  
  result <- list()
  result[["model"]] <- model
  result[["alpha"]] <- alpha
  result[["evaluation"]] <- eval
  
  return(result)

}



#' evaluation for one elastic net model
#'
#' calculates accuracy, auc etc (logistic regression) or MSE (LM) for one elastic net model for several lambda values
#' @param pred matrix with predictions
#' @param y real y of predicitons
#' @import dplyr caret checkmate pROC
#' @export
evaluation_elnet <- function(pred, y){
  
  checkmate::assert_matrix(pred)
  checkmate::assert_numeric(y)
  checkmate::assert_true(nrow(pred) == length(y))
  
  if(sum(y == 0 | y == 1) == length(y)){ # logistic regression 
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
  } else{ # LM -> MSE on test data
    MSE <- rep(0, ncol(pred))
    for(i in 1:ncol(pred)){ # for all lambda values
      MSE[i] <- mean((pred[, i] - y)^2)
      }
    return(MSE)
  }
  
}


