# R functions for models

#' elastic net model
#'
#' @param data dataset
#' @param vars which variables should be used? If null, columns containing connectivity data are automatically extracted
#' @param test_IDs IDs for test data
#' @param alpha alpha value for glmnet. 0 = Lasso, 1 = Ridge
#' @param y_0 which levels of data$prmdiag use as 0
#' @param y_1 which levels of data$prmdiag use as 1
#' @param data_source which dataset is it? (in case there have to be made adjustments)
#' @param metric which metric is used to find the best lambda? !!! ToDo: add more metrics, think about advantages/disadvantages
#' @param ... Additional parameters for glmnet function, e.g. nlambda
#' @import dplyr glmnet checkmate
#' @export
el_net <- function(data, vars = NULL, test_IDs, alpha, y_0, y_1, metric = "accuracy", ...){
  
  
  checkmate::assert_data_frame(data)
  checkmate::assert_character(vars, null.ok = TRUE)
  checkmate::check_integer(test_IDs)
  checkmate::assert_number(alpha, lower = 0, upper = 1)
  checkmate::assert_true(y_0 %in% levels(data$prmdiag))
  checkmate::assert_true(y_1 %in% levels(data$prmdiag))
  checkmate::assert_choice(metric, choices = c("accuracy")) # ToDo: add more metrics
  if(!is.null(vars)){
    checkmate::assert_true(vars %in% colnames(data))
  } else{ 
    # extract all columns with connectivity data automatically
    vars <- names_conn(colnames(data))
  }
  
  # prepare data
  
  data <- data %>%
    mutate(y = case_when(
      data$prmdiag %in% y_0 ~ 0,
      data$prmdiag %in% y_1 ~ 1,
      TRUE ~ NA_real_
    )) %>%
    filter(!is.na(y)) %>%
    select(IDs, y, vars)
    
  data_test <- data %>% filter(IDs %in% test_IDs)
  data_train <- setdiff(data, data_test)
  
  # calculate model (for several lambda values)
  model <- glmnet(x = data_train[, vars], y = data_train$y, family = "binomial", alpha = alpha, ...)
  
  # find best model according to metric
  # 1. predict fitted values on test data for all lambda
  pred_test <- predict(model, newx = as.matrix(data_test[, vars]), s = model$a0, # for all lambda values
                       type = "response")
  
  # 2. calculate metric for all lambda
  if(metric == "accuracy"){
    
  }
  
  
}

#' extract names of connectivity variables
#'
#' @param colnames colnames of dataset
#' @import dplyr stringr checkmate
#' @export
names_conn <- function(colnames){
  
  checkmate::assert_character(colnames)
  
  cols <- str_detect(colnames, "\\d+_\\d+") # search for one or more digits, _, one or more digits
  vars <- colnames[cols]
  return(vars)
}

