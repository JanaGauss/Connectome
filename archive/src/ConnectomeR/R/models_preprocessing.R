# functions for preprocessing for models

#' returns names of connectivity variables
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

#' prepare outcome variable
#'
#' returns dataset with column y added
#' @param data dataset
#' @param y_0 which levels of data$prmdiag use as 0
#' @param y_1 which levels of data$prmdiag use as 1
#' @import dplyr checkmate
#' @export
prep_y <- function(data, y_0, y_1){
  
  checkmate::assert_data_frame(data)
  checkmate::assert_true(all(y_0 %in% levels(data$prmdiag)))
  checkmate::assert_true(all(y_1 %in% levels(data$prmdiag)))
  
  
  data <- data %>%
    mutate(y = case_when(
      data$prmdiag %in% y_0 ~ 0,
      data$prmdiag %in% y_1 ~ 1,
      TRUE ~ NA_real_
    )) %>%
    filter(!is.na(y))
  
  return(data)
}


#' prepare test and training data - Not necessary anymore
#'
#' returns list with train and test data
#' @param data dataset
#' @param train_IDs IDs for training data
#' @param test_IDs IDs for test data, if null, use all data that is not in training data 
#' @import dplyr checkmate
#' @export
train_test_data <- function(data, train_IDs, test_IDs = NULL){
  
  checkmate::assert_data_frame(data)
  checkmate::check_integer(train_IDs)
  checkmate::check_integer(test_IDs, null.ok = TRUE)
  
  data_train <- data %>% filter(IDs %in% train_IDs)
  if(!is.null(test_IDs)){
    data_test <- data %>% filter(IDs %in% test_IDs)
  } else{
    data_test <- setdiff(data, data_train)
  }
  
  result <- list(data_train, data_test)
  names(result) <- c("train", "test")
  
  return(result)
}

