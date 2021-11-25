# R functions for data preprocessing

#' Read and prepare datasets. Returns data (and the data can be stored additionally)
#'
#' @param path path to test and train csv files
#' @param store_rds should the prepared data be stored as .rds file?
#' @param store_file name of the stored data file if store_rds is true (without .rds)
#' @param data_source which dataset is it? (in case there have to be made adjustments)
#' @import dplyr readr checkmate
#' @export
read_data <- function(path = "00_Data", store_rds = FALSE, store_file = NULL, data_source = "DELCODE"){
  checkmate::assert_choice(data_source, choices = c("DELCODE"))
  
  if(store_rds == TRUE){
    checkmate::assert_true(!is.null(store_file))
  }
  
  train <- read_csv(paste0(path, "/train.csv"))
  test <- read_csv(paste0(path, "/test.csv"))
  
  if(data_source == "DELCODE"){
    train$visdat <- as.Date(train$visdat, format = c("%d.%m.%Y"))
    train$sex <- factor(train$sex)
    train$prmdiag <- factor(train$prmdiag)
    
    test$visdat <- as.Date(test$visdat, format = c("%d.%m.%Y"))
    test$sex <- factor(test$sex)
    test$prmdiag <- factor(test$prmdiag)
  }
  
  data_list <- list(test, train)
  names(data_list) <- c("test", "train")
  
  if(store_rds == TRUE){
    saveRDS(train, file = paste0(store_file, "train.rds"))
    saveRDS(test, file = paste0(store_file, "test.rds"))
    return(list(test, train))
  } else{
    return(list(test, train))
  }
  
}

