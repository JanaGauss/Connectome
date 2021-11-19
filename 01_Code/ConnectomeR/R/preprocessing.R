# R functions for data preprocessing

#' Read and prepare datasets. Returns data (and the data can be stored additionally)
#'
#' @param file name of csv file of the dataset
#' @param store_rds should the prepared data be stored as .rds file?
#' @param store_file name of the stored data file if store_rds is true
#' @param data_source which dataset is it? (in case there have to be made adjustments)
#' @import dplyr readr checkmate
#' @export
read_data <- function(file, store_rds = FALSE, store_file = NULL, data_source = "DELCODE"){
  checkmate::assert_choice(data_source, choices = c("DELCODE"))
  
  if(store_rds == TRUE){
    checkmate::assert_true(!is.null(store_file))
    checkmate::assertPathForOutput(store_file, overwrite = TRUE, extension = "rds")
  }
  
  data <- read_csv(file)
  
  if(data_source == "DELCODE"){
    data$visdat <- as.Date(data$visdat, format = c("%d.%m.%Y"))
    data$sex <- factor(data$sex)
    data$prmdiag <- factor(data$prmdiag)
  }
  
  if(store_rds == TRUE){
    saveRDS(data, file = store_file)
    return(data)
  } else{
    return(data)
  }
  
}

