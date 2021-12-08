# example code for reading and preparing data in R


# devtools::install("01_Code/ConnectomeR") # only necessary if there are changes
library(ConnectomeR)

# prepare Delcode-data
storefile <- paste0("00_Data/Delcode_prepared_", Sys.Date())
data_list <- read_data(path = "00_Data", store_rds = TRUE, store_file = storefile, 
                  data_source = "DELCODE") # takes a while ...

str(data_list$train[, 1:20])
table(data_list$train$prmdiag)
table(data_list$train$sex) 

# load saved data
file_delcode <- "00_Data/Delcode_prepared_2021-11-25train.rds"
train <- readRDS(file_delcode)
