# example code for reading and preparing data in R


devtools::install("01_Code/ConnectomeR") # only necessary if there are changes
library(ConnectomeR)

# prepare Delcode-data
storefile <- paste0("00_Data/Delcode_prepared_", Sys.Date(), ".rds")
data <- read_data(file = "00_Data/merged_matrices.csv", store_rds = TRUE, store_file = storefile, 
                  data_source = "DELCODE") # takes a while ...

str(data[, 1:20])
table(data$prmdiag)
table(data$sex) # Boris' excel says 1 = female, 2 = male but that seems unrealistic, 2 is maybe divers? ToDo: ask him if 0 = male or female


# load saved data
file_delcode <- "00_Data/Delcode_prepared_2021-11-19.rds"
data <- readRDS(storefile)
