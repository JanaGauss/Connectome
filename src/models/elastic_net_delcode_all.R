# All elastic net models for delcode data

library(dplyr)
library(glmnet)
library(caret)
# devtools::install("src/ConnectomeR")
library(ConnectomeR)

# set wd to source file location first
setwd("../../data/delcode")


#### read data ####

train_conn <- readRDS("conn_matrix/train.rds")
train_agg_zero <- readRDS("aggregated_regions/GreaterZero/train.rds")
train_agg_max <- readRDS("aggregated_regions/Max/train.rds")
train_agg_mean <- readRDS("aggregated_regions/Mean/train.rds")
train_gm_only <- readRDS("graph_metrics/gm_only/train.rds")
train_gm_conn <- readRDS("graph_metrics/gm_conn/train.rds")

test_conn <- readRDS("conn_matrix/test.rds")
test_agg_zero <- readRDS("aggregated_regions/GreaterZero/test.rds")
test_agg_max <- readRDS("aggregated_regions/Max/test.rds")
test_agg_mean <- readRDS("aggregated_regions/Mean/test.rds")
test_gm_only <- readRDS("graph_metrics/gm_only/test.rds")
test_gm_conn <- readRDS("graph_metrics/gm_conn/test.rds")


train_list <- list(train_conn, train_agg_zero, train_agg_max, train_agg_mean, train_gm_only, train_gm_conn)
test_list <- list(test_conn, test_agg_zero, test_agg_max, test_agg_mean, test_gm_only, test_gm_conn)
names(train_list) <- c("conn", "agg_zero", "agg_max", "agg_mean", "gm_only", "gm_conn")
names(test_list) <- c("conn", "agg_zero", "agg_max", "agg_mean", "gm_only", "gm_conn")


#### prepare parameters for calculation ####

# data frame containing name of model, data, option
params <- rbind(
  c("elnet_conn", "conn", "standard"),
  c("elnet_agg_zero", "agg_zero", "standard"),
  c("elnet_agg_max", "agg_max", "standard"),
  c("elnet_agg_mean", "agg_mean", "standard"),
  c("elnet_gm_only", "gm_only", "standard"),
  c("elnet_gm_conn", "gm_conn", "standard"),
  c("elnet_conn_abs", "conn", "abs"),
  c("elnet_conn_squ", "conn", "squ"),
  c("elnet_conn_quadratic", "conn", "quadratic"),
  c("elnet_agg_zero_inter", "agg_zero", "interactions"),
  c("elnet_agg_max_inter", "agg_max", "interactions"),
  c("elnet_agg_mean_inter", "agg_mean", "interactions") #,
  # c("elnet_conn_inter", "conn", "interactions")
  )

params <- as.data.frame(params)
colnames(params) <- c("model", "data", "option")

# variables that have to be removed from the features
vars_remove <- c("ConnID", "Repseudonym", "siteid", "visdat", "prmdiag", "IDs", "Apoe", "MEM_score")



#### calculation models ####

setwd("results")

for(i in 1:nrow(params)){
  
  name_model <- params$model[i]
  train <- train_list[[params$data[i]]]
  test <- test_list[[params$data[i]]]
  option_model <- params$option[i]
  
  vars_model <- colnames(train)[!colnames(train) %in% vars_remove]
  
  print(name_model)
  
  model <- el_net(test = test, train = train, 
                  y_0 = c("0"), y_1 = c("2", "3"), 
                  vars = vars_model,
                  option = option_model)
  
  saveRDS(model, file = paste0(name_model, ".rds"))
  
}



#### intercept model accuraccy and model without connectivity data ####

vars_without_conn <- c("age", "sex", "edyears")

model_without_conn <- el_net(test = test_conn, train = train_conn,
                             y_0 = c("0"), y_1 = c("2", "3"),
                             vars = vars_without_conn)

result_table_elnet(model_without_conn) # 71.8 (best accuracy)
acc_intercept(model_without_conn) # 56.5
