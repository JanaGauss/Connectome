
library(dplyr)
# library(glmnet)
# library(caret)
# devtools::install("src/ConnectomeR")
library(ConnectomeR)

setwd("data/delcode/results")

#### read and prepare results ####
elnet_conn <- readRDS("elnet_conn.rds")
elnet_conn_abs <- readRDS("elnet_conn_abs.rds")
elnet_conn_squ <- readRDS("elnet_conn_squ.rds")
elnet_conn_quadratic <- readRDS("elnet_conn_quadratic.rds")
elnet_agg_zero <- readRDS("elnet_agg_zero.rds")
elnet_agg_max <- readRDS("elnet_agg_max.rds")
elnet_agg_mean <- readRDS("elnet_agg_mean.rds")
elnet_gm_only <- readRDS("elnet_gm_only.rds")
elnet_gm_conn <- readRDS("elnet_gm_conn.rds")
elnet_agg_zero_inter <- readRDS("elnet_agg_zero_inter.rds")
elnet_agg_max_inter <- readRDS("elnet_agg_max_inter.rds")
elnet_agg_mean_inter <- readRDS("elnet_agg_mean_inter.rds")
# elnet_conn_inter <- readRDS("elnet_conn_inter.rds")

result_list <- list(elnet_conn, elnet_conn_abs, elnet_conn_squ, elnet_conn_quadratic, 
                    elnet_agg_zero, elnet_agg_max, elnet_agg_mean,
                    elnet_gm_only, elnet_gm_conn,
                    elnet_agg_zero_inter, elnet_agg_max_inter, elnet_agg_mean_inter# ,
                    # elnet_conn_inter
                    )
names(result_list) <- c("elnet_conn", "elnet_conn_abs", "elnet_conn_squ", "elnet_conn_quadratic",
                        "elnet_agg_zero", "elnet_agg_max", "elnet_agg_mean", 
                        "elnet_gm_only", "elnet_gm_conn", 
                        "elnet_agg_zero_inter", "elnet_agg_max_inter", "elnet_agg_mean_inter" #,
                        # "elnet_conn_inter" 
                        )




#### create table with accuracy, auc ... ####

result_table <- data.frame()

for(i in 1:length(result_list)){
  
  model <- result_list[[i]]
  
  model_name <- names(result_list)[i]
  
  eval <- result_table_elnet(model)
  
  alpha <- eval$accuracy[which(eval$accuracy$value == max(eval$accuracy$value)), "alpha"][1]
  
  
  
  acc_test <- get_confMatrix_elnet(model, 
                                    ind_alpha = which(eval$accuracy$value == max(eval$accuracy$value))[1], 
                                    ind_lambda = eval$accuracy[which(eval$accuracy$value == max(eval$accuracy$value)), "ind_lambda"][1], 
                                    test = TRUE, print = FALSE)$overall["Accuracy"]  %>% round(., 3)*100
  
  auc_test <- get_auc_elnet(model, 
                             ind_alpha = which(eval$accuracy$value == max(eval$accuracy$value))[1], 
                             ind_lambda = eval$accuracy[which(eval$accuracy$value == max(eval$accuracy$value)), "ind_lambda"][1]) %>% round(., 3)*100
  
  acc_train <- get_confMatrix_elnet(model, 
                                    ind_alpha = which(eval$accuracy$value == max(eval$accuracy$value))[1], 
                                    ind_lambda = eval$accuracy[which(eval$accuracy$value == max(eval$accuracy$value)), "ind_lambda"][1], 
                                    test = FALSE, print = FALSE)$overall["Accuracy"]  %>% round(., 3)*100
  
  n_par <- nrow(model$results_models[[1]]$model$beta)
  
  result_table <- result_table %>% 
    rbind(c(model_name, alpha, acc_test, auc_test, acc_train, n_par))
  
}

names(result_table) <- c("model", "alpha", "accuracy_test", "auc_test", "accuracy_train", "n_params")
write.csv(result_table, file = "../../../results/result_table_elnet.csv")


