# baseline model: elastic net/lasso/ridge
# first try

# https://glmnet.stanford.edu/articles/glmnet.html

library(dplyr)
library(glmnet)
library(caret)
# devtools::install("01_Code/ConnectomeR") # only necessary if there are changes
library(ConnectomeR)


test <- readRDS("00_Data/Delcode_prepared_2021-11-30test.rds") %>% filter(!is.na(Apoe))
train <- readRDS("00_Data/Delcode_prepared_2021-11-30train.rds") %>% filter(!is.na(Apoe))
table(train$prmdiag)


elastic_net <- el_net(test = test, train = train, y_0 = c("0"), y_1 = c("2", "3")) # only with connectivity variables
result_table_elnet(elastic_net)




vars_remove <- c("ConnID", "Repseudonym", "siteid", "visdat", "prmdiag", "IDs")
vars_model <- colnames(train)[!colnames(train) %in% vars_remove]

elastic_net_2 <- el_net(test = test, train = train, y_0 = c("0"), y_1 = c("2", "3"),
                        vars = vars_model) # for alpha = seq(0, 1, by = 0.1)

results_eval <- result_table_elnet(elastic_net_2)
results_eval
max(results_eval$accuracy$value)
results_eval$accuracy[which(results_eval$accuracy$value == max(results_eval$accuracy$value)), ]

get_confMatrix_elnet(elastic_net_2, ind_alpha = 11, ind_lambda = 7)

new_x <- as.matrix(elastic_net_2$data_list$test[, dimnames(elastic_net_2$results_models[[11]]$model$beta)[[1]]])
class(new_x) <- "numeric"
pred <- predict(elastic_net_2$results_models[[11]]$model, newx = new_x, type = "response")[, 7] 



#### old stuff, may not work #####
new_x <- as.matrix(elnet_result$data_list$test[, vars_model])
class(new_x) <- "numeric"
pred_2 <- predict(elnet_result$results_models[[11]]$model,
                newx = new_x, 
                type = "response")
confusionMatrix(data = factor(elnet_result$data_list$test$y), reference = factor(as.integer(pred_2[, 100]>0.5), levels = c("0", "1")),
                positive = "1")

elastic_net_2$metric_values
elastic_net_2$model_best$alpha

confMat <- get_confMatrix_elnet(elastic_net_2)
confMat
confMat$table

xy <- calc_el_net(train, test, vars_model)

# old stuff
data <- readRDS("00_Data/Delcode_prepared_2021-11-19.rds")
table(data$prmdiag)

# try preprocessing functions
vars <- names_conn(colnames(data))
d2 <- prep_y(data, y_0 = c("0"), y_1 = c("2", "3"))
table(data$prmdiag)
table(d2$y)

ids <- sample(d2$IDs, size = 470/2)

data_list <- train_test_data(d2, train_IDs = ids)
train <- data_list[["train"]] %>% filter(!is.na(Apoe))
test <- data_list[["test"]] %>% filter(!is.na(Apoe))

# try model
model <- glmnet(x = train[, vars_model], y = train$y, family = "binomial", alpha = 0)

new_x <- as.matrix(test[, vars_model])
class(new_x) <- "numeric"
pred <- predict(model, newx = new_x, type = "response")


t <- evaluation_elnet(pred, test$y)

test_1 <- calc_el_net(train, test, vars_model, alpha = 0, metric = "accuracy")
test_2 <- calc_el_net(train, test, vars, alpha = 1, metric = "accuracy", nlambda = 200)
test_1$metric_value
test_2$metric_value




# function doing all that stuff at once:
elastic_net_2 <- el_net(test = test, train = train, y_0 = c("0"), y_1 = c("2", "3")) # for alpha = seq(0, 1, by = 0.1)
elastic_net_2$metric_values
elastic_net_2$model_best$alpha

confMat <- get_confMatrix_elnet(elastic_net_2)
confMat
confMat$table
