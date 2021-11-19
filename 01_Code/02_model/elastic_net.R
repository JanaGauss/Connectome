# baseline model: elastic net/lasso/ridge
# first try

# https://glmnet.stanford.edu/articles/glmnet.html

library(dplyr)

devtools::install("01_Code/ConnectomeR") # only necessary if there are changes
library(ConnectomeR)

data <- readRDS("00_Data/Delcode_prepared_2021-11-19.rds")
table(data$prmdiag)




# try preprocessing functions
vars <- names_conn(colnames(data))
d2 <- prep_y(data, y_0 = c("0"), y_1 = c("2", "3"))
table(data$prmdiag)
table(d2$y)

ids <- sample(d2$IDs, size = 470/2)

data_list <- train_test_data(d2, train_IDs = ids)
train <- data_list[["train"]]
test <- data_list[["test"]]

# try model
model <- glmnet(x = train[, vars], y = train$y, family = "binomial", alpha = 0)

pred <- predict(model, newx = as.matrix(test[, vars]), type = "response")
acc <- calc_acc_elnet(pred, test$y)

test_1 <- calc_el_net(train, test, vars, alpha = 0, metric = "accuracy")
test_2 <- calc_el_net(train, test, vars, alpha = 1, metric = "accuracy", nlambda = 200)
test_1$metric_value
test_2$metric_value

# function doing all that stuff at once:
elastic_net <- el_net(data, train_IDs = ids, alpha = c(0, 1), y_0 = c("0"), y_1 = c("2", "3"))
elastic_net$metric_values

elastic_net_2 <- el_net(data, train_IDs = ids, y_0 = c("0"), y_1 = c("2", "3")) # for alpha = seq(0, 1, by = 0.1)
elastic_net_2$metric_values
# -> Lasso (alpha = 0) works best
elnet_best <- elastic_net_2$results_models[[1]]


