library(ConnectomeR)
library(dplyr)

# connectivity data
data_conn <- read_data(path = "data/delcode/conn_matrix", store_rds = TRUE, store_file = "data/delcode/conn_matrix/", 
                       data_source = "DELCODE")


# data aggregated by regions
data_agg_zero <- read_data(path = "data/delcode/aggregated_regions/GreaterZero", store_rds = TRUE, 
                           store_file = "data/delcode/aggregated_regions/GreaterZero/", 
                           data_source = "DELCODE")
data_agg_max <- read_data(path = "data/delcode/aggregated_regions/Max", store_rds = TRUE, 
                           store_file = "data/delcode/aggregated_regions/Max/", 
                           data_source = "DELCODE")
data_agg_mean <- read_data(path = "data/delcode/aggregated_regions/Mean", store_rds = TRUE, 
                           store_file = "data/delcode/aggregated_regions/Mean/", 
                           data_source = "DELCODE")

# graph metrics
test_gm <- read_csv("data/delcode/graph_metrics/test.csv")
train_gm <- read_csv("data/delcode/graph_metrics/train.csv")

test_gm_0 <- data_conn[[1]] %>%
  filter(prmdiag %in% c(0, 2, 3)) %>%
  select(prmdiag)
  
train_gm_0 <- data_conn[[2]] %>%
  filter(prmdiag %in% c(0, 2, 3)) %>%
  select(prmdiag)

test_gm$prmdiag <- test_gm_0$prmdiag
train_gm$prmdiag <- train_gm_0$prmdiag

test_gm <- test_gm %>% select(-"...1")
train_gm <- train_gm %>% select(-"...1")
test_gm <- test_gm %>% select(-target)
train_gm <- train_gm %>% select(-target)
# contain graph metrics and connectivity data
saveRDS(test_gm, file = "data/delcode/graph_metrics/gm_conn/test.rds")
saveRDS(train_gm, file = "data/delcode/graph_metrics/gm_conn/train.rds")


# select only graph metrics data (remove connectivity data)
cols_gm <- colnames(train_gm)[!colnames(train_gm) %in% names_conn(colnames(train_gm))]
train_gm_only <- train_gm %>%
  select(cols_gm)
test_gm_only <- test_gm %>%
  select(cols_gm)
saveRDS(test_gm_only, file = "data/delcode/graph_metrics/gm_only/test.rds")
saveRDS(train_gm_only, file = "data/delcode/graph_metrics/gm_only/train.rds")


