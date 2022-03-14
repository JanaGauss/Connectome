library(ConnectomeR)
library(dplyr)
library(stringr)

train_conn <- readRDS("data/delcode/conn_matrix/train.rds")
regions_dat <- read.csv("references/subregion_func_network_Yeo_updated.csv", sep = ";")[, c(1, 4)]

beta <- names_conn(colnames(train_conn)) # extract connectivity variables

dat <- data.frame(beta = beta, 
                  var1 = as.numeric(str_extract(beta, "\\d+")),
                  var2 = as.numeric(str_replace(str_extract(beta, "_\\d+"), "_", "")))

dat <- dat %>% rbind(data.frame(beta = beta,
                                var1 = dat$var2,
                                var2 = dat$var1))

dat <- dat %>% left_join(data.frame(var1 = regions_dat[,1], region1 = regions_dat[,2]))
dat <- dat %>% left_join(data.frame(var2 = regions_dat[,1], region2 = regions_dat[,2]))

dat$region_smaller <- pmin(dat$region1, dat$region2)
dat$region_bigger <- pmax(dat$region1, dat$region2)


dat$region <- paste0(dat$region_smaller, "_", dat$region_bigger)      

unique(dat$region)

result <- dat %>%
  select(conn_name = beta, region)

write.csv(result, file = "references/colnames_to_yeo7.csv")
