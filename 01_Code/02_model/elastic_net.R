# baseline model: elastic net/lasso/ridge
# first try

# https://glmnet.stanford.edu/articles/glmnet.html
library(glmnet)
library(dplyr)

devtools::install("01_Code/ConnectomeR") # only necessary if there are changes
library(ConnectomeR)

data <- readRDS("00_Data/Delcode_prepared_2021-11-19.rds")
table(data$prmdiag)

# first model: healthy controls (0) vs MCI and AD (2, 3)
data <- data %>%
  mutate(y = case_when(
  data$prmdiag == "0" ~ 0,
  data$prmdiag %in% c("2", "3") ~ 1,
  TRUE ~ NA_real_
  )) %>%
  filter(!is.na(y))

table(data$y)

model_1 <- glmnet(x = data[, 13:(ncol(data) - 1)], # ToDo: extract columns with connectivity matrix data automatically
                  y = data$y,
                  family = "binomial",
                  alpha = 1 # Lasso
                  )
print(model_1)
plot(model_1)
coef(model_1, s = 0.1)

predict(model_1, newx = as.matrix(data[, 13:(ncol(data) - 1)]), s = model_1$a0, type = "response")

