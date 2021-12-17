# R functions for evaluation/interpretation of models

#' get result table based on metric for el_net result
#'
#' returns best values (mretric, lambda) for each alpha
#' @param elnet_result result of el_net function
#' @import dplyr 
#' @export
result_table_elnet <- function(elnet_result){
  
  if(sum(elnet_result$data_list$test$y %in% c(0, 1)) == nrow(elnet_result$data_list$test)){ # logistic regression
    result <- list()
    
    for(i in c("accuracy", "auc", "precision", "recall", "F1")){
      dat_best <- data.frame(value = NA, ind_lambda = NA, alpha = NA)
      for(j in 1:length(elnet_result$results_models)){
        max_metric_j <- max(elnet_result$results_models[[j]]$evaluation[[i]])
        ind_best_j <- which(elnet_result$results_models[[j]]$evaluation[[i]] == max_metric_j)[1] # [1] -> take only one lambda if there several best ones
        alpha_j <- elnet_result$results_models[[j]]$alpha
        
        dat_best <- rbind(dat_best, c(max_metric_j, ind_best_j, alpha_j))
      }
      
      result[[length(result) + 1]] <- dat_best[-1, ]
    }
    
    names(result) <- c("accuracy", "auc", "precision", "recall", "F1")
    return(result)
  } else{ # LM
    
    dat_best <- data.frame(MSE_test = NA, ind_lambda = NA, alpha = NA)
    for(j in 1:length(elnet_result$results_models)){
      min_mse_j <- min(elnet_result$results_models[[j]]$evaluation)
      ind_best_j <- which(elnet_result$results_models[[j]]$evaluation == min_mse_j)[1] # [1] -> take only one lambda if there several best ones
      alpha_j <- elnet_result$results_models[[j]]$alpha
      
      dat_best <- rbind(dat_best, c(min_mse_j, ind_best_j, alpha_j))
    }
    
    return(dat_best[-1,])
    
  }
  
}


#' get confusion matrix for elastic net model
#'
#' @param elnet_result result of el_net function
#' @param ind_alpha index of alpha that should be used
#' @param ind index of lambda value that should be used
#' @import dplyr caret checkmate
#' @export
get_confMatrix_elnet <- function(elnet_result, ind_alpha, ind_lambda){
  
  print(paste("alpha: ", elnet_result$results_models[[ind_alpha]]$alpha))
  print(paste("lambda: ", elnet_result$results_models[[ind_alpha]]$model$lambda[ind_lambda]))
  
  new_x <- as.matrix(elnet_result$data_list$test[, dimnames(elnet_result$results_models[[ind_alpha]]$model$beta)[[1]]])
  class(new_x) <- "numeric"
  pred <- predict(elnet_result$results_models[[ind_alpha]]$model, newx = new_x, type = "response") 
  
  x <- confusionMatrix(reference = factor(elnet_result$data_list$test$y), 
                       data = factor(as.integer(pred[, ind_lambda]>0.5), levels = c("0", "1")),
                       positive = "1")
  
  return(x)
}



#' accuracy intercept model
#'
#' calculates the accuracy of an intercept model
#' @param elnet_result result of el_net function
#' @import dplyr 
#' @export
acc_intercept <- function(elnet_result){
  
  test <- elnet_result$data_list$test
  train <- elnet_result$data_list$train
  
  perc_1_train <- sum(train$y == 1)/nrow(train) # percentage of 1 in training data
  pred <- as.integer(perc_1_train>0.5) # prediction of intercept model on training data
  
  acc <- sum(test$y == pred)/nrow(test)
  
  return(acc)
  
}


#' plot matrix of coefficients
#'
#' @param beta coefficients of one model
#' @import ggplot2 stringr dplyr
#' @export
plot_matrix_coeffs <- function(beta){
  
  beta <- beta[names(beta) %in% names_conn(names(beta))] # extract connectivity variables
  
  dat_plot <- data.frame(beta = beta,
                         var1 = as.numeric(str_extract(names(beta), "\\d+")),
                         var2 = as.numeric(str_replace(str_extract(names(beta), "_\\d+"), "_", "")))
  
  dat_plot <- dat_plot %>% rbind(data.frame(beta = beta,
                                            var1 = dat_plot$var2,
                                            var2 = dat_plot$var1))
  
  plot <- ggplot() + 
    geom_tile(data = dat_plot, aes(var1, var2, fill = beta)) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
    labs(x = "", y = "") +
    scale_y_reverse()
  
  return(plot)
  
}