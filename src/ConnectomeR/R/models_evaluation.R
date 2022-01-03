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


#' plot matrix of coefficients, probably doesn't work for when beta contains interaction terms!
#'
#' @param beta coefficients of one model
#' @param regions_dat data on regions of labels. First column contains labels, second column contains regions
#' @import ggplot2 stringr dplyr
#' @export
plot_matrix_coeffs <- function(beta, regions_dat = NULL){
  
  beta <- beta[names(beta) %in% names_conn(names(beta))] # extract connectivity variables
  
  dat_plot <- data.frame(beta = beta,
                         var1 = as.numeric(str_extract(names(beta), "\\d+")),
                         var2 = as.numeric(str_replace(str_extract(names(beta), "_\\d+"), "_", "")))
  
  dat_plot <- dat_plot %>% rbind(data.frame(beta = beta,
                                            var1 = dat_plot$var2,
                                            var2 = dat_plot$var1))
  
  if(!is.null(regions_dat)){ # add regions
    dat_plot <- dat_plot %>% left_join(data.frame(var1 = regions_dat[,1], region1 = regions_dat[,2]))
    dat_plot <- dat_plot %>% left_join(data.frame(var2 = regions_dat[,1], region2 = regions_dat[,2]))
    
    # new numbering based on regions
    dat_plot <- dat_plot[order(dat_plot$region1),] # order by regions
    new_numbers <- data.frame(var_old = unique(dat_plot$var1), var_new = 1:length(unique(dat_plot$var1))) # create new number for each variable (from 1 to 246)
    
    dat_plot <- dat_plot %>% left_join(data.frame(var1 = new_numbers$var_old, var1_new = new_numbers$var_new), by = "var1") # add new variable numbers
    dat_plot <- dat_plot %>% left_join(data.frame(var2 = new_numbers$var_old, var2_new = new_numbers$var_new), by = "var2")
    # -> result: if the heatmatrix is plotted, the variables are grouped by regions
  } else {
    dat_plot$var1_new <- dat_plot$var1
    dat_plot$var2_new <- dat_plot$var2
  }
  
  plot <- ggplot() + 
    geom_tile(data = dat_plot, aes(var1_new, var2_new, fill = beta)) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
    labs(x = "", y = "") +
    scale_y_reverse()
  
  if(!is.null(regions_dat)){ # add lines as boundaries between regions
    
    # calculate lines
    regions_freq <- table(regions_dat[, 2]) %>% as.integer()
    lines <- cumsum(regions_freq)
    lines <- lines[1:(length(lines) - 1)] # remove last line (246)
    
    ticks <- dat_plot %>% group_by(region1) %>% summarise(m = mean(var1_new))
    
    plot <- plot + 
      geom_vline(xintercept = lines) +
      geom_hline(yintercept = lines) +
      scale_x_continuous(breaks = ticks$m, labels = ticks$region1) +
      scale_y_continuous(breaks = ticks$m, labels = ticks$region1, trans = "reverse") +
      theme(axis.ticks = element_blank()) +
      labs(x = "Region", y = "Region")
  }
  
  return(plot)
  
}
