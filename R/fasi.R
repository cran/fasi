#' Create a ranking score model to implement the fasi classification algorithm.
#'
#' This function implements the Fair Adjusted Selective Inference method. It assumes that you have an observed data set that includes all variables needed for your ranking score model and class labels. The user is able to pick from a set of popular ML algorithms when estimating the ranking scores and is able to provide the algorithm their own model. If desired, the user can also directly provide their own ranking scores without using the functions pre-set algorithms. These will be directly used in the predict step when estimating the r-scores.
#'
#'
#' @param observed_data The observed data set that will be split into a testing and calibration data for you by proportion split_p - which is user-specified. If you are providing your own ranking scores, the observed data should just be the calibration data.
#' @param model_formula A formula that will be provided to a specified ML model used to produce ranking scores. Please be sure to follow the exact notation of each package and wrap your formula in the as.formula function.
#' @param split_p The proportion of your observed data that should be used for the training data set.
#' @param alg A specified algorithm used to produce ranking scores. The options are "gam", "logit", "adaboost", "nonparametric_nb", and "user-provided".
#' @param class_label The name of the class label variable in your data set. Defaults to "y".
#' @param niter_adaboost The number of weak learners you want to use for the adaboost algorithm. Defaults to 10. This parameter is useless if you did not select the adaboost algorithm.
#' @return A list where the first element is the observed data with an extra variable denoting which observation was selected for the training and calibration data set, second is the model fit, third the training data. fourth the calibration data and lastly the chosen ranking score algorithm.
#' @author Bradley Rava. PhD Candidate at the University of Southern California's Marshall School of Business.
#' Department of Data Sciences and Operations.
#' @importFrom gam gam
#' @importFrom stats glm
#' @importFrom fastAdaboost adaboost
#' @importFrom naivebayes nonparametric_naive_bayes
#' @examples
#' \donttest{
#' fasi(observed_data, model_formula, split_p=0.5, alg="gam", class_label="y")
#' }
#' @export
fasi <- function(observed_data, model_formula, split_p=0.5, alg="gam", class_label="y", niter_adaboost=10) {

  ## Make sure the observed data set is a data frame
  observed_data <- as.data.frame(observed_data)

  class_label_given <- observed_data[,which(colnames(observed_data)==class_label)]
  class_label_given_unique <- sort(as.numeric(unique(class_label_given)))

  if (class_label_given_unique[1] != 1 | class_label_given_unique[2] != 2) {
    return(warning("The class labels must be 1 or 2. Please recode your class lable variable to be in this format."))
  }

  ## Split the observed data into observed / calibrate by proportion split.
  obs_rows <- 1:nrow(observed_data)
  train_rows <- sample(obs_rows, split_p*length(obs_rows))
  calibrate_rows <- obs_rows[-train_rows]

  ## Create the train and calibrate data set from the splits above.
  train_data <- observed_data[train_rows,]
  calibrate_data <- observed_data[calibrate_rows,]

  ## For the model fit, separate the class label "y"
  y_train <- train_data[,which(colnames(train_data) == class_label)]
  train_data_noy <- train_data[,-which(colnames(train_data) == class_label)]

  ## GAM Fit
  if (alg == "gam") {
    ## For GAM, 0<=y<=1
    train_data_gam <- train_data
    train_data_gam$y <- as.factor(as.numeric(train_data_gam$y) - 1)
    ## Create the GAM fit
    model_fit <- gam::gam(model_formula, train_data_gam, family = "binomial")
  }
  ## Logistic Regression Fit
  else if (alg == "logit") {
    ## For logistic regression, 0<=y<=1
    train_data_logit <- train_data
    train_data_logit$y <- as.factor(as.numeric(train_data_logit$y) - 1)
    model_fit <- stats::glm(model_formula, data = train_data_logit, family = "binomial")
  }
  ## Adaboost fit
  else if (alg == "adaboost") {
    dta_train_adaboost <- cbind.data.frame(data.matrix(train_data_noy), y = y_train)
    model_fit <- fastAdaboost::adaboost(model_formula, data=as.data.frame(dta_train_adaboost), nIter = niter_adaboost)
  }
  ## Nonparametric Naive Bayes
  else if (alg == "nonparametric_nb") {
    model_fit <- naivebayes::nonparametric_naive_bayes(x = data.matrix(train_data_noy), y = as.factor(y_train))
  }
  ## User provided their own ranking score model
  else if (alg == "user-provided") {
    model_fit <- NA
  }

  ## Combine the calibration and testing data with an indicator of which data set they were split into.
  train_indicator <- vector(mode="character", length = nrow(observed_data))
  train_indicator[train_rows] <- "train"
  train_indicator[calibrate_rows] <- "calibrate"
  observed_data_return <- cbind.data.frame(observed_data, train_indicator)

  ## Return the model fit, calibration data and testing data.
  return_object <- list(observed_data = observed_data_return,
                        model_fit = model_fit,
                        train_data = train_data,
                        calibrate_data = calibrate_data,
                        algorithm = alg)
  ## Specify the class of the return object
  class(return_object) <- "fasi"

  return(return_object)

}


#' Prediction of a FASI Object
#'
#' After a model is trained with the fasi function, predict estimates the r-scores and classification of all observations in the test data set.
#'
#' @param object An object of class fasi. It can be created from the fasi function.
#' @param test_data The test data set that contains new observations to be classified.
#' @param alpha_1 User specified group and overall FSR control for class 1.
#' @param alpha_2 User specified group and overall FSR control for class 2.
#' @param rscore_plus A logical variable that indicates if the r-score or r-score plus is calculated. By default the r-score plus is calculated.
#' @param ptd_group_var The name of the protected group variable in your data set. Defaults to "a".
#' @param class_label The name of the class label variable in your data set. Defaults to "y".
#' @param ranking_score_calibrate A vector of ranking scores for the calibration data set. This should only be used if the built in ranking score algorithms are not used.
#' @param ranking_score_test A vector of ranking scores for the test data set. This should only be used if the built in ranking score algorithms are not used.
#' @param indecision_choice A number, 1, 2, or 3. This determines how the indecision cases are treated if we are equally confident in placing them in both class 1 and 2. Defaults to the scenario where class 2 is preferred
#' @param ... Additional arguments
#' @importFrom stats predict
#' @return A list where the first element is the r-scores for both class 1 and class 2. The second element is the actual classifications, class 1, class 2 or the indecision class. The third element is a logical value True/False that denotes if the r-score or r-score plus was calculated. The last element in the list is the values of alpha for both classes. Alpha can directly be compared to the r-scores to obtain the classifications.
#' @author Bradley Rava. PhD Candidate at the University of Southern California's Marshall School of Business.
#' Department of Data Sciences and Operations.
#' @examples
#' \donttest{
#' fasi_object <- fasi(observed_data, model_formula, split_p=0.5, alg="gam", class_label="y")
#' predict(fasi_object, test_data, alpha_1=0.1, alpha_2=0.1)
#' }
#' @export
predict.fasi <- function(object, test_data, alpha_1, alpha_2, rscore_plus=TRUE, ptd_group_var="a", class_label="y",
                         ranking_score_calibrate, ranking_score_test, indecision_choice = "2", ...) {
  ######################
  ## Warning Messages ##
  ######################

  if (alpha_1 < 0 | alpha_1 > 1) {
    return(warning("alpha_1 must be a number between 0 and 1."))
  }
  if (alpha_2 < 0 | alpha_2 > 1) {
    return(warning("alpha_2 must be a number between 0 and 1."))
  }
  if (rscore_plus != TRUE & rscore_plus != FALSE) {
    return(warning("rscore_plus must be a logical. T/F."))
  }

  ######################

  ## Pull out the calibrate data
  calibrate_data <- object$calibrate_data

  ## Make sure the test data set is a data frame
  if (is.data.frame(test_data) == FALSE) {
    test_data <- as.data.frame(test_data)
  }

  ## Preset the class 2 and class 2 label. Future iterations of the package will allow a user specified version.
  class_1_label = 1
  class_2_label = 2

  ## Alphas given
  alpha_all <- c(alpha_1, alpha_2)
  names(alpha_all) <- c("alpha_1", "alpha_2")

  calibrate_data_withy <- object$calibrate_data
  calibrate_data_noy <- calibrate_data_withy[,-which(colnames(calibrate_data_withy)==class_label)]

  ## Use the model provided to estimate ranking scores
  model_fit <- object$model_fit

  ## GAM Fit
  if (object$algorithm == "gam") {
    ## Calibrate ranking scores
    logit_calibrate <- predict(model_fit, newdata = calibrate_data_noy)
    calibrate_data$s <- exp(logit_calibrate) / (1+exp(logit_calibrate))

    ## Testing ranking scores
    logit_test <- predict(model_fit, newdata = test_data)
    test_data$s <- exp(logit_test) / (1+exp(logit_test))
  }
  ## Logistic Regression Fit
  else if (object$algorithm == "logit") {
    ## Calibrate ranking scores
    logit_calibrate <- predict(model_fit, newdata = calibrate_data_noy)
    calibrate_data$s <- exp(logit_calibrate) / (1+exp(logit_calibrate))

    ## Testing ranking scores
    logit_test <- predict(model_fit, newdata = test_data)
    test_data$s <- exp(logit_test) / (1+exp(logit_test))
  }
  ## Adaboost fit
  else if (object$algorithm == "adaboost") {
    calibrate_data$s <- predict(model_fit, newdata = calibrate_data_noy)$prob[,2]
    test_data$s <- predict(model_fit, newdata = test_data)$prob[,2]
  }
  ## Nonparametric Naive Bayes
  else if (object$algorithm == "nonparametric_nb") {
    calibrate_data$s <- predict(model_fit, newdata = data.matrix(calibrate_data_noy), type = "prob")[,2]
    test_data$s <- predict(model_fit, newdata = data.matrix(test_data), type = "prob")[,2]
  }
  ## User provided their own ranking score model
  else if (object$algorithm == "user-provided") {
    calibrate_data <- object$observed_data

    calibrate_data$s <- ranking_score_calibrate
    test_data$s <- ranking_score_test
  }

  ## Subset the data on the information we need for the fasi algorithm. ##

  ## Important column's in the x_observed data frame. These are the same for the train and calibrate data set.
  ptd_group_col_cal <- which(colnames(calibrate_data) == ptd_group_var)
  class_label_col_cal <- which(colnames(calibrate_data) == class_label)
  rank_col_cal <- which(colnames(calibrate_data) == "s")

  ## Important column's in the x_test data frame.
  ptd_group_col_test <- which(colnames(test_data) == ptd_group_var)
  class_label_col_test <- which(colnames(test_data) == class_label)
  rank_col_test <- which(colnames(test_data) == "s")

  z_calibrate <- as.data.frame(calibrate_data[,c(class_label_col_cal, rank_col_cal, ptd_group_col_cal)])
  colnames(z_calibrate) <- c("y", "s", "a")
  z_test <- as.data.frame(test_data[,c(class_label_col_test, rank_col_test, ptd_group_col_test)])
  colnames(z_test) <- c("y", "s", "a")

  ## Determine the protected groups in the dataset.
  a_test_unique <- unique(z_test$a)

  ## Determine the class labels in the dataset.
  y_test_unique <- unique(z_test$y)
  if (sort(as.numeric(y_test_unique))[1] != 1 & sort(as.numeric(y_test_unique))[2] != 2) {
    return(warning("The class labels MUST be coded as 1 and 2. Please change your class label names to reflect this."))
  }
  class_1_label <- y_test_unique[which(y_test_unique != class_2_label)]

  ## Calculate the r-scores for every observation in the test data set.
  r_score2 <- sapply(1:nrow(z_test), function(ii) fasi::rscore(z_test$s[ii], class_2_label, z_test$a[ii], z_calibrate, z_test, rscore_plus, r2_indicator=T))
  r_score1 <- sapply(1:nrow(z_test), function(ii) fasi::rscore(z_test$s[ii], class_1_label, z_test$a[ii], z_calibrate, z_test, rscore_plus, r2_indicator=F))
  r_scores <- cbind.data.frame(r_score1, r_score2)

  ## Classify the observations according to the calculated r-scores (3-options)
  if (indecision_choice == "1") {
    classification_vector <- vector(mode="numeric", length = nrow(z_test))
    classification_vector[r_scores$r_score2 <= alpha_2] <- class_2_label
    classification_vector[r_scores$r_score1 <= alpha_1] <- class_1_label
  } else if (indecision_choice == "2") {
    classification_vector <- vector(mode="numeric", length = nrow(z_test))
    classification_vector[r_scores$r_score1 <= alpha_1] <- class_1_label
    classification_vector[r_scores$r_score2 <= alpha_2] <- class_2_label
  } else if (indecision_choice == "3") {
    classification_vector <- vector(mode="numeric", length = nrow(z_test))
    classification_vector[r_scores$r_score1 <= alpha_1] <- class_1_label
    classification_vector[r_scores$r_score2 <= alpha_2] <- class_2_label
    classification_vector[r_scores$r_score2 <= alpha_2 & r_scores$r_score1 <= alpha_1] <- 0
  } else {
    return(warning("Please use a valid indecision_choice (1-3)."))
  }

  return_object <- list(r_scores=r_scores,
                        classification=classification_vector,
                        rscore_plus=rscore_plus,
                        alpha=alpha_all)

  class(return_object) <- "fasi"

  return(return_object)
}

#' Calculate the r1 or r2 score for a new observation in ones test data.
#'
#' This function calculates an r-score for a given ranking score. It requires a calibration and testing dataset.
#' Both the r-score+ and r-score can be implemented.
#'
#' Do not call this function externally. It is only meant to be called from within the fasi function.
#'
#'
#' @param s_test_cur The current ranking score from the test data to be evaluated.
#' @param y_class_cur The class label you want to generate the r-scores for.
#' @param a_cur The current protected group from the test data to be evaluated.
#' @param z_cal The calibration data set.
#' @param z_test The test data set.
#' @param rscore_plus Logical variable, TRUE/FALSE, that determines if the r-score or r-score plus is calculated.
#' @param r2_indicator Logical variable, TRUE/FALSE, that determines if the r1 or r2 score is calculated.
#' @return The r-score corresponding to s_test_cur.
#' @author Bradley Rava. PhD Candidate at the University of Southern California's Marshall School of Business.
#' Department of Data Sciences and Operations.
#' @examples
#' \donttest{
#' rscore(s_test_cur, y_class_cur, a_cur, z_cal, z_test, rscore_plus, r2_indicator)
#' }
#' @export
rscore <- function(s_test_cur, y_class_cur, a_cur, z_cal, z_test, rscore_plus, r2_indicator) {
  n_a_cal <- length(which(z_cal$a == a_cur))
  m_a <- length(which(z_test$a == a_cur))

  if (r2_indicator == TRUE) {
    ## Numerator of r-score
    z_cal_acur <- z_cal[which(z_cal$a == a_cur),]
    z_cal_acur_thresh <- z_cal_acur[which(z_cal_acur$s >= s_test_cur),]
    z_cal_acur_thresh_new <- z_cal_acur_thresh[which(z_cal_acur_thresh$y != y_class_cur),]
    ## Denominator of r-score (includes z_cal_acur_thresh above)
    z_test_acur <- z_test[which(z_test$a == a_cur),]
    z_test_acur_thresh <- z_test_acur[which(z_test_acur$s >= s_test_cur),]

    ## Calculate the r-score or r-score+
    if (rscore_plus == TRUE) {
      r_score_term <- ((1/(n_a_cal+1)) * (nrow(z_cal_acur_thresh_new) + 1)) / ((1/(m_a + n_a_cal + 1)) * (nrow(z_cal_acur_thresh) + nrow(z_test_acur_thresh) + 1))
      ## Technical adjustment: If the r-score is greater than 1, set it to 1.
      r_score_term <- ifelse(r_score_term > 1, 1, r_score_term)
    } else {
      r_score_term <- ((1/(n_a_cal+1)) * (nrow(z_cal_acur_thresh_new) + 1)) / ((1/(m_a)) * (nrow(z_test_acur_thresh)))
      ## Technical adjustment: If the r-score is greater than 1, set it to 1.
      r_score_term <- ifelse(r_score_term > 1, 1, r_score_term)
    }
  } else {
    ## Numerator of r-score
    z_cal_acur <- z_cal[which(z_cal$a == a_cur),]
    z_cal_acur_thresh <- z_cal_acur[which((1-z_cal_acur$s) >= (1-s_test_cur)),]
    z_cal_acur_thresh_new <- z_cal_acur_thresh[which(z_cal_acur_thresh$y != y_class_cur),]
    ## Denominator of r-score (includes z_cal_acur_thresh above)
    z_test_acur <- z_test[which(z_test$a == a_cur),]
    z_test_acur_thresh <- z_test_acur[which((1-z_test_acur$s) >= (1-s_test_cur)),]

    ## Calculate the r-score or r-score+
    if (rscore_plus == TRUE) {
      r_score_term <- ((1/(n_a_cal+1)) * (nrow(z_cal_acur_thresh_new) + 1)) / ((1/(m_a + n_a_cal + 1)) * (nrow(z_cal_acur_thresh) + nrow(z_test_acur_thresh) + 1))
      ## Technical adjustment: If the r-score is greater than 1, set it to 1.
      r_score_term <- ifelse(r_score_term > 1, 1, r_score_term)
    } else {
      r_score_term <- ((1/(n_a_cal+1)) * (nrow(z_cal_acur_thresh_new) + 1)) / ((1/(m_a)) * (nrow(z_test_acur_thresh)))
      ## Technical adjustment: If the r-score is greater than 1, set it to 1.
      r_score_term <- ifelse(r_score_term > 1, 1, r_score_term)

    }
  }

  return(r_score_term)
}
