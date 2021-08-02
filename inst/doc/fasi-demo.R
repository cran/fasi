## ----message=FALSE, warning=FALSE, eval=FALSE---------------------------------
#  library(devtools)
#  install_github("bradleyrava/fasi@master")

## ----message=FALSE, warning=FALSE, eval=FALSE---------------------------------
#  library(fasi)

## ----message=FALSE, warning=FALSE---------------------------------------------
z_full <- fasi::adult

## Subset the data so the package runs faster
z <- z_full[sample(1:nrow(z_full), 0.1*nrow(z_full)), ]

## -----------------------------------------------------------------------------
obs_rows <- sample(1:nrow(z), 0.5*nrow(z))
test_rows <- (1:nrow(z))[-obs_rows]

observed_data <- z[obs_rows,]
test_data <- z[test_rows,]

## -----------------------------------------------------------------------------
model_formula <- as.formula("y ~ age")
fasi_object <- fasi::fasi(observed_data = observed_data, model_formula = model_formula, alg = "logit")

## -----------------------------------------------------------------------------
fasi_object$model_fit

## -----------------------------------------------------------------------------
fasi_predict <- predict(object = fasi_object, test_data = test_data, alpha_1 = 0.1, alpha_2 = 0.1, ptd_group_var = "sex")
head(fasi_predict$r_scores)
head(fasi_predict$classification)

## -----------------------------------------------------------------------------
## Random ranking scores
calibrate_scores <- rnorm(nrow(observed_data))
test_scores <- rnorm(nrow(test_data))

fasi_object <- fasi::fasi(observed_data = observed_data, alg = "user-provided")
fasi_predict <- predict(object = fasi_object, test_data = test_data, alpha_1 = 0.1, alpha_2 = 0.1, ptd_group_var = "sex",
                        ranking_score_calibrate = calibrate_scores, ranking_score_test = test_scores)


