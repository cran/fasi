test_that("fasi main function check", {

  z <- fasi::adult

  obs_rows <- sample(1:nrow(z), 0.5*nrow(z))
  test_rows <- (1:nrow(z))[-obs_rows]

  observed_data <- z[obs_rows,]
  test_data <- z[test_rows,]

  model_formula <- as.formula("y ~ `hours-per-week`")
  fasi_object <- fasi::fasi(observed_data = observed_data, model_formula = model_formula, alg = "logit")

  fasi_predict <- predict(object = fasi_object, test_data = test_data, alpha_1 = 0.1, alpha_2 = 0.1, ptd_group_var = "sex")



  expect_equivalent(c(0.1,0.1), fasi_predict$alpha)
  expect_equivalent("fasi", class(fasi_object))
})
