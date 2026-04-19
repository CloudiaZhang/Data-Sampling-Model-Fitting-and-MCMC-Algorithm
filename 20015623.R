
### Question 1

#generate covariate data and ¦Â values
set.seed(5623)

n <- 150 # number of observations
d <- 10 # number of beta parameters

# create matrix to populate with covariates
X <- matrix(nrow = n, ncol = d)
X[,1] <- rep(1, n) # first column is an intercept

# create base uncorrelated random numbers to turn into x_i's
z <- matrix(rnorm(n*(d-1)), nrow = n, ncol = d-1)

# create x_i's (ith row of matrix corresponds to variable x_i)
X[,2] <- z[,1]
X[,3] <- z[,1] + 0.2*z[,2]
X[,4] <- 0.5*z[,3]
X[,5] <- z[,4]
X[,6] <- 2*z[,5] + 20*z[,6]
X[,7] <- z[,6]
X[,8] <- 0.5 * (z[,7] + z[,4] + z[,8] + z[,1])
X[,9] <- z[,8] + 10*z[,4]
X[,10] <- z[,5] + 0.5*z[,9]

# create true beta values
beta <- seq(-2,2, length = 10)




## Question 2

set.seed(5623)
# calculate the linear predictor eta
eta <- X %*% beta

# draw n latent variables Y* from a Cauchy distribution
Y_star_cauchit <- eta + rt(n, df=1)

# transform Y* to binary outcomes Y_i (1 if Y* > 0, 0 otherwise)
Y_cauchit <- as.numeric(Y_star_cauchit > 0)

cat("Yi from Cauchy distribution:")
print(Y_cauchit)





## Question 3

set.seed(5623)
# density of target distribution (Logistic)
fL <- function(x) {
  exp(-x) / (1 + exp(-x))^2
}

# density of candidate distribution (Cauchy)
fC <- function(x) {
  1 / (pi * (1 + x^2))
}

# ratio of the two densities
ratio <- function(x) {
  fL(x) / fC(x)
}

# plot the ratio
par(mar=c(2.1, 2.1, 2.1, 2.1))
plot(ratio, xlim = c(-10,10), ylim = c(0, 2), type = 'l', main = "Ratio of Densities of Logistic over Cauchy", xlab = "x", ylab = "Ratio")

x_values <- seq(-10, 10, 0.01)
# evaluate the ratio at these x values
ratio_values <- sapply(x_values, ratio)

# find the maximum of ratio
M <- max(ratio_values)


# rejection sampling function
set.seed(5623)
n <- 150
n_samples <- n
residual_samples <- rep(NA, n_samples) # create storage vector for accepted samples
n <- 1
iteration_count <- 0

while (n < n_samples + 1) {
  # draw a candidate sample from the Cauchy distribution
  x <- rt(1, df=1)
  
  # generate a uniform random number for the acceptance probability
  u <- runif(1)
  
  # calculate the acceptance probability
  accept_prob <- fL(x) / (M * fC(x))
  
  # if u is smaller than the acceptance probability, accept
  if (u < accept_prob) {
    residual_samples[n] <- x
    n <- n + 1
  }
  
  iteration_count <- iteration_count + 1
}

# compute Y* from logistic distribution
Y_star_logistic <- X %*% beta + residual_samples

# convert to binary outcomes
Y_logistic <- as.numeric(Y_star_logistic > 0)

cat("Yi from Logistic distribution:")
print(Y_logistic)






## Question 4

library(mvtnorm)
# log likelihood of standard Gaussian prior
log_prior_Gaussian <- function(beta) {
  sum(dnorm(beta, mean = 0, sd = 1, log = TRUE))
}

# log-likelihood of logistic regression model
loglikelihood_logistic <- function(beta, X, y) {
  p <- 1 / (1 + exp(-X %*% as.numeric(beta)))
  sum(y * log(p) + (1 - y) * log(1 - p), na.rm = TRUE)  #weighted sum of log likelihood
}

# log-likelihood of cauchit regression model
loglikelihood_cauchit <- function(beta, X, y) {
  sum(log(1/(pi * (1 + (y - X %*% as.numeric(beta))^2))), na.rm = TRUE)
}


# metropolis-hastings algorithm
RWM  <- function(log_likelihood, log_prior, X, y, initial_beta, nits, step_size) {
  d <- length(initial_beta)    # number of parameters being estimated
  x_store <- matrix(NA, nrow = nits, ncol = d)     #  A matrix to store the generated samples of parameters
  x_curr <- initial_beta      # starting point for the parameters in the MCMC simulation.
  logpi_curr <- log_likelihood(x_curr, X, y) + log_prior(x_curr)
  accepted <- 0
  
  for (i in 1:nits) {
    # propose a candidate move from the current parameters in x_curr to the next proposed new parameters
    x_prop <- x_curr + step_size * rnorm(d)
    # evaluate the log-posterior of the proposed parameters
    logpi_prop <- log_likelihood(x_prop, X, y) + log_prior(x_prop)
    
    # calculate the acceptance probability
    loga <- logpi_prop - logpi_curr
    
    # generate a random number from U[0,1], and compare log(u) with loga
    if (log(runif(1)) < loga) {    # accept
      x_curr <- x_prop
      logpi_curr <- logpi_prop
      accepted <- accepted + 1
    }
    
    # store the current accepted parameters
    x_store[i, ] <- x_curr
  }
  
  return(list(x_store = x_store, acceptance_rate = accepted / nits))
}


# with Gaussian prior
set.seed(5623)
initial_beta <- seq(-2,2, length = 10)
nits <- 10000
proposal_sd <- 0.1

output_logisticG <- RWM (log_likelihood = loglikelihood_logistic,
                         log_prior = log_prior_Gaussian, X = X, y = Y_logistic, initial_beta = initial_beta, nits = nits, step_size = 0.085)
output_cauchitG <- RWM (log_likelihood = loglikelihood_cauchit,
                        log_prior = log_prior_Gaussian, X = X, y = Y_logistic, initial_beta = initial_beta, nits = nits, step_size = 0.0075)

parameter_samples1 <- output_logisticG$x_store
acceptance_rate1 <- output_logisticG$acceptance_rate

parameter_samples2 <- output_cauchitG$x_store
acceptance_rate2 <- output_cauchitG$acceptance_rate

cat("first 30 parameters samples for fitting logistic regression with gaussian prior:\n")
print(parameter_samples1[1:3, ])
cat("\n") 
cat("first 30 parameters samples for fitting cauchit regression with gaussian prior:\n")
print(parameter_samples2[1:3, ])
cat("\n") 
print(paste("Acceptance rate (logistic regression with gaussian prior):", acceptance_rate1))
print(paste("Acceptance rate (cauchit regression with gaussian prior):", acceptance_rate2))


# with Unit Information Prior
set.seed(5623)
library(mvtnorm)
initial_beta <- seq(-2,2, length = 10)
nits <- 10000
proposal_sd <- 0.1

set.seed(5623)
n <- 150  # number of observations
d <- 10   # number of predictors
X <- matrix(rnorm(n * d), ncol = d)

# log likelihood of Unit Information Prior
log_prior_Unit <- function(beta, X) {
  n <- nrow(X) 
  var <- n * solve(t(X) %*% X)
  sum(dmvnorm(as.numeric(beta), mean = rep(0, length(beta)), sigma = var, log = TRUE))
}

log_prior = log_prior_Unit

output_logisticU <- RWM (log_likelihood = loglikelihood_logistic,
                         log_prior = function(beta) log_prior_Unit(beta, X), X = X, y = Y_logistic, initial_beta = initial_beta, nits = nits, step_size = 0.12)
output_cauchitU <- RWM (log_likelihood = loglikelihood_cauchit,
                        log_prior = function(beta) log_prior_Unit(beta, X), X = X, y = Y_cauchit, initial_beta = initial_beta, nits = nits, step_size = 0.03)

parameter_samples3 <- output_logisticU$x_store
acceptance_rate3 <- output_logisticU$acceptance_rate

parameter_samples4 <- output_cauchitU$x_store
acceptance_rate4 <- output_cauchitU$acceptance_rate

cat("first 30 parameters samples for fitting logistic regression with unit information prior:\n")
print(parameter_samples3[1:3, ])

cat("first 30 parameters samples for fitting cauchit regression with unit information prior:\n")
print(parameter_samples4[1:3, ])

print(paste("Acceptance rate (logistic regression with Information Unit prior):", acceptance_rate3))
print(paste("Acceptance rate (cauchit regression with Information Unit prior):", acceptance_rate4))








## Question 5

par(mfrow = c(2, 2))
par(mar=c(5.1, 4.1, 4.1, 2.1)) 

# traceplots for each model

plot(output_logisticG$x_store[,1], type = 'l',
     main = "Logistic (Gaussian Prior)",
     xlab = "Iteration", ylab = "Parameter Values")

plot(output_cauchitG$x_store[,1], type = 'l', 
     main = "Cauchit (Gaussian Prior)",
     xlab = "Iteration", ylab = "Parameter Values")

plot(output_logisticU$x_store[,1], type = 'l', 
     main = "Logistic (IUP)",
     xlab = "Iteration", ylab = "Parameter Values")

plot(output_cauchitU$x_store[,1], type = 'l', 
     main = "Cauchit (IUP)",
     xlab = "Iteration", ylab = "Parameter Values")


library(coda)

mc_logisticG <- mcmc(output_logisticG$x_store)
mc_cauchitG <- mcmc(output_cauchitG$x_store)
mc_logisticU <- mcmc(output_logisticU$x_store)
mc_cauchitU <- mcmc(output_cauchitU$x_store)

par(mfrow = c(2, 2))
par(mar=c(5.1, 4.1, 4.1, 2.1)) 

# histograms of effective sample sizes
xlims <- range(c(effectiveSize(mc_logisticG), effectiveSize(mc_cauchitG), effectiveSize(mc_logisticU), effectiveSize(mc_cauchitU)))

hist(effectiveSize(mc_logisticG), main = "ESS: Logistic (Gaussian Prior)",
     xlab = "Effective Sample Size", xlim = c(0,300), ylim = c(0,5))
hist(effectiveSize(mc_cauchitG), main = "ESS: Cauchit (Gaussian Prior)",
     xlab = "Effective Sample Size", xlim = c(0,300), ylim = c(0,5))
hist(effectiveSize(mc_logisticU), main = "ESS: Logistic (IUP)",
     xlab = "Effective Sample Size",xlim = c(0,300), ylim = c(0,5))
hist(effectiveSize(mc_cauchitU), main = "ESS: Cauchit (IUP)",
     xlab = "Effective Sample Size", xlim = c(0,300), ylim = c(0,5))


# ACF values for parameter samples
par(mfrow = c(2, 2))
par(mar=c(5.1, 4.1, 4.1, 2.1)) 
acf(parameter_samples1[, 1], main="ACF: Logistic (Gaussian prior)")
acf(parameter_samples2[, 1], main="ACF: Cauchit (Gaussian prior)")
acf(parameter_samples3[, 1], main="ACF: Logistic (IUP)")
acf(parameter_samples4[, 1], main="ACF: Cauchit (IUP)")


# brier score function
brier_score <- function(y, m) {
  n <- length(y)
  sum((y - m)^2) / n
}

# predicted probability for logistic
predict_logistic <- function(beta, X) {
  p <- 1 / (1 + exp(-X %*% beta))
  return(p)
}

# predicted probability for cauchit
predict_cauchit <- function(beta, X) {
  p <- 1 / pi * atan(X %*% beta) + 1/2
  return(p)
}

# get the predicted values for each model
predicted_logisticG <- colMeans(apply(parameter_samples1, 1, function(beta) predict_logistic(beta, X)))
predicted_cauchitG <- colMeans(apply(parameter_samples2, 1, function(beta) predict_cauchit(beta, X)))
predicted_logisticU <- colMeans(apply(parameter_samples3, 1, function(beta) predict_logistic(beta, X)))
predicted_cauchitU <- colMeans(apply(parameter_samples4, 1, function(beta) predict_cauchit(beta, X)))

# get the brier scores
brier_logisticG <- brier_score(Y_logistic, predicted_logisticG[0:150])
brier_cauchitG <- brier_score(Y_cauchit, predicted_cauchitG[0:150])
brier_logisticU <- brier_score(Y_logistic, predicted_logisticU[0:150])
brier_cauchitU <- brier_score(Y_cauchit, predicted_cauchitU[0:150])

cat("Brier Score - Logistic Regression with Gaussian Prior:", brier_logisticG, "\n")
cat("Brier Score - Cauchit Regression with Gaussian Prior:", brier_cauchitG, "\n")
cat("Brier Score - Logistic Regression with Information Unit Prior:", brier_logisticU, "\n")
cat("Brier Score - Cauchit Regression with Information Unit Prior:", brier_cauchitU, "\n")





