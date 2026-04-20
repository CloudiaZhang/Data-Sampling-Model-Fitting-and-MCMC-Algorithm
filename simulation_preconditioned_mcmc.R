###############################################################################
# Bayesian Binary Classification via Logistic and Cauchit Regression
#
# This script implements a simulation and MCMC study for two
# binary classification models:
#   1. Logistic regression
#   2. Cauchit regression
#
# The aim is to:
#   - generate simulated covariate data X and true coefficient values beta
#   - simulate binary responses from logistic and cauchit models
#   - use rejection sampling to generate logistic latent errors
#     with Cauchy proposals
#   - fit Bayesian logistic and cauchit regression models using
#     Metropolis-Hastings MCMC
#   - compare model behaviour under two priors:
#       * standard Gaussian prior
#       * Unit Information Prior
#   - assess posterior sampling quality using diagnostics such as
#     trace plots, autocorrelation, and effective sample size
#   - compare predictive performance using the Brier score
#
# Notes:
#   - No external dataset is required; all data are simulated in this script.
###############################################################################




###############################################################################
# Part 1: Generate covariate data and true coefficient values
#
# Simulate a design matrix X with n = 150 observations and
# d = 10 predictors. The first column is an intercept term.
# The remaining predictors are generated from Gaussian random
# variables and transformed so that some covariates are correlated.
#
# Define the true coefficient vector beta, which is later
# used to generate binary outcomes from the logistic and cauchit
# regression models.
###############################################################################

# generate covariate data and ¦Â values
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




############################################################
# Part 2: Simulate binary outcomes from the cauchit regression model
#
# Use the latent-variable representation
#   Y*_i = x_i^T beta + epsilon_i
# where epsilon_i follows a Cauchy distribution.
#
# The observed binary response is then defined by
#   Y_i = 1 if Y*_i > 0
#   Y_i = 0 otherwise
#
# Since the Cauchy distribution is equivalent to a t
# distribution with 1 degree of freedom, we generate the
# latent errors using rt(n, df = 1).
############################################################

# compute the linear predictor x_i^T beta for each observation
eta <- as.vector(X %*% beta)

# draw Cauchy-distributed latent errors
epsilon_cauchy <- rt(n, df = 1)

# construct the latent variable Y*_i
Y_star_cauchit <- eta + epsilon_cauchy

# convert the latent variable into binary outcomes
Y_cauchit <- as.numeric(Y_star_cauchit > 0)

cat("Binary outcomes generated from the cauchit model:\n")
print(Y_cauchit)




############################################################
# Part 3: Simulate logistic latent errors using rejection sampling
#
# To generate binary outcomes from the logistic regression model,
# we use the latent-variable representation
#   Y*_i = x_i^T beta + epsilon_i
# where epsilon_i follows a logistic distribution.
#
# Because the task requires logistic errors to be generated via
# rejection sampling, we use the Cauchy distribution as the
# proposal distribution. This is appropriate because the Cauchy
# distribution has heavier tails than the logistic distribution,
# so there exists a finite constant M such that
#   f_L(x) <= M f_C(x)
# for all x.
#
# Firstly approximate M numerically by evaluating the ratio
# f_L(x) / f_C(x) on a dense grid. We then draw candidate values
# from the Cauchy distribution and accept them with probability
#   f_L(x) / (M f_C(x)).
#
# After obtaining n accepted logistic errors, we construct the
# latent variables and convert them into binary outcomes.
############################################################

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

# plot the ratio to inspect the envelope constant M
par(mar=c(2.1, 2.1, 2.1, 2.1))
plot(ratio, xlim = c(-10,10), ylim = c(0, 2), type = 'l', main = "Ratio of Densities of Logistic over Cauchy", xlab = "x", ylab = "Ratio")

# approximate M numerically on a fine grid
x_values <- seq(-10, 10, 0.01)
# evaluate the ratio at these x values
ratio_values <- sapply(x_values, ratio)

# find the maximum of ratio
M <- max(ratio_values)

# rejection sampling to generate logistic latent errors
n_samples <- 150
epsilon_logistic <- rep(NA, n_samples)
accepted <- 1
iteration_count <- 0

while (accepted <= n_samples) {
  # draw a candidate from the Cauchy proposal
  x_candidate <- rt(1, df = 1)
  
  # draw a uniform random number
  u <- runif(1)
  
  # compute acceptance probability
  accept_prob <- fL(x_candidate) / (M * fC(x_candidate))
  
  # accept or reject
  if (u < accept_prob) {
    epsilon_logistic[accepted] <- x_candidate
    accepted <- accepted + 1
  }
  
  iteration_count <- iteration_count + 1
}

# construct the latent variable for logistic regression
Y_star_logistic <- as.vector(X %*% beta) + epsilon_logistic

# convert latent variables into binary outcomes
Y_logistic <- as.numeric(Y_star_logistic > 0)

cat("Binary outcomes generated from the logistic model:\n")
print(Y_logistic)

# report empirical acceptance rate
cat("Acceptance rate:", n_samples / iteration_count, "\n")
cat("Estimated M:", M, "\n")




############################################################
# Part 4: Bayesian model fitting using Metropolis-Hastings MCMC
#
# Fit two binary regression models:
#   1. Logistic regression
#   2. Cauchit regression
#
# Each model is fitted under two priors:
#   - independent standard Gaussian prior
#   - Unit Information Prior
#
# This produces four fitted models in total:
#   - logistic + Gaussian prior
#   - cauchit + Gaussian prior
#   - logistic + Unit Information Prior
#   - cauchit + Unit Information Prior
#
# Posterior inference is performed using a Random-Walk
# Metropolis algorithm. To improve mixing relative to a basic
# isotropic random walk, a pre-conditioned Gaussian proposal
# is used, based on the geometry of the design matrix.
############################################################

library(mvtnorm)

############################################################
# Log-priors
############################################################

# independent N(0,1) prior for each beta_j
log_prior_gaussian <- function(beta) {
  sum(dnorm(beta, mean = 0, sd = 1, log = TRUE))
}

# unit Information Prior: beta ~ N_d(0, n (X'X)^(-1))
log_prior_unit <- function(beta, X) {
  n_obs <- nrow(X)
  Sigma <- n_obs * solve(t(X) %*% X)
  dmvnorm(beta, mean = rep(0, length(beta)), sigma = Sigma, log = TRUE)
}

############################################################
# Log-likelihoods
############################################################

# logistic regression:
# p_i = 1 / (1 + exp(-x_i^T beta))
loglik_logistic <- function(beta, X, y) {
  eta <- as.vector(X %*% beta)
  p <- plogis(eta)
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  sum(y * log(p) + (1 - y) * log(1 - p))
}

# cauchit regression:
# p_i = 1/pi * atan(x_i^T beta) + 1/2
loglik_cauchit <- function(beta, X, y) {
  eta <- as.vector(X %*% beta)
  p <- 0.5 + atan(eta) / pi
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  sum(y * log(p) + (1 - y) * log(1 - p))
}

############################################################
############################################################
# Pre-conditioned Random-Walk Metropolis algorithm
#
# This function implements a Random-Walk Metropolis sampler
# for Bayesian posterior inference.
#
# At each iteration, a new candidate parameter vector is
# proposed by adding a multivariate Gaussian perturbation to
# the current parameter vector:
#
#   beta_prop = beta_curr + scale * z
#
# where z has covariance structure given by proposal_cov.
#
# Unlike a basic random-walk proposal that uses independent
# N(0, step_size^2) moves in each coordinate, this version
# uses a pre-conditioned proposal. This means that the shape
# of the proposal distribution reflects the scale and
# dependence structure of the parameters. In practice, this
# often improves mixing and increases effective sample size,
# especially when predictors are correlated or parameters
# have different posterior scales.
############################################################

RWM_preconditioned <- function(log_likelihood,
                               log_prior,
                               X,
                               y,
                               initial_beta,
                               nits,
                               scale,
                               proposal_cov) {
  
  # number of parameters
  d <- length(initial_beta)
  
  # matrix to store the MCMC samples
  samples <- matrix(NA, nrow = nits, ncol = d)
  
  # initialise the chain at the chosen starting value
  beta_curr <- initial_beta
  
  # cholesky factor of the proposal covariance matrix
  # this is used to generate correlated Gaussian proposals
  chol_prop <- chol(proposal_cov)
  
  # evaluate the current log-posterior
  logpost_curr <- log_likelihood(beta_curr, X, y) + log_prior(beta_curr)
  
  # counter for accepted proposals
  accepted <- 0
  
  for (i in 1:nits) {
    
    # generate a multivariate Gaussian proposal step
    proposal_step <- drop(t(chol_prop) %*% rnorm(d))
    
    # propose a new parameter vector
    beta_prop <- beta_curr + scale * proposal_step
    
    # evaluate the proposed log-posterior
    logpost_prop <- log_likelihood(beta_prop, X, y) + log_prior(beta_prop)
    
    # log acceptance ratio
    log_alpha <- logpost_prop - logpost_curr
    
    # accept or reject the proposal
    if (log(runif(1)) < log_alpha) {
      beta_curr <- beta_prop
      logpost_curr <- logpost_prop
      accepted <- accepted + 1
    }
    
    # store the current state of the chain
    samples[i, ] <- beta_curr
  }
  
  # return posterior samples and acceptance rate
  list(
    samples = samples,
    acceptance_rate = accepted / nits
  )
}

############################################################
# Proposal covariance for pre-conditioning
#
# A simple choice is based on (X'X)^(-1), which partially
# accounts for scaling and dependence among predictors.
############################################################

proposal_cov <- solve(t(X) %*% X + diag(1e-6, ncol(X)))

# run the four models
set.seed(5623)

initial_beta <- seq(-2, 2, length = 10)
nits <- 50000

# logistic regression with Gaussian prior
fit_logistic_G <- RWM_preconditioned(
  log_likelihood = loglik_logistic,
  log_prior = log_prior_gaussian,
  X = X,
  y = Y_logistic,
  initial_beta = initial_beta,
  nits = nits,
  scale = 0.8,
  proposal_cov = proposal_cov
)

# cauchit regression with Gaussian prior
fit_cauchit_G <- RWM_preconditioned(
  log_likelihood = loglik_cauchit,
  log_prior = log_prior_gaussian,
  X = X,
  y = Y_cauchit,
  initial_beta = initial_beta,
  nits = nits,
  scale = 0.8,
  proposal_cov = proposal_cov
)

# logistic regression with Unit Information Prior
fit_logistic_U <- RWM_preconditioned(
  log_likelihood = loglik_logistic,
  log_prior = function(beta) log_prior_unit(beta, X),
  X = X,
  y = Y_logistic,
  initial_beta = initial_beta,
  nits = nits,
  scale = 0.8,
  proposal_cov = proposal_cov
)

# cauchit regression with Unit Information Prior
fit_cauchit_U <- RWM_preconditioned(
  log_likelihood = loglik_cauchit,
  log_prior = function(beta) log_prior_unit(beta, X),
  X = X,
  y = Y_cauchit,
  initial_beta = initial_beta,
  nits = nits,
  scale = 0.8,
  proposal_cov = proposal_cov
)


# store outputs
parameter_samples1 <- fit_logistic_G$samples
parameter_samples2 <- fit_cauchit_G$samples
parameter_samples3 <- fit_logistic_U$samples
parameter_samples4 <- fit_cauchit_U$samples

acceptance_rate1 <- fit_logistic_G$acceptance_rate
acceptance_rate2 <- fit_cauchit_G$acceptance_rate
acceptance_rate3 <- fit_logistic_U$acceptance_rate
acceptance_rate4 <- fit_cauchit_U$acceptance_rate

cat("Acceptance rate (logistic regression with Gaussian prior):", acceptance_rate1, "\n")
cat("Acceptance rate (cauchit regression with Gaussian prior):", acceptance_rate2, "\n")
cat("Acceptance rate (logistic regression with Unit Information prior):", acceptance_rate3, "\n")
cat("Acceptance rate (cauchit regression with Unit Information prior):", acceptance_rate4, "\n")




############################################################
# Part 5: MCMC diagnostics and comparison of fitted models
#
# This section continues from the fitted objects:
#   - fit_logistic_G
#   - fit_cauchit_G
#   - fit_logistic_U
#   - fit_cauchit_U
#
# Assess MCMC quality using:
#   - trace plots
#   - effective sample sizes
#   - autocorrelation plots
#
# Compare fitted models using:
#   - posterior mean estimates versus the true beta
#   - Brier scores based on posterior predictive probabilities
############################################################

library(coda)

# extract posterior samples and acceptance rates
parameter_samples1 <- fit_logistic_G$samples
parameter_samples2 <- fit_cauchit_G$samples
parameter_samples3 <- fit_logistic_U$samples
parameter_samples4 <- fit_cauchit_U$samples

acceptance_rate1 <- fit_logistic_G$acceptance_rate
acceptance_rate2 <- fit_cauchit_G$acceptance_rate
acceptance_rate3 <- fit_logistic_U$acceptance_rate
acceptance_rate4 <- fit_cauchit_U$acceptance_rate


# remove burn-in
burn_in <- 10000

samples_logisticG <- parameter_samples1[(burn_in + 1):nrow(parameter_samples1), ]
samples_cauchitG  <- parameter_samples2[(burn_in + 1):nrow(parameter_samples2), ]
samples_logisticU <- parameter_samples3[(burn_in + 1):nrow(parameter_samples3), ]
samples_cauchitU  <- parameter_samples4[(burn_in + 1):nrow(parameter_samples4), ]

mc_logisticG <- mcmc(samples_logisticG)
mc_cauchitG  <- mcmc(samples_cauchitG)
mc_logisticU <- mcmc(samples_logisticU)
mc_cauchitU  <- mcmc(samples_cauchitU)


# trace plots for selected parameters
# inspect beta_1 and beta_10 as representative examples.
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(samples_logisticG[, 1], type = "l",
     main = expression("Trace plot: Logistic + Gaussian (" * beta[1] * ")"),
     xlab = "Iteration", ylab = expression(beta[1]))

plot(samples_cauchitG[, 1], type = "l",
     main = expression("Trace plot: Cauchit + Gaussian (" * beta[1] * ")"),
     xlab = "Iteration", ylab = expression(beta[1]))

plot(samples_logisticU[, 10], type = "l",
     main = expression("Trace plot: Logistic + IUP (" * beta[10] * ")"),
     xlab = "Iteration", ylab = expression(beta[10]))

plot(samples_cauchitU[, 10], type = "l",
     main = expression("Trace plot: Cauchit + IUP (" * beta[10] * ")"),
     xlab = "Iteration", ylab = expression(beta[10]))


# effective sample size (ESS)
ess_logisticG <- effectiveSize(mc_logisticG)
ess_cauchitG  <- effectiveSize(mc_cauchitG)
ess_logisticU <- effectiveSize(mc_logisticU)
ess_cauchitU  <- effectiveSize(mc_cauchitU)

ess_summary <- rbind(
  "Logistic + Gaussian" = c(min = min(ess_logisticG),
                            median = median(ess_logisticG),
                            max = max(ess_logisticG)),
  "Cauchit + Gaussian"  = c(min = min(ess_cauchitG),
                            median = median(ess_cauchitG),
                            max = max(ess_cauchitG)),
  "Logistic + IUP"      = c(min = min(ess_logisticU),
                            median = median(ess_logisticU),
                            max = max(ess_logisticU)),
  "Cauchit + IUP"       = c(min = min(ess_cauchitU),
                            median = median(ess_cauchitU),
                            max = max(ess_cauchitU))
)

cat("ESS summary table:\n")
print(round(ess_summary, 2))

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

hist(ess_logisticG,
     main = "ESS: Logistic + Gaussian",
     xlab = "Effective sample size")

hist(ess_cauchitG,
     main = "ESS: Cauchit + Gaussian",
     xlab = "Effective sample size")

hist(ess_logisticU,
     main = "ESS: Logistic + IUP",
     xlab = "Effective sample size")

hist(ess_cauchitU,
     main = "ESS: Cauchit + IUP",
     xlab = "Effective sample size")


# autocorrelation plots
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

acf(samples_logisticG[, 1],
    main = expression("ACF: Logistic + Gaussian (" * beta[1] * ")"))

acf(samples_cauchitG[, 1],
    main = expression("ACF: Cauchit + Gaussian (" * beta[1] * ")"))

acf(samples_logisticU[, 1],
    main = expression("ACF: Logistic + IUP (" * beta[1] * ")"))

acf(samples_cauchitU[, 1],
    main = expression("ACF: Cauchit + IUP (" * beta[1] * ")"))


# posterior mean estimates compared with the true beta
post_mean_logisticG <- colMeans(samples_logisticG)
post_mean_cauchitG  <- colMeans(samples_cauchitG)
post_mean_logisticU <- colMeans(samples_logisticU)
post_mean_cauchitU  <- colMeans(samples_cauchitU)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(beta, post_mean_logisticG,
     main = "Posterior mean vs true beta\nLogistic + Gaussian",
     xlab = "True beta", ylab = "Posterior mean")
abline(0, 1, lty = 2)

plot(beta, post_mean_cauchitG,
     main = "Posterior mean vs true beta\nCauchit + Gaussian",
     xlab = "True beta", ylab = "Posterior mean")
abline(0, 1, lty = 2)

plot(beta, post_mean_logisticU,
     main = "Posterior mean vs true beta\nLogistic + IUP",
     xlab = "True beta", ylab = "Posterior mean")
abline(0, 1, lty = 2)

plot(beta, post_mean_cauchitU,
     main = "Posterior mean vs true beta\nCauchit + IUP",
     xlab = "True beta", ylab = "Posterior mean")
abline(0, 1, lty = 2)


# posterior predictive probabilities and Brier scores
brier_score <- function(y, m) {
  mean((y - m)^2)
}

predict_logistic_prob <- function(beta, X) {
  as.vector(plogis(X %*% beta))
}

predict_cauchit_prob <- function(beta, X) {
  as.vector(0.5 + atan(X %*% beta) / pi)
}

# compute posterior predictive probabilities by averaging over the posterior samples
predicted_logisticG <- Reduce(`+`,
                              lapply(seq_len(nrow(samples_logisticG)), function(i) {
                                predict_logistic_prob(samples_logisticG[i, ], X)
                              })
) / nrow(samples_logisticG)

predicted_cauchitG <- Reduce(`+`,
                             lapply(seq_len(nrow(samples_cauchitG)), function(i) {
                               predict_cauchit_prob(samples_cauchitG[i, ], X)
                             })
) / nrow(samples_cauchitG)

predicted_logisticU <- Reduce(`+`,
                              lapply(seq_len(nrow(samples_logisticU)), function(i) {
                                predict_logistic_prob(samples_logisticU[i, ], X)
                              })
) / nrow(samples_logisticU)

predicted_cauchitU <- Reduce(`+`,
                             lapply(seq_len(nrow(samples_cauchitU)), function(i) {
                               predict_cauchit_prob(samples_cauchitU[i, ], X)
                             })
) / nrow(samples_cauchitU)

brier_logisticG <- brier_score(Y_logistic, predicted_logisticG)
brier_cauchitG  <- brier_score(Y_cauchit, predicted_cauchitG)
brier_logisticU <- brier_score(Y_logistic, predicted_logisticU)
brier_cauchitU  <- brier_score(Y_cauchit, predicted_cauchitU)

brier_table <- rbind(
  "Logistic + Gaussian" = brier_logisticG,
  "Cauchit + Gaussian"  = brier_cauchitG,
  "Logistic + IUP"      = brier_logisticU,
  "Cauchit + IUP"       = brier_cauchitU
)

cat("\nBrier scores:\n")
print(round(brier_table, 6))


# acceptance-rate summary
acceptance_table <- rbind(
  "Logistic + Gaussian" = acceptance_rate1,
  "Cauchit + Gaussian"  = acceptance_rate2,
  "Logistic + IUP"      = acceptance_rate3,
  "Cauchit + IUP"       = acceptance_rate4
)

cat("\nAcceptance rates:\n")
print(round(acceptance_table, 4))





