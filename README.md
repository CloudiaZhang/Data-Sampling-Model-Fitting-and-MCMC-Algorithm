This repository contains code and report materials for a Bayesian binary classification project based on logistic and cauchit regression models.

## Project overview

The project studies two binary regression models:
- logistic regression
- cauchit regression

Binary outcomes are generated from latent-variable representations. For the cauchit model, latent errors are sampled directly from the Cauchy distribution. For the logistic model, latent errors are generated using rejection sampling with a Cauchy proposal distribution.

Posterior inference is then carried out under two prior specifications:
- standard Gaussian prior
- Unit Information Prior

This gives four fitted models in total. A pre-conditioned Random-Walk Metropolis algorithm is used to improve sampling efficiency compared with a basic isotropic random-walk proposal.

## Files

- `simulation_preconditioned_mcmc.R` — main R script for data generation, model fitting, diagnostics, and model comparison
- `simulation_preconditioned_mcmc_logistic_cauchit.pdf` — polished report version prepared as an application sample

## Methods used

- simulation of correlated covariates and binary outcomes
- latent-variable formulation for logistic and cauchit models
- rejection sampling for logistic latent errors
- Bayesian inference using Metropolis–Hastings MCMC
- pre-conditioned proposal covariance based on \((X^T X)^{-1}\)
- trace plots, autocorrelation, effective sample size, posterior mean comparison, and Brier score

## Reproducibility

No external dataset is required. All data used in this project are simulated within the R script.

## Note

This repository contains an improved and cleaned version of coursework originally completed for STAT0044 at University College London.
