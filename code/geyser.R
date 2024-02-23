library(mclust)
library(dplyr)

### Model-based Clustering

# Gaussian mixture model (GMM)
# GMM model with 3 components

data(geyser, package = 'MASS')
geyser_mc <- Mclust(geyser, G = 3)

# results
plot(geyser_mc, what = "density")
plot(geyser_mc, what = "uncertainty")

# observations with high uncertainty 
sort(geyser_mc$uncertainty, decreasing = TRUE) %>% 
  head()

# model selection
summary(geyser_mc)

# Bayesian information criterion(BIC) to choose a best model
# leaving G = NULL forces Mclust() to evaluate 1–9 clusters
# then select the optimal # of components based on BIC

geyser_optimal_mc <- Mclust(geyser)
summary(geyser_optimal_mc)
legend_args <- list(x = "bottomright", ncol = 5)
plot(geyser_optimal_mc, what = "BIC", legendArgs = legend_args)
plot(geyser_optimal_mc, what = "classification")
plot(geyser_optimal_mc, what = "uncertainty")
