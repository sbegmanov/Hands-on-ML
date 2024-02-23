# – Problem type: unsupervised basket analysis
# – response variable: NA
# – features: 42
# – observations: 2,000
# – objective: use attributes of each basket to identify common groupings
# of items purchased together.
# – access: available on the companion website for this book

library(readr)
library(dplyr)
library(ggplot2)
library(h2o)

url <- "https://raw.githubusercontent.com/koalaverse/homlr/master/data/my_basket.csv"
my_basket <- readr::read_csv(url)
dim(my_basket)


# PCA
# 1. Data are in tidy format
# 2. Any missing values in the data must be removed or imputed
# 3. Typically, the data must all be numeric values (e.g., one-hot, label, ordinal encoding categorical features)
# 4. Numeric data should be standardized (e.g., centered and scaled) to make features comparable 


h2o.no_progress() # turn off progress bars for brevity
h2o.init(max_mem_size = "5g")

# if data contains more numeric variables, use method = "GramSVD"
# if data contains more categorical variables, use method = "GLRM"

# convert data to h20 object
my_basket.h2o <- as.h2o(my_basket)

# PCA
my_pca <- h2o.prcomp(
  training_frame = my_basket.h2o,
  pca_method = "GramSVD",
  k = ncol(my_basket.h2o),
  transform = "STANDARDIZE",
  impute_missing = TRUE,
  max_runtime_secs = 1000
)

my_pca

my_pca@model$importance

# first PC (PC1) captures the most variance
my_pca@model$eigenvectors %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>% 
  ggplot(aes(pc1, reorder(feature, pc1))) +
  geom_point()

# how the different features contribute to PC1 and PC2
my_pca@model$eigenvectors %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>% 
  ggplot(aes(pc1, pc2, label = feature)) +
  geom_text()

# 3 approaches to decide the number of PCs to keep
# 1. Eigenvalue criterion
# 2. Proportion of variance explained criterion
# 3. Scree plot criterion

# Eigenvalue criterion
# h2o.prcomp() automatically computes the SD of the PCs

eigen <- my_pca@model$importance["Standard deviation", ] %>% 
  as.vector() %>%
  as.numeric() %>% 
  .^2

# sum of all eigenvalues equals number of variables
sum(eigen)

# plot
plot(eigen)

# Find PCs where the sums of eigenvalues is greater than or equal to 1
which(eigen >= 1)

# Proportion of variance explained criterion (PVE)
# h2o.prcomp() provides with PVE and cumulative variance explained (CVE)

# extract and plot PVE and CVE
data.frame(
  PC = my_pca@model$importance %>% seq_along(),
  PVE = my_pca@model$importance %>% .[2, ] %>% unlist(),
  CVE = my_pca@model$importance %>% .[3, ] %>% unlist()
) %>% 
  tidyr::gather(metric, variance_explained, -PC) %>% 
  ggplot(aes(PC, variance_explained)) +
  geom_point() +
  facet_wrap(~metric, ncol = 1, scales = "free")

# How many PCs required to explain at least 75% of total variability
min(which(ve$CVE >= 0.75))

# Scree plot criterion
data.frame(
  PC = my_pca@model$importance %>% seq_along,
  PVE = my_pca@model$importance %>% .[2,] %>% unlist()
) %>% 
  ggplot(aes(PC, PVE, group = 1, label = PC)) +
  geom_point() +
  geom_line() +
  geom_text(nudge_y = -.002)


### Generalized Low Rank Models(GLRMs)
h2o.no_progress()
h2o.init(max_mem_size = "5g")

# convert data to h2o object
my_basket.h2o <- as.h2o(my_basket)

# basic GLRM
basic_glrm <- h2o.glrm(
  training_frame = my_basket.h2o,
  k = 20,
  loss = "Quadratic",
  regularization_x = "None",
  regularization_y = "None",
  transform = "STANDARDIZE",
  max_iterations = 2000,
  seed = 123
)

# top level summary information on model
summary(basic_glrm)

# the model converged at 1000 iterations
plot(basic_glrm)

str(basic_glrm) 

# amount of variance explained by each archetype
basic_glrm@model$importance


data.frame(
  PC = basic_glrm@model$importance %>% seq_along(),
  PVE = basic_glrm@model$importance %>% .[2,] %>% unlist(),
  CVE = basic_glrm@model$importance %>% .[3,] %>% unlist()
) %>% 
  tidyr::gather(metric, variance_explained, -PC) %>% 
  ggplot(aes(PC, variance_explained)) +
  geom_point() +
  facet_wrap(~ metric, ncol = 1, scales = "free")

# how each feature aligns to the different archetypes
t(basic_glrm@model$archetypes)[1:5, 1:5]


# grid arrange
p1 <- t(basic_glrm@model$archetypes) %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>% 
  ggplot(aes(Arch1, reorder(feature, Arch1))) +
  geom_point()

p2 <- t(basic_glrm@model$archetypes) %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>% 
  ggplot(aes(Arch1, Arch2, label = feature)) +
  geom_text()
gridExtra::grid.arrange(p1, p2, nrow = 1)

#  based on scree plot approach, k = 8
# re-run model with k = 8

k8_glrm <- h2o.glrm(
  training_frame = my_basket.h2o,
  k = 8,
  loss = "Quadratic",
  regularization_x = "None",
  regularization_y = "None",
  transform = "STANDARDIZE",
  max_iterations = 2000,
  seed = 123
)

# reconstruct to see how well the model did
my_reconstruction <- h2o.reconstruct(k8_glrm, my_basket.h2o,
                                     reverse_transform = TRUE)
# raw predicted values
my_reconstruction[1:5, 1:4]

# round values to whole integers
my_reconstruction[1:5, 1:4] %>% round(0)

# tuning to optimize for unseen data
# non-negative regularization
k8_glrm_regularized <- h2o.glrm(
  training_frame = my_basket.h2o,
  k = 8,
  loss = "Quadratic",
  regularization_x = "NonNegative",
  regularization_y = "NonNegative",
  gamma_x = 0.5,
  gamma_y = 0.5,
  transform = "STANDARDIZE",
  max_iterations = 2000,
  seed = 123
)

# predicted values
predict(k8_glrm_regularized, my_basket.h2o)[1:5, 1:4]

# compare regularized vs non-regularized loss
par(mfrow = c(1, 2))
plot(k8_glrm)
plot(k8_glrm_regularized)


# automated tuning grid
# split data into train and validation
split <- h2o.splitFrame(my_basket.h2o, ratios = 0.75, seed = 123)
train <- split[[1]]
valid <- split[[2]]

# hyperparameter search grid
params <- expand.grid(
  regularization_x = c("None", "NonNegative", "L1"),
  regularization_y = c("None", "NonNegative", "L1"),
  gamma_x = seq(0, 1, by = .25),
  gamma_y = seq(0, 1, by = .25),
  error = 0,
  stringsAsFactors = FALSE
)

# perform grid search
for(i in seq_len(nrow(params))) {
  # model
  glrm_model <- h2o.glrm(
    training_frame = train,
    k = 8,
    loss = "Quadratic",
    regularization_x = params$regularization_x[i],
    regularization_y = params$regularization_y[i],
    gamma_x = params$gamma_x[i],
    gamma_y = params$gamma_y[i],
    transform = "STANDARDIZE",
    max_runtime_secs = 1000,
    seed = 123
  )
  
  # predict on validation set and extract error
  validate <- h2o.performance(glrm_model, valid)
  params$error[i] <- validate@metrics$numerr
}

# top 10 models with the lowest error rate
params %>% 
  arrange(error) %>% 
  head(10)

# finding an optimal model, rerun on entire training set
# final model with optimal hyperparameters

final_glrm_model <- h2o.glrm(
  training_frame = my_basket.h2o,
  k = 8,
  loss = "Quadratic",
  regularization_x = "L1",
  regularization_y = "NonNegative",
  gamma_x = 1,
  gamma_y = 0.25,
  transform = "STANDARDIZE",
  max_iterations = 2000,
  seed = 123
)

# two new observation to score
new_observations <- as.h2o(sample_n(my_basket, 2))

# basic scoring
score <- predict(final_glrm_model, new_observations) %>% 
  round(0)
score[, 1:4]


# denoising autoencoder


# Train a denoise autoencoder
denoise_ae <- h2o.deeplearning(
  x = seq_along(features),
  training_frame = inputs_currupted_gaussian,
  validation_frame = features,
  autoencoder = TRUE,
  hidden = 100,
  activation = 'Tanh',
  sparse = TRUE
)
# performance
h2o.performance(denoise_ae, valid = TRUE)

# anomaly detection
# Extract reconstruction errors
(reconstruction_errors <- h2o.anomaly(best_model, features))


# Plot distribution
reconstruction_errors <- as.data.frame(reconstruction_errors)
ggplot(reconstruction_errors, aes(Reconstruction.MSE)) +
  geom_histogram()

### Model-based Clustering
library(mclust)

url <- "https://koalaverse.github.io/homlr/data/my_basket.csv"
my_basket <- readr::read_csv(url)

# Gaussian mixture model (GMM)
# search across all 14 GMM models across 1–20 clusters
my_basket_mc <- Mclust(my_basket, 1:20)
summary(my_basket)

plot(my_basket_mc, what = 'BIC',
     legendArgs = list(x = "bottomright", ncol = 5))

# bimodal distributions of probabilities
# Distribution of probabilities for all observations 
# aligning to each of 6 clusters

probabilities <- my_basket_mc$z
colnames(probabilities) <- paste0('C', 1:6)

probabilities <- probabilities %>%
  as.data.frame() %>%
  mutate(id = row_number()) %>%
  tidyr::gather(cluster, probability, -id)

ggplot(probabilities, aes(probability)) +
  geom_histogram() +
  facet_wrap(~ cluster, nrow = 2)

uncertainty <- data.frame(
  id = 1:nrow(my_basket),
  cluster = my_basket_mc$classification,
  uncertainty = my_basket_mc$uncertainty
)
uncertainty %>%
  group_by(cluster) %>%
  filter(uncertainty > 0.25) %>%
  ggplot(aes(uncertainty, reorder(id, uncertainty))) +
  geom_point() +
  facet_wrap(~ cluster, scales = 'free_y', nrow = 1)

# extract cluster membership
# average standardized consumption
cluster2 <- my_basket %>%
  scale() %>%
  as.data.frame() %>%
  mutate(cluster = my_basket_mc$classification) %>%
  filter(cluster == 2) %>%
  select(-cluster)
cluster2 %>%
  tidyr::gather(product, std_count) %>%
  group_by(product) %>%
  summarize(avg = mean(std_count)) %>%
  ggplot(aes(avg, reorder(product, avg))) +
  geom_point() +
  labs(x = "Average standardized consumption", y = NULL)







