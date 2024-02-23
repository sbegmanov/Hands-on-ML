# – Problem type: supervised multinomial classification
# – response variable: V785 (i.e., numbers to predict: 0, 1, …, 9)
# – features: 784
# – observations: 60,000 (train) / 10,000 (test)
# – objective: use attributes about the “darkness” of each of the 784
#   pixels in images of handwritten numbers to predict if the number is 0, 1, …, or 9.
# – access: provided by the dslabs package (Irizarry, 2018)
# – more details: See ?dslabs::read_mnist() and online MNIST documentation
library(tidyverse)
library(dslabs)

# MNIST training data
mnist <- dslabs::read_mnist()
names(mnist)
dim(mnist$train$images)
head(mnist$train$labels)

# MNIST example
index <- sample(nrow(mnist$train$images), size = 10000)
mnist_x <- mnist$train$images[index, ]
mnist_y <- factor(mnist$train$labels[index])

library(purrr)

mnist_x %>% 
  as.data.frame() %>% 
  map_df(sd) %>% 
  gather(feature, sd) %>% 
  ggplot(aes(sd)) +
  geom_histogram(binwidth = 1)

# rename features
colnames(mnist_x) <- paste0("V", 1:ncol(mnist_x))

# remove near zero variance features manually
nzv <- nearZeroVar(mnist_x)
index <- setdiff(1:ncol(mnist_x), nzv)
mnist_x <- mnist_x[, index]


# use train/validate resampling method
cv <- trainControl(
  method = "LGOCV",
  p = 0.7,
  number = 1,
  savePredictions = TRUE
)

# create a hyperparameter grid search
hyper_grid <- expand.grid(k = seq(3, 25, by = 2))

# execute grid search
knn_mnist <- train(
  mnist_x,
  mnist_y,
  method = "knn",
  tuneGrid = hyper_grid,
  preProc = c("center", "scale"),
  trControl = cv
)

ggplot(knn_mnist)

# create confusion matrix
cm <- confusionMatrix(knn_mnist$pred$pred, knn_mnist$pred$obs)
# sensitivity, specificity, accuracy
cm$byClass[, c(1:2, 11)]

library(caret)
library(dplyr)

# top 20 most important features
vi <- varImp(knn_mnist)
print(vi)

library(tibble)

# get median value for feature importance
imp <- vi$importance %>% 
  rownames_to_column(var = "feature") %>% 
  gather(response, imp, -feature) %>% 
  group_by(feature) %>% 
  summarize(imp = median(imp))

# create tibble for all edge pixels
edges <- tibble(
  feature = paste0("V", nzv),
  imp = 0
)

library(stringr)
# combine and plot
imp <- rbind(imp, edges) %>% 
  mutate(ID = as.numeric(str_extract(feature, "\\d+"))) %>% 
  arrange(ID)

image(matrix(imp$imp, 28, 28), col = gray(seq(0, 1, 0.05)),
      xaxt = "n", yaxt = "n")


# get a few accurate predictions
set.seed(9)
good <- knn_mnist$pred %>% 
  filter(pred == obs) %>% 
  sample_n(4)

# get a few inaccurate predictions
set.seed(9)
bad <- knn_mnist$pred %>% 
  filter(pred != obs) %>% 
  sample_n(4)

combine <- bind_rows(good, bad)

# get original feature set with all pixel features
set.seed(123)
index <- sample(nrow(mnist$train$images), 10000)
X <- mnist$train$images[index,]

# plot results
par(mfrow = c(4, 2), mar = c(1, 1, 1, 1))
layout(matrix(seq_len(nrow(combine)), 4, 2, byrow = FALSE))

for(i in seq_len(nrow(combine))) {
  image(matrix(X[combine$rowIndex[i],], 28, 28)[, 28:1],
        col = gray(seq(0, 1, 0.05)),
        main = paste("Actual: ", combine$obs[i], " ",
                     "Predicted:", combine$pred[i]),
        xaxt = "n", yaxt = "n")
}

### deep learning
# import MNIST training data
mnist <- dslabs::read_mnist()
mnist_x <- mnist$train$images
mnist_y <- mnist$train$labels

# rename columns and standardize feature values
colnames(mnist_x) <- paste0("V", 1:ncol(mnist_x))
mnist_x <- mnist_x / 255

library(keras)
# one-hot encode response
mnist_y <- to_categorical(mnist, 10)

# get number of feature in the model
p <- ncol(mnist_x)

# network architecture
# activation
model <- keras_model_sequential() %>% 
  layer_dense(units = 128,activation = "relu",input_shape = p) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax") %>% 
  # backpropagation
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

# train the model
fit1 <- model %>% 
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = FALSE
  )
print(fit1)
plot(fit1)

# model tuning
# batch normalization

model_w_norm <- keras_model_sequential() %>% 
  
  # network architecture with batch normalization
  layer_dense(units = 256, activation = "relu", input_shape = p) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 128,activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

# regularization

model_w_reg <- keras_model_sequential() %>% 
  
  # network architecture with L1 regularization and batch normalization
  layer_dense(units = 256, activation = "relu", input_shape = p,
              kernel_regularizer = regularizer_l2(0.001)) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 128,activation = "relu",
              kernel_regularizer = regularizer_l2(0.001)) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 64, activation = "relu",
              kernel_regularizer = regularizer_l2(0.001)) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

# dropout as additional regularization
model_w_reg <- keras_model_sequential() %>% 
  
  # network architecture with L1 regularization and batch normalization
  layer_dense(units = 256, activation = "relu", input_shape = p) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 128,activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

# adjusting learning rate
model_w_adj_lrn <- keras_model_sequential() %>% 
  
  # network architecture with batch normalization
  layer_dense(units = 256, activation = "relu", input_shape = p) %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128,activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  ) %>% 
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 35,
    batch_size = 128,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = 0.05)
    ),
    verbose = FALSE
  )

print(model_w_adj_lrn)

# optimal
min(model_w_adj_lrn$metrics$val_loss)
max(model_w_adj_lrn$metrics$val_acc)
max(model_w_adj_lrn$metrics$val_acc)

# learning rate
plot(model_w_adj_lrn)


# grid search
FLAGS <- flags(
  # nodes
  flag_numeric("nodes1", 256),
  flag_numeric("nodes2", 128),
  flag_numeric("nodes3", 64),
  
  # dropouts
  flag_numeric("dropout1", 0.4),
  flag_numeric("dropout2", 0.3),
  flag_numeric("dropout3", 0.2),
  
  # learning parameters
  flag_string("optimizer", "rmsprop"),
  flag_numeric("lr_annealing", 0.1)
)



model <- keras_model_sequential() %>% 
  layer_dense(units = FLAGS$nodes1, activation = "relu", input_shape = p) %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = FLAGS$dropout1) %>% 
  layer_dense(units = FLAGS$nodes2,activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = FLAGS$dropout2) %>% 
  layer_dense(units = FLAGS$nodes3, activation = "relu") %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = FLAGS$dropout3) %>% 
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy'),
    optimizer = FLAGS$optimizer
  ) %>% 
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 35,
    batch_size = 128,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = FLAGS$lr_annealing)
    ),
    verbose = FALSE
  )

# running of combinations of dropout1 and dropout2
runs <- tuning_run("scripts/mnist_grid_search.R",
                   flags = list(
                     nodes1 = c(64, 128, 256),
                     nodes2 = c(64, 128, 256),
                     nodes3 = c(64, 128, 256),
                     dropout1 = c(0.2, 0.3, 0.4),
                     dropout2 = c(0.2, 0.3, 0.4),
                     dropout3 = c(0.2, 0.3, 0.4),
                     optimizer = c("rmsprop", "adam"),
                     lr_annealing = c(0.1, 0.05)
                   ),
                   sample = 0.05
                   )
runs %>% 
  filter(metric_val_loss == min(metric_val_loss)) %>% 
  glimpse()

### autoencoders

# convert features to an h2o input data set
features <- as.h2o(mnist$train$images)

# train an autoencoder

ae1 <- h2o.deeplearning(
  x = seq_along(features),
  training_frame = features,
  autoencoder = TRUE,
  hidden = 2,
  activation = 'Tanh',
  sparse = TRUE #80% of the data elements are zeros.
)

# extract the deep features
ae1_codings <- h2o.deepfeatures(ae1, features, layer = 1)
ae1_codings


# five undercomplete autoencoder architecture
# hyperparameter serach grid
hyper_grid <- list(hidden = list(
  c(50),
  c(100),
  c(300, 100, 300),
  c(100, 50, 100),
  c(250, 100, 50, 100, 250)
))

# grid search
ae_grid <- h2o.grid(
  algorithm = 'deeplearning',
  x = seq_along(features),
  training_frame = features,
  grid_id = 'autoencoder_grid',
  autoencoder = TRUE,
  activation = 'Tanh',
  hyper_params = hyper_grid,
  sparse = TRUE,
  ignore_const_cols = TRUE,
  seed = 123
)

# grid details
h2o.getGrid('autoencoder_grid', sort_by = 'mse', decreasing = FALSE)

# plot the reconstruction
# get sampled test images

index <- sample(1:nrow(mnist$test$images), 4)
sampled_digits <- mnist$test$images[index, ]
colnames(sampled_digits) <- paste0("V", seq_len(ncol(sampled_digits)))

# predict reconstructed pixel values
best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
recon_digits <- predict(best_model, as.h2o(sampled_digits))
names(recon_digits) <- paste0("V", seq_len(ncol(recon_digits)))

combine <- rbind(sampled_digits, as.matrix(recon_digits))

# Plot original versus reconstructed
par(mfrow = c(1, 3), mar=c(1, 1, 1, 1))
layout(matrix(seq_len(nrow(combine)), 4, 2, byrow = FALSE))

for(i in seq_len(nrow(combine))) {
  image(matrix(combine[i, ], 28, 28)[, 28:1], xaxt="n", yaxt="n")
}

# For  best_model with 100 codings, the sparsity level is zero
ae100_codings <- h2o.deepfeatures(best_model, features, layer = 1)
ae100_codings %>%
  as.data.frame() %>%
  tidyr::gather() %>%
  summarize(average_activation = mean(value))

# hyperparameter search grid
hyper_grid <- list(sparsity_beta = c(0.01, 0.05, 0.1, 0.2))

# execute grid search
ae_sparsity_grid <- h2o.grid(
  algorithm = 'deeplearning',
  x = seq_along(features),
  training_frame = features,
  grid_id = 'sparsity_grid',
  autoencoder = TRUE,
  hidden = 100,
  activation = 'Tanh',
  hyper_params = hyper_grid,
  sparse = TRUE,
  average_activation = -0.1,
  ignore_const_cols = FALSE,
  seed = 123
)

# grid details
h2o.getGrid('sparsity_grid', sort_by = 'mse', decreasing = FALSE)

### Clustering k-means
library(stats)

features <- mnist$train$images

# k-means model, k number based on priori knowledge
mnist_clustering <- kmeans(features, 
                           centers = 10, 
                           nstart = 10)

# model output
str(mnist_clustering)

# Extract cluster centers
mnist_centers <- mnist_clustering$centers

# Plot typical cluster digits
par(mfrow = c(2, 5), mar=c(0.5, 0.5, 0.5, 0.5))
layout(matrix(seq_len(nrow(mnist_centers)), 2, 5, 
              byrow = FALSE))
for(i in seq_len(nrow(mnist_centers))) {
  image(matrix(mnist_centers[i, ], 28, 28)[, 28:1],
        col = gray.colors(12, rev = TRUE), xaxt="n", yaxt="n")
}

# compare the cluster digits with the actual digit
# Create mode function

mode_fun <- function(x){
  which.max(tabulate(x))
}
mnist_comparison <- data.frame(
  cluster = mnist_clustering$cluster,
  actual = mnist$train$labels
) %>%
  group_by(cluster) %>%
  mutate(mode = mode_fun(actual)) %>%
  ungroup() %>%
  mutate_all(factor, levels = 0:9)
# Create confusion matrix and plot results
yardstick::conf_mat(
  mnist_comparison,
  truth = actual,
  estimate = mode
) %>%
  autoplot(type = 'heatmap')

# clustering into 1-5 clusters
library(factoextra)
fviz_nbclust(
  my_basket,
  kmeans,
  k.max = 25,
  method = "wss", # elbow method
  diss = get_dist(my_basket, method = "spearman") # correlation-based distrance measure
)


# for larger data set use clustering large applications(CLARA)
library(cluster)

# k-means computation time on MNIST data
system.time(kmeans(features, centers = 10))
# CLARA computation time on MNIST data
system.time(clara(features, k = 10))

### hierarchical clustering
library(dplyr)

ames_scale <- AmesHousing::make_ames() %>% 
  select_if(is.numeric) %>% 
  select(-Sale_Price) %>% 
  mutate_all(as.double) %>% 
  scale()
  
library(stats)
set.seed(123)

# dissimilarity matrix
d <- dist(ames_scale, method = "euclidean")

# complete linkage
hc1 <- hclust(d, method = "complete")

# agnes clustering
library(cluster)
set.seed(123)

# maximum or complete linkage clustering with agnes
hc2 <- agnes(ames_scale, method = "complete")

# agglomerative coeff.
hc2$ac


# methods to assess
m <- c("average", "single", "complete", "ward")
names(m) <- c("average", "single", "complete", "ward")

# compute coeff.
ac <- function(x) {
  agnes(ames_scale, method = x)$ac
}

# agglomerative coeff. for each linakage method
purrr::map_dbl(m, ac)

# divisive hiearchical clustering
library(cluster)

hc4 <- diana(ames_scale)
# divise coeff.
hc4$dc

# determine optimal clusters
library(factoextra)

p1 <- fviz_nbclust(ames_scale, FUN = hcut, method = "wss",
                   k.max = 10) +
  ggtitle("(A) Elbow method")
p2 <- fviz_nbclust(ames_scale, FUN = hcut, method = "silhouette",
                   k.max = 10) +
  ggtitle("(B) Silhouette method")
p3 <- fviz_nbclust(ames_scale, FUN = hcut, method = "gap_stat",
                   k.max = 10) +
  ggtitle("(C) Gap statistic")

gridExtra::grid.arrange(p1, p2, p3, nrow = 1)

# dendrograms
hc5 <- hclust(d, method = "ward.D2" )
dend_plot <- fviz_dend(hc5)
dend_data <- attr(dend_plot, "dendrogram")
dend_cuts <- cut(dend_data, h = 8)
fviz_dend(dend_cuts$lower[[2]])

# Ward’s method
hc5 <- hclust(d, method = "ward.D2")

# Cut tree into 4 groups
sub_grp <- cutree(hc5, k = 8)

# Number of members in each cluster
table(sub_grp)

#  full dendogram
fviz_dend(
  hc5,
  k = 8,
  horiz = TRUE,
  rect = TRUE,
  rect_fill = TRUE,
  rect_border = "jco",
  k_colors = "jco",
  cex = 0.1
)

# legible dendogramddsdfsw
dend_plot <- fviz_dend(hc5) # create full dendogram
dend_data <- attr(dend_plot, "dendrogram") # extract plot info
dend_cuts <- cut(dend_data, h = 70.5) # cut the dendogram at
# designated height
# Create sub dendrogram plots
p1 <- fviz_dend(dend_cuts$lower[[1]])
p2 <- fviz_dend(dend_cuts$lower[[1]], type = 'circular')

gridExtra::grid.arrange(p1, p2, nrow = 1)




