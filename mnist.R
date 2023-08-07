# – Problem type: supervised multinomial classification
# – response variable: V785 (i.e., numbers to predict: 0, 1, …, 9)
# – features: 784
# – observations: 60,000 (train) / 10,000 (test)
# – objective: use attributes about the “darkness” of each of the 784
#   pixels in images of handwritten numbers to predict if the number is 0, 1, …, or 9.
# – access: provided by the dslabs package (Irizarry, 2018)
# – more details: See ?dslabs::read_mnist() and online MNIST documentation

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














