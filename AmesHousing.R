# – problem type: supervised regression
# – response variable: Sale_Price (i.e., $195,000, $215,000)
# – features: 80
# – observations: 2,930
# – objective: use property attributes to predict the sale price of a home
# – access: provided by the AmesHousing package (Kuhn, 2017a)
# – more details: See ?AmesHousing::ames_raw
library(tidyverse)
library(AmesHousing)
library(h2o)

ames <- AmesHousing::make_ames()
dim(ames)
head(ames$Sale_Price)

h2o.init()
ames.h2o <- as.h2o(ames)

# raw data
raw_ames <- AmesHousing::ames_raw


### Simple random sampling
# using base R
set.seed(123)
index_1 <- sample(1:nrow(ames), round(nrow(ames) * 0.7))
train_1 <- ames[index_1, ]
test_1 <- ames[-index_1, ]

# using caret package
library(ggplot2)
library(caret)
require(lattice)

set.seed(123)
index_2 <- createDataPartition(ames$Sale_Price, p = 0.7, list = FALSE)
train_2 <- ames[index_2, ]
test_2 <- ames[-index_2, ]

### using rsample package
library(rsample)
set.seed(123)
split_1 <- initial_split(ames, prop = 0.7)
train_3 <- training(split_1)
test_3 <- testing(split_1)

# using h2o package
split_2 <- h2o.splitFrame(ames.h2o, ratios = 0.7, seed = 123)
train_4 <- split_2[[1]]
test_4 <- split_2[[2]]

# same linear regression model output
lm_gm <- lm(Sale_Price ~ ., data = ames)
glm_glm <- glm(Sale_Price ~ ., data = ames, family = gaussian)
# meta engine (aggregator)
lm_caret <- train(Sale_Price ~ ., data = ames, method = "lm")

# resampling using h2o
h2o.cv <- h2o.glm(x = x, y = y,
                  training_frame = ames.h2o,
                  nfolds = 10)

vfold_cv(ames, v = 10)

# resampling using bootstraps
bootstraps(ames, times = 10)

# stratified sampling with the rsample package
set.seed(123)
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)


# k-nearest neighbor regressor application
# To so
# 1. Resampling method: we use 10-fold CV repeated 5 times.
# 2. Grid search: we specify the hyperparameter values to assess (𝑘 = 2, 4, 6, … , 25).
# 3. Model training & Validation: we train a k-nearest neighbor (method = ”knn”) 
# model using our pre-specified resampling procedure (trControl = cv), 
# grid search (tuneGrid = hyper_grid), and preferred loss 
# function (metric = ”RMSE”).

# specify resampling strategy
cv <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5
)

# create grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1))

# tune a knn model using grid search
knn_fit <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "knn",
  trControl = cv,
  tuneGrid = hyper_grid,
  metric = "RMSE"
)

knn_fit
ggplot(knn_fit)

# target engineering
# option-1: normalize with a log transformation

transformed_response <- log(ames_train$Sale_Price)

library(recipes)
# create a blueprint to be reapplied strategically
ames_recipe <- recipes::recipe(Sale_Price ~ ., data = ames_train) %>% 
  recipes::step_log(all_outcomes())

# option-2: use a Box Cox transformation
# here, we need to compute the lambda on the training set to
# training and test set to minimize data leakage

# dealing with missingness
# visualizing missing values from raw AmesHousing
sum(is.na(AmesHousing::ames_raw))

AmesHousing::ames_raw %>% 
  is.na() %>% 
  reshape2::melt() %>% 
  ggplot(aes(Var2, Var1, fill = value)) +
    geom_raster() +
    coord_flip() +
    scale_y_continuous(NULL, expand = c(0, 0)) +
    scale_fill_grey(name = "",
                    labels = c("Present", "Missing")) +
    xlab("Observation") +
    theme(
      axis.text.y = element_text(size = 4)
      )


AmesHousing::ames_raw %>%
  filter(is.na("Garage Type")) %>%
  select("Garage Type", "Garage Cars", "Garage Area")

library(visdat)

vis_miss(AmesHousing::ames_raw, cluster = TRUE)

# imputation should be performed within the resampling process
# step_modeimpute() to impute categorical features 
# with the most common value.

ames_recipe %>% 
  step_impute_median(Gr_Liv_Area)


# KNN imputation
# best used on small to moderate sized data

ames_recipe %>% 
  step_impute_knn(all_predictors(), neighbors = 6)

# Tree-based imputation
ames_recipe %>% 
  step_impute_bag(all_predictors())

vis_miss(ames_recipe, cluster = TRUE)


# Feature filtering

