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

# resampling using h2o just sample
h2o.cv <- h2o.glm(x = x, y = y,
                  training_frame = ames.h2o,
                  nfolds = 10)
# V-fold cross-validation randomly splitting using rsample
vfold_cv(ames, v = 10)

# resampling using bootstraps
bootstraps(ames, times = 10)

# stratified sampling with the rsample package
set.seed(123)
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)


# k-nearest neighbor regressor application
# To do so
# 1. Resampling method: use 10-fold CV repeated 5 times.
# 2. Grid search: specify the hyperparameter values to assess (𝑘 = 2, 4, 6, … , 25).
# 3. Model training & Validation: train a k-nearest neighbor (method = ”knn”) 
# model using the pre-specified resampling procedure (trControl = cv), 
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


# Feature filtering to speed up training time
# zero and near-zero variables - single unique value with no useful info to a model
# fraction of unique values =<10%
# ratio of freq. of the most prevalent value to the 2nd freq. of the 2nd most prevalent
# value => 20%

caret::nearZeroVar(ames_train, saveMetrics = TRUE) %>% 
  rownames_to_column() %>% 
  filter(nzv)

# Numeric feature engineering
# skewness: when normalizing, use Box-Cox when feature values - strictly positive
# use Yeo-Johnson when feature values - not strictly positive

# normalize all numeric columns
recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_YeoJohnson(all_numeric())

# standardization
# centering and scaling - numeric variables have zero mean and unit variance
# better standardize within recipe blueprint so that both training and test data
# standardization are based on the same mean and variance

ames_recipe %>% 
  step_center(all_numeric(), -all_outcomes()) %>% 
  step_scale(all_numeric(), -all_outcomes())


# categorical feature engineering
# Lumping
# contains levels that have very few observations
count(ames_train, Neighborhood) %>% arrange(n)

count(ames_train, Screen_Porch) %>% arrange(n)

# lump levels for two features
lumping <- recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_other(Neighborhood, threshold = 0.01, other = "other") %>% 
  step_other(Screen_Porch, threshold = 0.01, other = ">0")

apply_2_training <- prep(lumping, training = ames_train) %>% 
  bake(ames_train)

# New distributions
count(apply_2_training, Neighborhood) %>% arrange(n)
count(apply_2_training, Screen_Porch) %>% arrange(n)

# one-hot and dummy encoding
# lump levels for two features, it creates perfect collinearity which bad for prediction
recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_dummy(all_nominal(), one_hot = TRUE)

# label encoding
# Original categories
count(ames_train, MS_SubClass)
# Label encoded
recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_integer(MS_SubClass) %>% 
  prep(ames_train) %>% 
  bake(ames_train) %>% 
  count(MS_SubClass)

# be careful with unordered categorical features
# ordinal encoding is Ames housing

ames_train %>% select(contains("Qual"))

# ordered factors check
count(ames_train, Overall_Qual)

# Label encoded
recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_integer(Overall_Qual) %>% 
  prep(ames_train) %>% 
  bake(ames_train) %>% 
  count(Overall_Qual)

# Dimension reduction
# PCA and retain components explaining,say 95% of the variance
recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_center(all_numeric()) %>% 
  step_scale(all_numeric()) %>% 
  step_pca(all_numeric(), threshold = .95)

# Proper implementation
# potential steps
# 1. Filter out zero or near-zero variance features.
# 2. Perform imputation if required.
# 3. Normalize to resolve numeric feature skewness.
# 4. Standardize (center and scale) numeric features.
# 5. Perform dimension reduction (e.g., PCA) on numeric features.
# 6. One-hot or dummy encode categorical features.


# three main steps in creating and applying feature engineering with recipes:
# 1. recipe: where you define your feature engineering steps to create your blueprint.
# 2. prepare: estimate feature engineering parameters based on training data.
# 3. bake: apply the blueprint to new data.

# the following defines Sale_Price as the target variable and then uses all .
# the remaining columns as features based on ames_train.
# 1. Remove near-zero variance features that are categorical (aka nominal).
# 2. Ordinal encode our quality-based features (which are inherently ordinal).
# 3. Center and scale (i.e., standardize) all numeric features.
# 4. Perform dimension reduction by applying PCA to all numeric features.

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_nzv(all_nominal()) %>% 
  step_integer(matches("Qual|Cond|QC|Qu")) %>% 
  step_center(all_numeric(), -all_outcomes()) %>% 
  step_scale(all_numeric(), -all_outcomes()) %>% 
  step_pca(all_numeric(), -all_outcomes())
blueprint

# train this blueprint on some training data
prepare <- prep(blueprint, training = ames_train)
prepare

# apply the blueprint to new data
baked_train <- bake(prepare, new_data = ames_train)
baked_test <- bake(prepare, new_data = ames_test)
baked_train

# caret package simplifies this process
# 1. Filter out near-zero variance features for categorical features.
# 2. Ordinally encode all quality features, which are on a 1–10 Likert scale.
# 3. Standardize (center and scale) all numeric features.
# 4. One-hot encode our remaining categorical features.

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_nzv(all_nominal()) %>% 
  step_integer(matches("Qual|Cond|QC|Qu")) %>% 
  step_center(all_numeric(), -all_outcomes()) %>% 
  step_scale(all_numeric(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

# resampling method and hyperparameter search grid
# specify resampling plan
cv <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5
)

# construct grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1))

# tune a knn model using grid search
knn_fit2 <- train(
  blueprint,
  data = ames_train,
  method = "knn",
  trControl = cv,
  tuneGrid = hyper_grid,
  metric = "RMSE"
)

# print model results
knn_fit2

# plot cross validation results
ggplot(knn_fit2)

# supeverised learning
# linear relationship between total above ground living space and sale price
model1 <- lm(Sale_Price ~ Gr_Liv_Area, data = ames_train)
summary(model1)

#RMSE
sigma(model1)
# MSE
sigma(model1)^2










