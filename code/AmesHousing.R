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

### supervised learning
# linear relationship between total above ground living space and sale price
# OLS regression model
model1 <- lm(Sale_Price ~ Gr_Liv_Area, data = ames_train)
summary(model1)

# coefficients to extract
coef(model1)
#RMSE = Residual standard error in summary()
sigma(model1)
# MSE
sigma(model1)^2

# CI for each coefficient
confint(model1, level = 0.95)

# Multiple linear regression
model2 <- lm(Sale_Price ~ Gr_Liv_Area + Year_Built,
             data = ames_train)
# + or - used to add or remove terms from the original model
model2 <- update(model1, . ~ . + Year_Built)
summary(model2)

# library(lm.beta)
# lm.beta(model2)
beta_values <- coef(model2)
beta_values[2] # beta_1
beta_values[3] # beta_2

# : operator to include an interaction (* could be used )
lm(Sale_Price ~ Gr_Liv_Area + Year_Built + Gr_Liv_Area:Year_Built, 
   data = ames_train)

# include all possible main effects
model3 <- lm(Sale_Price ~ ., data = ames_train)
# print estimated coefficients in a tidy df
broom::tidy(model3)

# assessing model accuracy
# train model using 10-fold cross-validation
set.seed(123)
cv_model1 <- train(
  form = Sale_Price ~ Gr_Liv_Area,
  data = ames_train,
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)
print(cv_model1)

# model 2 CV
set.seed(123)
cv_model2 <- train(
  Sale_Price ~ Gr_Liv_Area + Year_Built,
  data = ames_train,
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)
print(cv_model2)

### model 3 CV
set.seed(123)
cv_model3 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

# Extract out of sample performance measures
summary(resamples(list(
  model1 = cv_model1,
  model2 = cv_model2,
  model3 = cv_model3
)))

### model concerns
# linear relationship

p1 <- ggplot(ames_train, aes(Year_Built, Sale_Price)) +
  geom_point(size = 1, alpha = .4) +
  geom_smooth(se = FALSE) +
  scale_y_continuous("Sale price", labels = scales::dollar) +
  xlab("Year built") +
  ggtitle(paste("Non-transformed variables with a\n",
                "non-linear relationship."))
p2 <- ggplot(ames_train, aes(Year_Built, Sale_Price)) +
  geom_point(size = 1, alpha = .4) +
  geom_smooth(method = "lm", se = FALSE) +
  scale_y_log10("Sale price", labels = scales::dollar,
                breaks = seq(0, 400000, by = 100000)) +
  xlab("Year built") +
  ggtitle(paste("Transforming variables can provide a\n",
                "near-linear relationship"))
gridExtra::grid.arrange(p1, p2, nrow = 1)

# Constant variance among residuals
# variance among error terms are constant(homoscedasticity)
df1 <- broom::augment(cv_model1$finalModel, data = ames_train)

p1 <- ggplot(df1, aes(.fitted, .std.resid)) +
  geom_point(size = 1, alpha = .4) +
  xlab("Predicted values") +
  ylab("Residuals") +
  ggtitle("Model 1", subtitle = "Sale_Price ~ Gr_Liv_Area")

df2 <- broom::augment(cv_model3$finalModel, data = ames_train)

p2 <- ggplot(df2, aes(.fitted, .std.resid)) +
  geom_point(size = 1, alpha = .4) +
  xlab("Predicted values") +
  ylab("Residuals") +
  ggtitle("Model 3", subtitle = "Sale_Price ~ .")

gridExtra::grid.arrange(p1, p2, nrow = 1)

# No autocorrelation
# the errors are independent and uncorrelated

df1 <- mutate(df1, id = row_number())
df2 <- mutate(df2, id = row_number())

p1 <- ggplot(df1, aes(id, .std.resid)) +
  geom_point(size = 1, alpha = .4) +
  xlab("Row ID") +
  ylab("Residuals") +
  ggtitle("Model 1", subtitle = "Correlated residuals.")

p2 <- ggplot(df2, aes(id, .std.resid)) +
  geom_point(size = 1, alpha = .4) +
  xlab("Row ID") +
  ylab("Residuals") +
  ggtitle("Model 3", subtitle = "Uncorrelated residuals.")


gridExtra::grid.arrange(p1, p2, nrow = 1)

# More observations than predictors
# the number of features exceeds the number of observations (p > n)
# one can remove variables one-at-a-time until p < n

# No or little multicollinearity
# collinearity can cause predictor variables to appear as
# statistically insignificant when in fact they are significant.

# fit with two strongly correlated variables
summary(cv_model3) %>% 
  broom::tidy() %>% 
  filter(term %in% c("Garage_Area", "Garage_Cars"))

# model without Garage_Area
set.seed(123)
mod_wo_Garage_Cars <- train(
  Sale_Price ~ .,
  data = select(ames_train, -Garage_Cars),
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

summary(mod_wo_Garage_Cars) %>% 
  broom::tidy() %>% 
  filter(term == "Garage_Area")

### Principal component regression
# represents correlated variables with a
# smaller number of uncorrelated features(principle components)

# perform 10-fold cv on a PCR model tuning the 
# number of principal components to use as predictors from 1-20
set.seed(123)
cv_model_pcr <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "pcr",
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("zv", "center", "scale"),
  tuneLength = 20
)

# model with lowest RMSE
cv_model_pcr$bestTune


# plot cv RMSE
ggplot(cv_model_pcr)


# alternative way to redude the impact of multicollinearity is
# partial least squres(PLS)
# PLS finds components that simultaneously summarize variation of the predictors
# while being optimally correlated with the outcome and then uses those PCs 
# as predictors.

# perform 10-fold cross validation on a PLS model tuning the
# number of principal components to use as predictors from 1-20
set.seed(123)
cv_model_pls <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "pls",
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("zv", "center", "scale"),
  tuneLength = 20
)

# model with lowest RMSE
cv_model_pls$bestTune

# plot cv RMSE
ggplot(cv_model_pls)

### Feature interpretation
library(vip)
vip(cv_model_pls, number_features = 20, method = "model")


# partial dependence plots(PDF)
library(pdp)
partial(cv_model_pls, "Gr_Liv_Area", grid.resolution = 20, plot = TRUE)


# ridge penalty - good for handling correlated features, retains all features
# lasso penalty - perform feature selection, when two strongly correlated features
#are pushed towards zero
#elastic net - enables effective regularization via the ridge penalty with
#feature selection characteristics of the lasso


X <- model.matrix(Sale_Price ~ ., ames_train)[, -1]
Y <- log(ames_train$Sale_Price)

# regularized model
library(glmnet)

# ridge regression to attrition data
ridge <- glmnet(
  x = X,
  y = Y,
  alpha = 0
)
plot(ridge, xvar = "lambda")
coef(ridge)
ridge$lambda

# lambdas applied to penalty parameter
ridge$lambda %>% 
  head()


colnames(coef(ridge))

# small lambda results in large coefficients
coef(ridge)[c("Latitude", "Overall_QualVery_Excellent"), 100]

# large lambda results in small coefficients
coef(ridge)[c("Latitude", "Overall_QualVery_Excellent"), 1]

# Tuning
# alpha is a tuning parameter to control the model
# k-fold CV optimizes alpha value

# CV ridge regression
ridge <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 0
)

# CV lasso regression
lasso <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 1
)

# plot results
par(mfrow = c(1, 2))
plot(ridge, main = "Ridge penalty\n\n")
plot(lasso, main = "Lasso penalty\n\n")

# Ridge model
# min MSE
min(ridge$cvm)
# lambda for this min MSE
ridge$lambda.min

# 1-SE rule
ridge$cvm[ridge$lambda == ridge$lambda.1se]
#lambda for this MSE
ridge$lambda.1se

# Lasso model
# min MSE
min(lasso$cvm)
# lambda for this MSE
lasso$lambda.min

# 1-SE rule
lasso$cvm[lasso$lambda == lasso$lambda.1se]
# lambda for this MSE
lasso$lambda.1se


# Ridge model
ridge_min <- glmnet(
  x = X,
  y = Y,
  alpha = 0
)

# Lasso model
lasso_min <- glmnet(
  x = X,
  y = Y,
  alpha = 1
)

par(mfrow = c(1, 2))
# plot ridge model
plot(ridge_min, xvar = "lambda", main = "Ridge pentaly\n\n")
abline(v = log(ridge$lambda.min), col = "red", lty = "dashed")
abline(v = log(ridge$lambda.1se), col = "blue", lty = "dashed")

# plot lasso model
plot(lasso_min, xvar = "lambda", main = "Lasso pentaly\n\n")
abline(v = log(lasso$lambda.min), col = "red", lty = "dashed")
abline(v = log(lasso$lambda.1se), col = "blue", lty = "dashed")


# Any alpha value between 0–1 will perform an elastic net. 
# When alpha = 0.5 we perform an equal combination of penalties 
# whereas alpha < 0.5 will have a heavier ridge penalty applied and
# alpha > 0.5 will have a heavier lasso penalty.

set.seed(123)
# grid search across
cv_glmnet <- train(
  x = X,
  y = Y,
  method = "glmnet",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# model with lowest RMSE
cv_glmnet$bestTune

# plot CV RMSE
ggplot(cv_glmnet)


# predict sales price on training data
pred <- predict(cv_glmnet, X)

# compute RMSE of transformed predicted
RMSE(exp(pred), exp(Y))

# feature interpretation
library(vip)
vip(cv_glmnet, num_features = 20, bar = FALSE)

# Enhanced Adaptive Regression Through Hinges(earth)
library(earth)

# fit a basic MARS model
# Generalized CV (GCV)
mars1 <- earth(
  Sale_Price ~ .,
  data = ames_train
)

print(mars1)

summary(mars1) %>% 
  .$coefficients %>% 
  head(10)

plot(mars1)


# fit a MARS model
# assess potential interactions

mars2 <- earth(
  Sale_Price ~ .,
  data = ames_train,
  degree = 2
)

# check out the first 10 coefficients terms
summary(mars2) %>% 
  .$coefficients %>% 
  head(10)

# create a tuning grid
hyper_grid <- expand.grid(
  degree = 1:3,
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

head(hyper_grid)
# cross-validated model
set.seed(123)

cv_mars <- train(
  x = subset(ames_train, select = - Sale_Price),
  y = ames_train$Sale_Price,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = hyper_grid
)

# View results
cv_mars$bestTune

ggplot(cv_mars)

# feature interpretation
# variable importance plots
library(vip)

# Generalized CV
p1 <- vip(cv_mars, num_features = 40, bar = FALSE, value = "gcv") +
  ggtitle("GCV")

# sums of squres(RSS)
p2 <- vip(cv_mars, num_features = 40, bar = FALSE, value = "rss") +
  ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)


# extract coefficients, convert to tidy data frame
# filter for interactions terms
cv_mars$finalModel %>% 
  coef() %>% 
  broom::tidy() %>% 
  filter(stringr::str_detect(names, "\\*"))

# construct partial dependence plots
library(pdp)
p1 <- partial(cv_mars, pred.var = "Gr_Liv_Area") %>% 
  autoplot()
p2 <- partial(cv_mars, pred.var = "Year_Built") %>% 
  autoplot()
p3 <- partial(cv_mars, pred.var = c("Gr_Liv_Area", "Year_Built"),
              chull = TRUE) %>% 
  plotPartial(palette = "inferno", contour = TRUE) %>% 
  ggplotify::as.grob() # convert to grob to plot with cowplot

# display plots in a grid
top_row <- cowplot::plot_grid(p1, p2)
cowplot::plot_grid(top_row, p3, nrow = 2, rel_heights = c(1, 2))


### Decision trees
library(rpart)
library(rpart.plot)

ames_dt1 <- rpart(
  formula = Sale_Price ~ .,
  data = ames_train,
  method = "anova"
)

rpart.plot(ames_dt1)
plotcp(ames_dt1)

ames_dt2 <- rpart(
  formula = Sale_Price ~ .,
  data = ames_train,
  method = "anova",
  control = list(cp = 0, xval = 10)
  
)

plotcp(ames_dt2)
abline(v = 11, lty = "dashed")


# rpart cross validation results
ames_dt1$cptable

# caret cross validation results
library(caret)

ames_dt3 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 20
)

ggplot(ames_dt3)

# feature interpretation
library(vip)
vip(ames_dt3, num_features = 40, bar = FALSE)

# construct partial dependence plots
library(pdp)

p1 <- partial(ames_dt3, pred.var = "Gr_Liv_Area") %>% 
  autoplot()
p2 <- partial(ames_dt3, pred.var = "Year_Built") %>% 
  autoplot()
p3 <- partial(ames_dt3, pred.var = c("Gr_Liv_Area", "Year_Built")) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE,
              colorkey = TRUE, screen = list(z = -20, x = -60))
# display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)


### bagging
library(ipred)
set.seed(123)

# train bagged model
ames_bag1 <- bagging(
  formula = Sale_Price ~ .,
  data = ames_train,
  nbagg = 100,
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

ames_bag1

# with caret
library(caret)

ames_bag2 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 10),
  nbagg = 200,
  control = rpart.control(minsplit = 2, cp = 0)
)

ames_bag2

# parallelize
# create a parallel socket cluster
library(parallel)
library(doParallel)
require(iterators)

cl <- makeCluster(8)
registerDoParallel(cl)

# fit trees in parallel and compute predictions on the test set
predictions <- foreach(
  icount(160),
  .packages = "rpart",
  .combine = cbind
) %dopar% {
  # bootstrap copy of training data
  index <- sample(nrow(ames_train), replace = TRUE)
  ames_train_boot <- ames_train[index, ]
  
  # fit tree to bootstrap copy
  bagged_tree <- rpart(
    Sale_Price ~ .,
    control = rpart.control(minsplit = 2, cp = 0),
    data = ames_train_boot
  )
  predict(bagged_tree, newdata = ames_test)
}

predictions[1:5, 1:7]

# plot
predictions %>% 
  as.data.frame() %>% 
  mutate(
    observation = 1:n(),
    actual = ames_test$Sale_Price
  ) %>% 
  tidyr::gather(tree, predicted, -c(observation, actual)) %>% 
  group_by(observation) %>% 
  mutate(tree = stringr::str_extract(tree, '\\d+') %>% 
  as.numeric()) %>% 
  ungroup() %>% 
  arrange(observation, tree) %>% 
  group_by(observation) %>% 
  mutate(avg_prediction = cummean(predicted)) %>% 
  group_by(tree) %>% 
  summarize(RMSE = RMSE(avg_prediction, actual)) %>% 
  ggplot(aes(tree, RMSE)) +
  geom_line() +
  xlab('Number of trees')

# shutdown parallel cluster
stopCluster(cl)

vip::vip(ames_bag2, num_features = 40, bar = FALSE)

### random forest
library(ranger)

# number of features
n_features <- length(setdiff(names(ames_train), "Sale_Price"))

# train a default random forest model
ames_rf1 <- ranger(
  Sale_Price ~ .,
  data = ames_train,
  mtry = floor(n_features / 3),
  respect.unordered.factors = "order",
  seed = 123
)

# get OOB RMSE
(default_rmse <- sqrt(ames_rf1$prediction.error))

# hyperparameter grid

hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10),
  replace = c(TRUE, FALSE),
  sample.fraction = c(.5, .63, .8),
  rmse = NA
)

# execute full Cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # fit model for i'th hyperparameter combination
  fit <- ranger(
    formula = Sale_Price ~ .,
    data = ames_train,
    num.trees = n_features * 10,
    mtry = hyper_grid$mtry[i],
    min.node.size = hyper_grid$min.node.size[i],
    replace = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose = FALSE,
    seed = 123,
    respect.unordered.factors = 'order'
  )
  # export OOB error
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

# top 10 models
hyper_grid %>% 
  arrange(rmse) %>% 
  mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>% 
  head(10)

h2o.no_progress()
h2o.init(max_mem_size = "5g")


# convert training data to h20 object
train_h2o <- as.h2o(ames_train)

# set the response column to Sale_Price
response <- "Sale_Price"

# set the predictor names
predictors <- setdiff(colnames(ames_train), response)

# similar to baseline ranger
h2o_rf1 <- h2o.randomForest(
  x = predictors,
  y = response,
  training_frame = train_h2o,
  ntrees = n_features * 10,
  seed = 123
)

h2o_rf1


# 240 hyperparameter combinations
# hyperparameter grid
hyper_grid <- list(
  mtries = floor(n_features * c(.05, .15, .25, .3333, .4)),
  min_rows = c(1, 3, 5, 10),
  max_depth = c(10, 20, 30),
  sample_rate = c(.55, .632, .70, .80)
)

# random grid search strategy
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.001,  # stop if improvement is < 0.1%
  stopping_rounds = 10,       # over the last 10 models
  max_runtime_secs = 60 * 5   # or stop search after 5 min
)

# perform grid search
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_random_grid",
  x = predictors,
  y = response,
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  ntrees = n_features * 10,
  seed = 123,
  stopping_metric = "RMSE",
  stopping_rounds = 10,           # stop if last 100 trees added
  stopping_tolerance = 0.005,     # don't improve RMSE by 0.5%
  search_criteria = search_criteria
)


# collect the results and start by our model performance metric
# of choice

random_grid_perf <- h2o.getGrid(
  grid_id = "rf_random_grid",
  sort_by = "mse",
  decreasing = FALSE
)

random_grid_perf


# after identification optimal parameter values from the grid search
rf_impurity <- ranger(
  formula = Sale_Price ~ .,
  data = ames_train,
  num.trees = 2000,
  mtry = 32,
  min.node.size = 1,
  sample.fraction = .80,
  replace = FALSE,
  importance = "impurity",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed = 123
)

# re-run model with permutation-based variable importance
rf_permutation <- ranger(
  formula = Sale_Price ~ .,
  data = ames_train,
  num.trees = 2000,
  mtry = 32,
  min.node.size = 1,
  sample.fraction = .80,
  replace = FALSE,
  importance = "permutation",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed = 123
)

p1 <- vip::vip(rf_impurity, num_features = 25, bar = FALSE)
p2 <- vip::vip(rf_permutation, num_features = 25, bar = FALSE)

gridExtra::grid.arrange(p1, p2, nrow = 1)

### GBM model
# run a basic GBM model
library(gbm)

set.seed(123)
ames_gbm1 <- gbm(
  formula = Sale_Price ~ .,
  data = ames_train,
  distribution = "gaussian", # SSE loss function
  n.trees = 5000,
  shrinkage = 0.1, 
  interaction.depth = 3,
  n.minobsinnode = 10,
  cv.folds = 10
)

# find index for number trees with minimum CV error
best <- which.min(ames_gbm1$cv.error)

# get MSE and compute RMSE
sqrt(ames_gbm1$cv.error[best])

# plot error plot curve
gbm.perf(ames_gbm1, method = "cv")

# create grid search
hyper_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  RMSE = NA,
  trees = NA,
  time = NA
)

# execute grid search
for(i in seq_len(nrow(hyper_grid))){
  
  # fit gbm
  set.seed(123)
  train_time <- system.time({
    m <- gbm(
      formula = Sale_Price ~ .,
      data = ames_train,
      distribution = "gaussian",
      n.trees = 5000,
      shrinkage = hyper_grid$learning_rate[i],
      interaction.depth = 3,
      n.minobsinnode = 10,
      cv.folds = 10
    )
  })
  # add SSE, trees, and training time to results
  hyper_grid$RMSE[i] <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$time[i] <- train_time[["elapsed"]]
}

# results
arrange(hyper_grid, RMSE)

# from the results optimal learning rate is 0.05

# search grid
hyper_grid <- expand.grid(
  n.trees = 4000,
  shrinkage = 0.05,
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15)
)

# create model fit function
model_fit <- function(n.trees, shrinkage, interaction.depth,
                      n.minobsinnode) {
  set.seed(123)
  n <- gbm(
    formula = Sale_Price ~ .,
    data = ames_train,
    distribution = "gaussian",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    n.minobsinnode = n.minobsinnode,
    cv.folds = 10
  )
  # RMSE
  sqrt(min(m$cv.error))
}

# perform search gird with functional programming
hyper_grid$rmse <- purrr::pmap_dbl(
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,
    shrinkage = ..2,
    interaction.depth = ..3,
    n.minobsinnode = ..4
  )
)

# results, no improvements
arrange(hyper_grid, rmse)

# stochastic GBMs
# use optimal hyperparameters from the previous 

# refined hyperparameter grid
# sample_rate: row subsampling
# col_sample_rate: col subsampling for each split
# col_sample_rate_per_tree: col subsampling for each tree


hyper_grid <- list(
  sample_rate = c(0.5, 0.75, 1),
  col_sample_rate = c(0.5, 0.75, 1),
  col_sample_rate_per_tree = c(0.5, 0.75, 1)
)

# random grid search strategy
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_tolerance = 0.001,
  stopping_metric = "mse",
  stopping_rounds = 10,
  max_runtime_secs = 60*60
)

# perform grid search
grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid",
  x = predictors,
  y = response,
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  ntrees = 5000,
  learn_rate = 0.05,
  max_depth = 5,
  min_rows = 5,
  nfolds = 10,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  search_criteria = search_criteria,
  seed = 123
  
)


# collect the results and sort by model performance
# metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "gbm_grid",
  sort_by = "mse",
  decreasing = FALSE
)

grid_perf


# get model_id for the top model, chosen by cross validation error
best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# performance metrics on the best model
h2o.performance(model = best_model, xval = TRUE)


# XGBoost

library(recipes)

xgb_prep <- recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_integer(all_nominal()) %>% 
  prep(training = ames_train, retrain = TRUE) %>% 
  juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "Sale_Price")])
Y <- xgb_prep$Sale_Price

set.seed(123)
ames_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 6000,
  objective = "reg:linear",
  early_stopping_rounds = 50,
  nfold = 10,
  params = list(
    eta = 0.01,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.5,
    colsample_bytree = 0.5
  ),
  verbose = 0
)

# min test CV RMSE
min(ames_xgb$evaluation_log$test_rmse_mean)

# hyperparameter grid
hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = 3,
  min_child_weight  = 3,
  subsample = 0.5,
  colsample_bytree = 0.5,
  gamma = c(0, 1, 10, 100, 1000),
  lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  rmse = 0, # a place to dump RMSE results,
  trees = 0 # a place to dump required number of trees
)

# grid search
for(i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  m <- xgb.cv(
    data = X,
    label = Y,
    nrounds = 4000,
    objective = "reg:linear",
    early_stopping_rounds = 50,
    nfold = 10,
    verbose = 0,
    params = list(
      eta = hyper_grid$eta[i],
      max_depth = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i],
      gamma = hyper_grid$gamma[i],
      lambda = hyper_grid$lambda[i],
      alpha = hyper_grid$alpha[i]
    )
  )
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration
}

# results
hyper_grid %>% 
  filter(rmse > 0) %>% 
  arrange(rmse) %>% 
  glimpse()


# optimal parameter list
params <- list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5
)

# train final model model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 3944,
  objective = "reg:linear",
  verbose = 0
)

### stacked models

library(rsample)

set.seed(123)
split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

# consistent categorical levels
library(recipes)

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_other(all_nominal(), threshold = 0.005)

# training and test for h2o
library(h2o)

h2o.init()

train_h2o <- prep(blueprint, training = ames_train, retain = TRUE) %>% 
  juice() %>% 
  as.h2o()
test_h2o <- prep(blueprint, training = ames_train) %>% 
  bake(new_data = ames_test) %>% 
  as.h2o()

# response and feature
Y <- "Sale_Price"
X <- setdiff(names(ames_train), Y)


# train and cross validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, 
  nfolds = 10,
  fold_assignment = "Modulo", # same observations
  keep_cross_validation_predictions = TRUE,
  seed = 123
)

# train and cross validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 1000,
  mtries = 20, max_depth = 30, min_rows = 1, sample_rate = 0.8,
  nfolds = 10, 
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, 
  seed = 123,
  stopping_rounds = 50, 
  stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# train and cross validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 5000,
  learn_rate = 0.01, max_depth = 7, min_rows = 5, sample_rate = 0.8,
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# train and cross validate an XGBoost model
best_xgb <- h2o.xgboost(
  x = X, y = Y, training_frame = train_h2o, ntrees = 5000,
  learn_rate = 0.05, max_depth = 3, min_rows = 3, sample_rate = 0.8,
  categorical_encoding = "Enum",
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# train a stacked ensemble
ensemble_tree <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o,
  model_id = "my_tree_ensemble",
  base_models = list(best_glm, best_rf, best_gbm, best_xgb),
  metalearner_algorithm = "drf"
)

# results from base learners
get_rmse <- function(model) {
  results <- h2o.performance(model, newdata = test_h2o)
  results@metrics$RMSE
}

list(best_glm, best_rf, best_gbm, best_xgb) %>% 
  purrr::map_dbl(get_rmse)


# stacked results
h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$RMSE


# all base learners show a high correlation
# stacking in this case provides less advantage

glm_id <- best_glm@model$cross_validation_holdout_predictions_frame_id
rf_id <- best_rf@model$cross_validation_holdout_predictions_frame_id
gbm_id <- best_gbm@model$cross_validation_holdout_predictions_frame_id
xgb_id <- best_xgb@model$cross_validation_holdout_predictions_frame_id
data.frame(
  GLM_pred = as.vector(h2o.getFrame(glm_id$name)),
  RF_pred = as.vector(h2o.getFrame(rf_id$name)),
  GBM_pred = as.vector(h2o.getFrame(gbm_id$name)),
  XGB_pred = as.vector(h2o.getFrame(xgb_id$name))
) %>% cor()


# stacking a grid search
# GBM hyperparameter grid

hyper_grid <- list(
  max_depth = c(1, 3, 5),
  min_rows = c(1, 5, 10),
  learn_rate = c(0.01, 0.05, 0.1),
  learn_rate_annealing = c(0.99, 1),
  sample_rate = c(0.5, 0.75, 1),
  col_sample_rate = c(0.8, 0.9, 1)
)

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  max_models = 25
)

# random grid search
random_grid <- h2o.grid(
  algorithm = "gbm", grid_id = "gbm_grid", x = X, y = Y,
  training_frame = train_h2o, hyper_params = hyper_grid,
  search_criteria = search_criteria, ntrees = 5000,
  stopping_metric = "RMSE", stopping_rounds = 10,
  stopping_tolerance = 0, nfolds = 10, fold_assignment = "Modul",
  keep_cross_validation_predictions = TRUE, seed = 123
)

# results by RMSE
h2o.getGrid(
  grid_id = "gbm_grid",
  sort_by = "rmse"
)

# Grab the model_id for the top model, chosen by validation error
best_model_id <- random_grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
h2o.performance(best_model, newdata = test_h2o)

# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(x = X, y = Y,
                                training_frame = train_h2o, 
                                model_id = "ensemble_gbm_grid",
                                base_models = random_grid@model_ids, 
                                metalearner_algorithm = "gbm"
)

# Evaluate ensemble performance on a test set
h2o.performance(ensemble, newdata = test_h2o)

# AutoML to find a list of candidate 80 models
auto_ml <- h2o.automl(x = X, y = Y,
                      training_frame = train_h2o, nfolds = 5,
                      max_runtime_secs = 60 * 120, max_models = 50,
                      keep_cross_validation_predictions = TRUE, 
                      sort_metric = "RMSE",
                      stopping_rounds = 50, 
                      stopping_metric = "RMSE",
                      stopping_tolerance = 0,
                      seed = 123
)

# results of top 15 models
auto_ml@leaderboard %>%
  as.data.frame() %>%
  dplyr::select(model_id, rmse) %>%
  dplyr::slice(1:25)

### interpretable machine learning (IML)
# local interpretation

# predictions
predictions <- predict(ensemble_tree, train_h2o) %>% as.vector()

# highest and lowest predicted sales price
paste("Observation", which.max(predictions),
      "has a predicted sale price of", scales::dollar(max(predictions)))
#[1] ”Observation 1825 has a predicted sale price of $663,136”

paste("Observation", which.min(predictions),
       "has a predicted sale price of", scales::dollar(min(predictions)))
#[1] ”Observation 139 has a predicted sale price of $47,245.45”

# Grab feature values for observations with min/max predicted sales price

high_ob <- as.data.frame(train_h2o)[which.max(predictions), ] %>%
  select(-Sale_Price)
low_ob <- as.data.frame(train_h2o)[which.min(predictions), ] %>%
  select(-Sale_Price)

# model-specific vs. model-agnostic

# for example, ML algorithms have no natural way of measuring feature importance
vip(ensemble_tree, method = "model")


# modelling agnostic procedures

# 1) data frame with features
features <- as.data.frame(train_h2o) %>% select(-Sale_Price)

# 2) a vector with the actual responses
response <- as.data.frame(train_h2o) %>% pull(Sale_Price)

# 3) custom predict function that returns the predicted values as a vector
pred <- function(object, newdata) {
  results <- as.vector(h2o.predict(object, as.h2o(newdata)))
  return(results)
}

# prediction output
pred(ensemble_tree, features) %>% head()

library(iml)
library(DALEX)

# iml model agnostic object
components_iml <- Predictor$new(
  model = ensemble_tree,
  data = features,
  y = response,
  predict.fun = pred
)

# DALEX model agnostic object
components_dalex <- DALEX::explain(
  model = ensemble_tree,
  data = features,
  y = response,
  predict_function = pred
)

### permutation-based feature importance
library(vip)

vip(
  ensemble_tree,
  train = as.data.frame(train_h2o),
  method = "permute",
  target = "Sale_Price",
  metric = "RMSE",
  nsim = 5,
  sample_frac = 0.5,
  pred_wrapper = pred
)


## partial dependence
library(pdp)

# Custom prediction function wrapper
pdp_pred <- function(object, newdata) {
  results <- mean(as.vector(h2o.predict(object, as.h2o(newdata))))
  return(results)
}

# Compute partial dependence values
pd_values <- partial(
  ensemble_tree,
  train = as.data.frame(train_h2o),
  pred.var = "Gr_Liv_Area",
  pred.fun = pdp_pred,
  grid.resolution = 20
)

head(pd_values)

# Partial dependence plot
autoplot(pd_values, rug = TRUE, train = as.data.frame(train_h2o))

## Individual conditional expectation

# Construct c-ICE curves
partial(
  ensemble_tree,
  train = as.data.frame(train_h2o),
  pred.var = "Gr_Liv_Area",
  pred.fun = pred,
  grid.resolution = 20,
  plot = TRUE,
  center = TRUE,
  plot.engine = "ggplot2"
)

# feature interactions
library(iml)

interact <- Interaction$new(components_iml)

interact$results %>%
  arrange(desc(.interaction)) %>%
  head()

plot(interact)

# feature of interest
feat <- "First_Flr_SF"
interact_2way <- Interaction$new(components_iml, feature = feat)
interact_2way$results %>%
  arrange(desc(.interaction)) %>%
  top_n(10)

# Two-way PDP using iml
interaction_pdp <- Partial$new(
  components_iml,
  c("First_Flr_SF", "Overall_Qual"),
  ice = FALSE,
  grid.size = 20
)
plot(interaction_pdp)


## Local interpretable model-agnostic explanations(LIME)
library(lime)

# explainer object
components_lime <- lime(
  x = features,
  model = ensemble_tree,
  n_bins = 10
)

class(components_lime)
summary(components_lime)

# Use LIME to explain previously defined instances: high_ob & low_ob

lime_explanation <- lime::explain(
  x = rbind(high_ob, low_ob),
  explainer = components_lime,
  n_permutations = 5000,
  dist_fun = "gower",
  kernel_width = 0.25,
  n_features = 10,
  feature_select = "highest_weights"
)

glimpse(lime_explanation)
plot_features(lime_explanation, ncol = 1)

# Tune the LIME algorithm a bit
lime_explanation2 <- explain(
  x = rbind(high_ob, low_ob),
  explainer = components_lime,
  n_permutations = 5000,
  dist_fun = "euclidean",
  kernel_width = 0.75,
  n_features = 10,
  feature_select = "lasso_path"
)

# results
plot_features(lime_explanation2, ncol = 1)

## Shapley values
library(iml)

# Compute (approximate) Shapley values
(shapley <- Shapley$new(components_iml, x.interest = high_ob,
                        sample.size = 1000))

# Plot results
plot(shapley)


# Reuse existing object
shapley$explain(x.interest = low_ob)

# Plot results
shapley$results %>%
  top_n(25, wt = abs(phi)) %>%
  ggplot(aes(phi, reorder(feature.value, phi), color = phi > 0)) +
  geom_point(show.legend = FALSE)


# XGBoost and built-in Shapley values

# Compute tree SHAP for a previously obtained XGBoost model
X <- readr::read_rds("data/xgb-features.rds")
xgb.fit.final <- readr::read_rds("data/xgb-fit-final.rds")


# Try to re-scale features (low to high)
feature_values <- X %>%
  as.data.frame() %>%
  mutate_all(scale) %>%
  gather(feature, feature_value) %>%
  pull(feature_value)

# Compute SHAP values, wrangle a bit, compute SHAP-based
# importance, etc.
shap_df <- xgb.fit.final %>%
  predict(newdata = X, predcontrib = TRUE) %>%
  as.data.frame() %>%
  select(-BIAS) %>%
  gather(feature, shap_value) %>%
  mutate(feature_value = feature_values) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)))

# SHAP contribution plot
p1 <- ggplot(shap_df,
             aes(x = shap_value,
                 y = reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX = FALSE, varwidth = TRUE,
                               size = 0.4, alpha = 0.25) +
  xlab("SHAP value") +
  ylab(NULL)

# SHAP importance plot
p2 <- shap_df %>%
  select(feature, shap_importance) %>%
  filter(row_number() == 1) %>%
  ggplot(aes(x = reorder(feature, shap_importance),
             y = shap_importance)) +
  geom_col() +
  coord_flip() +
  xlab(NULL) +
  ylab("mean(|SHAP value|)")

# Combine plots
gridExtra::grid.arrange(p1, p2, nrow = 1)

shap_df %>%
  filter(feature %in% c("Overall_Qual", "Gr_Liv_Area")) %>%
  ggplot(aes(x = feature_value, y = shap_value)) +
  geom_point(aes(color = shap_value)) +
  scale_colour_viridis_c(name = "Feature value\n(standardized)",
                         option = "C") +
  facet_wrap(~ feature, scales = "free") +
  scale_y_continuous('Shapley value', labels = scales::comma) +
  xlab('Normalized feature value')

# Localized step-wise procedure
# Break Down method, alternative is step down
library(DALEX)

high_breakdown <- prediction_breakdown(components_dalex,
                                       observation = high_ob)
# class of prediction_breakdown output
class(high_breakdown)

# check out the top 10 influential variables for this observation
high_breakdown[1:10, 1:5]

plot(high_breakdown)

### k-means clustering with mixed data

# Full ames data set --> recode ordinal variables to numeric
ames_full <- AmesHousing::make_ames() %>% 
  mutate_if(str_detect(names(.), 'Qual|Cond|QC|Qu'), as.numeric)

# One-hot encode --> retain only the features and not sale price
full_rank <- caret::dummyVars(Sale_Price ~ ., data = ames_full,
                              fullRank = TRUE)
ames_1hot <- predict(full_rank, ames_full)

# scale data
ames_1hot_scaled <- scale(ames_1hot)

# new dimensions
dim(ames_1hot_scaled)

# k-means clustering
set.seed(123)

fviz_nbclust(
  ames_1hot_scaled,
  kmeans,
  method = "wss", #elbow method
  k.max = 25,
  verbose = FALSE
)

# Gower distance
library(cluster)

# original data minus Sale_Price
ames_full <- AmesHousing::make_ames() %>% select(-Sale_Price)
gower_dst <- daisy(ames_full, metric = "gower")

# Gower distance matrix to several clustering algos
pam_gower <- pam(x = gower_dst, k = 8, diss = TRUE) 
diana_gower <- diana(x = gower_dst, diss = TRUE)
agnes_gower <- agnes(x = gower_dst, diss = TRUE)

# partitioning around medians(PAM)
fviz_nbclust(
  ames_1hot_scaled,
  pam,
  method = "wss", #elbow method
  k.max = 25,
  verbose = FALSE
)

# for larger data set use clustering large applications(CLARA)

























