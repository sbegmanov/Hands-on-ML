# – problem type: supervised binomial classification
# – response variable: Attrition (i.e., “Yes”, “No”)
# – features: 30
# – observations: 1,470
# – objective: use employee attributes to predict if they will attrit (leave the company)
# – access: provided by the rsample package (Kuhn and Wickham, 2019)
# – more details: See ?rsample::attrition

library(modeldata)
library(dplyr)
library(h2o)

attrition <- modeldata::attrition
dim(attrition)
head(attrition$Attrition)

churn <- attrition %>% 
  mutate_if(is.ordered, .funs = factor, ordered = FALSE)

# resampling and model training
h2o.init()
churn.h2o <- as.h2o(churn)

## Stratified sampling which is more common with classification problem
# original response distribution
table(churn$Attrition) %>% 
  prop.table()

# stratified sampling with the rsample package
library(rsample)
set.seed(123)
split_strat <- initial_split(churn, prop = 0.7, strata = "Attrition")
train_strat <- training(split_strat)
test_strat <- testing(split_strat)

# consistent response ratio between train and test
table(train_strat$Attrition) %>% 
  prop.table()

table(test_strat$Attrition) %>% 
  prop.table()


df <- attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE)

# create training(70%) and test(30%) sets
set.seed(123)
churn_split <- initial_split(df, prop = .7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test <- testing(churn_split)

### Logistic Regression
# simple logistic regression

model1 <- glm(Attrition ~ MonthlyIncome, family = "binomial", data = churn_train)
tidy(model1)

model2 <- glm(Attrition ~ OverTime, family = "binomial", data = churn_train)
tidy(model2)

# interpretation of coefficients
exp(coef(model1))
exp(coef(model2))

# CI
confint(model1)
# for odds
exp(confint(model1))

#CI
confint(model2)


# Multiple logistic regression

model3 <- glm(
  Attrition ~ MonthlyIncome + OverTime,
  family = "binomial",
  data = churn_train
)
tidy(model3)

# Assessing model accuracy
library(caret)

set.seed(123)
cv_model1 <- train(
  Attrition ~ MonthlyIncome,
  data = churn_train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model2 <- train(
  Attrition ~ MonthlyIncome + OverTime,
  data = churn_train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model3 <- train(
  Attrition ~ .,
  data = churn_train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

# extract out of sample performance measures
# third model is best
summary(
  resamples(
    list(
      model1 = cv_model1,
      model2 = cv_model2,
      model3 = cv_model3
    )
  )
)$statistics$Accuracy

# predict class
pred_class <- predict(cv_model3, churn_train)

# confusion matrix
confusionMatrix(
  data = relevel(pred_class, ref = "Yes"),
  reference = relevel(churn_train$Attrition, ref = "Yes")
)

# ratio on non-attrition vs attrition
table(churn_train$Attrition) %>% 
  prop.table()

library(ROCR)

# compute predicted probabilities
m1_prob <- predict(cv_model1, churn_train, type = "prob")$Yes
m3_prob <- predict(cv_model3, churn_train, type = "prob")$Yes

# compute AUC metrics for cv_model1 and cv_model3
perf1 <- prediction(m1_prob, churn_train$Attrition) %>% 
  performance(measure = "tpr", x.measure = "fpr")

perf2 <- prediction(m3_prob, churn_train$Attrition) %>% 
  performance(measure = "tpr", x.measure = "fpr")

# plot ROC curves for cv_model1 and cv_model3
plot(perf1, col = "black", lty = 2)
plot(perf2, add = TRUE, col = "blue")
legend(0.8, 0.2, legend = c("cv_model1", "cv_model3"),
       col = c("black", "blue"), lty = 2:1, cex = 0.6)


# perform 10-fold CV on a PLS tuning the number of PCs to
# use as predictors
set.seed(123)
cv_model_pls <- train(
  Attrition ~ .,
  data = churn_train,
  method = "pls",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("zv", "center", "scale"),
  tuneLength = 16
)
# model with lowest RMSE
cv_model_pls$bestTune

# plot cv RMSE
ggplot(cv_model_pls)

# model concerns
# feature interpretation

library(vip)
vip(cv_model3, num_features = 20)


# # partial dependence plots(PDF)
# library(pdp)
# partial(cv_model_pls, "OverTime", grid.resolution = 20, plot = TRUE)
# partial(cv_model_pls, "NumCompaniesWorked", grid.resolution = 20, plot = TRUE)


# attrition data additional 0.8% improvement in accuracy
df <- attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE)

set.seed(123)
churn_split <- initial_split(df, prop = 0.7, strata = "Attrition")
train <- training(churn_split)
test <- testing(churn_split)

# train logistic regression model
set.seed(123)
glm_mod <- train(
  Attrition ~ .,
  data = train,
  method = "glm",
  family = "binomial",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10)
)

# train regulized logistic regression model
set.seed(123)
penalized_mod <- train(
  Attrition ~ .,
  data = train,
  method = "glmnet",
  family = "binomial",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# extract out of sample performance measures
summary(resamples(list(
  logistic_model = glm_mod,
  penalized_model = penalized_mod
)))$statistics$Accuracy


## MARS model
df <- attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE)

set.seed(123)
churn_split <- initial_split(df, prop = 0.7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test <- testing(churn_split)


# create a tuning grid
hyper_grid <- expand.grid(
  degree = 1:3,
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

set.seed(123)
# cross validated model
tuned_mars <- train(
  x = subset(churn_train, select = -Attrition),
  y = churn_train$Attrition,
  method = "earth",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = hyper_grid
)

# best model
tuned_mars$bestTune

# plot results
ggplot(tuned_mars)


### KKN algorithm

attrit <- attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE)


set.seed(123)
churn_split <- initial_split(attrit, prop = 0.7, strata = "Attrition")

churn_train <- training(churn_split)

# MNIST training data
mnist <- dslabs::read_mnist()
names(mnist)

# simple example
two_houses <- ames_train %>% 
  select(Gr_Liv_Area, Year_Built) %>% 
  sample_n(2)
print(two_houses)

# euclidean distance
dist(two_houses, method = "euclidean")

# Manhattan distance
dist(two_houses, method = "manhattan")






