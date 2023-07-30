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
cv_model_pls
















