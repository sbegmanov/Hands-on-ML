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
split_strat <- initial_split(churn, prop = 0.7,
                             strata = "Attrition")
train_strat <- training(split_strat)
test_strat <- testing(split_strat)

# consistent response ratio between train and test
table(train_strat$Attrition) %>% 
  prop.table()

table(test_strat$Attrition) %>% 
  prop.table()








