# – Problem type: supervised multinomial classification
# – response variable: V785 (i.e., numbers to predict: 0, 1, …, 9)
# – features: 784
# – observations: 60,000 (train) / 10,000 (test)
# – objective: use attributes about the “darkness” of each of the 784
#   pixels in images of handwritten numbers to predict if the number is 0, 1, …, or 9.
# – access: provided by the dslabs package (Irizarry, 2018)
# – more details: See ?dslabs::read_mnist() and online MNIST documentation

library(dslabs)

mnist <- dslabs::read_mnist()
names(mnist)
dim(mnist$train$images)
head(mnist$train$labels)
