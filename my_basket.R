# – Problem type: unsupervised basket analysis
# – response variable: NA
# – features: 42
# – observations: 2,000
# – objective: use attributes of each basket to identify common groupings
# of items purchased together.
# – access: available on the companion website for this book

library(readr)

url <- "https://raw.githubusercontent.com/koalaverse/homlr/master/data/my_basket.csv"
my_basket <- readr::read_csv(url)
dim(my_basket)
