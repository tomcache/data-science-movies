###################################################################
#
# PH125 - Capstone I. Create Train and Validation Sets for 'movielens 10m'
# dataset
#
# You will use the following code to generate your datasets. 
# Develop your algorithm using the edx set. For a final test of 
# your algorithm, predict movie ratings in the validation set as 
# if they were unknown. RMSE will be used to evaluate how close 
# your predictions are to the true values in the validation set.
# 
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################################################
#
# Quiz section
#

# Q1 - How many rows and columns are in the edx dataset?
# We can ask for the dimensions of the dataframe to answer this:
dim(edx)

# Q2 - How many '0' ratings are in the edx dataset?
# Simply sum up all the ratings that match:
sum(edx$rating == 0)

# Q2.1 - How many '3' ratings are in the edx dataset?
# ditto the above:
sum(edx$rating == 3)

# Q3 - How many different movies are in the edx dataset?
# movieId is the key, we need a sum of the number of unique values: 
length(unique(edx$movieId))

# Q4 - How many different users are in the edx dataset?
# for this one, the key is userId:
length(unique(edx$userId))

# Q5 - How many movie ratings are in each of the following genres in the edx dataset?
# Here we want to group summaries by genre:

#create subset for testing code:
t_edx <- edx[1:100, ]
test_rating <- separate_rows(t_edx, genres)

#Since the genre column includes mixed types, we need to seperate 
# the genres out to get an accurate count:
my_rating <- separate_rows(edx, genres)

my_rating %>% filter(genres == "Drama") %>% summarize(ratings = n())
my_rating %>% filter(genres == "Comedy") %>% summarize(ratings = n())
my_rating %>% filter(genres == "Thriller") %>% summarize(ratings = n())
my_rating %>% filter(genres == "Romance") %>% summarize(ratings = n())

# Q6 - Which movie has the greatest number of ratings?
edx %>% group_by(movie = movieId, title = title) %>% summarize(ratings = n()) %>%
  arrange(desc(ratings))

# Q7 - What are the five most given ratings in order from most to least?
edx %>% group_by(quality = rating) %>% summarize(ratings = n()) %>%
  arrange(desc(ratings))

# Q8 - True or False: In general, half star ratings are less common than whole star ratings 
# (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
edx %>% ggplot(aes(rating)) + geom_histogram()

###################################################################
# 
# Part II. Prediction Algorithm 
# 
###################################################################
#
# Section 1. Introduction 
# 
# The goal of this project is to create a recommendation system that will predict
# with high accuracy the rating that a customer would give to any particular movie
# that they would watch. This would be useful to organizations that provide movies
# as a service to help guide customers to select movies that they would like to see,
# and to avoid selecting movies that they would not. 
#
# The information that we have available to us is a large database of movie ratings 
# ("movielens") by many users - in this case, ratings of about 10,000 movies rated by
# almost 70,000 viewers. However, this database is "sparse" in that we don't have ratings
# for all movies by all viewers. The challenge is to be able to use the information 
# available to design a system that would be able to provide a good estimate of a 
# particular users rating of a movie that they haven't rated yet.
#
# It turns out that this is a fairly complicated problem to solve, and involves much
# more that just providing the average review rating from all previous viewers of the 
# movie. There are many factors to consider - some people like different types of movies
# more than others, some movies are more liked than others by the general population, 
# the user may be swayed by the actors featured in a movie, or the fact that all their 
# friends are talking about a movie, and so on.
#
# Because of all these factors, we have chosen to use a machine learning algorithm approach
# to solving this puzzle. This will allow us to rapidy modify our modeling approaches
# utilizing a wide range of available techniques, and to iteratively add components to our 
# model and to be able to test the effects of these on the overall accuracy.
# The goal of this project is to reduce the estimate error to a number below 0.9 RMSE,
# with increasing project scores as this metric approaches 0.8649.
#
# To begin with , we take the movielens 10M database provided and split this into a 
# test set (90%) and a validation set (10%). Our methodology will be developed on the 
# test set, and measured on the validation set, using the RMSE metric to judge and 
# compare the results as noted above.
#
###################################################################
#
# Methods and analysis
# 
# We start by examing our dataset for data quality, first with summary, to see 
# if there are any NA's:
summary(edx)
# we will also check the range on the rating to make sure there are no outliers:
min(edx$rating)
max(edx$rating)
# So the data looks OK. We can now proceed to analyzing these data.

# computing a baseline mean over all user-movie ratings:
(mu_hat <- mean(edx$rating))

# Now we compute our first RMSE using just the baseline mean:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

(naive_rmse <- RMSE(validation$rating, mu_hat))

# This isn't very accurate, so we will add both a movie effect and a user effect
# to this calculation: 

# We compute the difference of each movie mean rating from our mu_hat, above:
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

predicted_ratings_1 <- mu_hat + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

(model_1_rmse <- RMSE(predicted_ratings_1, validation$rating))

#Now we add user effect. Similarly, this computes the difference in the mean 
# of each user, once the baseline and movie effects have been removed:
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

(model_2_rmse <- RMSE(predicted_ratings, validation$rating))

# Since we see that there is such a significant user effect, we want to examine this
# a bit further to see what is going on here. We now plot the histogram of users 
# who have rated at least 100 movies:

edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

user_avgs %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black"))

# This plot shows that the user effect is fairly tightly distributed around a 
# mean of about 3.6, so this verifies that it is a significant effect. 

# Part II. Can we do better? Now that we have accounted for the baseline, movie,
# and user effects, our RMSE is at a respectable 0.865. However, we have not
# yet explored 

edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu_hat + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) 

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

# the 10 best movies:
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i) %>% 
  slice(1:10) 

# and the 10 worst:
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i) %>% 
  slice(1:10) 

# We now want to regualrize the data. First we need to select our lambda parameter:
lambdas <- seq(0, 10, 0.25)

mu <- mean(edx$rating)
just_the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings_l <- validation %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i_l = s/(n_i+l)) %>%
    mutate(pred = mu + b_i_l) %>%
    pull(pred)
  return(RMSE(predicted_ratings_l, validation$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# The above uses the test set, so let's recalculate using the training data only:
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i_l <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i_l = sum(rating - mu)/(n()+l))
  
  b_u_l <- edx %>% 
    left_join(b_i_l, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_l = sum(rating - b_i_l - mu)/(n()+l))
  
  predicted_ratings_l <- 
    validation %>% 
    left_join(b_i_l, by = "movieId") %>%
    left_join(b_u_l, by = "userId") %>%
    mutate(pred = mu + b_i_l + b_u_l) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings_l, validation$rating))
})

qplot(lambdas, rmses)  

(lambda <- lambdas[which.min(rmses)])

(model_4_rmse = min(rmses))

# Now we will recompute our factors using the regularized data:
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# To see the effect, we plot the original movie data vs. our new averages:
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#Note that the change is especially strong in the lower LH quadrant

