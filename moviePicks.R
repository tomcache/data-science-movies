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

# Clean up data structures no longer needed

rm(dl, ratings, movies, test_index, temp, removed)

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

#create test subset (can be handy when checking code):
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
# Methods and analysis
# 

# We start by examing our dataset for data quality, checking to see 
# if there are any NA's, ranges and outliers, etc.:
summary(movielens)
glimpse(movielens)
# So the data looks OK. We can now proceed to analyzing these data to come up with 
# an accurate prediction methodology. From the reference literature and coursework 
# that we recently completed, we know that our model will require at least a 
# computation of the following:
#
# A. A baseline rating (e.g., the mean of all ratings)
# B. A movie-specific effect (e.g., how this movie is rated vs. all of the other movies)
# C. A user-specific effect (similar to the above, how this user compares to the others)
#
# In addition, it may be helpful to regularize the ratings data to reduce the 
# noisiness of movie ratings with low sampling counts. We will use this technique and
# compare the results to the unmodified dataset.


# computing a baseline mean over all user-movie ratings:
(mu_hat <- mean(edx$rating))

# let's also build a function to compute the RMSE of our model against a
# test data set:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Now we compute our first RMSE using just the baseline mean:
(baseline_rmse <- RMSE(validation$rating, mu_hat))

# let's put the results of our calculations into a table, so that we can 
# compare them in the final section:
rmse_results <- data_frame(method = "Baseline (mean)", RMSE = baseline_rmse)

# This isn't very accurate, so we will add both a movie effect and a user effect
# to this calculation: 

# We begin with the movie effect.
# We compute the difference of each movie mean rating from our mu_hat, above:
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# Now we create a set of predictions for the validation set by using the baseline
# and movie effect calculations above for each observation:
predicted_ratings <- mu_hat + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# now we see how much of an improvement this yields:
(movie_rmse <- RMSE(predicted_ratings, validation$rating))

# and we save that into our results table:
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = movie_rmse))

#Now we add user effect. Similarly, this computes the difference in the mean 
# of each user, once the baseline and movie effects have been removed:

#Let's look at the distribution of user ratings:

edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")


user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

(user_rmse <- RMSE(predicted_ratings, validation$rating))

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = user_rmse))

# Since we see that there is such a significant user effect, we want to examine this
# a bit further to see what is going on here. We now plot the histogram of users 
# who have rated at least 100 movies:

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

user_avgs %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black"))




# Part II. Can we do better? Now that we have accounted for the baseline, movie,
# and user effects, our RMSE is at a respectable 0.865. However, we have not
# yet explored whether removing the noise from ratings of movies with low sample
# counts will improve our score, so we will take another look at the data.


# by looking at the movies with the largest errors, we see that
edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu_hat + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% distinct() %>% slice(1:25) 

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

# We now look at how often these movies were rated:
edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) 

edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10)


# We now want to test the effect of regularizing the data. We will compute this
# for a range of penalty parameters (lambda):
lambdas <- seq(0, 10, 0.25)

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

# Now we will recompute our factors using the regularized data:
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu_hat)/(n()+lambda), n_i = n()) 

# To see the effect, we plot the original movie data vs. our new averages:
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#Note that the change is especially strong in the lower LH quadrant

edx %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(residual = rating - (mu_hat + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% distinct() %>% slice(1:25) 

# finally, we update our table with the best rmse from the regularized data:
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# Discussion of results.

# We'll recalculate our final model:

b_i_l <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i_l = sum(rating - mu_hat)/(n()+lambda))

b_u_l <- edx %>% 
  left_join(b_i_l, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_l = sum(rating - b_i_l - mu_hat)/(n()+lambda))

predicted_ratings_l <- 
  validation %>% 
  left_join(b_i_l, by = "movieId") %>%
  left_join(b_u_l, by = "userId") %>%
  mutate(pred = mu_hat + b_i_l + b_u_l)

# Now lets take a look at some of our predictions. We'll create a sample set
# of predictions using our validation set:

set.seed(1, sample.kind="Rounding")

index <- sample(1:nrow(predicted_ratings_l), 50)

movie_sample <- predicted_ratings_l[index, ] %>% mutate(error = rating - pred, mu = mu_hat)

movie_plot <- gather(movie_sample, key = effect, value = value, b_i_l:mu, -pred)

# Now lot's take a look at the errors in our predictions

movie_sample %>% ggplot(aes(x = title, y = error)) + geom_bar( stat = "identity") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6))    

# We can summarize the predictions, and compare to the ratings distribution:

summary(predicted_ratings_l)

# Finally, we compare the histograms of the actual ratings to our model. It
# shows us that while the RMSE's have been improved, the actual characteristics
# between the two sets of data are quite different. 

predicted_ratings_l %>% ggplot(aes(x = rating)) + 
  geom_histogram(binwidth = .5)

predicted_ratings_l %>% ggplot(aes(x = pred)) + 
  geom_histogram(binwidth = .5)


                                     
                                     