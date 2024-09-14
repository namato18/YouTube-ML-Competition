library(dplyr)
library(fastDummies)
library(xgboost)

# Reading in our training data
df = read.csv("data/train.csv")

# Check structure
str(df)

# Turn integers into numeric
df_numeric = df %>%
  mutate_if(is.integer, as.numeric)
str(df_numeric)


# Identify uniform columns
uniform_vars = lapply(df, function(x){
  length(unique(x)) == 1
}) %>%
  unlist()

# Select only non uniform variables
df_numeric_filt = df_numeric %>%
  select(!which(uniform_vars))

# Add one hot encoding
df_onehot = dummy_cols(df_numeric_filt, select_columns = c("GameRulesetName", "agent1", "agent2"))

# Check variable names
names(df_onehot)

# Remove unwanted variables
df_onehot_filt = df_onehot %>%
  select(-Id, -EnglishRules, -LudRules, -num_wins_agent1,
         -num_draws_agent1, -num_losses_agent1, -GameRulesetName,
         -agent1, -agent2)


# Check Structure
str(df_onehot_filt)

# Create a split of 80% TRUE and 20% FALSE values
# with the length of our df_onehot_filt
sample.split = sample(size = nrow(df_onehot_filt), c(TRUE,FALSE), prob = c(0.8,0.2), replace = TRUE)

# Split our data into train/test
train_data = df_onehot_filt[sample.split,]
test_data = df_onehot_filt[!sample.split,]

# Extract our train/test targets (lables)
train_labels = train_data$utility_agent1
test_labels = test_data$utility_agent1

# Remove the targets (labels) so our model can't cheat when training
train_data = train_data %>%
  select(-utility_agent1)
test_data = test_data %>%
  select(-utility_agent1)

# Creat our basic model
bst = xgboost(data = as.matrix(train_data),
              label = train_labels,
              nrounds = 100,
              max_depth = 6,
              objective = 'reg:squarederror',
              verbose = 1
              )

# Make predictions on our testing data
predictions = predict(bst, as.matrix(test_data))

# Build a data.frame to compare
df_compare = data.frame(
  Actual = test_labels,
  Predicted = predictions
)

# Calculate the RMSE of our results
RMSE = sqrt(mean((df_compare$Actual - df_compare$Predicted)^2))
RMSE
  
  
  
