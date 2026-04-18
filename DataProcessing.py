import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# Remove missing/redundant data

# Drop rows where the Finishing Time is missing
train_df = train_df.dropna(subset=['actual_finish_time_minutes'])
test_df = test_df.dropna(subset=['actual_finish_time_minutes'])


# List the columns to be dropped entirely
# We remove medal_outcome because it's only in the training set and won't be 
# available for the test set(or be able to be used to predict a time), 
# runner_id because it's just an identifier,
# weekly_mileage_km because we already have it in miles,
# and marathon_date because it won't be useful for prediction since we have weather
cols_to_drop = ['runner_id', 'weekly_mileage_km', 'marathon_date'] 

# Drop them from training data
train_df = train_df.drop(columns=cols_to_drop)
train_df = train_df.drop(columns=['medal_outcome'])

# Drop them from test data
test_df = test_df.drop(columns=cols_to_drop)


# Handle missing values

# features that need mean value filled in for missing values
mean_impute_list = ['personal_best_minutes', 'vo2_max', 'sleep_hours_avg', 
                    'nutrition_score', 'hydration_consistency']

# Calculate mean from train data and fill in both train and test with that mean
for col in mean_impute_list:
    train_mean = train_df[col].mean()
    train_df[col] = train_df[col].fillna(train_mean)
    test_df[col] = test_df[col].fillna(train_mean)


# For cross-training, missing likely means 0 hours.
# For injury_severity, missing means no injury ('None').
train_df['cross_training_hours_per_week'] = train_df['cross_training_hours_per_week'].fillna(0)
test_df['cross_training_hours_per_week'] = test_df['cross_training_hours_per_week'].fillna(0)

train_df['injury_severity'] = train_df['injury_severity'].fillna('None')
test_df['injury_severity'] = test_df['injury_severity'].fillna('None')


# Set numerical values for non numerical features
# Define mappings for features with numerical meaning
program_map = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
injury_map = {'None': 0, 'Minor': 1, 'Moderate': 2, 'Severe': 3}

# Apply mapping
train_df['training_program'] = train_df['training_program'].map(program_map)
test_df['training_program'] = test_df['training_program'].map(program_map)

train_df['injury_severity'] = train_df['injury_severity'].map(injury_map)
test_df['injury_severity'] = test_df['injury_severity'].map(injury_map)


# Feature Scaling

# Take all numeric columns 
features_to_scale = train_df.select_dtypes(include=['number']).columns.tolist()

# Don't scale the target variable
features_to_scale.remove('actual_finish_time_minutes') 

# Now run your scaling logic
scaler = StandardScaler()
train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

# Use One-Hot encoding for the remaining categorical features that don't have a natural order

# List remaining features
nominal_cols = ['gender', 'marathon_weather', 'course_difficulty']

# Apply One-Hot Encoding, dropping the first category to remove redundancy
train_df = pd.get_dummies(train_df, columns=nominal_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=nominal_cols, drop_first=True)

# Align train and test sets to have the same columns after one-hot encoding
# If a category exists in train but not in test, get_dummies will create different columns.
# Align test set to the train set, filling missing columns with 0s (since those categories don't exist in test)
# Only matters if there is a category in train that doesn't exist in test such as a weather 
# condition or course difficulty level that only appears in the training data.
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)


train_df.to_csv('train_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)


# Checks for missing values and data types after processing
#print(train_df.dtypes)
#print(train_df.isnull().sum())

print("Pre-processing complete.")
print(f"Final training samples: {len(train_df)}")
print(f"Features processed: {len(features_to_scale)}")