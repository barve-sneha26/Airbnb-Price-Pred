# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

# use random forest, decision trees, Gradient Boosting
# further ElasticNet Regression
# further ahead neural networks

# graphs which can be useful
# reviews per month vs price
# no of reviews vs price
# neighbourhood group vs price
# neighbourhood vs price
# room type vs price
# host listings count vs review
# host lisitngs count vs price


# importing dataset
path = "/Users/snehabarve/Datasets/NewYork_Airbnb.csv"
df = pd.read_csv(path)

print(df.head())

print(df.columns)

print(df.isnull().sum())

print(df.shape)

df = df.drop('license', axis=1)

print(df.columns)

df['host_name'] = df['host_name'].fillna("John Doe")

df['room_type'] = np.where(df['room_type'] == "Entire home/apt", "Entire House", df['room_type'])
print(df.head())

df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].mean())

df = df.drop('last_review', axis=1)
df = df.drop('number_of_reviews_ltm', axis=1)

df['name'] = df['name'].fillna("New York Neighbourhood")

print(df.isnull().sum())

print(df.columns)

# corr_matrix = df.corr()
# f, ax = plt.subplots(figsize=(12, 12))
# sn.heatmap(corr_matrix, vmax=0.8, square=True)
# sn.set(font_scale=0.8)

# feature engineering

df['availability_365'] = df['availability_365'].astype(int)

# Function to categorize values
def categorize_value(value):
    if value < 15:
        return "low"
    elif 16 <= value <= 364:
        return "moderate"
    else:
        return "high"


# Apply the categorization function to the DataFrame column
df['availability_365'] = df['availability_365'].apply(categorize_value)

print(df['availability_365'].head)


df = df.drop('name', axis=1)
df = df.drop('latitude', axis=1)
df = df.drop('longitude', axis=1)
df = df.drop('id', axis=1)
df = df.drop('host_id', axis=1)
df = df.drop('host_name', axis=1)

print(df.head())

print(df['neighbourhood_group'].value_counts())

print(df['neighbourhood'].value_counts())

print(df['availability_365'].value_counts())

print(df.shape)

print(df.columns)

# In[30]:


print(df.dtypes)

# Handling categorical data

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Apply label encoding to the 'Category' column
df['neighbourhood_group'] = label_encoder.fit_transform(df['neighbourhood_group'])
df['neighbourhood'] = label_encoder.fit_transform(df['neighbourhood'])
df['room_type'] = label_encoder.fit_transform(df['room_type'])
df['availability_365'] = label_encoder.fit_transform(df['availability_365'])

print(df.head())

# features(x) and target(y)

X = df.drop('price', axis=1)
y = df['price']

print(X.columns)

print("x.shape: ", X.shape)
print("y.shape: ", y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
linear_reg_model = LinearRegression()

# Create the RFE model and specify the number of features to select
num_features_to_select = 5
rfe = RFE(linear_reg_model, n_features_to_select=num_features_to_select)

# Fit the RFE model to the training data
X_train_rfe = rfe.fit_transform(X_train, y_train)

# Get the selected feature indices
selected_indices = rfe.support_

# Get the selected feature names
selected_feature_names = [df.columns[i] for i, selected in enumerate(selected_indices) if selected]

print("Selected features:", selected_feature_names)

# List of column names to select
selected_columns = ['neighbourhood_group', 'room_type', 'number_of_reviews', 'calculated_host_listings_count']

# Create a subset DataFrame using selected column names
new_X = df[selected_columns]

print(new_X.shape)
print(y.shape)

print(new_X.columns)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.2, random_state=42)

# Model evaluation

linear_reg_model.fit(X_train, y_train)

model_eval_pred = linear_reg_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, model_eval_pred)
print("Mean Squared Error:", mse)
# acc_score = accuracy_score(y_test, model_eval_pred)
# print("Accuracy: ", acc_score)
r2 = r2_score(y_test, model_eval_pred)
print("R2 score: ", r2)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(linear_reg_model, new_X, y, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE scores to positive
cv_scores = -cv_scores

# Calculate the mean and standard deviation of cross-validation scores
mean_cv_score = np.mean(cv_scores)
sd_cv_score = np.std(cv_scores)

print("Mean Cross-Validation Score:", mean_cv_score)
print("Standard Deviation of Cross-Validation Scores:", sd_cv_score)


# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model on the training set
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print("MSE DT: ", mse_dt)

# Calculate Mean Absolute Error (MAE)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
print("MAE DT: ", mae_dt)

# Calculate R-squared (R2) score
r2_dt = r2_score(y_test, y_pred_dt)
print("R2 score DT: ", r2_dt)

# confusion matrix
# accuracy
# precision
# recall
# f1 score

# # Random Forest

# Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("MSE RF: ", mse_rf)

# Calculate Mean Absolute Error (MAE)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("MAE RF: ", mae_rf)

# Calculate R-squared (R2) score
r2_rf = r2_score(y_test, y_pred_rf)
print("R2 score RF: ", r2_rf)

# # Gradient Boost
# Gradient Boost model
gbr_model = GradientBoostingRegressor()

# Train the model on the training set
gbr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_gbr = gbr_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_gbr = mean_squared_error(y_pred_gbr, y_test)
print("MSE GB:", mse_gbr)

# Calculate Mean Absolute Error (MAE)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
print("MAE GB: ", mae_gbr)

# Calculate R-squared (R2) score
r2_gbr = r2_score(y_test, y_pred_gbr)
print("R2 score GB: ", r2_gbr)

# # ElasticNet Regression

# ElasticNet model
elasticNet_reg_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

# Train the model on the training set
elasticNet_reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_enr = elasticNet_reg_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_enr = mean_squared_error(y_test, y_pred_enr)

print("MSE ENR:", mse_enr)

# Calculate Mean Absolute Error (MAE)
mae_enr = mean_absolute_error(y_test, y_pred_enr)
print("MAE ENR: ", mae_enr)

# Calculate R-squared (R2) score
r2_enr = r2_score(y_test, y_pred_enr)
print("R2 score ENR: ", r2_enr)

# displayed on basis of MSE
model_list_mse = pd.DataFrame({
    'Model Names': ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Elastic Net'],
    'MSE': [mse_dt, mse_rf, mse_gbr, mse_enr]})

model_list_mse.sort_values(by='MSE', ascending=True)

# displayed on basis of MAE
model_list_mae = pd.DataFrame({
    'Model Names': ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Elastic Net'],
    'MAE': [mae_dt, mae_rf, mae_gbr, mae_enr]})

model_list_mae.sort_values(by='MAE', ascending=True)

# displayed on basis of R2 score
model_list_r2score = pd.DataFrame({
    'Model Names': ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Elastic Net'],
    'R2 Score': [r2_dt, r2_rf, r2_gbr, r2_enr]})

model_list_r2score.sort_values(by='R2 Score', ascending=True)

with open("rf.model", "wb") as f:
    pickle.dump(rf_model, f)


