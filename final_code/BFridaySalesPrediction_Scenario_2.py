# Scenario_2

# Excluding product category3


# import dependencies/libraries  

import numpy as np  # Array Operations and Mathematical Operations
import pandas as pd  # Analyzing and manipulating the data, especially for DataFrames
import seaborn as sb  # to visualize random distributions/statistical graphics
from sklearn.impute import SimpleImputer # Statistical data visualization and plotting
import matplotlib # support for data exploration through visualization
import matplotlib.pyplot as plt # For plotting graphs and visualizations 
import sklearn # Importing the main scikit-learn library for machine learning functions
import os  # For handling operating system functionality
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding categorical data and feature scaling
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.ensemble import RandomForestRegressor  # Random Forest regression model
from xgboost import XGBRegressor  # XGBoost regression model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Metrics for evaluating model performance
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Splitting data and hyperparameter tuning
import xgboost as xgb  # Additional functionality for XGBoost
from IPython.display import display  # For displaying outputs

# Get the current directory path
current_directory = os.getcwd()
print(f"Current Directory: {current_directory}")

# Print each library version
print(f"The numpy version is {np.__version__}.")
print(f"The matplotlib version is {matplotlib.__version__}.")
print(f"The scikit-learn version is {sklearn.__version__}.")
print(f"The pandas version is {pd.__version__}.")
print(f"The seaborn version is {sb.__version__}.")

#  Dataset Loading
file_name = 'train.csv'
file_path = os.path.join(current_directory, file_name)

# Load the dataset into a DataFrame
bfriday_sales_train_df = pd.read_csv(file_path)
print("The train dataset has been loaded")

#displays the initial data of train data
print(bfriday_sales_train_df.head(10))

# display the number of rows and columns from the train dataset (dimensions)
print(bfriday_sales_train_df.shape)

# displays complete information about the dataset
print(bfriday_sales_train_df.info())

# Dropping the User_ID and Product_ID columns
bfriday_sales_train_df = bfriday_sales_train_df.drop(['User_ID', 'Product_ID'], axis=1)

# verify the initial data
print(bfriday_sales_train_df.head())

# Handling special characters in 'Age' and 'Stay_In_Current_City_Years' columns

# Remove the '+' character from 'Age'
bfriday_sales_train_df['Age'] = bfriday_sales_train_df['Age'].str.replace('+', '', regex=False)

# Remove the '+' character from 'Stay_In_Current_City_Years' and convert to float
bfriday_sales_train_df['Stay_In_Current_City_Years'] = (
    bfriday_sales_train_df['Stay_In_Current_City_Years']
    .str.replace('+', '', regex=False)
    .astype(float)
)


# verify the initial data
print(bfriday_sales_train_df.head())


# To check existing null values in the columns
print(bfriday_sales_train_df.isnull().sum())


# Transform the categorical columns into numeric values using LabelEncoder, making the data suitable for analysis and machine learning models
# List of columns to encode
columns_to_encode = ['Gender', 'Age', 'City_Category']

# Create a LabelEncoder object
cencoder = LabelEncoder()

# Apply label encoding to each column in the list
for column in columns_to_encode:
    bfriday_sales_train_df[column] = cencoder.fit_transform(bfriday_sales_train_df[column])

# verify the initial data of the transformed DataFrame
print(bfriday_sales_train_df.head())

# Function to fill missing values with mean and convert to integer
def fill_missing_and_convert_to_int(df, column_name):
    df[column_name] = df[column_name].fillna(df[column_name].mean()).astype('int64')

# Apply the function to the desired columns
fill_missing_and_convert_to_int(bfriday_sales_train_df, 'Product_Category_2')
fill_missing_and_convert_to_int(bfriday_sales_train_df, 'Product_Category_3')

# Verify the changes
print(bfriday_sales_train_df.info())

# Convert the Product_Category_2 and Product_Category_3 data types to int64
bfriday_sales_train_df['Product_Category_2'] =bfriday_sales_train_df['Product_Category_2'].astype('int64')
bfriday_sales_train_df['Product_Category_3'] =bfriday_sales_train_df['Product_Category_3'].astype('int64')
print(bfriday_sales_train_df.info())

# Fill missing values in the DataFrame with the mean of each column to handle NaN values
df_filled = bfriday_sales_train_df.fillna(bfriday_sales_train_df.mean())

# Checking the null values in each column
print(bfriday_sales_train_df.isnull().sum())

# verify the initial data
print(bfriday_sales_train_df.head())

# Observing the complete information about the dataframe
print(bfriday_sales_train_df.info())


#  Data Split

# Dividing the dataset into train & test 
#from sklearn.model_selection import train_test_split

# Split the DataFrame into features (X) and target (y), using 'Purchase' as the target
X = bfriday_sales_train_df.drop(columns=['Purchase', 'Product_Category_3'], axis=1) # dropping Product_Category_3
y = bfriday_sales_train_df['Purchase']

# Assuming X and y are already defined with features and target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the DataFrame into features (X1) and target (y1), using 'Purchase' as the target
X1 = bfriday_sales_train_df.drop(columns=['Purchase', 'Product_Category_3'], axis=1)
y1 = bfriday_sales_train_df['Purchase']

# splitting the data into 70% - 30%
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)


# Feature Scaling

# Check for any non-numeric columns (should raise an issue if any non-numeric data exists)
print(X_train.apply(pd.to_numeric, errors='coerce').isna().sum())  # This will show columns with conversion issues


# Check for non-numeric entries by attempting to convert each value to numeric
# 'errors="coerce"' will turn non-numeric values into NaN
non_numeric_rows = X_train[~X_train.apply(pd.to_numeric, errors='coerce').notna().all(axis=1)]
print("Non-numeric rows:\n", non_numeric_rows)

#It wll gives the complete information about dataframe
bfriday_sales_train_df.info()


#from sklearn.preprocessing import LabelEncoder

# 1. Encode 'Gender' as numeric: 'M' -> 1, 'F' -> 0
X_train['Gender'] = X_train['Gender'].map({'M': 1, 'F': 0})
X_test['Gender'] = X_test['Gender'].map({'M': 1, 'F': 0})

# 2. Encode 'City_Category' as numeric: 'A' -> 0, 'B' -> 1, 'C' -> 2
X_train['City_Category'] = X_train['City_Category'].map({'A': 0, 'B': 1, 'C': 2})
X_test['City_Category'] = X_test['City_Category'].map({'A': 0, 'B': 1, 'C': 2})

# Now apply the StandardScaler to the numeric data
#from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Convert the scaled data back to DataFrame format
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Verify the result
print("Scaled X_train after encoding and scaling:\n", X_train.head())
print("Scaled X_test after encoding and scaling:\n", X_test.head())

# Check for any NaN values in the dataset after encoding and before scaling
print("NaN values in X_train after encoding:", X_train.isna().sum())
print("NaN values in X_test after encoding:", X_test.isna().sum())

# Fill any remaining NaN values in X_train and X_test (if any)
X_train = X_train.fillna(0)  # You can fill with 0 or any appropriate value
X_test = X_test.fillna(0)

# Verify that there are no NaN values remaining
print("NaN values in X_train after filling:", X_train.isna().sum())
print("NaN values in X_test after filling:", X_test.isna().sum())

# Now apply StandardScaler to the data again
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Convert the scaled data back to DataFrame format
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Verify the result
print("Scaled X_train after encoding and scaling:\n", X_train.head())
print("Scaled X_test after encoding and scaling:\n", X_test.head())

#verify the initial data
bfriday_sales_train_df.head()


# Model Training and Evalutaion

# Linear Regression(80-20)

# Import necessary libraries
#from sklearn.linear_model import LinearRegression

# Define a function to initialize and train the Linear Regression model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()  # Initialize the model
    model.fit(X_train, y_train)  # Train the model
    return model

# Train the model using the training data
linear_reg_model = train_linear_regression(X_train, y_train)

# Define a function to make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)

# Make predictions using the trained model
predictions = make_predictions(linear_reg_model, X_test)

# Bar Plot for Actual vs Predicted Purchases with numeric labels
def plot_actual_vs_predicted_bar_numeric(y_test, predictions, num_samples=20):
    # Convert to NumPy arrays for slicing
    y_test = np.array(y_test)
    predictions = np.array(predictions)
    
    # Select a subset of data for better visualization
    indices = np.arange(len(y_test))[:num_samples]
    actual = y_test[indices]
    predicted = predictions[indices]
    
    # Bar positions
    bar_width = 0.35
    positions_actual = np.arange(len(actual))
    positions_predicted = positions_actual + bar_width

    # Plot bars
    plt.figure(figsize=(12, 6))
    plt.bar(positions_actual, actual, width=bar_width, label='Actual Purchases', alpha=0.7)
    plt.bar(positions_predicted, predicted, width=bar_width, label='Predicted Purchases', alpha=0.7)

    # Add labels, title, and legend
    plt.xlabel(' Test data Samples')
    plt.ylabel('Purchases')
    plt.title('Actual vs Predicted Purchases - Linear Regression(80-20 Split))')
    plt.xticks(positions_actual + bar_width / 2, indices, rotation=45)  # Numeric labels 0, 1, 2...
    plt.legend()
    plt.show()

# Call the function
plot_actual_vs_predicted_bar_numeric(y_test, predictions, num_samples=20)

# Import the necessary metrics
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define a function to evaluate the model
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # Return a dictionary with evaluation metrics
    return {
        "Mean Absolute Error (MAE)": mae,
        "Root Mean Squared Error (RMSE)": rmse,
        "R-squared (R²)": r2
    }

# Evaluate the model's performance
metrics = evaluate_model(y_test, predictions)

# Display the metrics
print(metrics)

# 70-30
# Train the model using the 70% training data
linear_reg_model_70 = train_linear_regression(X1_train, y1_train)

# Make predictions using the trained model
predictions_70 = make_predictions(linear_reg_model_70, X1_test)

# Bar Plot for Actual vs Predicted Purchases with numeric labels and custom bar colors (70-30 split)
def plot_actual_vs_predicted_bar_numeric_70_30(y1_test, predictions_70, num_samples=20):
    # Convert to NumPy arrays for slicing
    y1_test = np.array(y1_test)
    predictions_70 = np.array(predictions_70)
    
    # Select a subset of data for better visualization
    indices = np.arange(len(y1_test))[:num_samples]
    actual = y1_test[indices]
    predicted = predictions_70[indices]
    
    # Bar positions
    bar_width = 0.35
    positions_actual = np.arange(len(actual))
    positions_predicted = positions_actual + bar_width

    # Plot bars with custom colors
    plt.figure(figsize=(12, 6))
    plt.bar(positions_actual, actual, width=bar_width, color='pink', label='Actual Purchases', alpha=0.8)
    plt.bar(positions_predicted, predicted, width=bar_width, color='red', label='Predicted Purchases', alpha=0.8)

    # Add labels, title, and legend
    plt.xlabel('Test Data Samples')
    plt.ylabel('Purchases')
    plt.title('Actual vs Predicted Purchases -Linear Regression(70-30 Split)')
    plt.xticks(positions_actual + bar_width / 2, indices, rotation=45)  # Numeric labels 0, 1, 2...
    plt.legend()
    plt.show()

# Call the function
plot_actual_vs_predicted_bar_numeric_70_30(y1_test, predictions_70, num_samples=20)

# Evaluate the model's performance for 70-30 split
metrics_70 = evaluate_model(y1_test, predictions_70)

# Display the metrics
print("Metrics for 70-30 Split:")
print(metrics_70)


#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to compute metrics for a given split
def calculate_metrics(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)  # RMSE
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2

# Assuming predictions_80_20 and predictions_70_30 are the predictions for respective splits
mae_80_20, rmse_80_20, r2_80_20 = calculate_metrics(y_test, predictions)
mae_70_30, rmse_70_30, r2_70_30 = calculate_metrics(y1_test, predictions_70)

# Combine results into lists for plotting
metrics_80_20 = [mae_80_20, rmse_80_20, r2_80_20]
metrics_70_30 = [mae_70_30, rmse_70_30, r2_70_30]

# Define labels and x positions
x_labels = ['MAE', 'RMSE', 'R^2']
x = np.arange(len(x_labels))  # Numeric positions for the metrics

# Create subplots for side-by-side visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot 80-20 Split
axes[0].plot(x, metrics_80_20, marker='o', label='80-20 Split', color='blue', linestyle='-', linewidth=2)
axes[0].set_title('80-20 Split Metrics')
axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Values')
axes[0].set_xticks(x)
axes[0].set_xticklabels(x_labels)
axes[0].grid(alpha=0.3)

# Plot 70-30 Split
axes[1].plot(x, metrics_70_30, marker='o', label='70-30 Split', color='orange', linestyle='--', linewidth=2)
axes[1].set_title('70-30 Split Metrics')
axes[1].set_xlabel('Metrics')
axes[1].set_xticks(x)
axes[1].set_xticklabels(x_labels)
axes[1].grid(alpha=0.3)

# Adjust layout and show plot
plt.suptitle('Comparison of Evaluation Metrics for 80-20 and 70-30 Splits')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the super title
plt.show()


# Random Forest regressor( 80-20)

#from sklearn.ensemble import RandomForestRegressor
# Define a function to train the Random Forest Regressor model
def train_random_forest(X_train, y_train):
    rf_model = RandomForestRegressor(random_state=30, n_estimators=200)  # Initialize the model
    rf_model.fit(X_train, y_train)  # Train the model
    return rf_model

# Train the model for 80-20 split
rf_model_80 = train_random_forest(X_train, y_train)

# Make predictions for 80-20 split
predictions_80 = make_predictions(rf_model_80, X_test)

# Bar Plot for Actual vs Predicted Purchases
def plot_actual_vs_predicted_bar(y_test, predictions_80, num_samples=20):
    # Convert to NumPy arrays for slicing
    y_test = np.array(y_test)
    predictions_80 = np.array(predictions_80)
    
    # Select a subset of data for better visualization
    indices = np.arange(len(y_test))[:num_samples]
    actual = y_test[indices]
    predicted = predictions_80[indices]
    
    # Bar positions
    bar_width = 0.35
    positions_actual = np.arange(len(actual))
    positions_predicted = positions_actual + bar_width

    # Plot bars
    plt.figure(figsize=(12, 6))
    plt.bar(positions_actual, actual, width=bar_width, label='Actual Values', color='grey', alpha=0.8)
    plt.bar(positions_predicted, predicted, width=bar_width, label='Predicted Values', color='purple', alpha=0.8)

    # Add labels, title, and legend
    plt.xlabel('Test data Samples ')
    plt.ylabel('Purchase Values')
    plt.title('Actual vs Predicted Purchases - Random Forest Regressor(80-20 Split)')
    plt.xticks(positions_actual + bar_width / 2, indices, rotation=45)
    plt.legend()
    plt.show()

# Call the function for the Random Forest predictions
plot_actual_vs_predicted_bar(y_test, predictions_80, num_samples=20)

# Import the necessary metrics
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define a function to evaluate the model
def evaluate_model(y_test, predictions_80):
    mae = mean_absolute_error(y_test, predictions_80)
    rmse = mean_squared_error(y_test, predictions_80, squared=False)
    r2 = r2_score(y_test, predictions_80)
    
    # Return a dictionary with evaluation metrics
    return {
        "Mean Absolute Error (MAE)": mae,
        "Root Mean Squared Error (RMSE)": rmse,
        "R-squared (R²)": r2
    }

# Evaluate the model's performance
metrics = evaluate_model(y_test, predictions_80)

# Display the metrics
print(metrics)


# 70-30

# Train the model for 70-30 split
rf_model_70 = train_random_forest(X1_train, y1_train)

# Make predictions for 70-30 split
predictions_70 = make_predictions(rf_model_70, X1_test)

# Bar Plot for Actual vs Predicted Purchases (70-30 Split)
def plot_actual_vs_predicted_bar_70_30(y1_test, predictions_70, num_samples=20):
    # Convert to NumPy arrays for slicing
    y1_test = np.array(y1_test)
    predictions_70 = np.array(predictions_70)
    
    # Select a subset of data for better visualization
    indices = np.arange(len(y_test))[:num_samples]
    actual = y1_test[indices]
    predicted = predictions_70[indices]
    
    # Bar positions
    bar_width = 0.35
    positions_actual = np.arange(len(actual))
    positions_predicted = positions_actual + bar_width

    # Plot bars
    plt.figure(figsize=(12, 6))
    plt.bar(positions_actual, actual, width=bar_width, label='Actual Values', color='lightblue', alpha=0.8)
    plt.bar(positions_predicted, predicted, width=bar_width, label='Predicted Values', color='green', alpha=0.8)

    # Add labels, title, and legend
    plt.xlabel('Test Data Samples')
    plt.ylabel('Purchase Values')
    plt.title('Actual vs Predicted Purchases - Random Forest Regressor(70-30 Split)')
    plt.xticks(positions_actual + bar_width / 2, indices, rotation=45)
    plt.legend()
    plt.show()

# Call the function for the 70-30 Random Forest predictions
plot_actual_vs_predicted_bar_70_30(y1_test, predictions_70, num_samples=20)

# Evaluate the model's performance for 70-30 split
metrics_70 = evaluate_model(y1_test, predictions_70)

# Display the metrics
print("Metrics for 70-30 Split:")
print(metrics_70)


#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to compute metrics for a given split
def calculate_metrics(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)  # RMSE
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2

# Assuming predictions_80_20 and predictions_70_30 are the predictions for respective splits
mae_80_20, rmse_80_20, r2_80_20 = calculate_metrics(y_test, predictions)
mae_70_30, rmse_70_30, r2_70_30 = calculate_metrics(y1_test, predictions_70)

# Combine results into lists for plotting
metrics_80_20 = [mae_80_20, rmse_80_20, r2_80_20]
metrics_70_30 = [mae_70_30, rmse_70_30, r2_70_30]

# Define labels and x positions
x_labels = ['MAE', 'RMSE', 'R^2']
x = np.arange(len(x_labels))  # Numeric positions for the metrics

# Create subplots for side-by-side visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot 80-20 Split
axes[0].plot(x, metrics_80_20, marker='o', label='80-20 Split', color='grey', linestyle='-', linewidth=2)
axes[0].set_title('80-20 Split Metrics')
axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Values')
axes[0].set_xticks(x)
axes[0].set_xticklabels(x_labels)
axes[0].grid(alpha=0.3)

# Plot 70-30 Split
axes[1].plot(x, metrics_70_30, marker='o', label='70-30 Split', color='indigo', linestyle='--', linewidth=2)
axes[1].set_title('70-30 Split Metrics')
axes[1].set_xlabel('Metrics')
axes[1].set_xticks(x)
axes[1].set_xticklabels(x_labels)
axes[1].grid(alpha=0.3)

# Adjust layout and show plot
plt.suptitle('Comparison of Evaluation Metrics for 80-20 and 70-30 Splits')
plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()

#  XGBoost regressor(80-20)
#from xgboost import XGBRegressor

# Define a function to train the XGBoost Regressor
def train_xgboost_regressor(X_train, y_train):
    xgb_model = XGBRegressor(random_state=30)  # Initialize the model
    xgb_model.fit(X_train, y_train)  # Train the model
    return xgb_model

#Train the model and make predictions
# 80-20 split
xgb_model_80 = train_xgboost_regressor(X_train, y_train)
predictions_80 = make_predictions(xgb_model_80, X_test)

# Bar Plot for Actual vs Predicted Purchases
def plot_actual_vs_predicted_bar(y_test, predictions_80, num_samples=20):
    # Convert to NumPy arrays for slicing
    y_test = np.array(y_test)
    predictions_80 = np.array(predictions)
    
    # Select a subset of data for better visualization
    indices = np.arange(len(y_test))[:num_samples]
    actual = y_test[indices]
    predicted = predictions_80[indices]
    
    # Bar positions
    bar_width = 0.35
    positions_actual = np.arange(len(actual))
    positions_predicted = positions_actual + bar_width

    # Plot bars
    plt.figure(figsize=(12, 6))
    plt.bar(positions_actual, actual, width=bar_width, label='Actual Values', color='grey', alpha=0.8)
    plt.bar(positions_predicted, predicted, width=bar_width, label='Predicted Values', color='Cyan', alpha=0.8)

    # Add labels, title, and legend
    plt.xlabel('Test data Samples ')
    plt.ylabel('Purchase Values')
    plt.title('Actual vs Predicted Purchases - XGBoost Regressor(80-20 Split)')
    plt.xticks(positions_actual + bar_width / 2, indices, rotation=45)
    plt.legend()
    plt.show()

# Call the function for the Random Forest predictions
plot_actual_vs_predicted_bar(y_test, predictions_80, num_samples=20)

# Import the necessary metrics
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define a function to evaluate the model
def evaluate_model(y_test, predictions_80):
    mae = mean_absolute_error(y_test, predictions_80)
    rmse = mean_squared_error(y_test, predictions_80, squared=False)
    r2 = r2_score(y_test, predictions_80)
    
    # Return a dictionary with evaluation metrics
    return {
        "Mean Absolute Error (MAE)": mae,
        "Root Mean Squared Error (RMSE)": rmse,
        "R-squared (R²)": r2
    }

# Evaluate the model's performance
metrics = evaluate_model(y_test, predictions_80)

# Display the metrics
print(metrics)

# 70-30
# Train the model for 70-30 split
xgb_model_70 = train_xgboost_regressor(X1_train, y1_train)


# Make predictions for 70-30 split
predictions_70 = make_predictions(xgb_model_70, X1_test)


# Bar Plot for Actual vs Predicted Purchases
def plot_actual_vs_predicted_bar(y_test, predictions_70, num_samples=20):
    # Convert to NumPy arrays for slicing
    y_test = np.array(y_test)
    predictions_70 = np.array(predictions_70)
    
    # Select a subset of data for better visualization
    indices = np.arange(len(y_test))[:num_samples]
    actual = y_test[indices]
    predicted = predictions_70[indices]
    
    # Bar positions
    bar_width = 0.35
    positions_actual = np.arange(len(actual))
    positions_predicted = positions_actual + bar_width

    # Plot bars
    plt.figure(figsize=(12, 6))
    plt.bar(positions_actual, actual, width=bar_width, label='Actual Values', color='orange', alpha=0.8)
    plt.bar(positions_predicted, predicted, width=bar_width, label='Predicted Values', color='Plum', alpha=0.8)

    # Add labels, title, and legend
    plt.xlabel('Test data Samples ')
    plt.ylabel('Purchase Values')
    plt.title('Actual vs Predicted Purchases - XGBoost Regressor(70-30 Split)')
    plt.xticks(positions_actual + bar_width / 2, indices, rotation=45)
    plt.legend()
    plt.show()

# Call the function for the Random Forest predictions
plot_actual_vs_predicted_bar(y_test, predictions_70, num_samples=20)

# Evaluate the model's performance for 70-30 split
metrics_70 = evaluate_model(y1_test, predictions_70)

# Display the metrics
print("Metrics for 70-30 Split:")
print(metrics_70)


#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to compute metrics for a given split
def calculate_metrics(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)  # RMSE
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2

# Assuming predictions_80_20 and predictions_70_30 are the predictions for respective splits
mae_80_20, rmse_80_20, r2_80_20 = calculate_metrics(y_test, predictions)
mae_70_30, rmse_70_30, r2_70_30 = calculate_metrics(y1_test, predictions_70)

# Combine results into lists for plotting
metrics_80_20 = [mae_80_20, rmse_80_20, r2_80_20]
metrics_70_30 = [mae_70_30, rmse_70_30, r2_70_30]

# Define labels and x positions
x_labels = ['MAE', 'RMSE', 'R^2']
x = np.arange(len(x_labels))  # Numeric positions for the metrics

# Create subplots for side-by-side visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot 80-20 Split
axes[0].plot(x, metrics_80_20, marker='o', label='80-20 Split', color='brown', linestyle='-', linewidth=2)
axes[0].set_title('80-20 Split Metrics')
axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Values')
axes[0].set_xticks(x)
axes[0].set_xticklabels(x_labels)
axes[0].grid(alpha=0.3)

# Plot 70-30 Split
axes[1].plot(x, metrics_70_30, marker='o', label='70-30 Split', color='teal', linestyle='--', linewidth=2)
axes[1].set_title('70-30 Split Metrics')
axes[1].set_xlabel('Metrics')
axes[1].set_xticks(x)
axes[1].set_xticklabels(x_labels)
axes[1].grid(alpha=0.3)

# Adjust layout and show plot
plt.suptitle('Comparison of Evaluation Metrics for 80-20 and 70-30 Splits')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the super title
plt.show()

# Performance Metrics Comparison of Regression Models (80-20 vs 70-30 Splits)

# Define models and metrics for both splits
models = ['Linear Regression', 'Random Forest', 'XGBoost']

# Metrics for 80-20 split
mae_80 = [3602.129818687039, 2248.510346821977, 2248.510346821977]  # Mean Absolute Error
rmse_80 = [4703.7248756890285, 3041.2425572696498, 3041.2425572696498]  # Root Mean Squared Error
r2_80 = [0.11944323591270134, 0.6318916638970715, 0.6318916638970715]  # R² Score

# Metrics for 70-30 split
mae_70 = [3597.9620945825905, 2216.435901542471, 2188.8989553498386]  # Mean Absolute Error
rmse_70 = [4701.822378495528, 3032.236002234281, 2930.438108715919]  # Root Mean Squared Error
r2_70 = [0.12281198787587633, 0.6351736062611392, 0.6592582438402976]  # R² Score

# Function to plot side-by-side metrics for both splits
def plot_metrics_comparison(models, metrics_80, metrics_70, metric_names):
    """
    Function to plot side-by-side metrics for 80-20 and 70-30 splits.

    Parameters:
        models (list): List of model names.
        metrics_80 (list of lists): Metrics for 80-20 split.
        metrics_70 (list of lists): Metrics for 70-30 split.
        metric_names (list): Names of the metrics (e.g., ['MAE', 'RMSE', 'R²']).
    """
    x = np.arange(len(models))  # Positions for the bars
    width = 0.35  # Width of each bar

    plt.figure(figsize=(22, 10))

    for i, metric_name in enumerate(metric_names):
        plt.subplot(1, 3, i + 1)  # Create subplots for each metric

        # Bars for 80-20 and 70-30 splits
        plt.bar(x - width/2, metrics_80[i], width, label='80-20 Split', color='brown')
        plt.bar(x + width/2, metrics_70[i], width, label='70-30 Split', color='pink')

        # Add labels and title
        plt.xlabel('Models', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'Comparison of {metric_name}', fontsize=14)
        plt.xticks(x, models, rotation=15, fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout 
    plt.tight_layout()
    plt.show()

# Prepare the metrics data for input into the function
metrics_80 = [mae_80, rmse_80, r2_80]
metrics_70 = [mae_70, rmse_70, r2_70]
metric_names = ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R²)']

# Call the function to plot comparison
plot_metrics_comparison(models, metrics_80, metrics_70, metric_names)


# Hyperparameter tuning

#from sklearn.model_selection import train_test_split, RandomizedSearchCV
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Linear Regression - we can’t tune standard linear regression because it has no settings to adjust.

# Random forest

#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb

# Define RandomForest model
rf_model = RandomForestRegressor(random_state=30, n_estimators=200)

# Define the XGBoost model
xgb_model = XGBRegressor(random_state=30)

# Reduced hyperparameter grid
param_dist_rf = {
    "n_estimators": [50, 200],
    "max_depth": [12, None],
    "min_samples_split": [2, 5],
}

param_dist_xgb = {
    "n_estimators": [50, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.1, 0.3],
}

# RandomizedSearchCV for RandomForest with parallel processing
random_search_rf = RandomizedSearchCV(
    rf_model, 
    param_distributions=param_dist_rf, 
    n_iter=1,  # Use 1 iteration for faster testing
    cv=2, 
    scoring="neg_mean_absolute_error", 
    random_state=30,
    n_jobs=-1  # Parallelize across available CPU cores
)

# RandomizedSearchCV for XGBoost with parallel processing
random_search_xgb = RandomizedSearchCV(
    xgb_model, 
    param_distributions=param_dist_xgb, 
    n_iter=1,  # Use 1 iteration for faster testing
    cv=2, 
    scoring="neg_mean_absolute_error", 
    random_state=30,
    n_jobs=-1  # Parallelize across available CPU cores
)

# Function to perform RandomizedSearchCV and model evaluation
def tune_and_evaluate(model, param_dist, X_train, y_train, X_test, y_test):
    random_search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist, 
        n_iter=1, 
        cv=2, 
        scoring="neg_mean_absolute_error", 
        random_state=30,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return best_model, best_params, mae, rmse, r2

# Evaluation (with 1 iteration for faster testing)
results_80_rf = tune_and_evaluate(rf_model, param_dist_rf, X_train, y_train, X_test, y_test)
results_70_rf = tune_and_evaluate(rf_model, param_dist_rf, X1_train, y1_train, X1_test, y1_test)

results_80_xgb = tune_and_evaluate(xgb_model, param_dist_xgb, X_train, y_train, X_test, y_test)
results_70_xgb = tune_and_evaluate(xgb_model, param_dist_xgb, X1_train, y1_train, X1_test, y1_test)


#from IPython.display import display

# Results for 80-20 and 70-30 splits
results = {
    "Model": ["Random Forest (80-20)", "XGBoost (80-20)", 
              "Random Forest (70-30)", "XGBoost (70-30)"],
    "MAE (80-20)": [results_80_rf[2], results_80_xgb[2], results_70_rf[2], results_70_xgb[2]],
    "RMSE (80-20)": [results_80_rf[3], results_80_xgb[3], results_70_rf[3], results_70_xgb[3]],
    "R2 (80-20)": [results_80_rf[4], results_80_xgb[4], results_70_rf[4], results_70_xgb[4]],
    "MAE (70-30)": [results_80_rf[2], results_80_xgb[2], results_70_rf[2], results_70_xgb[2]],
    "RMSE (70-30)": [results_80_rf[3], results_80_xgb[3], results_70_rf[3], results_70_xgb[3]],
    "R2 (70-30)": [results_80_rf[4], results_80_xgb[4], results_70_rf[4], results_70_xgb[4]]
}

# Create the DataFrame
results_df = pd.DataFrame(results)
