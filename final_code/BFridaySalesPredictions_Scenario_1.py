# Scenario_1

# Mean substitution for missing data.

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


