#  Title: Black Friady Sales Prediction Using Machine Learning.

'''
The goal of this project is to identify the best regression model for predicting Black Friday sales. 
The first step involved thoroughly examining the data to understand its structure and the various features it contains. 
I performed data cleaning to ensure there were no inconsistencies, such as missing values, and filled in any gaps using appropriate methods, 
like filling missing values with the mean or null values where needed. Additionally, I checked the descriptive statistics of the data to get a better 
sense of its distribution and identify any unusual patterns.

Next, I conducted an Exploratory Data Analysis (EDA) to gain deeper insights into the dataset. 
This involved analyzing both numerical and categorical variables to understand their distributions, trends, and any potential outliers. 
I also looked into the relationships between the independent features (such as user demographics, product details, etc.) and the target variable, 
which in this case is the 'purchase' amount. Based on the findings from EDA, I enhanced the dataset by creating new features or modifying existing ones 
to improve model performance. This included feature encoding for categorical variables and feature scaling to ensure that the data was ready for machine learning
models.

After preparing the data, I split the dataset into training and testing sets. 
I then tested five different regression models—Linear Regression, XGBoost Regression and Random Forest Regression. 
The goal was to compare their performance and identify the model that would provide the most accurate predictions for Black Friday sales. 
Each model was evaluated based on key metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score, and the best-performing model 
was selected for further use. This approach allowed me to explore various regression techniques and determine the most suitable model for the task at hand.

'''

#  Dataset Description 

'''
  * Dataset Source: Kaggle (Black Friday Sales)
  * blackfriday_dataset.csv: The model and purchase prediction will be built using this file(Target variable)
  * The data set consists of following Columns:
  * User_ID : User id of the customer
  * Product_ID: Product id of the product
  * Gender: male or female
  * Age: Age in bins i.e 0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+
  * Occupation: Occupation (Masked)
  * City_Category: Category of the City (A,B,C)
  * Stay_In_Current_City_Years: Total number of years stay in current city
  * Marital_Status: 0-Unmarried, 1-Married
  * Product_Category_1: Product Category (Masked)
  * Product_Category_2: Product may belongs to other category also (Masked)
  * Product_Category_3: Product may belongs to other category also (Masked)
  * Purchase: Purchase Amount (Target Variable)

'''


# import dependencies/libraries  


import numpy as np  # Array Operations and Mathematical Operations
import pandas as pd  # Analyzing and manipulating the data, especially for DataFrames
import seaborn as sb  # to visualize random distributions/statistical graphics
from sklearn.impute import SimpleImputer # Statistical data visualization and plotting
import matplotlib # support for data exploration through visualization
import matplotlib.pyplot as plt # For plotting graphs and visualizations 
import sklearn # Importing the main scikit-learn library for machine learning functions
import os


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


# display the total number of rows and columns of dataset (dimensions)
bfriday_sales_train_df.shape

# shows all the information about the dataset
bfriday_sales_train_df.info()


print("**********************************")
print("The dataset uniquness/distinct details")
print("**********************************")
print("Total number of Transactions -> {}".format(bfriday_sales_train_df.shape[0]))

# displays the number of  unique user_ID's
distinct_users = len(bfriday_sales_train_df.User_ID.unique())
print("Total distinct users -> {}".format(distinct_users))

#displays the number of distinct products in a dataframe
distinct_products = len(bfriday_sales_train_df.Product_ID.unique())
print("Total distinct products -> {}".format(distinct_products))


# displays the shape and descriptive statistics for each of the training dataset's numerical columns.
bfriday_sales_train_df.describe()

# Dispaly the descriptive statistics of column "Purchase"
bfriday_sales_train_df['Purchase'].describe()


# Verify which data types are included in the train dataset
bfriday_sales_train_df.dtypes
