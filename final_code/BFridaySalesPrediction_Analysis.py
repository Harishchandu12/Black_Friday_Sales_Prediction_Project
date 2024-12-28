#  Title: Black Friady Sales Prediction Using Machine Learning.

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


# Data Quality Assessment
'''
 1. Handling Missing Values or Null values 
2. Special Characters
3. Outlier Analysis
4. Skewness and Distribution Checks
'''

# 1. Handling Misiing Values

#This shows the null values in the each column of dtaset
print(bfriday_sales_train_df.isnull().sum()) 


# Use a heatmap to Visualize the null values.
sb.heatmap(bfriday_sales_train_df.isnull(), cbar=False)



# We saw that both "Product_Category_2" and "Product_Category_3" contain NULL values. 
# We will now determine the percentage of NULL values in each column.
print(bfriday_sales_train_df.isnull().sum() / bfriday_sales_train_df.shape[0] * 100)



# **2. Special Characters/Range values**

# Display the Unique values in each attribute/column
print(bfriday_sales_train_df.nunique())

#It will give how many times 'Age' value appears in a dataset 
print(bfriday_sales_train_df['Age'].value_counts())

# Since the features "Age" and "Stay_In_Current_City_Years" has some values with "+" and ranges "-"
# so will try to find the count of each values in those 2 columns
print(round((bfriday_sales_train_df['Age'].value_counts(normalize = True).mul(100)), 2).astype(str) + ' %' ) # unique values of Age cloumn

#It will give count of each unique value in 'Stay_In_Current_City_Years' column appears in a dataset 
print(bfriday_sales_train_df['Stay_In_Current_City_Years'].value_counts())


#It will roundoff  each unique value in 'Stay_In_Current_City_Years' column in the form of percentages
print(round((bfriday_sales_train_df['Stay_In_Current_City_Years'].value_counts(normalize = True).mul(100)), 2).astype(str) + ' %' ) # unique values of Age Stay_In_Current_City_Years


# Display the duplicate values in each attribute/column
duplicate_data= bfriday_sales_train_df[bfriday_sales_train_df.duplicated()]
print(duplicate_data.count())


# Display duplicate User_IDs
Unique_User_IDs = len(set(bfriday_sales_train_df.User_ID))
Total_User_IDs = bfriday_sales_train_df.shape[0]
Dup_User_IDs = Total_User_IDs - Unique_User_IDs
print("There are " + str(Dup_User_IDs) + " duplicate User_ID for " + str(Total_User_IDs) + " total number of transactions in the dataset")


# 3. Outlier Analysis

#Diplay 25% and 75% Quartile values
Q1= bfriday_sales_train_df["Purchase"].quantile(0.25)
Q3= bfriday_sales_train_df["Purchase"].quantile(0.75)
print(Q1,Q3)


# Display Inter quartile range
IQR = Q3 - Q1
print(IQR)


# Display the outlier with 1.5 standard range
lower_lmt = Q1 - 1.5 *IQR
upper_lmt = Q3 + 1.5 *IQR

print("Upper limit of outlier is: ",upper_lmt)
print("Lower limit of outlier is: ",lower_lmt)


# Display the outliers purchase max and min values
outliers_df=bfriday_sales_train_df[(bfriday_sales_train_df.Purchase < lower_lmt)|(bfriday_sales_train_df.Purchase > upper_lmt)]
print("number of outliers: ",outliers_df['Purchase'].count())
print("max purcahse value from the outliers is: ",outliers_df['Purchase'].max())
print("min purchase value from the outliers: ",outliers_df['Purchase'].min())


# to show the outliers in the purchases using boxplot
sb.boxplot(data=bfriday_sales_train_df, x="Purchase").set(title='Boxplot of Purchase')


# 4. Skewness and Distribution Checks

# Display the skew data from the dataset
print(bfriday_sales_train_df['Purchase'].skew())


# Plotting the histogram and KDE of 'Purchase' column to visually inspect skewness and distribution
plt.figure(figsize=(16, 6))

# Histogram plot
plt.subplot(1, 3, 1)
sb.histplot(bfriday_sales_train_df['Purchase'], kde=False, bins=30, color='purple')
plt.title('Histogram of Purchase')
plt.xlabel('Purchase')
plt.ylabel('Frequency')

# KDE (Kernel Density Estimate) Plot
plt.subplot(1, 3, 2)
sb.kdeplot(bfriday_sales_train_df['Purchase'], shade=True, color='orange')
plt.title('KDE Plot of Purchase')
plt.xlabel('Purchase')
plt.ylabel('Density')

plt.tight_layout()
plt.show()




# Analyzing the data

# Aggregate the 'Purchase' values by grouping the data on 'Product'
top_products = bfriday_sales_train_df.groupby('Product_ID')['Purchase'].sum()

# Arrange the products from highest to lowest based on total purchase amount
top_products_sorted = top_products.sort_values(ascending=False)

# Get the top 5 products
top_5_products = top_products_sorted.head(5)

# Display the top 5 products
print("Top 5 Products based on total Purchase:")
print(top_5_products)



# Group by 'Occupation' and sum the 'Purchase' values
top_occupations = bfriday_sales_train_df.groupby('Occupation')['Purchase'].sum()

# Sort the occupations by total purchase amount in descending order
top_occupations_sorted = top_occupations.sort_values(ascending=False)

# Get the top 5 occupations
top_5_occupations = top_occupations_sorted.head(5)

# Display the top 5 occupations
print("Top 5 Occupations based on total Purchase:")
print(top_5_occupations)


# Display top Cities
print(bfriday_sales_train_df['City_Category'].value_counts().head())


# display Gender counts
print(bfriday_sales_train_df['Gender'].value_counts())


# Print the gender unique values count by dividing all values by the sum of values
print(bfriday_sales_train_df['Gender'].value_counts(normalize=True)*100)


# Display the count of marital status of customers
print(bfriday_sales_train_df['Marital_Status'].value_counts())


# Married men made more purchases
# Now, we'll check the highest and lowest purchase amounts for both men and women

max_price = bfriday_sales_train_df['Purchase'].max() # max price 
min_price = bfriday_sales_train_df['Purchase'].min() # min price
print("Maximum price range of purchase in both the male & female catogries is: " ,(max_price))
print("Minimum price range of purchase in both the male & female catogries is: " ,(min_price))


# Display mean/average purchase values for overall Gender category
print(bfriday_sales_train_df.groupby("Gender")["Purchase"].mean())



# Exploratory Data Analysis(EDA)

# Gender Distribution Analysis
# Plot a bar graph for counts of Gender

plt.figure(figsize=(6, 6))
gender_counts = bfriday_sales_train_df['Gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values, color=['purple', 'red'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Transactions by Gender')
plt.show()

# Show the percentage of each gender in the data.
print(bfriday_sales_train_df['Gender'].value_counts(normalize=True)*100)

# Set the plot size
plt.figure(figsize=(8, 6))

# Plot the barplot for Gender vs Purchase
gender_purchase_plot = sb.barplot(data=bfriday_sales_train_df, x='Gender', y='Purchase' , palette='Blues')

# Adjust the title of the plot
gender_purchase_plot.set_title('Gender vs Purchase')

# display the plot
plt.show()


# Age Distribution Analysis
# Plot number of transactions by age group
age_group_data = bfriday_sales_train_df['Age'].value_counts().reset_index()
age_group_data.columns = ['Age Group', 'Transaction Count']

plt.figure(figsize=(10, 6))
sb.barplot(data=age_group_data, x='Age Group', y='Transaction Count', palette='Blues')
plt.title('Transactions by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Transaction Count')
plt.show()

# Group by Age and calculate the sum of Purchase
age_purchase_sum = bfriday_sales_train_df.groupby("Age")["Purchase"].sum()

# Plot the data as a line chart
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(age_purchase_sum.index, age_purchase_sum.values, marker="o", linestyle="-", color="blue", label="Total Purchase")

# Add title and axis labels
plt.title("Total Purchase Amount by Age", fontsize=16, fontweight="bold")
plt.xlabel("Age", fontsize=14)
plt.ylabel("Total Purchase Amount", fontsize=14)

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid and legend
plt.grid(linestyle="--", alpha=0.7)
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Stay_In_Current_City_Years Analysis
# Distribution of 'Stay_In_Current_City_Years'
plt.figure(figsize=(10, 6))
sb.countplot(x='Stay_In_Current_City_Years', data=bfriday_sales_train_df)
plt.title('Transactions by Stay in Current City')
plt.show()


# Boxplot of 'Purchase' vs 'Stay_In_Current_City_Years'
plt.figure(figsize=(10, 6))
sb.boxplot(x='Stay_In_Current_City_Years', y='Purchase', data=bfriday_sales_train_df)
plt.title('Purchase Distribution by Stay in Current City (Years)')
plt.show()

# Maritial status Analysis
# Display a count plot for marital status in the dataset.
print(sb.countplot(x='Marital_Status', data=bfriday_sales_train_df))


# City Category Analysis

# Display a count plot for City_Category status in the dataset.
print(sb.countplot(x='City_Category', data=bfriday_sales_train_df))


# # Product Category Analysis

# Count plot for occurrences of each Product Category 1
plt.figure(figsize=(12, 6))
sb.countplot(data=bfriday_sales_train_df, x='Product_Category_1', palette='viridis')
plt.title('Transaction Count by Product Category 1')
plt.xlabel('Product Category 1')
plt.ylabel('Transaction Count')
plt.show()

# Count plot for occurrences of each Product Category 2
plt.figure(figsize=(12, 6))
sb.countplot(data=bfriday_sales_train_df, x='Product_Category_2', palette='crest')
plt.title('Transaction Count by Product Category 2')
plt.xlabel('Product Category 2')
plt.ylabel('Transaction Count')
plt.show()


# Count plot for occurrences of each Product Category 3
plt.figure(figsize=(12, 6))
sb.countplot(data=bfriday_sales_train_df, x='Product_Category_3', palette='coolwarm')
plt.title('Transaction Count by Product Category 3')
plt.xlabel('Product Category 3')
plt.ylabel('Transaction Count')
plt.show()


# Count most popular Product Category 1 by Age and Gender
popular_products = bfriday_sales_train_df.groupby(['Age', 'Gender', 'Product_Category_1']).size().reset_index(name='Count')
top_products = popular_products.sort_values(['Age', 'Gender', 'Count'], ascending=[True, True, False]).groupby(['Age', 'Gender']).head(1)

plt.figure(figsize=(14, 8))
sb.barplot(data=top_products, x='Age', y='Count', hue='Gender', palette='magma', dodge=True)
plt.title('Most Popular Products by Age and Gender (Top Product Category 1)')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()


# Occupation Analysis

# Aggregate total purchases by Occupation
occupation_data = bfriday_sales_train_df.groupby('Occupation').agg(
    total_purchase=('Purchase', 'sum')
).reset_index()

# Plot total purchases by Occupation
plt.figure(figsize=(10, 6))
sb.barplot(data=occupation_data, x='Occupation', y='total_purchase', palette='viridis')
plt.title('Total Purchase by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Total Purchase')
plt.show()


# Aggregate average purchase by Occupation
occupation_data = bfriday_sales_train_df.groupby('Occupation').agg(
    avg_purchase=('Purchase', 'mean')
).reset_index()

# Plot Average Purchase by Occupation
plt.figure(figsize=(10, 6))
sb.barplot(data=occupation_data, x='Occupation', y='avg_purchase', palette='cividis')
plt.title('Average Purchase by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Average Purchase')
plt.show()


#  Relationship between target variable and others

# Display a boxplot comparing purchase amounts by gender
print(sb.boxplot(x='Gender', y='Purchase', data=bfriday_sales_train_df))
plt.show()

# Display a boxplot comparing purchase amounts by gender
print(sb.boxplot(x='Age', y='Purchase', data=bfriday_sales_train_df))
plt.show()

# Display a boxplot comparing purchase amounts by gender
print(sb.boxplot(x='City_Category', y='Purchase', data=bfriday_sales_train_df))
plt.show()

# purchase trends for different product categories

# verify the inital data
bfriday_sales_train_df.head()

#  display a barplot between average purchase and Product_Category_1'
sb.barplot(x='Product_Category_1', y='Purchase', data=bfriday_sales_train_df, estimator='mean', ci='sd')
plt.title('Average Purchase by Product Category')
plt.xlabel('Product Category 1')
plt.ylabel('Average Purchase')
plt.show()

# display a barplot between average purchase and Product_Category_2'
sb.pointplot(x='Product_Category_2', y='Purchase', data=bfriday_sales_train_df, estimator='mean', ci='sd')
plt.title('Point Plot of Purchase by Product Category')
plt.xlabel('Product Category 2')
plt.ylabel('Average Purchase')
plt.show()


# display a barplot between average purchase and Product_Category_2'
sb.boxplot(x='Product_Category_3', y='Purchase', data=bfriday_sales_train_df)
plt.title('Average Purchase by Product Category')
plt.xlabel('Product Category 3')
plt.ylabel('Average Purchase')
plt.show()
