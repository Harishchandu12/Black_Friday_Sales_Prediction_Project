# Overview

A retail company seeks to understand customer purchasing patterns, specifically the purchase amounts across different product categories. The dataset provided includes a summary of purchases made by various customers over the past month for a selection of high-demand products. In addition to purchase amounts, the data includes customer demographics—such as age, gender, marital status, city type, and length of residence in the current city—and product details, including product ID and product category.

This project uses the Black Friday dataset to explore the key factors influencing Black Friday transactions, such as demographics (age, gender, marital status, residency) and build a prediction model, applying machine learning techniques like Linear Regression, Random Forest Regression and XG Boost Regression. The performance and accuracy of these methods are compared, with a focus on data pre-processing and visualization for optimal results.

MAE, RMSE and R2  helps to measure the accuracy and errors.


# Project Structure

 **Abstract**: Overview of the project's goals, methods, and findings.

 **Acknowledgement**: Expressing gratitude to individuals and organizations for their support throughout the project.
 
 **Introduction**: Background, problem scope, and research aims.
 
 **Background**: Overview of the dataset, related studies, and algorithms.
 
 **Methodology**: Description of tools, techniques, and the research workflow.
 
 **Results and Conclusion**: Summary of experimental outcomes and insights.
 
 **Legal, Ethical, and Professional Issues**: Discussion on the responsible use of data.
 
 **References**: List of all sources and references used.
 
 **Appendices**: Overall code of the project.

 
# Dataset

This Project uses the Black Friday Dataset from Kaggle, which has 550,068 records and 12 features. It includes data on retail sales, covering customer details like Gender, Age, and Occupation, as well as product information and purchase amounts. The dataset is useful for understanding sales patterns and predicting future trends.


# Models
Models like Linear Regression, Random Forest Regression and XGBoost Regression help to predict sales and analyze factors influencing them.


# Metrics 
MAE, RMSE, and R² are key metrics for evaluating Black Friday sales prediction models

MAE shows the average prediction error, helping measure overall accuracy.

RMSE emphasizes large errors, making it useful for high-value sales predictions.

R² indicates how well the model explains sales variability and overall trends.

# Tools and Libraries

numpy

pandas

seaborn

matplotlib

scikit-learn

xgboost

ipython


# How To Use
1. Clone the repository:

         https://github.com/Harishchandu12/Black_Friday_Sales_Prediction_Project.git

2. Install the required dependencies:

         pip install -r requirements.txt

4. From final_code folder,

   (a)Run BFridaySalesPrediction_Analysis.py to pre-processing the data and perform Explore Data Analaysis(EDA).
   
   (b)Run BFridaySalesPredictions_Scenario_1.py to train the model, evaluate metrics, and visualize the results.
   
   (c)Run BFridaySalesPrediction_Scenario_2.py to train the model, evaluate metrics, and visualize the results.

5. Review the output and results.
