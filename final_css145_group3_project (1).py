# -*- coding: utf-8 -*-
"""Final - CSS145_Group3_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ehcMxXQievvhp9ynlQcAZTcb5TKkKugC

Group 3, Section BM7

Members:


*   LANCE NATHANIEL B. MACALALAD
*   RUSKIN GIAN A. LAUGUICO
*   MARC DAVE D. CONSTANTINO
*   CRAIG ZYRUS B. MANUEL
*   JEAN L. LOPEZ

---

**ABOUT**
The project will focus on Customer prediction, to be more specific, the most purchased product (Given by the dataset). The group decisded to use K-Means Clustering and Decision Tree to predict our data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import builtins

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor

!pip install opendatasets --upgrade
!pip install matplotlib
!pip install scikit-learn

import opendatasets as od
import streamlit as st


od.download("https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis")

dataset_df = pd.read_csv("customer-personality-analysis/marketing_campaign.csv", delimiter="\t")
display(dataset_df.head())
print("Total Rows:", len(dataset_df), "\n")

# Filter rows with any null values in any column
null_df = dataset_df[dataset_df.isnull().any(axis=1)]
display(null_df)

print("Total Rows:", len(dataset_df))
print("Total Null rows:", len(null_df), "\n")

# Drop all rows with null values
clean_pd = dataset_df.dropna()
display(clean_pd)


print("Total Rows:", len(dataset_df))
print("Total Null rows:", len(null_df), "\n")
print("Total Not Null rows:", len(clean_pd),"\n")

#Cast columns to proper data types, eliminate unneeded columns

clean_pd.loc[:, 'ID'] = clean_pd['ID'].astype('int64')
clean_pd.loc[:, 'Year_Birth'] = pd.to_datetime(clean_pd['Year_Birth'], format='%Y').dt.year
clean_pd.loc[:, 'Income'] = clean_pd['Income'].astype('float64')
clean_pd.loc[:, 'Kidhome'] = clean_pd['Kidhome'].astype('int32')
clean_pd.loc[:, 'Teenhome'] = clean_pd['Teenhome'].astype('int32')
clean_pd.loc[:, 'Dt_Customer'] = pd.to_datetime(clean_pd['Dt_Customer'], format='%d-%m-%Y').dt.date
clean_pd.loc[:, 'Recency'] = clean_pd['Recency'].astype('int32')
clean_pd.loc[:, 'MntWines'] = clean_pd['MntWines'].astype('float64')
clean_pd.loc[:, 'MntFruits'] = clean_pd['MntFruits'].astype('float64')
clean_pd.loc[:, 'MntMeatProducts'] = clean_pd['MntMeatProducts'].astype('float64')
clean_pd.loc[:, 'MntFishProducts'] = clean_pd['MntFishProducts'].astype('float64')
clean_pd.loc[:, 'MntSweetProducts'] = clean_pd['MntSweetProducts'].astype('float64')
clean_pd.loc[:, 'MntGoldProds'] = clean_pd['MntGoldProds'].astype('float64')
clean_pd.loc[:, 'NumDealsPurchases'] = clean_pd['NumDealsPurchases'].astype('int32')
clean_pd.loc[:, 'NumWebPurchases'] = clean_pd['NumWebPurchases'].astype('int32')
clean_pd.loc[:, 'NumCatalogPurchases'] = clean_pd['NumCatalogPurchases'].astype('int32')
clean_pd.loc[:, 'NumStorePurchases'] = clean_pd['NumStorePurchases'].astype('int32')
clean_pd.loc[:, 'NumWebVisitsMonth'] = clean_pd['NumWebVisitsMonth'].astype('int32')

clean_pd = clean_pd[['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
                     'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                     'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                     'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]
display(clean_pd)

print(clean_pd.dtypes)

#calculate total sales per product
prodsales_pd = pd.DataFrame({
    'MntWines': [clean_pd['MntWines'].sum()],
    'MntFruits': [clean_pd['MntFruits'].sum()],
    'MntMeatProducts': [clean_pd['MntMeatProducts'].sum()],
    'MntFishProducts': [clean_pd['MntFishProducts'].sum()],
    'MntSweetProducts': [clean_pd['MntSweetProducts'].sum()],
    'MntGoldProds': [clean_pd['MntGoldProds'].sum()]
})

prodsales_pivot = prodsales_pd.melt(var_name="Product", value_name="TotalSales")
st.bar_chart(prodsales_pivot.set_index('Product')['TotalSales'])

# Plot using Seaborn barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='Product', y='TotalSales', data=prodsales_pivot, palette='viridis')

# Adding labels and title
plt.title('Total Sales per Product')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.show()

#use heatmap to correlate the data
heatmap_pd = clean_pd[['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
correlation_matrix = heatmap_pd.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation on Income vs Product Spending")

#use a pie chart to show the sales distribution of purchases per product
plt.figure(figsize=(8, 8))
plt.pie(prodsales_pivot['TotalSales'], labels=prodsales_pivot['Product'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))

plt.title('Total Sales Distribution by Product')
plt.show()

#sum all purchases per product per year
clean_pd['Dt_Customer'] = pd.to_datetime(clean_pd['Dt_Customer'], format='%Y-%m-%d')
year_purchase_pd = clean_pd.groupby(clean_pd['Dt_Customer'].dt.year).agg({
    'MntWines': 'sum',
    'MntFruits': 'sum',
    'MntMeatProducts': 'sum',
    'MntFishProducts': 'sum',
    'MntSweetProducts': 'sum',
    'MntGoldProds': 'sum'
}).reset_index()

# Rename the year column to "FiscalYear"
year_purchase_pd.rename(columns={'Dt_Customer': 'FiscalYear'}, inplace=True)

pivot_year = year_purchase_pd.melt(id_vars=["FiscalYear"],
                             value_vars=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],
                             var_name="Product",
                             value_name="TotalSales")

plt.figure(figsize=(10, 20))
sns.barplot(x="FiscalYear", y="TotalSales", data=pivot_year, hue="Product", palette='viridis')

plt.title('Total Purchases per Product per Year')
plt.xlabel('Fiscal Year')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

#sum all purchases per year for pie chart
year_purchase_pd = clean_pd.groupby(clean_pd['Dt_Customer'].dt.year).apply(
    lambda x: x['MntWines'].sum() + x['MntFruits'].sum() + x['MntMeatProducts'].sum() +
              x['MntFishProducts'].sum() + x['MntSweetProducts'].sum() + x['MntGoldProds'].sum()
).reset_index(name='TotalSales')

# Rename the year column to "FiscalYear"
year_purchase_pd.rename(columns={'Dt_Customer': 'FiscalYear'}, inplace=True)

plt.figure(figsize=(8, 8))
plt.pie(year_purchase_pd['TotalSales'], labels=year_purchase_pd['FiscalYear'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))

plt.title('Total Sales Distribution by Year')
plt.show()

#sum all product purchases by marital status
marital_purchase_pd = clean_pd.groupby('Marital_Status').apply(
    lambda x: x['MntWines'].sum() + x['MntFruits'].sum() + x['MntMeatProducts'].sum() +
              x['MntFishProducts'].sum() + x['MntSweetProducts'].sum() + x['MntGoldProds'].sum()
).reset_index(name='TotalSales')

display(marital_purchase_pd)

plt.figure(figsize=(10, 6))
sns.barplot(x='Marital_Status', y='TotalSales', data=marital_purchase_pd, palette='viridis')

plt.title('Total Purchases by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Total Purchases')
plt.xticks(rotation=45)
plt.show()

#Unsupervised Machine Learning K-means Clustering Method
kmeans_pd = clean_pd

#Label Encoder since KMeans does not allow non-numeric values
columns_to_encode = ['Year_Birth','Marital_Status', 'Education', "Dt_Customer"]
label_encoders = {}

# Apply LabelEncoder and store each encoder in the dictionary
for col in columns_to_encode:
    le = LabelEncoder()
    kmeans_pd[col] = le.fit_transform(kmeans_pd[col])
    label_encoders[col] = le  # Store the encoder

#Get the columns needed for scaling
kmeans_df = kmeans_pd[['ID','Year_Birth', 'Education','Marital_Status', 'Dt_Customer', 'Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

#apply scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(kmeans_df)

# Elbow Method to find the optimal number of clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

#Using the optimal number of cluster based on the elbow method above
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_pd['Cluster'] = kmeans.fit_predict(scaled)

cluster_summary = kmeans_pd.groupby('Cluster').mean()
print(cluster_summary)

#Decode The Columns
kmeans_pd['Year_Birth'] = label_encoders['Year_Birth'].inverse_transform(kmeans_pd['Year_Birth'])
kmeans_pd['Marital_Status'] = label_encoders['Marital_Status'].inverse_transform(kmeans_pd['Marital_Status'])
kmeans_pd['Education'] = label_encoders['Education'].inverse_transform(kmeans_pd['Education'])
kmeans_pd['Dt_Customer'] = label_encoders['Dt_Customer'].inverse_transform(kmeans_pd['Dt_Customer'])
print(kmeans_pd)

pivot_cluster = cluster_summary.melt(id_vars="Income",
                                 value_vars=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],
                                 var_name="Product",
                                 value_name="TotalSales")
print(pivot_cluster)

plt.figure(figsize=(12, 6))

# Create the bar plot
sns.barplot(data=pivot_cluster, x="Income", y="TotalSales", hue="Product", palette="viridis")

# Customize the plot
plt.title("Cluster Summary Showing Income and Product Spending")
plt.xlabel("Cluster Mean Income")
plt.ylabel("Average Spending")
plt.legend(title="Product")
plt.show()

#Use Multi-Output Regression Model to classify and predict a customer's product spending
classify_pd = clean_pd[['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

#One-hot encoding for non-numerical columns (not accepted by the model)
classify_pd = pd.get_dummies(classify_pd, columns=['Marital_Status', 'Education'], drop_first=True)

#Define X as features (customer description) and y as targets (product columns)
#To get the customer description, drop their spending on products -- their marital_status, education, birthyear, etc. shall be retained
X = classify_pd.drop(['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], axis=1)
y = classify_pd[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

#Use 30% as test data, 70% as training dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict spending on the test set
y_pred = model.predict(X_test)

# Determine the importance of each customer description to the products
feature_importances = {}
individual_feature_importance_list = []
overall_feature_importances = {feature: 0 for feature in X.columns}

# Loop through each target model and get feature importances
for i, target in enumerate(y.columns):
    # Get feature importances from the individual model for each target
    importances = model.estimators_[i].feature_importances_

    # Store the importances in the dictionary
    feature_importances[target] = dict(zip(X.columns, importances))
    for feature, importance in zip(X.columns, importances):
        overall_feature_importances[feature] += importance

# Average the feature importances by dividing by the number of targets
num_targets = len(y.columns)
for feature in overall_feature_importances:
    overall_feature_importances[feature] /= num_targets

# Display feature importances for each target
for target, importances in feature_importances.items():
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importances:
        # Append as a dictionary to the list, specifying the target, feature, and its importance
        individual_feature_importance_list.append({"Target": target, "Feature": feature, "Importance": importance})

individual_feature_importance_df = pd.DataFrame(individual_feature_importance_list)
display(individual_feature_importance_df)

# For overall importances
sorted_overall_importances = sorted(overall_feature_importances.items(), key=lambda x: x[1], reverse=True)
overall_feature_importance_df = pd.DataFrame(sorted_overall_importances, columns=["Feature", "Importance"])

display(overall_feature_importance_df)

#create pie chart for the overall feature importance

# Threshold for small values
threshold = 0.012

# Filter categories based on threshold
filtered_data = overall_feature_importance_df[overall_feature_importance_df["Importance"] >= threshold]

# Optionally add an "Others" category if there are values below the threshold
if len(overall_feature_importance_df[overall_feature_importance_df["Importance"] < threshold]) > 0:
    others_sum = overall_feature_importance_df[overall_feature_importance_df["Importance"] < threshold]["Importance"].sum()
    # Create a DataFrame for the "Others" row with explicit label
    others_row = pd.DataFrame({"Feature": "Others", "Importance": [others_sum]})
    filtered_data = pd.concat([filtered_data, others_row], ignore_index=True)

plt.figure(figsize=(8, 8))
plt.pie(filtered_data['Importance'], labels=filtered_data['Feature'], autopct='%1.0f%%', startangle=140, colors=sns.color_palette('viridis'))

plt.title('Overall Feature Importance')
plt.show()

#decode the columns
X_test['Marital_Status'] = X_test[ ['Marital_Status_YOLO', 'Marital_Status_Together', 'Marital_Status_Married', 'Marital_Status_Widow', 'Marital_Status_Divorced',
     'Marital_Status_Alone', 'Marital_Status_Single']].idxmax(axis=1).str.replace('Marital_Status_', '')
X_test['Education'] = X_test[['Education_PhD', 'Education_Master', 'Education_Graduation', 'Education_Basic']].idxmax(axis=1).str.replace('Education_', '')

# Drop the one-hot encoded columns
X_test = X_test.drop(['Marital_Status_YOLO', 'Marital_Status_Together', 'Marital_Status_Married', 'Marital_Status_Widow', 'Marital_Status_Divorced',
                                        'Marital_Status_Alone', 'Marital_Status_Single',
                                        'Education_PhD', 'Education_Master', 'Education_Graduation', 'Education_Basic'], axis=1)

# Column names for reference
target_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
mae_list = []
mae_per_column = []
results_list = []

for i in range(len(y_test)):
    data_point_result = {'Data_Point': i + 1}

    # Add the feature data for this data point
    data_point_result.update(X_test.iloc[i].to_dict())

    for j, col in enumerate(target_columns):
        # Use .iloc to access data by position for y_test and y_pred
        actual = y_test.iloc[i, j]
        predicted = y_pred[i, j]

        # Explicitly use the built-in Python abs function
        absolute_error = __builtins__.abs(actual - predicted)
        mae_list.append(absolute_error)

        # Store actual, predicted, Absolute Error, and MAPE for each target
        data_point_result[f"{col}_Actual"] = actual
        data_point_result[f"{col}_Predicted"] = predicted
        data_point_result[f"{col}_Absolute_Error"] = absolute_error

    # Append this data point's results to the list
    results_list.append(data_point_result)

# Convert the list of dictionaries into a DataFrame
results_df = pd.DataFrame(results_list)

# Display the DataFrame
results_df = results_df[['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 'Recency',
                   'MntWines_Predicted', 'MntWines_Actual', 'MntWines_Absolute_Error',
                   'MntFruits_Predicted', 'MntFruits_Actual', 'MntFruits_Absolute_Error',
                   'MntMeatProducts_Predicted', 'MntMeatProducts_Actual', 'MntMeatProducts_Absolute_Error',
                   'MntFishProducts_Predicted', 'MntFishProducts_Actual', 'MntFishProducts_Absolute_Error',
                   'MntSweetProducts_Predicted', 'MntSweetProducts_Actual', 'MntSweetProducts_Absolute_Error',
                   'MntGoldProds_Predicted', 'MntGoldProds_Actual', 'MntGoldProds_Absolute_Error']]
display(results_df)

#pivot for scatterplot
melted_df = results_df.melt(
    id_vars=['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 'Recency'],
    value_vars=[
        'MntWines_Predicted', 'MntWines_Actual',
        'MntFruits_Predicted', 'MntFruits_Actual',
        'MntMeatProducts_Predicted', 'MntMeatProducts_Actual',
        'MntFishProducts_Predicted', 'MntFishProducts_Actual',
        'MntSweetProducts_Predicted', 'MntSweetProducts_Actual',
        'MntGoldProds_Predicted', 'MntGoldProds_Actual'
    ],
    var_name='Category_Variable', value_name='Value'
)

# Split `Category_Variable` into `Category` and `Type` (Predicted, Actual, Absolute_Error)
melted_df['Category'] = melted_df['Category_Variable'].apply(lambda x: x.split('_')[0])
melted_df['Type'] = melted_df['Category_Variable'].apply(lambda x: x.split('_')[1])

# Pivot the melted DataFrame to get Actual and Predicted columns
pivoted_df = melted_df.pivot_table(index=['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 'Recency','Category'],
                                   columns='Type', values='Value').reset_index()

# Displaying the final DataFrame with separate columns for Actual, Predicted, and Absolute_Error
display(pivoted_df)

# Scatterplot for actual vs predicted values for each category
# Scatter Plot: Show actual vs. predicted values, with a reference line.
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Actual", y="Predicted", hue="Category", data=pivoted_df, s=100, palette="deep")

# Adding a reference line for perfect prediction
plt.plot([pivoted_df["Actual"].min(), pivoted_df["Actual"].max()],
         [pivoted_df["Actual"].min(), pivoted_df["Actual"].max()], 'k--', lw=2)

# Adding titles and labels
plt.title("Actual vs. Predicted Values Across Multiple Categories")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

#Using correlation heatmap on income and products actual values
# Red: High positive correlation -> Variables increase together
# Blue: High negative correlation -> One variable increases while the other decreases
feature_columns = ['Income']
target_columns = ['MntWines_Actual', 'MntFruits_Actual', 'MntMeatProducts_Actual', 'MntFishProducts_Actual', 'MntSweetProducts_Actual', 'MntGoldProds_Actual']

# Calculate the correlation matrix
correlation_matrix = results_df[feature_columns + target_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Using correlation heatmap on income and products predicted values
feature_columns = ['Income']
target_columns = ['MntWines_Predicted', 'MntFruits_Predicted', 'MntMeatProducts_Predicted', 'MntFishProducts_Predicted', 'MntSweetProducts_Predicted', 'MntGoldProds_Predicted']

# Calculate the correlation matrix
correlation_matrix = results_df[feature_columns + target_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Calculate the Mean Absolute Error for each target variable
#Lower MAE is better, higher MAE indicates more error
mae_wines = mean_absolute_error(y_test['MntWines'], y_pred[:, 0])
mae_fish = mean_absolute_error(y_test['MntFishProducts'], y_pred[:, 1])
mae_meat = mean_absolute_error(y_test['MntMeatProducts'], y_pred[:, 2])
mae_sweets = mean_absolute_error(y_test['MntSweetProducts'], y_pred[:, 3])
mae_fruits = mean_absolute_error(y_test['MntFruits'], y_pred[:, 4])
mae_gold = mean_absolute_error(y_test['MntGoldProds'], y_pred[:, 5])
average_mae = (mae_wines + mae_fish + mae_meat + mae_sweets + mae_fruits + mae_gold) / 6

# Store MAE values in a dictionary
mae_values = {
    "MntWines": mae_wines,
    "MntFishProducts": mae_fish,
    "MntMeatProducts": mae_meat,
    "MntSweetProducts": mae_sweets,
    "MntFruits": mae_fruits,
    "MntGoldProds": mae_gold,
    "Overall MAE": average_mae
}

mae_df = pd.DataFrame(list(mae_values.items()), columns=["Category", "MAE"])
display(mae_df)

target_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
accuracy_data = []
weighted_accuracy_sum = 0
total_weight = 0

# Calculate and print percentage accuracy for each category
for col in target_columns:
    # Average actual value for the category
    avg_actual = y_test[col].mean()

    # Calculate percentage accuracy
    percentage_accuracy = (1 - (mae_values[col] / avg_actual)) * 100
    accuracy_data.append({"Category": col, "Percentage Accuracy": percentage_accuracy})

    # Add to the weighted accuracy sum and total weight
    weighted_accuracy_sum += percentage_accuracy * avg_actual
    total_weight += avg_actual

# Calculate the overall percentage accuracy across all categories
overall_percentage_accuracy = weighted_accuracy_sum / total_weight
accuracy_data.append({"Category": "Overall Percentage Accuracy", "Percentage Accuracy": overall_percentage_accuracy})

# Convert accuracy data to DataFrame
accuracy_df = pd.DataFrame(accuracy_data)
display(accuracy_df)

prod_acc_df = accuracy_df[accuracy_df['Category'].isin(['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'])]
plt.figure(figsize=(8, 8))
plt.pie(prod_acc_df['Percentage Accuracy'], labels=prod_acc_df['Category'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))

plt.title('Percentage Accuracy Per Product')
plt.show()
