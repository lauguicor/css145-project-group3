#######################
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import opendatasets as od
import streamlit as st
import builtins

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor


#######################
# Page configuration
st.set_page_config(
    page_title="Customer Personality Analysis", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Customer Personality Analysis')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Marc Dave D. Constantino\n2. Ruskin Gian A. Lauguico\n3. Jean L. Lopez \n4. Lance Nathaniel B. Macalalad \n5. Craig Zyrus B. Manuel")

#######################
# Data

# Load data

dataset_df = pd.read_csv("marketing_campaign.csv", delimiter="\t")
print("Total Rows:", len(dataset_df), "\n")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here
    st.markdown(""" 

    The project will focus on Customer prediction, to be more specific, the most purchased product (Given by the dataset). The group decisded to use K-Means Clustering and Decision Tree to predict our data.

    #### Pages
    - `Dataset` - Introduction to the Customer Personality dataset, providing demographic and behavioral details that help with customer segmentation and spending habit analysis.
    - `EDA` - Exploratory Data Analysis of the dataset to uncover patterns in demographics and purchase behaviors. Includes visualizations such as bar charts, and a correlation heatmap.
    - `Data Cleaning / Pre-processing` - Data cleaning steps to handle missing values, encode categorical features, and scale numerical features. Prepares the dataset for model training by splitting it into training and testing sets.
    - `Machine Learning` - Application of machine learning models: K-Means Clustering for customer segmentation and a Random Forest Regressor to predict spending. Includes visualizations of clusters and model evaluation using metrics such as Mean Absolute Error.
    - `Prediction` - Interactive page allowing users to input customer demographic and preference data to predict spending or cluster membership based on the trained models.
    - `Conclusion` - Summarizes key findings from the EDA and machine learning analyses, highlighting trends in customer spending and segmentation. Suggests potential future enhancements to improve accuracy and insights.

    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")
    st.markdown("""Here is a preview of the dataset that we used in this project. The Customer Personality analysis is an analysis of what a company's ideal customer is.
    `Link:` https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis 

    """) 

    # Your content for your DATASET page goes here
    st.write(dataset_df)

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here
    # Filter rows with any null values in any column
    st.write("### Rows with Null Values")
    null_df = dataset_df[dataset_df.isnull().any(axis=1)]
    st.dataframe(null_df)

    # Display total rows and null rows
    st.write("### Dataset Summary")
    st.write("Total Rows:", len(dataset_df))
    st.write("Total Null Rows:", len(null_df))

    # Drop all rows with null values
    clean_pd = dataset_df.dropna()
    st.write("### Cleaned Dataset")
    st.dataframe(clean_pd)

    # Display counts after cleaning
    st.write("Total Not Null Rows:", len(clean_pd))

    # Cast columns to proper data types and display
    st.write("### Dataset with Corrected Data Types")
    clean_pd['ID'] = clean_pd['ID'].astype('int64')
    clean_pd['Year_Birth'] = pd.to_datetime(clean_pd['Year_Birth'], format='%Y').dt.year
    clean_pd['Income'] = clean_pd['Income'].astype('float64')
    clean_pd['Kidhome'] = clean_pd['Kidhome'].astype('int32')
    clean_pd['Teenhome'] = clean_pd['Teenhome'].astype('int32')
    clean_pd['Dt_Customer'] = pd.to_datetime(clean_pd['Dt_Customer'], format='%d-%m-%Y').dt.date
    clean_pd['Recency'] = clean_pd['Recency'].astype('int32')
    clean_pd['MntWines'] = clean_pd['MntWines'].astype('float64')
    clean_pd['MntFruits'] = clean_pd['MntFruits'].astype('float64')
    clean_pd['MntMeatProducts'] = clean_pd['MntMeatProducts'].astype('float64')
    clean_pd['MntFishProducts'] = clean_pd['MntFishProducts'].astype('float64')
    clean_pd['MntSweetProducts'] = clean_pd['MntSweetProducts'].astype('float64')
    clean_pd['MntGoldProds'] = clean_pd['MntGoldProds'].astype('float64')
    clean_pd['NumDealsPurchases'] = clean_pd['NumDealsPurchases'].astype('int32')
    clean_pd['NumWebPurchases'] = clean_pd['NumWebPurchases'].astype('int32')
    clean_pd['NumCatalogPurchases'] = clean_pd['NumCatalogPurchases'].astype('int32')
    clean_pd['NumStorePurchases'] = clean_pd['NumStorePurchases'].astype('int32')
    clean_pd['NumWebVisitsMonth'] = clean_pd['NumWebVisitsMonth'].astype('int32')

    # Select relevant columns
    clean_pd = clean_pd[['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
                         'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                         'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                         'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]
    st.dataframe(clean_pd)

    # Store clean_pd in st.session_state for use in other pages
    st.session_state.clean_pd = clean_pd  # Save clean_pd in session state

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    # Access clean_pd from st.session_state
    clean_pd = st.session_state.get('clean_pd')

    if clean_pd is not None:
        # Perform EDA tasks
        col = st.columns(1, gap='large')  # Adjusted layout to 2 columns for better space management

        with col[0]:
            st.markdown('#### Correlation Heatmap')
            st.markdown("""The heatmap shows the correlation between income and spending on different product categories. Higher income is generally associated with higher spending across most categories, with some exceptions like gold products.""")
            heatmap_pd = clean_pd[['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
            correlation_matrix = heatmap_pd.corr()
            plt.figure(figsize=(10, 6))  # Increased size for better visibility
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation on Income vs Product Spending")
            st.pyplot()

        with col[0]:
            st.markdown('#### Total Product Sales')
            st.markdown("""The bar chart shows the total sales for different product categories. Wine has the highest sales, followed by meat products. Gold products and sweet products have the lowest sales.""")
            sales_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
            if all(col in clean_pd.columns for col in sales_columns):
                prodsales_pd = pd.DataFrame({
                    'MntWines': [clean_pd['MntWines'].sum()],
                    'MntFruits': [clean_pd['MntFruits'].sum()],
                    'MntMeatProducts': [clean_pd['MntMeatProducts'].sum()],
                    'MntFishProducts': [clean_pd['MntFishProducts'].sum()],
                    'MntSweetProducts': [clean_pd['MntSweetProducts'].sum()],
                    'MntGoldProds': [clean_pd['MntGoldProds'].sum()]
                })
                prodsales_pivot = prodsales_pd.melt(var_name="Product", value_name="TotalSales")
                plt.figure(figsize=(10, 6))  # Increased size for better visibility
                sns.barplot(x='Product', y='TotalSales', data=prodsales_pivot, palette='viridis')
                st.pyplot()

        # Second row of columns with larger plots for better organization
        col2 = st.columns(1, gap='large')  # Adjusted layout to 2 columns for better space management

        with col2[0]:
            st.markdown('#### Total Purchases by Marital Status')
            st.markdown("""The bar chart shows total purchases by marital status. Married individuals have the highest purchases, followed by single individuals. Divorced and "Together" have similar levels, while other categories have significantly lower purchases.""")
            marital_purchase_pd = pd.DataFrame({
                'MntWines': clean_pd.groupby('Marital_Status')['MntWines'].sum(),
                'MntFruits': clean_pd.groupby('Marital_Status')['MntFruits'].sum(),
                'MntMeatProducts': clean_pd.groupby('Marital_Status')['MntMeatProducts'].sum(),
                'MntFishProducts': clean_pd.groupby('Marital_Status')['MntFishProducts'].sum(),
                'MntSweetProducts': clean_pd.groupby('Marital_Status')['MntSweetProducts'].sum(),
                'MntGoldProds': clean_pd.groupby('Marital_Status')['MntGoldProds'].sum()
            })
            marital_purchase_pd['TotalSales'] = marital_purchase_pd.sum(axis=1)
            plt.figure(figsize=(10, 6))  # Increased size for better visibility
            sns.barplot(x=marital_purchase_pd.index, y='TotalSales', data=marital_purchase_pd, palette='viridis')
            plt.title('Total Purchases by Marital Status')
            plt.xlabel('Marital Status')
            plt.ylabel('Total Purchases')
            plt.xticks(rotation=45)
            st.pyplot()

        with col2[0]:
            st.markdown('#### Total Purchases per Product per Year')
            st.markdown("""This bar chart shows the total purchases per product per year. It allows you to see trends and changes in consumer behavior over time.""")

            # Prepare the data for the plot
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

            # Melt the data for plotting
            pivot_year = year_purchase_pd.melt(id_vars=["FiscalYear"],
                                                value_vars=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],
                                                var_name="Product",
                                                value_name="TotalSales")

            # Plot the data
            plt.figure(figsize=(10, 6))  # Increased size for better visibility
            sns.barplot(x="FiscalYear", y="TotalSales", data=pivot_year, hue="Product", palette='viridis')

            plt.title('Total Purchases per Product per Year')
            plt.xlabel('Fiscal Year')
            plt.ylabel('Total Sales')
            plt.xticks(rotation=45)
            st.pyplot()

        # Third row for the pie chart with larger size
        col3 = st.columns(1)  # One column for the pie chart

        with col3[0]:
            st.markdown('#### Total Sales Distribution by Year')
            st.markdown("""This pie chart shows the distribution of total sales by year, helping to visualize the contribution of each year to overall sales.""")

            # Sum all purchases per year for pie chart
            year_purchase_pd = clean_pd.groupby(clean_pd['Dt_Customer'].dt.year).apply(
                lambda x: x['MntWines'].sum() + x['MntFruits'].sum() + x['MntMeatProducts'].sum() +
                          x['MntFishProducts'].sum() + x['MntSweetProducts'].sum() + x['MntGoldProds'].sum()
            ).reset_index(name='TotalSales')

            # Rename the year column to "FiscalYear"
            year_purchase_pd.rename(columns={'Dt_Customer': 'FiscalYear'}, inplace=True)

            # Plot the pie chart
            plt.figure(figsize=(10, 6))  # Increased size for better visibility
            plt.pie(year_purchase_pd['TotalSales'], labels=year_purchase_pd['FiscalYear'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))

            plt.title('Total Sales Distribution by Year')
            st.pyplot()

    else:
        st.warning("Cleaned dataset (clean_pd) is not available! Please run the Data Cleaning page first.")


# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Ensure that clean_pd is available in the session state
    if 'clean_pd' in st.session_state:
        kmeans_pd = st.session_state.clean_pd  # Access it from session state
    else:
        st.error("Cleaned data is not available. Please run the Data Cleaning page first.")
        kmeans_pd = None

    if kmeans_pd is not None:
        # Label Encoder since KMeans does not allow non-numeric values
        columns_to_encode = ['Year_Birth', 'Marital_Status', 'Education', 'Dt_Customer']
        label_encoders = {}

        # Encode categorical columns
        for col in columns_to_encode:
            le = LabelEncoder()
            kmeans_pd[col] = le.fit_transform(kmeans_pd[col])
            label_encoders[col] = le  # Store the encoder for possible later use
        
        # Scaling for K-Means clustering
        kmeans_df = kmeans_pd[['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Dt_Customer', 'Income', 
                               'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
                               'MntGoldProds']]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(kmeans_df)

        # Elbow method to determine the optimal number of clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled)
            wcss.append(kmeans.inertia_)

        # Plot the Elbow Method to find the optimal K (number of clusters)
        st.subheader("Elbow Method for Optimal K")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.xticks(range(1, 11))
        plt.grid()
        st.pyplot()

        # Using the optimal number of clusters based on the elbow method
        optimal_k = 3  # You can modify this if a different number of clusters is chosen based on the plot
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_pd['Cluster'] = kmeans.fit_predict(scaled)

        # Show cluster summary (average features per cluster)
        cluster_summary = kmeans_pd.groupby('Cluster').mean()
        st.subheader("Cluster Summary")
        st.write(cluster_summary)

        # Decode the columns back to their original labels
        kmeans_pd['Year_Birth'] = label_encoders['Year_Birth'].inverse_transform(kmeans_pd['Year_Birth'])
        kmeans_pd['Marital_Status'] = label_encoders['Marital_Status'].inverse_transform(kmeans_pd['Marital_Status'])
        kmeans_pd['Education'] = label_encoders['Education'].inverse_transform(kmeans_pd['Education'])
        kmeans_pd['Dt_Customer'] = label_encoders['Dt_Customer'].inverse_transform(kmeans_pd['Dt_Customer'])
        
        # Show the decoded DataFrame
        st.write(kmeans_pd)

        # Visualize the clusters in a bar chart showing average spending per product by cluster
        pivot_cluster = cluster_summary.melt(id_vars="Income",
                                             value_vars=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                                                         'MntSweetProducts', 'MntGoldProds'],
                                             var_name="Product", value_name="TotalSales")
        st.subheader("Cluster Summary Showing Income and Product Spending")
        plt.figure(figsize=(12, 6))
        sns.barplot(data=pivot_cluster, x="Income", y="TotalSales", hue="Product", palette="viridis")
        plt.title("Cluster Summary Showing Income and Product Spending")
        plt.xlabel("Cluster Mean Income")
        plt.ylabel("Average Spending")
        plt.legend(title="Product")
        st.pyplot()

        # Model training: Random Forest Regressor to predict spending on products
        classify_pd = kmeans_pd[['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 
                                 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                                 'MntSweetProducts', 'MntGoldProds']]
        classify_pd = pd.get_dummies(classify_pd, columns=['Marital_Status', 'Education'], drop_first=True)
        
        # Split the dataset into features and target
        X = classify_pd.drop(['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], axis=1)
        y = classify_pd[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest Regressor
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display feature importances
        feature_importances = {}
        individual_feature_importance_list = []
        overall_feature_importances = {feature: 0 for feature in X.columns}
        
        # Calculate feature importance for each target
        for i, target in enumerate(y.columns):
            importances = model.estimators_[i].feature_importances_
            feature_importances[target] = dict(zip(X.columns, importances))
            for feature, importance in zip(X.columns, importances):
                overall_feature_importances[feature] += importance
        
        # Average feature importance
        num_targets = len(y.columns)
        for feature in overall_feature_importances:
            overall_feature_importances[feature] /= num_targets

    else:
        st.error("K-means cannot run because the data is missing. Please check the data cleaning page.")

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Ensure that clean_pd is available in the session state
    if 'clean_pd' in st.session_state:
        classify_pd = st.session_state.clean_pd[['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 
                                                 'Education', 'Income', 'Recency', 'MntWines', 'MntFruits', 
                                                 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
                                                 'MntGoldProds']]  # Extract relevant columns
    else:
        st.error("Cleaned data is not available. Please run the Data Cleaning page first.")
        classify_pd = None

    if classify_pd is not None:
        # One-hot encoding for non-numerical columns (not accepted by the model)
        classify_pd = pd.get_dummies(classify_pd, columns=['Marital_Status', 'Education'], drop_first=True)

        # Define X as features (customer description) and y as targets (product columns)
        X = classify_pd.drop(['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                              'MntSweetProducts', 'MntGoldProds'], axis=1)
        y = classify_pd[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                         'MntSweetProducts', 'MntGoldProds']]

        # Use 30% as test data, 70% as training dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the model
        model = RandomForestRegressor(random_state=42)

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Predict spending on the test set
        y_pred = model.predict(X_test)

        # Store y_test and mae_values in session_state for access on the conclusion page
        st.session_state.y_test = y_test
        st.session_state.mae_values = pd.DataFrame(abs(y_test - y_pred), columns=y.columns)  # Calculate MAE values

        # Determine the importance of each customer description to the products
        feature_importances = {}
        individual_feature_importance_list = []
        overall_feature_importances = {feature: 0 for feature in X.columns}

        # Loop through each target model and get feature importances
        for i, target in enumerate(y.columns):
            # Get feature importances from the individual model for each target
            importances = model.estimators_[i].feature_importances_

            # Store the importances in the dictionary for each target
            feature_importances[target] = dict(zip(X.columns, importances))

            # Add to the overall feature importance
            for feature, importance in zip(X.columns, importances):
                overall_feature_importances[feature] += importance

            # Append the importances to the individual list for later display
            for feature, importance in zip(X.columns, importances):
                individual_feature_importance_list.append({"Target": target, "Feature": feature, "Importance": importance})

        # Average the feature importances by dividing by the number of targets
        num_targets = len(y.columns)
        for feature in overall_feature_importances:
            overall_feature_importances[feature] /= num_targets

        # Display feature importances for each target
        individual_feature_importance_df = pd.DataFrame(individual_feature_importance_list)
        st.subheader("Feature Importances for Each Target")
        st.write(individual_feature_importance_df)

        # For overall importances, sort and display
        sorted_overall_importances = sorted(overall_feature_importances.items(), key=lambda x: x[1], reverse=True)
        overall_feature_importance_df = pd.DataFrame(sorted_overall_importances, columns=["Feature", "Importance"])
        st.subheader("Overall Feature Importance")
        st.write(overall_feature_importance_df)

        # Create pie chart for overall feature importance
        threshold = 0.012
        filtered_data = overall_feature_importance_df[overall_feature_importance_df["Importance"] >= threshold]

        if len(overall_feature_importance_df[overall_feature_importance_df["Importance"] < threshold]) > 0:
            others_sum = overall_feature_importance_df[overall_feature_importance_df["Importance"] < threshold]["Importance"].sum()
            others_row = pd.DataFrame({"Feature": "Others", "Importance": [others_sum]})
            filtered_data = pd.concat([filtered_data, others_row], ignore_index=True)

        st.subheader("Feature Importance (Pie Chart)")
        plt.figure(figsize=(8, 8))
        plt.pie(filtered_data['Importance'], labels=filtered_data['Feature'], autopct='%1.0f%%', startangle=140, colors=sns.color_palette('viridis'))
        plt.title('Overall Feature Importance')
        st.pyplot()

        # Decode columns for X_test
        X_test['Marital_Status'] = X_test[['Marital_Status_YOLO', 'Marital_Status_Together', 'Marital_Status_Married', 
                                           'Marital_Status_Widow', 'Marital_Status_Divorced', 'Marital_Status_Alone', 
                                           'Marital_Status_Single']].idxmax(axis=1).str.replace('Marital_Status_', '')
        X_test['Education'] = X_test[['Education_PhD', 'Education_Master', 'Education_Graduation', 'Education_Basic']].idxmax(axis=1).str.replace('Education_', '')

        # Drop the one-hot encoded columns
        X_test = X_test.drop(['Marital_Status_YOLO', 'Marital_Status_Together', 'Marital_Status_Married', 
                              'Marital_Status_Widow', 'Marital_Status_Divorced', 'Marital_Status_Alone', 
                              'Marital_Status_Single', 'Education_PhD', 'Education_Master', 'Education_Graduation', 
                              'Education_Basic'], axis=1)

        # Columns for predictions
        target_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        mae_list = []
        results_list = []

        for i in range(len(y_test)):
            data_point_result = {'Data_Point': i + 1}
            data_point_result.update(X_test.iloc[i].to_dict())

            for j, col in enumerate(target_columns):
                actual = y_test.iloc[i, j]
                predicted = y_pred[i, j]
                absolute_error = abs(actual - predicted)
                mae_list.append(absolute_error)

                data_point_result[f"{col}_Actual"] = actual
                data_point_result[f"{col}_Predicted"] = predicted
                data_point_result[f"{col}_Absolute_Error"] = absolute_error

            results_list.append(data_point_result)

        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)
        st.subheader("Prediction Results")
        st.write(results_df)

        # Display scatter plot of actual vs predicted values
        melted_df = results_df.melt(id_vars=['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 'Recency'],
                                    value_vars=[f'{col}_Predicted' for col in target_columns] + [f'{col}_Actual' for col in target_columns],
                                    var_name='Category_Variable', value_name='Value')

        melted_df['Category'] = melted_df['Category_Variable'].apply(lambda x: x.split('_')[0])
        melted_df['Type'] = melted_df['Category_Variable'].apply(lambda x: x.split('_')[1])

        pivoted_df = melted_df.pivot_table(index=['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 'Recency', 'Category'],
                                           columns='Type', values='Value').reset_index()

        st.subheader("Actual vs Predicted Values (Scatter Plot)")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Actual", y="Predicted", hue="Category", data=pivoted_df, s=100, palette="deep")
        plt.plot([pivoted_df["Actual"].min(), pivoted_df["Actual"].max()],
                 [pivoted_df["Actual"].min(), pivoted_df["Actual"].max()], 'k--', lw=2)
        plt.title("Actual vs. Predicted Values Across Multiple Categories")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        st.pyplot()

        # Correlation heatmap for actual values
        feature_columns = ['Income']
        target_columns = ['MntWines_Actual', 'MntFruits_Actual', 'MntMeatProducts_Actual', 'MntFishProducts_Actual', 'MntSweetProducts_Actual', 'MntGoldProds_Actual']

        correlation_matrix = results_df[feature_columns + target_columns].corr()

        st.subheader("Correlation Heatmap (Actual Values)")
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap for Actual Values")
        st.pyplot()

        # Correlation heatmap for predicted values
        target_columns = ['MntWines_Predicted', 'MntFruits_Predicted', 'MntMeatProducts_Predicted', 
                          'MntFishProducts_Predicted', 'MntSweetProducts_Predicted', 'MntGoldProds_Predicted']

        correlation_matrix = results_df[feature_columns + target_columns].corr()

        st.subheader("Correlation Heatmap (Predicted Values)")
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap for Predicted Values")
        st.pyplot()

        # Mean Absolute Error (MAE) table
        mae_table = pd.DataFrame(mae_list, columns=["Absolute_Error"])
        mae_table["Mean_Absolute_Error"] = mae_table["Absolute_Error"].mean()

        st.subheader("Mean Absolute Error (MAE)")
        st.write(mae_table)
        
    else:
        st.error("Data is missing, cannot run prediction.")

# Conclusion Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Ensure that predictions (y_test and mae_values) are available
    if 'y_test' in st.session_state and 'mae_values' in st.session_state:
        y_test = st.session_state.y_test
        mae_values = st.session_state.mae_values
        
        target_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        accuracy_data = []
        weighted_accuracy_sum = 0
        total_weight = 0

        # Calculate and store percentage accuracy for each category
        for col in target_columns:
            avg_actual = y_test[col].mean()  # Average actual value for the category
            percentage_accuracy = (1 - (mae_values[col] / avg_actual)) * 100  # Percentage accuracy
            accuracy_data.append({"Category": col, "Percentage Accuracy": percentage_accuracy})

            # Add to the weighted accuracy sum and total weight
            weighted_accuracy_sum += percentage_accuracy * avg_actual
            total_weight += avg_actual

        # Calculate the overall percentage accuracy
        overall_percentage_accuracy = weighted_accuracy_sum / total_weight
        accuracy_data.append({"Category": "Overall Percentage Accuracy", "Percentage Accuracy": overall_percentage_accuracy})

        # Convert accuracy data to DataFrame and display it
        accuracy_df = pd.DataFrame(accuracy_data)
        st.subheader("Percentage Accuracy Per Category")
        st.write(accuracy_df)

        # Ensure no NaN values in Percentage Accuracy before plotting
        prod_acc_df = accuracy_df[accuracy_df['Category'].isin(target_columns)]

        # Convert to numeric and drop any NaN values
        prod_acc_df['Percentage Accuracy'] = pd.to_numeric(prod_acc_df['Percentage Accuracy'], errors='coerce')
        prod_acc_df = prod_acc_df.dropna(subset=['Percentage Accuracy'])

    else:
        st.error("Prediction results or MAE values are missing, cannot calculate accuracy.")

    st.markdown("""
    * We used the People variables (it came with the dataset), to create the Machine Learning prediction results. The data set used was Customer Prediction Analysis by Akash Patel.
    * The data was thoroughly cleaned from any null variables that may cause biased predictions toward certain demographics. Upon cleaning the group noticed variables that had a small sample, which could be turned into biased predictions.
    * Exploratory Data Analysis (EDA) uses heatmaps and bar charts for visualization of data on the most prominent variables.
    * The Machine Learning model uses K-means Clustering to predict the possible customer product trends.
    * The Predictions had an overall prediction of 49.07%
        """)
