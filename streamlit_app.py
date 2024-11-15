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
        col = st.columns((3, 4, 3), gap='medium')

        with col[0]:
            st.markdown('#### Correlation Heatmap')
            st.markdown("""The heatmap shows the correlation between income and spending on different product categories. Higher income is generally associated with higher spending across most categories, with some exceptions like gold products.""")
            heatmap_pd = clean_pd[['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
            correlation_matrix = heatmap_pd.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation on Income vs Product Spending")
            st.pyplot()
            
        with col[1]:
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
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Product', y='TotalSales', data=prodsales_pivot, palette='viridis')
                st.pyplot()

        with col[2]:
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
            plt.figure(figsize=(10, 6))
            sns.barplot(x=marital_purchase_pd.index, y='TotalSales', data=marital_purchase_pd, palette='viridis')
            plt.title('Total Purchases by Marital Status')
            plt.xlabel('Marital Status')
            plt.ylabel('Total Purchases')
            plt.xticks(rotation=45)
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

        # Visualize the overall feature importance
        st.subheader("Overall Feature Importance")
        importance_df = pd.DataFrame(overall_feature_importances.items(), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Overall Feature Importance')
        st.pyplot()

    else:
        st.error("K-means cannot run because the data is missing. Please check the data cleaning page.")

# Prediction Page
# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Ensure that clean_pd is available in the session state
    if 'clean_pd' in st.session_state:
        kmeans_pd = st.session_state.clean_pd  # Access it from session state
    else:
        st.error("Cleaned data is not available. Please run the Data Cleaning page first.")
        kmeans_pd = None

    if kmeans_pd is not None:
        # Label Encoding for categorical columns in the dataset
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

        # Apply KMeans clustering
        optimal_k = 3  # Set the optimal number of clusters based on previous analysis
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_pd['Cluster'] = kmeans.fit_predict(scaled)

        # Cluster summary
        cluster_summary = kmeans_pd.groupby('Cluster').mean()
        st.subheader("Cluster Summary")
        st.write(cluster_summary)

        # Reverse label encoding for readability
        kmeans_pd['Year_Birth'] = label_encoders['Year_Birth'].inverse_transform(kmeans_pd['Year_Birth'])
        kmeans_pd['Marital_Status'] = label_encoders['Marital_Status'].inverse_transform(kmeans_pd['Marital_Status'])
        kmeans_pd['Education'] = label_encoders['Education'].inverse_transform(kmeans_pd['Education'])
        kmeans_pd['Dt_Customer'] = label_encoders['Dt_Customer'].inverse_transform(kmeans_pd['Dt_Customer'])
        st.write(kmeans_pd)

        # Visualizing cluster summary with average spending
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

        classify_pd = kmeans_pd[['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
        classify_pd = pd.get_dummies(classify_pd, columns=['Marital_Status', 'Education'], drop_first=True)
        X = classify_pd.drop(['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], axis=1)
        y = classify_pd[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        feature_importances = {}
        individual_feature_importance_list = []
        overall_feature_importances = {feature: 0 for feature in X.columns}

        for i, target in enumerate(y.columns):
            importances = model.estimators_[i].feature_importances_
            feature_importances[target] = dict(zip(X.columns, importances))
            for feature, importance in zip(X.columns, importances):
                overall_feature_importances[feature] += importance

        num_targets = len(y.columns)
        for feature in overall_feature_importances:
            overall_feature_importances[feature] /= num_targets

        for target, importances in feature_importances.items():
            sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importances:
                individual_feature_importance_list.append({"Target": target, "Feature": feature, "Importance": importance})

        individual_feature_importance_df = pd.DataFrame(individual_feature_importance_list)
        st.dataframe(individual_feature_importance_df)

        sorted_overall_importances = sorted(overall_feature_importances.items(), key=lambda x: x[1], reverse=True)
        overall_feature_importance_df = pd.DataFrame(sorted_overall_importances, columns=["Feature", "Importance"])
        st.dataframe(overall_feature_importance_df)

        threshold = 0.012
        filtered_data = overall_feature_importance_df[overall_feature_importance_df["Importance"] >= threshold]
        if len(overall_feature_importance_df[overall_feature_importance_df["Importance"] < threshold]) > 0:
            others_sum = overall_feature_importance_df[overall_feature_importance_df["Importance"] < threshold]["Importance"].sum()
            others_row = pd.DataFrame({"Feature": "Others", "Importance": [others_sum]})
            filtered_data = pd.concat([filtered_data, others_row], ignore_index=True)
        plt.figure(figsize=(8, 8))
        plt.pie(filtered_data['Importance'], labels=filtered_data['Feature'], autopct='%1.0f%%', startangle=140, colors=sns.color_palette('viridis'))

        plt.title('Overall Feature Importance')
        st.pyplot(plt)

        if 'X_test' in st.session_state:
            X_test = st.session_state.X_test  # Access X_test from session state
        else:
            st.error("X_test is not available. Please load the data first.")
            X_test = None

        if X_test is not None:
            # Decode the 'Marital_Status' columns (one-hot encoded to categorical)
            marital_status_columns = ['Marital_Status_YOLO', 'Marital_Status_Together', 'Marital_Status_Married', 'Marital_Status_Widow', 
                              'Marital_Status_Divorced', 'Marital_Status_Alone', 'Marital_Status_Single']
            education_columns = ['Education_PhD', 'Education_Master', 'Education_Graduation', 'Education_Basic']
    
            # Check if all the necessary columns exist in X_test
            if all(col in X_test.columns for col in marital_status_columns):
                X_test['Marital_Status'] = X_test[marital_status_columns].idxmax(axis=1).str.replace('Marital_Status_', '')
                st.write("Successfully decoded 'Marital_Status' columns!")
            else:
                st.error("Some 'Marital_Status' columns are missing in X_test.")
    
            if all(col in X_test.columns for col in education_columns):
                X_test['Education'] = X_test[education_columns].idxmax(axis=1).str.replace('Education_', '')
                st.write("Successfully decoded 'Education' columns!")
            else:
                st.error("Some 'Education' columns are missing in X_test.")
    
            # Drop the one-hot encoded columns
            columns_to_drop = marital_status_columns + education_columns
            X_test = X_test.drop(columns=columns_to_drop, axis=1)

            # Display the updated DataFrame
            st.write("Updated X_test DataFrame:")
            st.dataframe(X_test.head())  # Display the first few rows for preview

        else:
            st.error("X_test DataFrame is not available.")
# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
    st.markdown("""This is the content area""")
