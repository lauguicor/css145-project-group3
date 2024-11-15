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
    st.write(dataset_df)

    # Your content for your DATASET page goes here

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
            heatmap_pd = clean_pd[['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
            correlation_matrix = heatmap_pd.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation on Income vs Product Spending")
            st.pyplot()

        with col[1]:
            st.markdown('#### Total Product Sales')
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

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
