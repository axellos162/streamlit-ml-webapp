import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

# Load the Iris dataset
iris_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Set the page title
st.set_page_config(page_title='Iris EDA Dashboard')

# Define the sidebar
st.sidebar.title('Explore Iris Dataset')
plot_types = ['Histogram', 'Boxplot', 'Violinplot', 'Scatterplot', 'Pairplot', 'Heatmap']
plot_type = st.sidebar.selectbox('Select a plot type', plot_types)

# Define the main page content
st.title('Exploratory Data Analysis of Iris Dataset')

# Add a brief description of the dataset
st.write('The Iris dataset is a classic example of multivariate data analysis. It contains measurements of the sepal length, sepal width, petal length, and petal width of three different species of iris flowers.')

# Show a summary of the dataset
st.write('## Summary of the Dataset')
st.write(iris_df.describe())

# Show a correlation matrix
st.write('## Correlation Matrix')
corr = iris_df.corr()
st.write(corr.style.background_gradient(cmap='coolwarm'))

# Show a histogram of each feature
if plot_type == 'Histogram':
    st.write('## Histogram of Each Feature')
    for feature in iris_df.columns[:-1]:
        fig = px.histogram(iris_df, x=feature, color='species', marginal='rug')
        st.plotly_chart(fig)

# Show a boxplot of each feature
if plot_type == 'Boxplot':
    st.write('## Boxplot of Each Feature')
    for feature in iris_df.columns[:-1]:
        fig = px.box(iris_df, x='species', y=feature, points='all')
        st.plotly_chart(fig)

# Show a violinplot of each feature
if plot_type == 'Violinplot':
    st.write('## Violinplot of Each Feature')
    for feature in iris_df.columns[:-1]:
        fig = px.violin(iris_df, x='species', y=feature, box=True, points='all')
        st.plotly_chart(fig)

# Show a scatterplot of each feature
if plot_type == 'Scatterplot':
    st.write('## Scatterplot of Each Feature')
    for feature1 in iris_df.columns[:-1]:
        for feature2 in iris_df.columns[:-1]:
            if feature1 != feature2:
                fig = px.scatter(iris_df, x=feature1, y=feature2, color='species', marginal_x='rug', marginal_y='rug')
                st.plotly_chart(fig)

# Show a pairplot of all features
if plot_type == 'Pairplot':
    st.write('## Pairplot of All Features')
    fig = px.scatter_matrix(iris_df, dimensions=iris_df.columns[:-1], color='species')
    st.plotly_chart(fig)

# Show a heatmap of the correlation matrix
if plot_type == 'Heatmap':
    st.write('## Heatmap of the Correlation Matrix')
    fig = sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(fig.figure)
