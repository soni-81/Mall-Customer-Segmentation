import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

# Configure the app
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "About", "Customer Segmentation"])

if page == "Home":
    # Page title and header
    st.title("Welcome to the Mall Customer Segmentation App 🎯")
    st.markdown("""
    *Discover the power of customer segmentation with machine learning!*
    
    This application helps businesses understand their customers better by grouping them into meaningful clusters based on shared characteristics.
    """)

    # Adding an eye-catching image/banner
    st.image(
        "https://www.marketingevolution.com/hs-fs/hubfs/customer-segmentation.jpg?width=1650&name=customer-segmentation.jpg",
       
        caption="Understand your customers and make data-driven decisions."
    )

    # Section: What is Customer Segmentation
    st.markdown("## 🔍 What is Customer Segmentation?")
    st.markdown("""
    Customer segmentation is the process of dividing customers into groups based on their behaviors, preferences, and traits. 
    By understanding these groups, businesses can:
    - Tailor marketing strategies to specific customer needs.
    - Improve customer satisfaction and retention.
    - Enhance revenue by targeting the right audience.
    """)

    # Section: Why Use This App
    st.markdown("## 🚀 Why Use This App?")
    st.markdown("""
    - *Interactive Visualizations:* Understand your customers with dynamic and easy-to-understand charts.
    - *Custom Clustering:* Choose your own features and determine the number of clusters.
    - *Export Data:* Download your segmented data for further analysis.
    - *User-Friendly Design:* No coding skills required – just upload your dataset and start exploring!
    """)

    # Section: How It Works
    st.markdown("## 🛠️ How It Works")
    steps = """
    1. *Upload Your Dataset:* Click on the Customer Segmentation tab and upload your CSV file.
    2. *Customize Clustering:* Select features and the number of clusters to analyze.
    3. *View Results:* Explore insightful visualizations like scatter plots, bar charts, and more.
    4. *Export Data:* Download the segmented dataset with a single click.
    """
    st.markdown(steps)

    

# About Page
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    This app demonstrates how machine learning can be applied for customer segmentation using K-Means clustering.
    
    *Key Objectives*:
    - Group customers into clusters based on their similarities.
    - Provide actionable insights for better decision-making.

    *Features*:
    - Interactive visualizations using Plotly.
    - Dynamic selection of features and cluster count.
    - Easy-to-understand insights for each graph.

    *Tools & Technologies*:
    - Python
    - Streamlit (Frontend)
    - Plotly (Graphs and Visualizations)
    - Scikit-learn (K-Means Clustering)
    """)
    
# customer segmenation
elif page =="customer segmenttion":
    
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])
#Load dataset
#if uploaded_file is not None:
   
data = pd.read_csv("Mall_Customers.csv")
st.write("### Dataset view")

# Feature Selection
num_clusters = st.sidebar.slider("Select the number of clusters", 2, 10, value=3)
columns = st.multiselect("Select features for clustering:", data.columns)
if len(columns) >= 2:
            X = data[columns].values

            # Elbow Method
            st.write("### Elbow Method")
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            fig = px.line(x=range(1, 11), y=wcss, labels={"x": "Number of Clusters", "y": "WCSS"},
                        title="Elbow Method")
            st.plotly_chart(fig)
            st.markdown("""
            *Insight*: The Elbow Method helps determine the optimal number of clusters. The "elbow point" is where the WCSS (within-cluster sum of squares) curve starts to flatten. For this dataset, you can select the number of clusters at this point.
            """)

            # K-Means Clustering
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
            y_kmeans = kmeans.fit_predict(X)

            # Scatter Plot of Clusters
            st.write(f"### Scatter Plot (k={num_clusters})")
            cluster_data = pd.DataFrame(X, columns=columns)
            cluster_data['Cluster'] = y_kmeans
            fig = px.scatter(cluster_data, x=columns[0], y=columns[1], color='Cluster', title="Cluster Visualization",
                            color_continuous_scale=px.colors.sequential.Viridis)
            st.plotly_chart(fig)
            st.markdown(f"""
            *Insight*: Customers have been grouped into {num_clusters} clusters based on their similarities in the selected features. Each cluster represents a distinct group with shared characteristics. For example:
            - *Cluster 1* might represent high-spending customers.
            - *Cluster 2* could include customers with average income and spending.
            """)

            # 3D Scatter Plot (if more than 2 features are selected)
            if len(columns) >= 3:
                st.write("### 3D Scatter Plot")
                fig = px.scatter_3d(cluster_data, x=columns[0], y=columns[1], z=columns[2], color='Cluster',
                                    title="3D Cluster Visualization")
                st.plotly_chart(fig)
                st.markdown("""
                *Insight*: The 3D scatter plot provides a better visualization of clusters when analyzing three features simultaneously. This can help identify multi-dimensional relationships among customer groups.
                """)

            # Cluster Distribution
            st.write("### Cluster Size Distribution")
            cluster_sizes = cluster_data['Cluster'].value_counts().reset_index()
            cluster_sizes.columns = ['Cluster', 'Count']
            fig = px.bar(cluster_sizes, x='Cluster', y='Count', color='Cluster', title="Cluster Sizes")
            st.plotly_chart(fig)
            st.markdown("""
            *Insight*: The bar chart shows the size of each cluster. A larger cluster indicates more customers with similar traits, while smaller clusters may represent niche groups.
            """)

            # Heatmap of Correlations
            st.write("### Feature Correlation Heatmap")
            corr = data[columns].corr()
            fig = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
            st.plotly_chart(fig)
            st.markdown("""
            *Insight*: The heatmap shows the correlation between features. Strong correlations (closer to 1 or -1) indicate that these features are related. For example, if income is highly correlated with spending, targeting high-income customers may lead to higher sales.
            """)

            # Pie Chart of Clusters
            st.write("### Cluster Proportion")
            fig = px.pie(cluster_sizes, names='Cluster', values='Count', title="Cluster Proportion")
            st.plotly_chart(fig)
            st.markdown("""
            *Insight*: The pie chart provides a quick visual representation of the proportion of customers in each cluster. It helps identify which customer groups dominate.
            """)

            # Clustered Data
            st.write("### Clustered Data")
            clustered_data = data.copy()
            clustered_data['Cluster'] = y_kmeans
            st.dataframe(clustered_data)
            st.markdown("""
            *Insight*: This table includes the original data with the assigned cluster for each customer. You can analyze customer traits within each cluster to develop targeted strategies.
            """)

            # Download Clustered Data
            st.write("### Download Clustered Data")
            csv = clustered_data.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv, file_name='clustered_data.csv', mime='text/csv')
        