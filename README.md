# Customer Segmentation Using K-Means Clustering

This project demonstrates the application of K-Means Clustering, an unsupervised machine learning algorithm, to identify distinct customer segments based on their purchasing behavior. By grouping customers into clusters with similar traits, businesses can better understand customer preferences and tailor their marketing strategies accordingly.

## Table of Contents
 - [Overview](#project-overview)
 - [Features](#features)
 - [Installation](#installation)
 - [Usage](#usage)
 - [Examples](#examples)
 - [Result](#result)
 

## Project Overview
The goal of this project is to analyze customer purchasing data from a restaurant's satisfaction survey. The dataset includes various features that describe customer behavior, and K-Means clustering is used to group customers into segments. Visualizations are provided to help interpret the results, including scatter plots, silhouette plots, centroid plots, and more.


## Features

- Automated Feature Detection: The code automatically detects numeric columns from the dataset to use for clustering. You can also define specific keywords to filter relevant columns.
- Elbow Method: Determines the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS).
- Cluster Visualization: A scatter plot is generated to visualize the customer segments.
- Silhouette Plot: Evaluates the quality of the clustering by showing the silhouette coefficient for each cluster.
- Centroid Plot: Highlights the centroids of each cluster on the scatter plot.
- Pair Plot: Provides a detailed visualization of feature interactions for each cluster.
- Heatmap: Visualizes the cluster centers using a heatmap.
- Box Plot: Illustrates the distribution of each feature across clusters.


## Installation

The following Python libraries are used in this project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

To install these dependencies,run:

bash
  pip install pandas numpy scikit-learn matplotlib seaborn


## Usage
1. Clone the repository:

bash
git clone https://github.com/GDoomstar/Data-Driven-Segmentation.git
cd Data-Driven-Segmentation


2. Datset:

You may downlaod a dataset related to customer segmentation from kaggle or use any of the datasets provided in the repository.

3. Run the script:

bash
python segmodified.py



4. Visualize Results:
The script will generate multiple visualizations, including:

- Elbow Method: Helps to determine the optimal number of clusters.
- Scatter Plot: Displays customer segments.
- Silhouette Plot: Measures the consistency within clusters.
- Cluster Centroids Plot: Marks the center of each cluster.
- Pair Plot, Box Plot, Heatmap: Offers a variety of views to explore cluster features.


## Examples
1. Elbow Method Graph
- Purpose: This graph helps to determine the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS) against the number of clusters. The "elbow point" in the graph indicates where the addition of more clusters does not significantly improve the clustering.

- Formula:
  
  ![image](https://github.com/user-attachments/assets/5bccd7c4-d20b-424a-aafe-53e890744eac)


  where Ci is the set of points in cluster i, and μi​ is the centroid of cluster i.
  
  ![image](https://github.com/user-attachments/assets/e773d0be-4960-4bb6-9403-3ddb1dd62929)


- Interpretation: The number of clusters at the "elbow" point (where the graph starts to level off) is considered optimal. In your case, it's likely around 3 clusters based on the shape of the curve.

2. Scatter Plot of Clusters

- Purpose: This plot visualizes the customer segments based on the selected features (e.g., income and spending score), with each point representing a customer and the colors showing which cluster they belong to. 

  ![image](https://github.com/user-attachments/assets/21d6fb02-9b25-438c-82a8-2954cbf35d6f)


- Interpretation: By examining the scatter plot, you can see how customers are grouped into different segments based on their features. Clusters that are well-separated indicate that the algorithm is successful in distinguishing different groups.

- X-axis and Y-axis: The plot displays the two selected features, such as Annual Income and Spending Score.

3. Silhouette Plot

- Purpose: The silhouette plot visualizes how well each point fits into its assigned cluster, compared to other clusters. Higher silhouette values mean that points are better clustered.

- Formula: For a data point i, the silhouette coefficient is defined as:

    ![image](https://github.com/user-attachments/assets/9e38bd24-b4b8-4ec2-84df-cde85b627acf)


   where: a(i) is the average distance from i to the other points in its own cluster.
          b(i) is the minimum average distance from i to points in the nearest cluster.

  ![image](https://github.com/user-attachments/assets/205f6194-2836-4d5e-b160-0fe7d78267c7)

    
- Interpretation: The red dashed line in the silhouette plot represents the average silhouette score. Points above the line have better clustering quality. The closer the value is to 1, the better clustered the point is.

4. Cluster Centroid Plot

- Purpose: This scatter plot shows the clusters along with their centroids (the centers of the clusters) marked with a red 'X'. Centroids are the average position of all the points in a cluster.

- Formula:

  ![image](https://github.com/user-attachments/assets/1497534d-02f7-4372-a9ef-af4a0ec3e0e3)

  where μi is the centroid of cluster i, and Ci is the set of points in cluster i.

  ![image](https://github.com/user-attachments/assets/a786c633-8163-4d33-a798-8f0dfeeaf9e0)


- Interpretation: The centroids represent the "center" of each cluster, and they help in identifying the characteristics of each customer segment.

5. Pair Plot

- Purpose: The pair plot shows the relationships between each pair of selected features, color-coded by cluster. It's a grid of scatter plots, with each plot comparing two different features.

  ![image](https://github.com/user-attachments/assets/dad08832-33ed-45ce-86a8-3b7a09cbf848)


- Interpretation: By examining these plots, you can observe correlations between different features and how the clusters are separated based on these pairs of features.

6. Heatmap of Cluster Centers

- Purpose: This heatmap shows the values of the cluster centroids for each feature. It's useful for understanding how the average values of features differ between clusters.

  ![image](https://github.com/user-attachments/assets/81d4d2bf-136a-4601-8f26-3a2384d95000)


- Interpretation: Darker or lighter values represent higher or lower values for each feature within each cluster, helping to characterize each customer segment.

7. Box Plot by Cluster

-Purpose: The box plot shows the distribution of values for each feature within each cluster. It helps in identifying how different the features are across clusters.

   ![image](https://github.com/user-attachments/assets/ac2cc537-6916-4ded-a874-ef74a5e68e1a)


- Interpretation: You can observe the median (middle line), interquartile range (box), and outliers (points outside the whiskers) for each feature across different clusters.

## Result
The optimal number of clusters was determined using the Elbow Method.Customer clusters were visualized based on selected features, revealing patterns in spending, income, and satisfaction.Cluster centroids and silhouette scores were used to validate the clustering results.
    

Feel free to open an issue or submit a pull request if you'd like to contribute.
