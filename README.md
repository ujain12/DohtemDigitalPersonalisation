Dohtem Digital Personalisation Project
Overview

This project analyzes customer behavior in an e-commerce environment and builds a data-driven personalisation framework to improve customer engagement, retention, and revenue.

Using machine learning and behavioral analytics, the project identifies customer segments, detects churn risk, and proposes targeted personalisation strategies for each segment.

The goal is to demonstrate how data science can be applied to improve the digital customer experience and marketing effectiveness.

Business Problem

Many e-commerce companies struggle with:

• High customer churn
• Poor understanding of customer behavior
• Generic marketing strategies
• Lack of data-driven personalisation

This project helps answer the following questions:

Which customers are likely to churn?

What behavioral patterns exist among customers?

How can companies personalise marketing and retention strategies?

Which customer segments generate the most value?

Project Pipeline

The project follows a structured data science workflow.

1 Data Cleaning

The dataset is cleaned and prepared for analysis.

Steps include:
• Handling missing values
• Encoding categorical variables
• Feature scaling
• Outlier detection

Tools used:

Pandas

NumPy

Scikit-learn

2 Exploratory Data Analysis (EDA)

EDA helps understand customer behavior patterns.

Key insights explored:

• Customer tenure distribution
• Order frequency
• Cashback usage
• Coupon usage
• Satisfaction scores
• Churn distribution

Visualizations are created using:

Matplotlib

Seaborn

3 Dimensionality Reduction

Principal Component Analysis (PCA) is used to reduce the dataset into two dimensions for visualization.

Purpose:

• Identify hidden behavioral patterns
• Understand customer similarities
• Prepare the dataset for clustering

Output:

A 2D PCA plot that shows how customers group together.

4 Customer Segmentation

Customers are segmented using K-Means clustering.

Goal:

Group customers with similar behaviors.

Typical segments discovered:

High Value Loyal Customers

Discount Driven Customers

At Risk / Churn Customers

This segmentation helps businesses design targeted marketing strategies.

5 Churn Prediction

A machine learning model is used to predict whether a customer is likely to churn.

Algorithms used:

• Logistic Regression
• Random Forest
• Gradient Boosting

Model performance is evaluated using:

• Accuracy
• Precision
• Recall
• F1 Score
• Confusion Matrix

6 Personalisation Framework

Based on segmentation and churn prediction, the project proposes actionable marketing strategies.

Examples:

High Value Customers
• Loyalty programs
• Early access to products
• Premium offers

Discount Driven Customers
• Coupons and cashback incentives
• Flash sales

At Risk Customers
• Re-engagement campaigns
• Customer support outreach
• Personalized discounts

Technology Stack

Programming Language

Python

Libraries

Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Plotly

Deployment

Streamlit (Interactive dashboard)

Version Control

GitHub

Streamlit Application

The project includes an interactive Streamlit dashboard where users can:

• Explore customer behavior
• View PCA visualizations
• Analyze churn patterns
• Understand customer segments
• Review personalisation recommendations

The dashboard is designed to simulate how business stakeholders interact with analytics insights.

Key Insights

Some important findings from the analysis:

• Churn rate is approximately 16–17%
• Customers naturally group into 3 behavioral segments
• Engagement variables such as tenure, order count, coupons, and cashback usage strongly influence retention
• Poor service experience significantly increases churn probability

These insights help businesses design more effective retention strategies.

Business Impact

This framework helps companies:

Improve customer retention
Increase customer lifetime value
Create targeted marketing campaigns
Optimize digital customer experiences

Repository Structure
Dohtem-Personalisation-Project

data/
    ecommerce_customers.csv

notebooks/
    EDA.ipynb
    PCA_analysis.ipynb
    clustering.ipynb

app/
    streamlit_app.py

models/
    churn_model.pkl

visualizations/
    churn_distribution.png
    pca_plot.png
    cluster_plot.png

README.md
requirements.txt
Future Improvements

Possible extensions for this project include:

• Real-time recommendation systems
• Deep learning churn models
• Customer lifetime value prediction
• A/B testing framework for personalisation strategies

Author

Utkarsh Jain
MS Data Science & Business Analytics
UNC Charlotte

License

This project is for academic and educational purposes.
