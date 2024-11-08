# Data Science Codes

This repository contains various datasets and code used for data science tasks, including exploratory data analysis (EDA), regression analysis, outlier detection, hypothesis testing, and more. The project consists of several Jupyter notebooks and Python scripts for different exercises and analyses. It also includes datasets from multiple domains for modeling and analysis.

## Table of Contents

- [Overview](#overview) 
- [Datasets](#datasets)
- [How to Run the Code](#how-to-run-the-code)
- [Dependencies](#dependencies)

## Overview

-   **Exploratory Data Analysis (EDA)**: Analyzing datasets using visualization techniques and summary statistics.
-   **Outlier Detection**: Identifying and handling outliers in datasets.
-   **Regression Analysis**: Performing linear and logistic regression models.
-   **Hypothesis Testing**: Performing statistical tests such as t-tests and z-tests.

## Datasets

The following datasets are included in the repository:

1.  **Titanic Dataset** (train.csv, test.csv, gender_submission.csv): Contains data on passengers aboard the Titanic, including their survival status and various demographic features.
    
2.   **Air Quality Time Series Dataset** (city_hour.csv, city_day.csv, station_day.csv): Contains hourly and daily data for air quality measurements from various stations, including information on pollutants and environmental conditions.
    
3.   **Wholesale Customers Data** (Wholesale customers data.csv): Contains data on wholesale customers' buying behavior, segmented by product categories (e.g., fresh, milk, grocery).
    
4.  **Heart Disease Dataset** (heart.csv): Contains medical data on patients, used for predicting heart disease risk based on various health attributes.
    
5. **Breast Cancer Dataset** (breast_cancer_data.csv): Contains data on breast cancer patients, used for classification tasks to predict the malignancy of tumors.
6. **Diabetes Dataset** (diabetes.csv): Contains data on diabetes patients, including medical features used for classification and prediction of diabetes outcomes.
7.  **Housing Data** (housing.csv): Contains information on housing prices and features, including the number of rooms, location, and other factors affecting home prices.
8. **Actors Dataset** (actors.csv): Contains data about various actors, potentially including their names, genres, and other film-related attributes.
9.  **Adult  Dataset** (adult.csv): Contains demographic data used to predict whether a person earns above or below a certain threshold based on attributes such as age, education, and occupation.
10.  **Wholesale Customer Behavior** (wholesale_customers.csv): Contains data on wholesale customer demographics and purchasing patterns.
11. **Various Other Datasets**: Includes additional datasets related to various domains, useful for regression, classification, and clustering tasks.

The datasets are located in the `References-Otherclass/DataRepo/` folder.

## How to Run the Code

### Prerequisites:

Make sure you have the following installed:

-   Python 3.x
-   Jupyter Notebook
-   Git (for version control)

### Installation of Dependencies:

1.  Clone the repository to your local machine:
    
    
    `git clone https://github.com/ByteBender24/DAV_Codes.git`
    `cd DAV_Codes` 
    
2.  Create a virtual environment (optional but recommended):

    
    ``python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate` `` 
    
3.  Install the necessary dependencies:

    `pip install -r requirements.txt` 
    

### Running the Notebooks:

To run any Jupyter notebook (e.g., `StudentPerformance.ipynb`):

1.  Launch Jupyter Notebook:

    
    `jupyter notebook` 
    
2.  Open the desired notebook from the Jupyter interface in your browser and run the cells.
    

----------

## Dependencies

The project requires the following Python libraries:

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scikit-learn
-   statsmodels
-   plotly
-   jupyter

These dependencies can be installed using the `requirements.txt` file, or manually using `pip`.

----------



### Additional Notes:

-   Large files (e.g., `city_hour.csv`) exceed GitHub's recommended size limits. It is recommended to use [Git LFS](https://git-lfs.github.com/) for these files.
-   Some notebooks are checkpoints, which may contain incremental progress during analysis and can be ignored unless needed for historical reference.
