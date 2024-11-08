import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("adult.csv")

# Data Cleaning
df.columns = df.columns.str.replace('-', '_')  # Standardize column names
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert categorical columns to 'category' dtype
categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 
                       'relationship', 'race', 'gender', 'native_country', 'income']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Visualizing pairplot and boxplot for numeric columns
numeric_columns = ['age', 'fnlwgt', 'educational_num', 'capital_gain', 'capital_loss', 'hours_per_week']
sb.pairplot(df[numeric_columns], diag_kind='kde', kind='scatter')
plt.show()

# Boxplot before outlier handling
plt.figure(figsize=(10, 6))
df[numeric_columns].boxplot()
plt.title("Boxplot of Numeric Columns Before Outlier Handling")
plt.show()

# Outlier Detection and Handling using IQR method
for col in numeric_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    
    # Cap the outliers
    df[col] = np.where(df[col] > upper, upper, df[col])
    df[col] = np.where(df[col] < lower, lower, df[col])

# Boxplot after outlier handling
plt.figure(figsize=(10, 6))
df[numeric_columns].boxplot()
plt.title("Boxplot of Numeric Columns After Outlier Handling")
plt.show()

# Shapiro-Wilk Test for normality
print("\nShapiro-Wilk Test Results:")
for col in numeric_columns:
    stat, p = shapiro(df[col])
    print(f"{col} - Statistic: {stat}, p-value: {p}")
    if p > 0.05:
        print(f"{col} is normally distributed.\n")
    else:
        print(f"{col} is not normally distributed.\n")

# Kolmogorov-Smirnov Test for normality
print("\nKolmogorov-Smirnov Test Results:")
for col in numeric_columns:
    stat, p = kstest(df[col], 'norm')
    print(f"{col} - Statistic: {stat}, p-value: {p}")
    if p > 0.05:
        print(f"{col} is normally distributed.\n")
    else:
        print(f"{col} is not normally distributed.\n")

# Linear Regression to predict hours_per_week based on education level (as numeric)
x = df[['educational_num']]
y = df['hours_per_week']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Model evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\nLinear Regression Results:")
print(f"R² Score: {r2}")
print(f"Mean Squared Error: {mse}")

# Plot regression results
plt.scatter(x_test, y_test, label="Actual")
plt.plot(x_test, y_pred, color='r', label="Predicted")
plt.xlabel("Education Level (Numerical)")
plt.ylabel("Hours Per Week")
plt.legend()
plt.title("Linear Regression: Predicting Hours Worked Based on Education Level")
plt.show()

# One-Way ANOVA: Analyze the effect of workclass on hours-per-week
anova_model_one_way = ols('hours_per_week ~ C(workclass)', data=df).fit()
anova_table_one_way = sm.stats.anova_lm(anova_model_one_way, typ=2)
print("\nOne-Way ANOVA Results for Workclass on Hours-per-Week:")
print(anova_table_one_way)

# Two-Way ANOVA: Effect of workclass and marital_status on hours-per-week
anova_model_two_way = ols('hours_per_week ~ C(workclass) + C(marital_status) + C(workclass):C(marital_status)', data=df).fit()
anova_table_two_way = sm.stats.anova_lm(anova_model_two_way, typ=2)
print("\nTwo-Way ANOVA Results for Workclass and Marital Status on Hours-per-Week:")
print(anova_table_two_way)
