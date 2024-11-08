import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, boxcox, ttest_ind
from statsmodels.stats.weightstats import ztest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("actors.csv")

# Rename columns to replace spaces with underscores for easier referencing
df.columns = df.columns.str.replace(' ', '_')
print("Data Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# Pairplot for numeric columns
numeric_columns = ['Total_Gross', 'Number_of_Movies', 'Average_per_Movie', 'Gross']
sb.pairplot(df[numeric_columns], diag_kind='kde', kind='scatter')
plt.show()

# Boxplot for 'Total_Gross' before outlier removal
plt.boxplot(df['Total_Gross'])
plt.title("Before Outlier Removal for Total Gross")
plt.show()

# Outlier handling using IQR method
q1 = df['Total_Gross'].quantile(0.25)
q3 = df['Total_Gross'].quantile(0.75)
iqr = q3 - q1
upper = q3 + (1.5 * iqr)
lower = q1 - (1.5 * iqr)

# Cap outliers
df['Total_Gross'] = np.where(df['Total_Gross'] > upper, upper, df['Total_Gross'])
df['Total_Gross'] = np.where(df['Total_Gross'] < lower, lower, df['Total_Gross'])

# Boxplot after outlier removal
plt.boxplot(df['Total_Gross'])
plt.title("After Outlier Removal for Total Gross")
plt.show()

# Normality test for 'Number_of_Movies' using Shapiro-Wilk test
stat, p = shapiro(df['Number_of_Movies'])
if p > 0.05:
    print("Number of Movies is normally distributed")
else:
    print("Number of Movies is not normally distributed")

# Box-Cox Transformation for 'Average_per_Movie' to improve normality
df['Average_per_Movie'], fitted_lambda = boxcox(df['Average_per_Movie'])
print("Fitted lambda for Box-Cox Transformation:", fitted_lambda)

# Simple Linear Regression: Predicting Total Gross based on Average per Movie
x = df[['Average_per_Movie']]
y = df['Total_Gross']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluation of the regression model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2}")
print(f"Mean Squared Error: {mse}")

# Scatter plot of test set vs. predictions
plt.scatter(x_test, y_test, label="Actual")
plt.plot(x_test, y_pred, color='r', label="Predicted")
plt.xlabel("Average per Movie")
plt.ylabel("Total Gross")
plt.legend()
plt.title("Linear Regression: Predicting Total Gross Based on Average per Movie")
plt.show()

# Subset the data into two groups based on `Number_of_Movies` for testing
group_1 = df[df['Number_of_Movies'] <= 30]['Total_Gross']
group_2 = df[df['Number_of_Movies'] > 30]['Total_Gross']

# Z-Test between the two groups
z_stat, z_p_value = ztest(group_1, group_2, value=0)
print("Z-Test Results:")
print(f"Z-Statistic: {z_stat}, P-Value: {z_p_value}")
if z_p_value < 0.05:
    print("Reject the null hypothesis: Significant difference in Total Gross between groups.")
else:
    print("Fail to reject the null hypothesis: No significant difference in Total Gross between groups.")

# T-Test between the two groups
t_stat, t_p_value = ttest_ind(group_1, group_2, equal_var=False)
print("\nT-Test Results:")
print(f"T-Statistic: {t_stat}, P-Value: {t_p_value}")
if t_p_value < 0.05:
    print("Reject the null hypothesis: Significant difference in Total Gross between groups.")
else:
    print("Fail to reject the null hypothesis: No significant difference in Total Gross between groups.")

# One-Way ANOVA: Total Gross by Movie Count Category
df['movie_count_category'] = pd.cut(df['Number_of_Movies'], bins=[0, 20, 40, 60, 80], labels=['0-20', '20-40', '40-60', '60-80'])
anova_one_way = ols('Total_Gross ~ C(movie_count_category)', data=df).fit()
anova_table_one_way = sm.stats.anova_lm(anova_one_way, typ=2)
print("One-Way ANOVA Results:")
print(anova_table_one_way)

# Two-Way ANOVA: Total Gross by Movie Count Category and Gross Category
# Creating a gross category by binning the Gross column
df['gross_category'] = pd.cut(df['Gross'], bins=[0, 200, 400, 600, 800, 1000], labels=['0-200', '200-400', '400-600', '600-800', '800-1000'])
anova_two_way = ols('Total_Gross ~ C(movie_count_category) + C(gross_category) + C(movie_count_category):C(gross_category)', data=df).fit()
anova_table_two_way = sm.stats.anova_lm(anova_two_way, typ=2)
print("Two-Way ANOVA Results:")
print(anova_table_two_way)
