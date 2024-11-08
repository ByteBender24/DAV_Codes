import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest
from sklearn import datasets

# Load the Breast Cancer dataset
df = pd.read_csv("breast_cancer_data.csv")


# Display initial information about the dataset
print("Dataset shape:", df.shape)
print(df.head())
df.info()
print(df.describe())
print("Dataset shape after dropping missing values:", df.shape)

# Visualize pairplot for numeric columns
numeric_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                   'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
sb.pairplot(df[numeric_columns], diag_kind='kde', kind='scatter')
plt.show()

# Boxplot for 'radius_mean' before outlier removal
plt.boxplot(df['radius_mean'])
plt.title("Before Outlier Removal (radius_mean)")
plt.show()

# Calculate IQR for 'radius_mean' and remove outliers
q1 = df['radius_mean'].quantile(0.25)
q3 = df['radius_mean'].quantile(0.75)
iqr = q3 - q1
upper = q3 + (1.5 * iqr)
lower = q1 - (1.5 * iqr)

print(f"IQR: {iqr}")
print(f"Upper bound: {upper}, Lower bound: {lower}")

# Cap outliers in 'radius_mean'
df['radius_mean'] = np.where(df['radius_mean'] > upper, upper, df['radius_mean'])
df['radius_mean'] = np.where(df['radius_mean'] < lower, lower, df['radius_mean'])

# Boxplot for 'radius_mean' after outlier removal
plt.boxplot(df['radius_mean'])
plt.title("After Outlier Removal (radius_mean)")
plt.show()

# Check if 'radius_mean' is normally distributed using Shapiro-Wilk test
stat, p = shapiro(df['radius_mean'])
if p > 0.05:
    print("radius_mean is normally distributed")
else:
    print("radius_mean is not normally distributed")

# Logistic Regression to predict 'diagnosis' based on 'radius_mean'
# Encoding 'diagnosis' as binary (Malignant = 1, Benign = 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

x = df[['radius_mean']]  # Predictor variable
y = df['diagnosis']  # Target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='r')
plt.xlabel("Radius Mean")
plt.ylabel("Diagnosis (0: Benign, 1: Malignant)")
plt.title("Logistic Regression: Predicting Diagnosis Based on Radius Mean")
plt.show()

# Box-Cox transformation for specified columns
for col in ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']:
    df[col] = np.where(df[col] <= 0, np.nan, df[col])  # Handle non-positive values
    df[col] = df[col].fillna(df[col].mean())  # Impute missing values
    df[col], fitted_lambda = boxcox(df[col])

# Z-test for the 'radius_mean' between two random samples
a_1, _ = train_test_split(df['radius_mean'], test_size=0.4, random_state=0)
b_1, _ = train_test_split(df['radius_mean'], test_size=0.4, random_state=1)

ztest_result = ztest(a_1, b_1, value=0)
print("Z-test result:", ztest_result)

# Statistical calculations on the two samples
mu1 = np.mean(a_1)
mu2 = np.mean(b_1)
std_a_1 = np.std(a_1)
std_b_1 = np.std(b_1)
var_a_1 = np.var(a_1)
var_b_1 = np.var(b_1)

print("Mean of a_1:", mu1)
print("Mean of b_1:", mu2)
print("Standard deviation of a_1:", std_a_1)
print("Standard deviation of b_1:", std_b_1)
print("Variance of a_1:", var_a_1)
print("Variance of b_1:", var_b_1)

from scipy.stats import ttest_ind

# Separate the data by diagnosis (Malignant 'M' and Benign 'B')
malignant = df[df['diagnosis'] == 'M']
benign = df[df['diagnosis'] == 'B']

# Select the column of interest (e.g., 'radius_mean')
malignant_radius = malignant['radius_mean']
benign_radius = benign['radius_mean']

# Perform the t-test
t_stat, p_value = ttest_ind(malignant_radius, benign_radius)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("The difference between the two groups is statistically significant.")
else:
    print("The difference between the two groups is not statistically significant.")

# One-Way ANOVA: 'radius_mean' by 'diagnosis' category
anova_one_way = ols('radius_mean ~ C(diagnosis)', data=df).fit()
anova_table_one_way = sm.stats.anova_lm(anova_one_way, typ=2)
print("One-Way ANOVA Results:")
print(anova_table_one_way)

# Two-Way ANOVA: 'radius_mean' and 'texture_mean' as factors
anova_two_way = ols('radius_mean ~ C(diagnosis) + C(texture_mean) + C(diagnosis):C(texture_mean)', data=df).fit()
anova_table_two_way = sm.stats.anova_lm(anova_two_way, typ=2)
print("Two-Way ANOVA Results:")
print(anova_table_two_way)
