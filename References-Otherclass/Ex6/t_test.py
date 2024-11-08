import pandas as pd
from scipy import stats

# Load the dataset
df = pd.read_csv('./titanic_dataset.csv')

# Drop rows with missing Age values
df_clean = df.dropna(subset=['Age'])

# Group the ages based on survival status
age_survived = df_clean[df_clean['Survived'] == 1]['Age']
age_not_survived = df_clean[df_clean['Survived'] == 0]['Age']

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(age_survived, age_not_survived)

# Print the results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpretation of the results
if p_value < 0.05:
    print("There is a significant difference in age between those who survived and those who did not.")
else:
    print("There is no significant difference in age between those who survived and those who did not.")
