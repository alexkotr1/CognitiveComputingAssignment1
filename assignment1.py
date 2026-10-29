
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Task 2
df = pd.read_csv("loan_approval_dataset.csv", sep=";", skipinitialspace=True)

# clean up column names and text values (remove extra spaces)
df.columns = df.columns.str.strip()
for col in ["education", "self_employed", "loan_status"]:
    df[col] = df[col].str.strip()

pd.set_option("display.max_columns", None)

print("First 20 rows:")
print(df.head(20))

print("\nShape:", df.shape)
print("\nInfo:")
df.info()
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

print("\nLoan status counts:")
print(df["loan_status"].value_counts())

# Task 4
df["loan_status_num"] = np.where(df["loan_status"] == "Approved", 1, 0)

numeric_df = df.select_dtypes(include="number").drop(columns=["loan_id"])

corrmatrix = numeric_df.corr()
print("\nCorrelation matrix:")
print(corrmatrix.round(2))

plt.figure(figsize=(12, 9))
sns.heatmap(corrmatrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation heatmap of numeric features")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

# Task 5
print("\nCorrelation of each numeric feature with loan_status:")
print(corrmatrix["loan_status_num"].drop("loan_status_num").sort_values(ascending=False))

print("\nMean of each numeric feature for Approved vs Rejected:")
print(numeric_df.drop(columns=["loan_status_num"]).groupby(df["loan_status"]).mean().T.round(0))

# cibil_score looked like the strongest one so I plot it
plt.figure(figsize=(8, 5))
sns.boxplot(x="loan_status", y="cibil_score", data=df)
plt.title("CIBIL score by loan status")
plt.savefig("cibil_boxplot.png")
plt.show()

# Task 5
print("\nEducation vs loan_status (counts):")
table = pd.crosstab(df["education"], df["loan_status"])
print(table)

print("\nEducation vs loan_status (% per row):")
print((pd.crosstab(df["education"], df["loan_status"], normalize="index") * 100).round(1))

# chi-square test to check if education and loan_status are related
chi2, p, dof, expected = stats.chi2_contingency(table)
print("\nChi-square:", round(chi2, 3), " p-value:", round(p, 4))
if p < 0.05:
    print("p < 0.05 -> education and loan_status are related")
else:
    print("p >= 0.05 -> no significant relation between education and loan_status")

sns.countplot(x="education", hue="loan_status", data=df)
plt.title("Loan status by education")
plt.savefig("education_countplot.png")
plt.show()
