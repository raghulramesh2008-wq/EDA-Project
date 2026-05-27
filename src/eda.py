import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder automatically
os.makedirs("../images", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/dataset.csv")

# Display first 5 rows
print("\nFIRST 5 ROWS")
print(df.head())

# Dataset information
print("\nDATASET INFO")
print(df.info())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Histogram
plt.figure(figsize=(8,5))
df["Math"].hist(bins=5)
plt.title("Math Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.savefig("../images/histogram.png")
plt.show()

# Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("../images/heatmap.png")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df["Attendance"], df["Math"])
plt.title("Attendance vs Math Marks")
plt.xlabel("Attendance")
plt.ylabel("Math Marks")
plt.savefig("../images/scatterplot.png")
plt.show()

# Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(data=df[["Math","Science","English"]])
plt.title("Subject Marks Boxplot")
plt.savefig("../images/boxplot.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")