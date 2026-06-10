import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../dataset/titanic.csv")

print(data.head())

print("\nDataset Info")
print(data.info())

print("\nStatistical Summary")
print(data.describe())

print("\nMissing Values")
print(data.isnull().sum())

survival = data["survived"].value_counts()

plt.figure(figsize=(5, 4))
survival.plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Passengers")
plt.savefig("../images/survival_count.png")
plt.close()

gender = data["sex"].value_counts()

plt.figure(figsize=(5, 4))
gender.plot(kind="pie", autopct="%1.1f%%")
plt.title("Gender Distribution")
plt.ylabel("")
plt.savefig("../images/gender_distribution.png")
plt.close()

print("\nCorrelation Matrix")
print(data.corr(numeric_only=True))

print("\nEDA Completed Successfully")
