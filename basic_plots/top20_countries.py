import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Multiple Linear Regression\\WHR_2024.csv")

top20 = df.sort_values(by="happiness_score", ascending=False).head(25)

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    data=top20,
    x="happiness_score",
    y="country",
    palette="viridis"
)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

plt.title("Top 20 Countries by Happiness Index")
plt.xlabel("Happiness Index")
plt.ylabel("Country")

plt.tight_layout()
plt.show()
