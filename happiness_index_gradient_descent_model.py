import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))
df = pd.read_csv("WHR_2024.csv")
df = df.dropna() #dropping small percentage of rows have missing values
X_train = df.iloc[:, 3:9].values  # Features (columns 3-8)
y_train = df['happiness_score'].values  # Target variable (column 2)

X_features = ['gdp_per_capita','social_support', 'healthy_life_expectancy', 'freedom_to_make_life_choices',
              'generosity','perceptions_of_corruption']

fig,ax=plt.subplots(1, 6, figsize=(12, 6), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
    ax[i].set_ylabel("Happiness Index")
plt.show()
