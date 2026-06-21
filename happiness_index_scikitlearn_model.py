import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(__file__))
df = pd.read_csv("WHR_2024.csv")
df = df.dropna()
X_train = df.iloc[:, 3:9].values  # Features (columns 3-8)

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train, axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm, axis=0)}")

y_train = df['happiness_score'].values  # Target variable (column 2)

sgdr = SGDRegressor(max_iter=2000, alpha=0.01,epsilon=0)
sgdr.fit(X_norm, y_train)
print(sgdr.get_params())
print(f"Number of iterations completed: {sgdr.n_iter_}, Number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}")


# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:6]}" )
print(f"Target values \n{y_train[:6]}")

X_features = ['gdp_per_capita', 'social_support', 'healthy_life_expectancy', 'freedom_to_make_life_choices',
              'generosity', 'perceptions_of_corruption']

fig,ax=plt.subplots(1,6,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color='r', label = 'predict')
ax[0].set_ylabel("Happiness Index"); ax[0].legend();
fig.suptitle("Target versus prediction using z-score normalized model")
plt.show()