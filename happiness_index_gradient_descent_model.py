import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility_functions as uf
import os

os.chdir(os.path.dirname(__file__))
df = pd.read_csv("WHR_2024.csv")
df = df.dropna()  # dropping small percentage of rows have missing values
X_train = df.iloc[:, 3:9].values  # Features (columns 3-8)

# Feature Standardization (Z-score normalization)
# computes the mean of each column (feature) in X_train.
X_mean = X_train.mean(axis=0)  #store them for normalizing test data later on
# calculates the standard deviation of each feature (column) in X_train
X_std = X_train.std(axis=0)  #store them for normalizing test data later on
X_norm = (X_train - X_mean) / X_std

y_train = df['happiness_score'].values  # Target variable (column 2)

X_features = ['gdp_per_capita', 'social_support', 'healthy_life_expectancy', 'freedom_to_make_life_choices',
              'generosity', 'perceptions_of_corruption']

fig, ax = plt.subplots(1, 6, figsize=(12, 4), sharey=True) #sharey=True means that all subplots share the same y-axis scale.
for i in range(len(ax)):
    ax[i].scatter(X_norm[:, i], y_train)
    ax[i].set_xlabel(X_features[i])
    ax[i].set_ylabel("Happiness Index")
plt.show()


# initialize parameters
initial_w = np.zeros((6,))
initial_b = 0.
alpha = 0.01
w_final, b_final, J_hist = uf.gradient_descent(
    X_norm, y_train, initial_w, initial_b, alpha, 2000)

print(f"Final Weights: {w_final}, Bias : {b_final}\n")

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100]) # ax1.plot(range(len(J_hist[:100])), J_hist[:100])
ax1.set_title("Cost vs. iteration")
ax1.set_ylabel('Cost')
ax1.set_xlabel('iteration step')

ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax2.set_title("Cost vs. iteration (tail)")
ax2.set_ylabel('Cost')
ax2.set_xlabel('iteration step')

plt.show()

print("Predict the happiness index of country 'India' having features:")
test_row = np.array([1.166, 0.653, 0.417, 0.767, 0.174, 0.122])
print(f"{test_row}")
# Scale test data using training set statistics
test_row_scaled = (test_row - X_mean) / X_std
print("Scaled India features:\n", test_row_scaled)

prediction = (np.dot(test_row_scaled, w_final)) + b_final
print(f"Predicted Happiness Index: {prediction:0.2f}")
print(f"Actual Happiness Index: 4.05")
print(f"Prediction Error: {abs(prediction - 4.05):0.2f}")


# predict target using normalized features
# Note, the prediction is made using the normalized feature, 
# while the plot is shown using the original feature values.
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_final) + b_final

# plot predictions and targets versus original features    
fig,ax=plt.subplots(1,6,figsize=(12, 4),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].scatter(X_train[:,i],yp,color='r', label = 'predict')
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Happiness Index"); ax[0].legend();
fig.suptitle("Target versus prediction using z-score normalized model")
plt.show()

# A few points to note:
# with multiple features, we can no longer have a single plot showing results versus features.
# when generating the plot, the normalized features were used. 
# Any predictions using the parameters learned from a normalized training set must also be normalized.