from matplotlib import pyplot as plt
import numpy as np
import model_parameters as mp

X_train = np.array([[2104, 5, 1, 45],
                    [1416, 3, 2, 40],
                    [852, 2, 1, 35]
                    ])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

prediction_example0=mp.predict_simple_loop(X_train[0,:],w_init,b_init)
print(prediction_example0)

prediction_example0=mp.predict_vectorized(X_train[0,:],w_init,b_init)
print(prediction_example0)

cost=mp.compute_cost(X_train,y_train,w_init,b_init)
print(cost)

dj_dw,dj_db=mp.compute_gradient(X_train,y_train,w_init,b_init)
print(f"Dj_Dw={dj_dw} and Dj_Db={dj_db}")

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
alpha=5.0e-7
w_final,b_final,J_hist=mp.gradient_descent(X_train,y_train,initial_w,initial_b,alpha,1000)

print(w_final, b_final)

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()