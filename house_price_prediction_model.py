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