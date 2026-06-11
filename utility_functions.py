import math
import copy
import numpy as np


def predict_simple_loop(x, w, b):
    n = x.shape[0]
    p = 0
    for i in range(n):
        p = p + (w[i]*x[i])
    return p+b


def predict_vectorized(x, w, b):
    return np.dot(w, x) + b


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost = cost + (f_wb - y[i])**2
    return cost/(2*m)


def compute_gradient(X, y, w, b):
    m = X.shape[0]  # i
    n = X.shape[1]  # j
    dj_dw = np.zeros((n,))  # 1D array/vector
    dj_db = 0.0

    for i in range(m):  # i ranges from 0,1,2
        f_wb = np.dot(X[i], w) + b
        dj_db = dj_db + (f_wb-y[i])

        for j in range(n):  # j=0,1,2,3
            dj_dw[j] = dj_dw[j] + (f_wb-y[i])*X[i, j]
    return dj_dw/m, dj_db/m


def gradient_descent(X, y, w_init, b_init, alpha, no_of_iterations):
    J_history = []
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(no_of_iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w-alpha*dj_dw
        b = b-alpha*dj_db

        if (i < 100000):
            J_history.append(compute_cost(X, y, w, b))

        if i % math.ceil(no_of_iterations/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history
