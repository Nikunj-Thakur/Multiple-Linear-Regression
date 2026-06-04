import math
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
    m = X.shape[0]
    n = X.shape[1]
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for j in range(n):
        for i in range(m):
            f_wb = np.dot(X[i], w) + b
            dj_dw[j] = dj_dw[j] + (f_wb-y[i])*X[i,j]
        dj_db = dj_db + (f_wb-y[i])
    return dj_dw/m, dj_db/m