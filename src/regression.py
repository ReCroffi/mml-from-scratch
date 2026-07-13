import numpy as np


def add_bias(X):
    
    X = np.column_stack((np.ones(X.shape[0],), X))
    
    return X

def predict(X,w):

    X_aug = add_bias(X)
    y_pred = X_aug @ w
    return  y_pred

def mse_loss(X, y, w):
    y_pred = predict(X,w)
    resd = y_pred - y
    mse = np.mean(resd**2)
    return mse

def gradient(X, y, w):
    y_pred = predict(X,w)
    resid = y_pred - y
    X_aug = add_bias(X)
    return (2/X.shape[0]) * X_aug.T @ resid

def gradient_descent(X, y, lr, n_iters):
    w = np.zeros(X.shape[1] + 1)
    mse_list = []
    for _ in range(n_iters):
        grad = gradient(X, y, w)
        w = w - lr * grad
        mse_list.append(mse_loss(X, y, w))
    return w, mse_list

def normal_equation(X,y):
    X_aug = add_bias(X)       
    a = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    return a 