from src.regression import gradient_descent, normal_equation, mse_loss, gradient
import numpy as np


def test_gd_converge_pra_normal_equation():
    rng = np.random.default_rng(42)
    X =  rng.normal(size=(150,5))
    y = rng.normal(size= 150)
    w_ne = normal_equation(X,y)
    w_gd, _ = gradient_descent(X,y,lr=0.1,n_iters = 10000)
    assert np.allclose(w_gd, w_ne, atol=1.e-3)
    
def test_gradient_bate_com_numerico():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20,3))
    y = rng.normal(size=20)
    w = rng.normal(size=4)
    eps = 1e-6
    grad_num = np.zeros_like(w)
    grad_analitico = gradient(X,y,w)
    for i in range((len(w))):
        w_mais = w.copy()
        w_mais[i] += eps
        w_menos = w.copy()
        w_menos[i]-= eps
        grad_num[i] = (mse_loss(X, y, w=w_mais) - mse_loss(X, y, w=w_menos))/(2*eps)
    assert np.allclose(grad_analitico, grad_num)
    
        