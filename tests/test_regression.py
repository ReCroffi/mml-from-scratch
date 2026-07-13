from src.regression import gradient_descent, normal_equation, mse_loss, gradient
from src.pca import normalize, PCA
import numpy as np
from sklearn.datasets import load_wine

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
    
def test_regressao_nos_scores_pca():
    wine = load_wine()
    y = wine.data[:,12]
    X = wine.data[:, :12]
    Xn, _, _ = normalize(X)
    score, _, _, _, _ = PCA(X, num_components = X.shape[1])
    w_feat = normal_equation(Xn, y)
    mse_feat = mse_loss(Xn, y, w_feat)
    w_score = normal_equation(score, y)
    mse_score = mse_loss(score, y, w_score)
    assert np.allclose(mse_feat, mse_score)