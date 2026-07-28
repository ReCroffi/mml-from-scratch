from src.pca import normalize, PCA, reconstruct, PCA_svd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA as SklearnPCA



def test_normalize_centraliza():
    X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])   
    Xn, _, _ = normalize(X)
    assert np.allclose(Xn.mean(axis=0), 0) 
    

def test_pca():
    X = load_wine().data
    Xn, _, _ = normalize(X)
    _, _, pca_components, _, _ = PCA(Xn,num_components=2)
    sk = SklearnPCA(n_components = 2).fit(Xn)
    assert np.allclose(np.abs(pca_components), np.abs(sk.components_.T), atol=1e-6)


def test_reconstruct_roundtrip():
    X = load_wine().data                         
    score, _, components, mean, std = PCA(X, num_components=X.shape[1])   # todos!
    X_reconst = reconstruct(score, components, mean, std)   
    assert np.allclose(X_reconst, X)              

def test_pca_svd_bate_com_eigendecomposition():
    X = load_wine().data
    Xn, _, _, = normalize(X)
    _, eig_vals_pca, components_pca, _, _ = PCA(Xn, num_components = 2)
    _, eig_vals_svd, components_svd, _, _ = PCA_svd(Xn, num_components= 2)
    assert np.allclose(eig_vals_pca, eig_vals_svd)
    assert np.allclose(np.abs(components_pca), np.abs(components_svd))