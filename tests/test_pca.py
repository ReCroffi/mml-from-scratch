from src.pca import normalize, PCA, reconstruct
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
    X = load_wine().data                          # Arrange
    score, _, components, mean, std = PCA(X, num_components=X.shape[1])   # todos!
    X_reconst = reconstruct(score, components, mean, std)   # Act
    assert np.allclose(X_reconst, X)              # Assert

