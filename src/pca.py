import numpy as np


def normalize(X):
    mu = np.mean(X, axis=0)    
    X_bar = X - mu
    std = np.std(X, axis=0, ddof=1)
    X_bar = X_bar/std
    return X_bar, mu, std

#%% 
def cov_matrix(Xn):
    n = Xn.shape[0]
    S = Xn.T @ Xn / (n-1)
    return S

#%%
def eig(S):
    eigvals, eigvecs = np.linalg.eigh(S)
    sort_index = np.argsort(eigvals)[::-1]
    return eigvals[sort_index], eigvecs[:,sort_index]

#%% 
def PCA(X, num_components):
    Xn, mean, std = normalize(X)
    S = cov_matrix(Xn)
    eig_vals, eig_vecs = eig(S)
    components = eig_vecs[:, :num_components]
    score = Xn @ components
    return score, eig_vals, components, mean, std

#%% 
def reconstruct(score, components, mean, std):
    Xn_reconst = score @ components.T
    reconst = Xn_reconst * std + mean
    return reconst
