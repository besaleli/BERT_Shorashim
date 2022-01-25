from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch
import time

def tsne(embeddings, n_components=2, verbose=False):
    start = time.time()
    
    pca_model = PCA(n_components=50)
    tsne_model = TSNE(n_components=n_components)
    
    emb_pca = pca_model.fit_transform(embeddings)
    emb_tsne = tsne_model.fit_transform(emb_pca)
    
    end = time.time()
    
    if verbose:
        print('TIME ELAPSED: {}'.format(np.round(end-start, 2)))
        
    return zip(*emb_tsne)

def pca(embeddings, n_components=2, verbose=False):
    start = time.time()
    
    pca_model = PCA(n_components=n_components)
    
    emb_pca = pca_model.fit_transform(embeddings)
    
    end = time.time()
    
    if verbose:
        print('TIME ELAPSED: {}'.format(np.round(end-start, 2)))
        
    return zip(*emb_pca)