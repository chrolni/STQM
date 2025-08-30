import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import scipy.sparse as sp
import numpy as np
import pandas as pd 

def PositionalEncoder(image_shape,num_frequency_bands,max_frequencies=None):
    
    *spatial_shape, _ = image_shape
   
    coords = [ torch.linspace(-1, 1, steps=s) for s in spatial_shape ]
    pos = torch.stack(torch.meshgrid(*coords), dim=len(spatial_shape)) 
    
    encodings = []
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    frequencies = [ torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                                              for max_freq in max_frequencies ]
    
    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(pos[..., i:i+1] * frequencies_i[None, ...])

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    enc = torch.cat(encodings, dim=-1)
    enc = rearrange(enc, "... c -> (...) c")

    return enc

def cal_lape(adj_mx):
    lape_dim = 32
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    print(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


