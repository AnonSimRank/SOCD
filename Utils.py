import csv
import networkx as nx
import matplotlib.pyplot as plt
import os
import filecmp
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from scipy import sparse
import scipy


def test_files_identical(file1, file2):

    if filecmp.cmp(file1, file2):
        print("They are the same.")
    else:
        print("They are not the same.")


def SimpleNetwork2SpaMat(txtfile : str, head_skip_rows : int):

    """
    Convert a simple network to a sparse matrix, index start with 1.
    """
    txt_array = np.loadtxt(txtfile, dtype=int, skiprows=head_skip_rows)
    mat_dim = txt_array.max()

    txt_arr = txt_array - 1
    row_index_array = txt_arr[:,0]
    col_index_array = txt_arr[:,1]

    edge_num = txt_array.shape[0]
    val = np.ones(edge_num, dtype=float)

    spa_mat = sparse.csc_matrix((val, (row_index_array, col_index_array)), shape=(mat_dim, mat_dim))

    return spa_mat


def SimpleNetwork2SpaMat_0(txtfile : str, head_skip_rows : int):

    txt_array = np.loadtxt(txtfile, dtype=int, skiprows=head_skip_rows)
    edge_num = txt_array.shape[0]

    mat_dim = txt_array.max() + 1
    row_index_array = txt_array[:, 0]
    col_index_array = txt_array[:, 1]
    val = np.ones(edge_num, dtype=float)

    spa_mat = sparse.csc_matrix((val, (row_index_array, col_index_array)), shape=(mat_dim, mat_dim))
    return spa_mat


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def spamat_rowsum_diag_mat(spa_mat):
    row_num = spa_mat.shape[0]
    mat_row_sum = np.array(np.sum(spa_mat, axis=1).tolist())
    mat_row_sum_1d = mat_row_sum.reshape((row_num,))

    spa_diagonal_mat = scipy.sparse.diags(mat_row_sum_1d, offsets=0)
    lap_mat = spa_diagonal_mat - spa_mat - spa_mat.transpose()
    return lap_mat


def loadtxt_undirected_net(pre_txt_file, post_txt_file):
    pre_data = np.loadtxt(pre_txt_file, dtype=int)
    row_num, col_num = pre_data.shape

    post_data_list = []
    for i in range(row_num):
        if pre_data[i,0] < pre_data[i,1]:
            post_data_list.append(pre_data[i,:])
        else:
            continue

    post_data_array = np.array(post_data_list)
    np.savetxt(post_txt_file, post_data_array, fmt='%d')


