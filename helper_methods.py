"""Different helper methods extracted from rme_rec"""
import os

import numpy as np
import pandas as pd
from scipy import sparse
import argparse

def load_data(csv_file, shape):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['userId']), np.array(tp['movieId'])
    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int')), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp


def get_row(M, i):
    # get the row i of sparse matrix:
    lo, hi = M.indptr[i], M.indptr[i + 1]
    return lo, hi, M.data[lo:hi], M.indices[lo:hi]


def convert_to_SPPMI_matrix(M, max_row, shifted_K=1):
    # if we sum the co-occurrence matrix by row wise or column wise --> we have an array that contain the #(i) values
    obj_counts = np.asarray(M.sum(axis=1)).ravel()
    total_obj_pairs = M.data.sum()
    M_sppmi = M.copy()
    for i in xrange(max_row):
        lo, hi, data, indices = get_row(M, i)
        M_sppmi.data[lo:hi] = np.log(data * total_obj_pairs / (obj_counts[i] * obj_counts[indices]))
    M_sppmi.data[M_sppmi.data < 0] = 0
    M_sppmi.eliminate_zeros()
    if shifted_K == 1:
        return M_sppmi
    else:
        M_sppmi.data -= np.log(shifted_K)
        M_sppmi.data[M_sppmi.data < 0] = 0
        M_sppmi.eliminate_zeros()
    return M_sppmi

def get_args_parser():
    parser = argparse.ArgumentParser("Description: Running multi-embedding recommendation - RME model")
    parser.add_argument('--data_path', default='data', type=str, help='path to the data')
    parser.add_argument('--saved_model_path', default='MODELS', type=str,
                        help='path to save the optimal learned parameters')
    parser.add_argument('--s', default=1, type=int, help='a pre-defined shifted value for measuring SPPMI')
    parser.add_argument('--model', default='rme', type=str, help='the model to run: rme, cofactor')
    parser.add_argument('--n_factors', default=40, type=int,
                        help='number of hidden factors for user/item representation')
    parser.add_argument('--reg', default=1.0, type=float,
                        help='regularization for user and item latent factors (alpha, beta)')
    parser.add_argument('--reg_embed', default=1.0, type=float,
                        help='regularization for user and item context latent factors (gamma, delta, theta)')
    parser.add_argument('--dataset', default="ml10m", type=str, help='dataset')
    parser.add_argument('--neg_item_inference', default=0, type=int,
                        help='if there is no available disliked items, set this to 1 to infer '
                             'negative items for users using our user-oriented EM like algorithm')
    parser.add_argument('--neg_sample_ratio', default=0.2, type=float,
                        help='negative sample ratio per user. If a user consumed 10 items, and this'
                             'neg_sample_ratio = 0.2 --> randomly sample 2 negative items for the user')

    return parser


def get_unique_users_and_items(DATA_DIR):
    unique_uid = list()
    with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
    unique_movieId = list()
    with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_movieId.append(line.strip())

    return unique_movieId, unique_uid