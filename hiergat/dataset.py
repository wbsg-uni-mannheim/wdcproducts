import json
import numpy as np
import pandas as pd

import os
from pathlib import Path
import glob
import gzip
from copy import deepcopy

import pickle

import argparse
import re
import csv

def combine_row(left, right, label):
    def func(row):
        col_names = left.columns # assume left and right always have same attributes
        list_ = ['COL' + ' ' + str(b) + ' ' + 'VAL' + ' ' + str(a) + ' ' for a, b in zip(row, col_names.values.tolist())]
        list_ = ''.join(str(m) for m in list_)
        return list_

    left_list = list(map(func, left.values.tolist()))
    right_list = list(map(func, right.values.tolist()))
    label_list = [str(l) for l in label]

    left_df = pd.DataFrame({'left': pd.Series(left_list)})
    right_df = pd.DataFrame({'right': pd.Series(right_list)})
    label_df = pd.DataFrame({'label': pd.Series(label_list)})

    # using tab separator here
    # https://github.com/megagonlabs/ditto - <entry_1> \t <entry_2> \t <label>
    final_df = left_df.left.map(str) + '\t' + right_df.right
    final_df = final_df.map(str) + '\t' + label_df.label

    return final_df

def preprocess_part(path):
    with gzip.open(f'{path}', 'rb') as f:
        test_set = pd.read_pickle(f)

    test_set = test_set.drop(['pair_id'], axis=1)

    mask_left = test_set.columns.str.endswith('_left')
    mask_right = test_set.columns.str.endswith('_right')

    left = test_set.loc[:, mask_left]
    right = test_set.loc[:, mask_right]
    label = [int(x) for x in list(test_set['label'].values)]

    left.columns = left.columns.str.removesuffix('_left')
    left = left.drop(['id', 'cluster_id'], axis=1)
    left = left.fillna('')
    left = left[['brand', 'title', 'price', 'priceCurrency', 'description', 'specTableContent']]

    right.columns = right.columns.str.removesuffix('_right')
    right = right.drop(['id', 'cluster_id'], axis=1)
    right = right.fillna('')
    right = right[['brand', 'title', 'price', 'priceCurrency', 'description', 'specTableContent']]

    final_df = combine_row(left, right, label)

    return(final_df)

def preprocess_dataset():
    os.makedirs('./data/final_output/', exist_ok=True)

    test_path =  "../data/interim/wdc-lspc/" + "gold-standards" + "/"
    valid_path = "../data/interim/wdc-lspc/" + "validation-sets" + "/"
    train_path = "../data/interim/wdc-lspc/" + "training-sets" + "/"
    groups = {"gold-standards", "validation-sets", "training-sets"}

    if not (os.path.exists(test_path)):
        print('Dataset does not exist')
        return

    print(f'START BULDING FINAL DATASETS')
    
    for path in glob.glob(os.path.join(test_path, (r'preprocessed_wdcproducts*.pkl.gz'))):
        test_df = preprocess_part(path)
        np.savetxt(f'./data/final_output/{path[2:-7]}.txt', test_df.values, fmt = "%s")
    print(f'DONE TEST')

    for path in glob.glob(os.path.join(valid_path, (r'preprocessed_wdcproducts*.pkl.gz'))):
        valid_df = preprocess_part(path)
        np.savetxt(f'./data/final_output/{path[2:-7]}.txt', valid_df.values, fmt = "%s")
    print(f'DONE VALID')

    for path in glob.glob(os.path.join(train_path, (r'preprocessed_wdcproducts*.pkl.gz'))):
        train_df = preprocess_part(path)
        np.savetxt(f'./data/final_output/{path[2:-7]}.txt', train_df.values, fmt = "%s")
    print(f'DONE TRAIN')

    print(f'FINISHED BULDING FINAL DATASETS\n')

if __name__ == '__main__':
    preprocess_dataset()
