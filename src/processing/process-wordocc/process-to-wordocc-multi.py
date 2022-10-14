import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os
import glob
import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer

from pdb import set_trace

def process_df_columns_to_wordocc(file, columns_preprocess_wordcooc, feature_combinations):
    data_df = None
    if '.pkl.gz' in file:
        data_df = pd.read_pickle(file)
    if 'training' in file:
        valid = file.replace('training', 'validation')
        valid = valid.replace('train', 'valid')
        valid_df = pd.read_pickle(valid)
        data_df = pd.concat([data_df, valid_df])
        data_df = data_df.reset_index(drop=True)
    elif '.json.gz' in file:
        data_df = pd.read_json(file, lines=True)
    else:
        print(f'unrecognized file format: {Path(file).suffix}')
    data_df.fillna('', inplace=True)

    # preprocess selected columns
    for column in columns_preprocess_wordcooc:
        data_df[column] = data_df[column].astype(str)

    # build combined features for every feature combination
    for feature_combination in feature_combinations:
        feats_to_combine = feature_combination.split('+')
        data_df[feature_combination + '_wordocc'] = data_df[feats_to_combine[0]]

        for feat_to_combine in feats_to_combine[1:]:
            data_df[feature_combination + '_wordocc'] += (' ' + data_df[feat_to_combine])

        data_df[feature_combination + '_wordocc'] = data_df[feature_combination + '_wordocc'].str.strip()

    return data_df


def transform_columns_to_wordcount(data_df, feature_combinations, test_df):
    words = {}

    for feature_combination in feature_combinations:

        # build relevant strings for vocabulary
        all_strings = data_df[['id', feature_combination + '_wordocc']].copy()
        all_strings = all_strings.rename(
            columns={feature_combination + '_wordocc': feature_combination})
        all_unique_strings = all_strings
        all_unique_strings = all_unique_strings.drop_duplicates(subset='id')

        # learn vocabulary
        count_vectorizer = CountVectorizer(min_df=2, binary=True)
        count_vectorizer.fit(all_unique_strings[feature_combination])

        words[feature_combination] = count_vectorizer.get_feature_names()

        # apply binary word occurrence
        matrix = count_vectorizer.transform(data_df[feature_combination + '_wordocc'])
        data_df[feature_combination + '_wordocc'] = [x for x in matrix]

        if not isinstance(test_df, type(None)):
            matrix_test = count_vectorizer.transform(test_df[feature_combination + '_wordocc'])
            test_df[feature_combination + '_wordocc'] = [x for x in matrix_test]

    return data_df, test_df, words

def preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name, dataset_name, valid_set=None,
                        test_set=None):
    columns_preprocess_wordcooc = [col for col in columns_to_preprocess]

    main_df = process_df_columns_to_wordocc(file, columns_preprocess_wordcooc, feature_combinations)

    if not isinstance(test_set, type(None)):
        test_df = process_df_columns_to_wordocc(test_set, columns_preprocess_wordcooc, feature_combinations)
    else:
        test_df = None

    main_df, test_df, words = transform_columns_to_wordcount(main_df, feature_combinations, test_df)

    main_name = os.path.basename(file)
    new_main_name = main_name.replace('.pkl.gz', '_wordocc')
    new_main_name = new_main_name.replace('.json.gz', '_wordocc')

    out_path = f'../../../data/processed/{dataset_name}/wordocc/{experiment_name}/'

    os.makedirs(out_path + 'feature-names/', exist_ok=True)

    with open(out_path + 'feature-names/' + new_main_name + '_words.json', 'w') as f:
        json.dump(words, f, ensure_ascii=False)

    if isinstance(valid_set, type(None)):
        main_df.to_pickle(out_path + new_main_name + '.pkl.gz', compression='gzip')
    else:
        validation_ids_df = pd.read_pickle(valid_set)
        validation_df = main_df[main_df['id'].isin(validation_ids_df['id'].values)]

        main_df.to_pickle(out_path + new_main_name + '.pkl.gz', compression='gzip')
        valid_name = new_main_name.replace('train', 'valid')
        validation_df.to_pickle(out_path + valid_name + '.pkl.gz', compression='gzip')

    if not isinstance(test_df, type(None)):
        test_name = os.path.basename(test_set)
        test_name = test_name.replace('.pkl.gz', '')
        test_name = test_name.replace('.json.gz', '')
        new_test_name = new_main_name + '_' + test_name

        test_df.to_pickle(out_path + new_test_name + '.pkl.gz', compression='gzip')


if __name__ == '__main__':

    for file in glob.glob('../../../data/interim/wdc-lspc/training-sets/*'):
        if 'multi' not in file or 'wdcproducts' not in file:
            continue
            
        valid = file.replace('training', 'validation')
        valid = valid.replace('train', 'valid')

        columns_to_preprocess = ['title', 'description', 'brand', 'price', 'priceCurrency']
        feature_combinations = ['brand+title+price+priceCurrency+description']

        test_cat = os.path.basename(file).split('_')[1]
        test ='../../../data/interim/wdc-lspc/gold-standards/preprocessed_{}_gs.pkl.gz'.format(test_cat)

        preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name='learning-curve', dataset_name='wdc-lspc',  valid_set=valid, test_set=test)

        test = test.replace('000un','050un')
        preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name='learning-curve', dataset_name='wdc-lspc',  valid_set=valid, test_set=test)

        test = test.replace('050un','100un')
        preprocess_wordcooc(file, columns_to_preprocess, feature_combinations, experiment_name='learning-curve', dataset_name='wdc-lspc',  valid_set=valid, test_set=test)