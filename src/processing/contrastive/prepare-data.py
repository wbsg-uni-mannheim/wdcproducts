import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import itertools
import html

from pathlib import Path
import shutil

from src.data import utils

def _cut_lspc_multi(row):
    attributes = {'title': 50,
                  'brand': 5,
                  'description': 100,
                  'specTableContent': 200}

    for attr, value in attributes.items():
        try:
            row[attr] = ' '.join(row[attr].split(' ')[:value])
        except AttributeError:
            continue
    return row
   
def clean_price(price_input):
    price_input = price_input.fillna('')
    price_input = price_input.replace('nan', '')
    price_input = price_input.str.strip()
    return price_input

def update_price_currency(row):
    row_price = row[0]
    row_currency = row[1]
    price = ""
    currency = ""
    if row_price == "":
        return price, currency
    else:
        return row_price, row_currency


if __name__ == '__main__':

    categories = ['wdcproducts20cc80rnd000un', 'wdcproducts50cc50rnd000un', 'wdcproducts80cc20rnd000un']
    train_sizes = ['small', 'medium', 'large']
    valid_types = ['000un', '050un', '100un']

    data = pd.read_pickle('../../../data/interim/wdc-lspc/corpus/dedup_preprocessed_lspcV2020_only_en_strict_only_long_title_only_mainentity.pkl.gz')

    relevant_cols = ['id', 'cluster_id', 'brand', 'title', 'description', 'specTableContent', 'price', 'priceCurrency']

    for category in categories:
        for valid_type in valid_types:
            out_path = f'../../../data/processed/wdc-lspc/contrastive/pre-train/{category.replace("000un", valid_type)}/'
            shutil.rmtree(out_path, ignore_errors=True)
            Path(out_path).mkdir(parents=True, exist_ok=True)
            
            for train_size in train_sizes:
                try:
                    ids = pd.read_pickle(f'../../../data/interim/wdc-lspc/training-sets/preprocessed_{category}_train_{train_size}.pkl.gz')
                    ids_valid = pd.read_pickle(f'../../../data/interim/wdc-lspc/validation-sets/preprocessed_{category.replace("000un", valid_type)}_valid_{train_size}.pkl.gz')
                except FileNotFoundError:
                    continue
                relevant_ids = set()
                relevant_ids.update(ids['id_left'])
                relevant_ids.update(ids['id_right'])
                relevant_ids.update(ids_valid['id_left'])
                relevant_ids.update(ids_valid['id_right'])

                data_selection = data[data['id'].isin(relevant_ids)].copy()
                data_selection = data_selection[relevant_cols]
                data_selection = data_selection.reset_index(drop=True)

                data_selection['title'] = data_selection['title'].apply(utils.clean_string_2020)
                data_selection['description'] = data_selection['description'].apply(utils.clean_string_2020)
                data_selection['brand'] = data_selection['brand'].apply(utils.clean_string_2020)
                data_selection['price'] = data_selection['price'].apply(utils.clean_string_2020)
                data_selection['priceCurrency'] = data_selection['priceCurrency'].apply(utils.clean_string_2020)

                data_selection['price'] = clean_price(data_selection['price'])
                data_selection[['price', 'priceCurrency']] = data_selection[['price', 'priceCurrency']].apply(update_price_currency, axis=1, result_type="expand")

                data_selection = data_selection.fillna('')

                data_selection['title'] = data_selection['title'].apply(lambda x: html.unescape(x))
                data_selection['description'] = data_selection['description'].apply(lambda x: html.unescape(x))
                data_selection['brand'] = data_selection['brand'].apply(lambda x: html.unescape(x))

                data_selection = data_selection.apply(_cut_lspc_multi, axis=1)

                data_selection = data_selection.replace('', None)

                data_selection.to_pickle(f'{out_path}{category.replace("000un", valid_type)}_train_{train_size}.pkl.gz')