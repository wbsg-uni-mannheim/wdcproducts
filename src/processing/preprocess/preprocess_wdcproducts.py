import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import os
import glob

import html
import datetime

from tqdm.auto import tqdm

from src.data import utils

def _cut_lspc(row):
    attributes = {'title_left': 50,
                  'title_right': 50,
                  'brand_left': 5,
                  'brand_right': 5,
                  'description_left': 100,
                  'description_right': 100,
                  'specTableContent_left': 200,
                  'specTableContent_right': 200}

    for attr, value in attributes.items():
        try:
            row[attr] = ' '.join(row[attr].split(' ')[:value])
        except AttributeError:
            continue
    return row

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

    # preprocess training sets and gold standards
    print('BUILDING PREPROCESSED TRAINING SETS AND GOLD STANDARDS...')
    os.makedirs(os.path.dirname('../../../data/interim/wdc-lspc/training-sets/'), exist_ok=True)
    os.makedirs(os.path.dirname('../../../data/interim/wdc-lspc/gold-standards/'), exist_ok=True)
    os.makedirs(os.path.dirname('../../../data/interim/wdc-lspc/validation-sets/'), exist_ok=True)

    for file in tqdm(glob.glob('../../../data/raw/wdc-lspc/training-sets/*')):
        if 'wdcproducts' in file and 'multi' not in file:
            df = pd.read_json(file, lines=True)
            df['title_left'] = df['title_left'].apply(utils.clean_string_2020)
            df['description_left'] = df['description_left'].apply(utils.clean_string_2020)
            df['brand_left'] = df['brand_left'].apply(utils.clean_string_2020)
            df['price_left'] = df['price_left'].apply(utils.clean_string_2020)
            df['priceCurrency_left'] = df['priceCurrency_left'].apply(utils.clean_string_2020)
            df['title_right'] = df['title_right'].apply(utils.clean_string_2020)
            df['description_right'] = df['description_right'].apply(utils.clean_string_2020)
            df['brand_right'] = df['brand_right'].apply(utils.clean_string_2020)
            df['price_right'] = df['price_right'].apply(utils.clean_string_2020)
            df['priceCurrency_right'] = df['priceCurrency_right'].apply(utils.clean_string_2020)
            
            df['price_left'] = clean_price(df['price_left'])
            df[['price_left', 'priceCurrency_left']] = df[['price_left', 'priceCurrency_left']].apply(update_price_currency, axis=1, result_type="expand")

            df['price_right'] = clean_price(df['price_right'])
            df[['price_right', 'priceCurrency_right']] = df[['price_right', 'priceCurrency_right']].apply(update_price_currency, axis=1, result_type="expand")

            df = df.fillna('')

            df['title_left'] = df['title_left'].apply(lambda x: html.unescape(x))
            df['description_left'] = df['description_left'].apply(lambda x: html.unescape(x))
            df['brand_left'] = df['brand_left'].apply(lambda x: html.unescape(x))

            df['title_right'] = df['title_right'].apply(lambda x: html.unescape(x))
            df['description_right'] = df['description_right'].apply(lambda x: html.unescape(x))
            df['brand_right'] = df['brand_right'].apply(lambda x: html.unescape(x))
            
            df = df.apply(_cut_lspc, axis=1)

            df = df.replace('', np.nan)
            
            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'../../../data/interim/wdc-lspc/training-sets/{file}')

        elif 'wdcproducts' in file and 'multi' in file:

            df = pd.read_json(file, lines=True)
            df['title'] = df['title'].apply(utils.clean_string_2020)
            df['description'] = df['description'].apply(utils.clean_string_2020)
            df['brand'] = df['brand'].apply(utils.clean_string_2020)
            df['price'] = df['price'].apply(utils.clean_string_2020)
            df['priceCurrency'] = df['priceCurrency'].apply(utils.clean_string_2020)

            df['price'] = clean_price(df['price'])
            df[['price', 'priceCurrency']] = df[['price', 'priceCurrency']].apply(update_price_currency, axis=1, result_type="expand")

            df = df.fillna('')

            df['title'] = df['title'].apply(lambda x: html.unescape(x))
            df['description'] = df['description'].apply(lambda x: html.unescape(x))
            df['brand'] = df['brand'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc_multi, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'../../../data/interim/wdc-lspc/training-sets/{file}')

    for file in glob.glob('../../../data/raw/wdc-lspc/validation-sets/*'):
        if 'wdcproducts' in file and 'multi' not in file:
            df = pd.read_json(file, lines=True)
            df['title_left'] = df['title_left'].apply(utils.clean_string_2020)
            df['description_left'] = df['description_left'].apply(utils.clean_string_2020)
            df['brand_left'] = df['brand_left'].apply(utils.clean_string_2020)
            df['price_left'] = df['price_left'].apply(utils.clean_string_2020)
            df['priceCurrency_left'] = df['priceCurrency_left'].apply(utils.clean_string_2020)
            df['title_right'] = df['title_right'].apply(utils.clean_string_2020)
            df['description_right'] = df['description_right'].apply(utils.clean_string_2020)
            df['brand_right'] = df['brand_right'].apply(utils.clean_string_2020)
            df['price_right'] = df['price_right'].apply(utils.clean_string_2020)
            df['priceCurrency_right'] = df['priceCurrency_right'].apply(utils.clean_string_2020)

            df['price_left'] = clean_price(df['price_left'])
            df[['price_left', 'priceCurrency_left']] = df[['price_left', 'priceCurrency_left']].apply(update_price_currency, axis=1, result_type="expand")

            df['price_right'] = clean_price(df['price_right'])
            df[['price_right', 'priceCurrency_right']] = df[['price_right', 'priceCurrency_right']].apply(update_price_currency, axis=1, result_type="expand")

            df = df.fillna('')

            df['title_left'] = df['title_left'].apply(lambda x: html.unescape(x))
            df['description_left'] = df['description_left'].apply(lambda x: html.unescape(x))
            df['brand_left'] = df['brand_left'].apply(lambda x: html.unescape(x))

            df['title_right'] = df['title_right'].apply(lambda x: html.unescape(x))
            df['description_right'] = df['description_right'].apply(lambda x: html.unescape(x))
            df['brand_right'] = df['brand_right'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'../../../data/interim/wdc-lspc/validation-sets/{file}')

        elif 'wdcproducts' in file and 'multi' in file:

            df = pd.read_json(file, lines=True)
            df['title'] = df['title'].apply(utils.clean_string_2020)
            df['description'] = df['description'].apply(utils.clean_string_2020)
            df['brand'] = df['brand'].apply(utils.clean_string_2020)
            df['price'] = df['price'].apply(utils.clean_string_2020)
            df['priceCurrency'] = df['priceCurrency'].apply(utils.clean_string_2020)

            df['price'] = clean_price(df['price'])
            df[['price', 'priceCurrency']] = df[['price', 'priceCurrency']].apply(update_price_currency, axis=1, result_type="expand")

            df = df.fillna('')

            df['title'] = df['title'].apply(lambda x: html.unescape(x))
            df['description'] = df['description'].apply(lambda x: html.unescape(x))
            df['brand'] = df['brand'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc_multi, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'../../../data/interim/wdc-lspc/validation-sets/{file}')

    for file in glob.glob('../../../data/raw/wdc-lspc/gold-standards/*'):
        if 'wdcproducts' in file and 'multi' not in file:
            df = pd.read_json(file, lines=True)
            df['title_left'] = df['title_left'].apply(utils.clean_string_2020)
            df['description_left'] = df['description_left'].apply(utils.clean_string_2020)
            df['brand_left'] = df['brand_left'].apply(utils.clean_string_2020)
            df['price_left'] = df['price_left'].apply(utils.clean_string_2020)
            df['priceCurrency_left'] = df['priceCurrency_left'].apply(utils.clean_string_2020)
            df['title_right'] = df['title_right'].apply(utils.clean_string_2020)
            df['description_right'] = df['description_right'].apply(utils.clean_string_2020)
            df['brand_right'] = df['brand_right'].apply(utils.clean_string_2020)
            df['price_right'] = df['price_right'].apply(utils.clean_string_2020)
            df['priceCurrency_right'] = df['priceCurrency_right'].apply(utils.clean_string_2020)

            df['price_left'] = clean_price(df['price_left'])
            df[['price_left', 'priceCurrency_left']] = df[['price_left', 'priceCurrency_left']].apply(update_price_currency, axis=1, result_type="expand")

            df['price_right'] = clean_price(df['price_right'])
            df[['price_right', 'priceCurrency_right']] = df[['price_right', 'priceCurrency_right']].apply(update_price_currency, axis=1, result_type="expand")

            df = df.fillna('')

            df['title_left'] = df['title_left'].apply(lambda x: html.unescape(x))
            df['description_left'] = df['description_left'].apply(lambda x: html.unescape(x))
            df['brand_left'] = df['brand_left'].apply(lambda x: html.unescape(x))

            df['title_right'] = df['title_right'].apply(lambda x: html.unescape(x))
            df['description_right'] = df['description_right'].apply(lambda x: html.unescape(x))
            df['brand_right'] = df['brand_right'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'../../../data/interim/wdc-lspc/gold-standards/{file}')
        
        elif 'wdcproducts' in file and 'multi' in file:

            df = pd.read_json(file, lines=True)
            df['title'] = df['title'].apply(utils.clean_string_2020)
            df['description'] = df['description'].apply(utils.clean_string_2020)
            df['brand'] = df['brand'].apply(utils.clean_string_2020)
            df['price'] = df['price'].apply(utils.clean_string_2020)
            df['priceCurrency'] = df['priceCurrency'].apply(utils.clean_string_2020)

            df['price'] = clean_price(df['price'])
            df[['price', 'priceCurrency']] = df[['price', 'priceCurrency']].apply(update_price_currency, axis=1, result_type="expand")

            df = df.fillna('')

            df['title'] = df['title'].apply(lambda x: html.unescape(x))
            df['description'] = df['description'].apply(lambda x: html.unescape(x))
            df['brand'] = df['brand'].apply(lambda x: html.unescape(x))

            df = df.apply(_cut_lspc_multi, axis=1)

            df = df.replace('', np.nan)

            df = df.reset_index(drop=True)

            file = os.path.basename(file)
            file = file.replace('.json.gz', '.pkl.gz')
            file = f'preprocessed_{file}'
            df.to_pickle(f'../../../data/interim/wdc-lspc/gold-standards/{file}')

    print('FINISHED BUILDING PREPROCESSED TRAINING SETS AND GOLD STANDARDS...')

