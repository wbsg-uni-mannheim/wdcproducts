import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import itertools

from pathlib import Path
import shutil

from src.data import utils

from src.processing.preprocess.price_conversion.utils_number import parseNumber
from src.processing.preprocess.price_conversion.utils_currency_converter import CurrencyRates
import html
import datetime
import simplejson as json
from decimal import Decimal

import pandarallel

from pandarallel import pandarallel

from pdb import set_trace

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

def tryConvertFloat(value):
    if value.find('[eE]'):
        try:
            return format(float(value))
        except (ValueError, TypeError):
            pass
    return value

def clean_price(price_input):
    price_input = price_input.replace(' ', '', regex=True)
    # only take first number, maybe we can refine it at some point
    price_input = price_input.str.extract(r'(\d+([,.E]\d+)*)')[0]
    price_input = price_input.fillna('')
    price_input = price_input.apply(lambda x: tryConvertFloat(x))
    price_input = price_input.apply(lambda x: parseNumber(x))
    price_input = price_input.apply(lambda x: "{:.2f}".format(x))
    price_input = price_input.replace('nan', '')
    price_input = price_input.replace('0.00', '')
    return price_input

def clean_currency(currency_input):
    currency_input.replace({'Kč': 'CZK', 'Kn': 'HRK', 'eur': 'EUR', 'Euro': 'EUR', 'czk': 'CZK'}, inplace=True)
    currency_input.replace(to_replace=r'[0-9]', value='', regex=True, inplace=True)
    currency_input.replace(to_replace=r'\s+', value='', regex=True, inplace=True)
    return currency_input

def update_price_currency(row):
    row_price = row[0]
    row_currency = row[1]
    price = ""
    currency = ""
    if row_price != "" and row_currency != "":
        if row_currency == 'EUR':
            price = row_price
            currency = row_currency
        elif row_currency == 'AED':
            price = "{:.2f}".format((float(row_price) * 0.2255))
            currency = 'EUR'
        else:
            try:
                price = "{:.2f}".format(c.convert(str(row_currency), 'EUR', Decimal(row_price), date_obj))
                currency = 'EUR'
            except json.errors.JSONDecodeError:
                price = row_price
                currency = row_currency
    elif row_price != "":
        return row_price, currency
    return price, currency

if __name__ == '__main__':

    categories = ['wdcproducts20cc80rnd000un', 'wdcproducts50cc50rnd000un', 'wdcproducts80cc20rnd000un', 'wdcproducts20cc80rnd050un', 'wdcproducts50cc50rnd050un', 'wdcproducts80cc20rnd050un', 'wdcproducts20cc80rnd100un', 'wdcproducts50cc50rnd100un', 'wdcproducts80cc20rnd100un']
    train_sizes = ['small', 'medium', 'large']

    data = pd.read_pickle('../../../data/interim/wdc-lspc/corpus/dedup_preprocessed_lspcV2020_only_en_strict_only_long_title_only_mainentity.pkl.gz')

    relevant_cols = ['id', 'cluster_id', 'brand', 'title', 'description', 'specTableContent', 'price', 'priceCurrency']

    c = CurrencyRates()
    date_obj = datetime.datetime(2020, 12, 1, 18, 36, 28, 151012)
    pandarallel.initialize()

    for category in categories:
        out_path = f'../../../data/processed/wdc-lspc/contrastive/pre-train/{category}/'
        shutil.rmtree(out_path, ignore_errors=True)
        Path(out_path).mkdir(parents=True, exist_ok=True)
        for train_size in train_sizes:
            if '000un' not in category:
                break
            try:
                ids = pd.read_pickle(f'../../../data/interim/wdc-lspc/training-sets/preprocessed_{category}_train_{train_size}.pkl.gz')
                ids_valid = pd.read_pickle(f'../../../data/interim/wdc-lspc/validation-sets/preprocessed_{category}_valid_{train_size}.pkl.gz')
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
            data_selection['priceCurrency'] = clean_currency(data_selection['priceCurrency'])
            data_selection[['price', 'priceCurrency']] = data_selection[['price', 'priceCurrency']].parallel_apply(update_price_currency, axis=1, result_type="expand")

            data_selection = data_selection.fillna('')

            data_selection['title'] = data_selection['title'].apply(lambda x: html.unescape(x))
            data_selection['description'] = data_selection['description'].apply(lambda x: html.unescape(x))
            data_selection['brand'] = data_selection['brand'].apply(lambda x: html.unescape(x))

            data_selection = data_selection.apply(_cut_lspc_multi, axis=1)

            data_selection = data_selection.replace('', None)
            
            data_selection.to_pickle(f'{out_path}{category}_train_{train_size}.pkl.gz')
            
        # test set
        ids = pd.read_pickle(f'../../../data/interim/wdc-lspc/gold-standards/preprocessed_{category}_gs.pkl.gz')
        relevant_ids = set()
        relevant_ids.update(ids['id_left'])
        relevant_ids.update(ids['id_right'])

        data_selection = data[data['id'].isin(relevant_ids)].copy()
        data_selection = data_selection[relevant_cols]
        data_selection = data_selection.reset_index(drop=True)
        
        data_selection['title'] = data_selection['title'].apply(utils.clean_string_2020)
        data_selection['description'] = data_selection['description'].apply(utils.clean_string_2020)
        data_selection['brand'] = data_selection['brand'].apply(utils.clean_string_2020)
        data_selection['price'] = data_selection['price'].apply(utils.clean_string_2020)
        data_selection['priceCurrency'] = data_selection['priceCurrency'].apply(utils.clean_string_2020)

        data_selection['price'] = clean_price(data_selection['price'])
        data_selection['priceCurrency'] = clean_currency(data_selection['priceCurrency'])
        data_selection[['price', 'priceCurrency']] = data_selection[['price', 'priceCurrency']].parallel_apply(update_price_currency, axis=1, result_type="expand")

        data_selection = data_selection.fillna('')

        data_selection['title'] = data_selection['title'].apply(lambda x: html.unescape(x))
        data_selection['description'] = data_selection['description'].apply(lambda x: html.unescape(x))
        data_selection['brand'] = data_selection['brand'].apply(lambda x: html.unescape(x))

        data_selection = data_selection.apply(_cut_lspc_multi, axis=1)

        data_selection = data_selection.replace('', None)
        
        data_selection.to_pickle(f'{out_path}{category}_gs.pkl.gz')