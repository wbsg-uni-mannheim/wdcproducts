import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd

from pathlib import Path
import glob
import gzip
import pickle
from copy import deepcopy

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from sklearn.preprocessing import LabelEncoder

from pdb import set_trace

def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result

def serialize_sample_lspc_contrastive(sample, only_title, use_price=False):
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split(" ")[:50])}'.strip()
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand"].split(" ")[:5])}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split(" ")[:50])}'.strip()
    if use_price:
        string = f'{string} [COL] price [VAL] {sample[f"price"]}'.strip()
        string = f'{string} [COL] priceCurrency [VAL] {sample[f"priceCurrency"]}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split(" ")[:100])}'.strip()
    string = f'{string} [COL] specTableContent [VAL] {" ".join(sample[f"specTableContent"].split(" ")[:200])}'.strip()

    return string

def serialize_sample_abtbuy_contrastive(sample, only_title):
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"name"].split(" ")[:50])}'.strip()
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"name"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split()[:100])}'.strip()

    return string

def serialize_sample_amazongoogle_contrastive(sample, only_title):
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split(" ")[:50])}'.strip()
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split()[:100])}'.strip()

    return string

def serialize_sample_walmartamazon_contrastive(sample, only_title):
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split(" ")[:50])}'.strip()
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split())}'.strip()
    string = f'{string} [COL] modelno [VAL] {" ".join(sample[f"modelno"].split())}'.strip()
    string = f'{string} [COL] category [VAL] {" ".join(sample[f"category"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split()[:100])}'.strip()

    return string

def serialize_sample_dblpscholar_contrastive(sample, only_title):
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split(" ")[:50])}'.strip()
    string = f'{string}[COL] title [VAL] {" ".join(sample[f"title"].split())}'.strip()
    string = f'{string} [COL] authors [VAL] {" ".join(str(sample[f"authors"]).split())}'.strip()
    string = f'{string} [COL] venue [VAL] {" ".join(str(sample[f"venue"]).split())}'.strip()
    string = f'{string} [COL] year [VAL] {" ".join(str(sample[f"year"]).split())}'.strip()

    return string

def serialize_sample_beeradvoratebeer_contrastive(sample, only_title):
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"Beer_Name"].split(" ")[:50])}'.strip()
    string = f'{string}[COL] title [VAL] {" ".join(sample[f"Beer_Name"].split())}'.strip()
    string = f'{string} [COL] Brew_Factory_Name [VAL] {" ".join(sample[f"Brew_Factory_Name"].split())}'.strip()
    string = f'{string} [COL] Style [VAL] {" ".join(sample[f"Style"].split())}'.strip()
    string = f'{string} [COL] ABV [VAL] {" ".join(sample[f"ABV"].split()[:100])}'.strip()

    return string

def serialize_sample_company_contrastive(sample, only_title):
    string = ''
    string = f'{string}[COL] content [VAL] {" ".join(sample[f"content"].split())}'.strip()

    return string

def serialize_sample_lspc_pairwise(sample, side, only_title, use_price=False):
        
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split(" ")[:50])}'.strip()
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split(" ")[:5])}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split(" ")[:50])}'.strip()
    if use_price:
        string = f'{string} [COL] price [VAL] {sample[f"price_{side}"]}'.strip()
        string = f'{string} [COL] priceCurrency [VAL] {sample[f"priceCurrency_{side}"]}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split(" ")[:100])}'.strip()
    string = f'{string} [COL] specTableContent [VAL] {" ".join(sample[f"specTableContent_{side}"].split(" ")[:200])}'.strip()

    return string

def serialize_sample_abtbuy_pairwise(sample, side, only_title):
    
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"name_{side}"].split())}'.strip()
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"name_{side}"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()
    

    return string

def serialize_sample_amazongoogle_pairwise(sample, side, only_title):
    
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

    return string

def serialize_sample_walmartamazon_pairwise(sample, side, only_title):
    
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
    string = f'{string} [COL] modelno [VAL] {" ".join(sample[f"modelno_{side}"].split())}'.strip()
    string = f'{string} [COL] category [VAL] {" ".join(sample[f"category_{side}"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

    return string

def serialize_sample_dblpscholar_pairwise(sample, side, only_title):
    
    string = ''
    if only_title:
        return f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
    string = f'{string} [COL] authors [VAL] {" ".join(sample[f"authors_{side}"].split())}'.strip()
    string = f'{string} [COL] venue [VAL] {" ".join(sample[f"venue_{side}"].split())}'.strip()
    string = f'{string} [COL] year [VAL] {" ".join(str(sample[f"year_{side}"]).split())}'.strip()

    return string

def serialize_sample_beeradvoratebeer_pairwise(sample, side, only_title):
    
    string = ''
    if only_title:
        return f'{string}[COL] title [VAL] {" ".join(sample[f"Beer_Name_{side}"].split())}'.strip()
    string = f'{string}[COL] title [VAL] {" ".join(sample[f"Beer_Name_{side}"].split())}'.strip()
    string = f'{string} [COL] Brew_Factory_Name [VAL] {" ".join(sample[f"Brew_Factory_Name_{side}"].split())}'.strip()
    string = f'{string} [COL] Style [VAL] {" ".join(sample[f"Style_{side}"].split())}'.strip()
    string = f'{string} [COL] ABV [VAL] {" ".join(sample[f"ABV_{side}"].split()[:100])}'.strip()

    return string

def serialize_sample_company_pairwise(sample, side, only_title):
    
    string = ''
    string = f'{string}[COL] content [VAL] {" ".join(sample[f"content_{side}"].split())}'.strip()

    return string

class Augmenter():
    def __init__(self, aug):

        stopwords = ['[COL]', '[VAL]', 'title', 'name', 'description', 'manufacturer', 'brand', 'specTableContent']

        aug_typo = nac.KeyboardAug(stopwords=stopwords, aug_char_p=0.1, aug_word_p=0.1)
        aug_swap = naw.RandomWordAug(action="swap", stopwords=stopwords, aug_p=0.1)
        aug_del = naw.RandomWordAug(action="delete", stopwords=stopwords, aug_p=0.1)
        aug_crop = naw.RandomWordAug(action="crop", stopwords=stopwords, aug_p=0.1)
        aug_sub = naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=0.1)
        aug_split = naw.SplitAug(stopwords=stopwords, aug_p=0.1)

        aug = aug.strip('-')

        if aug == 'all':
            self.augs = [aug_typo, aug_swap, aug_split, aug_sub, aug_del, aug_crop, None]
        
        if aug == 'typo':
            self.augs = [aug_typo, None]

        if aug == 'swap':
            self.augs = [aug_swap, None]

        if aug == 'delete':
            self.augs = [aug_del, None]

        if aug == 'crop':
            self.augs = [aug_crop, None]

        if aug == 'substitute':
            self.augs = [aug_sub, None]

        if aug == 'split':
            self.augs = [aug_split, None]

    def apply_aug(self, string):
        aug = random.choice(self.augs)
        if aug is None:
            return string
        else:
            return aug.augment(string)

class ContrastivePretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='lspc', only_interm=False, aug=False, only_title=False):

        self.max_length = max_length
        if 'sigmod' not in dataset:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('lenovo','thinkpad','elitebook', 'toshiba', 'asus', 'acer', 'lexar', 'sandisk', 'tesco', 'intenso', 'transcend'))
        self.dataset = dataset
        self.aug = aug
        self.only_title = only_title

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        data = pd.read_pickle(path)
        
        if dataset == 'abt-buy':
            data['brand'] = ''

        if dataset == 'amazon-google' or dataset == 'walmart-amazon':
            data['description'] = ''
                
        if intermediate_set is not None:
            interm_data = pd.read_pickle(intermediate_set)
            max_cid = data['cluster_id'].max()
            interm_data['cluster_id'] = interm_data['cluster_id']+ max_cid + 1
            if only_interm:
                data = interm_data
            else:
                data = data.append(interm_data)
        
        data = data.reset_index(drop=True)

        data = data.fillna('')

        if 'wdcproducts' in path:
            data = self._prepare_data(data, True)
        else:
            data = self._prepare_data(data, False)

        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()
        selection = self.data[self.data['labels'] == example['labels']]
        # if len(selection) > 1:
        #     selection = selection.drop(idx)
        pos = selection.sample(1).iloc[0].copy()

        if self.aug:
            example['features'] = self.augmenter.apply_aug(example['features'])
            pos['features'] = self.augmenter.apply_aug(pos['features'])

        return (example, pos)

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data, use_price=False):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc_contrastive, args=(self.only_title, use_price), axis=1)

        elif self.dataset == 'abt-buy':
            data['features'] = data.apply(serialize_sample_abtbuy_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'walmart-amazon':
            data['features'] = data.apply(serialize_sample_walmartamazon_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'dblp-scholar':
            data['features'] = data.apply(serialize_sample_dblpscholar_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'beeradvo-ratebeer':
            data['features'] = data.apply(serialize_sample_beeradvoratebeer_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'company':
            data['features'] = data.apply(serialize_sample_company_contrastive, args=(self.only_title,), axis=1)

        label_enc = LabelEncoder()
        data['labels'] = label_enc.fit_transform(data['cluster_id'])

        self.label_encoder = label_enc

        data = data[['features', 'labels']]

        return data

class ContrastivePretrainDatasetDeepmatcher(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='abt-buy', only_interm=False, aug=False, split=True, only_title=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset
        self.aug = aug
        self.only_title = only_title

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        data = pd.read_pickle(path)

        if dataset == 'abt-buy':
            data['brand'] = ''

        if dataset == 'amazon-google' or dataset == 'walmart-amazon':
            data['description'] = ''
        
        if clean:
            train_data = pd.read_json(deduction_set, lines=True)
            
            if dataset == 'abt-buy':
                val = pd.read_csv('../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                val = pd.read_csv('../../data/interim/amazon-google/amazon-google-valid.csv')
            elif dataset == 'walmart-amazon':
                val = pd.read_csv('../../data/interim/walmart-amazon/walmart-amazon-valid.csv')
            elif dataset == 'dblp-scholar':
                val = pd.read_csv('../../data/interim/dblp-scholar/dblp-scholar-valid.csv')
            elif dataset == 'beeradvo-ratebeer':
                val = pd.read_csv('../../data/interim/beeradvo-ratebeer/beeradvo-ratebeer-valid.csv')
            elif dataset == 'company':
                val = pd.read_csv('../../data/interim/company/company-valid.csv')


            val_set = train_data[train_data['pair_id'].isin(val['pair_id'])]
            val_set_pos = val_set[val_set['label'] == 1]
            val_set_pos = val_set_pos.sample(frac=0.80)
            val_ids = set()
            val_ids.update(val_set['pair_id'])
            
            train_data = train_data[~train_data['pair_id'].isin(val_ids)]
            train_data = train_data[train_data['label'] == 1]
            train_data = train_data.sample(frac=0.80)

            train_data = train_data.append(val_set_pos)

            bucket_list = []
            for i, row in train_data.iterrows():
                left = f'{row["id_left"]}'
                right = f'{row["id_right"]}'
                found = False
                for bucket in bucket_list:
                    if left in bucket and row['label'] == 1:
                        bucket.add(right)
                        found = True
                        break
                    elif right in bucket and row['label'] == 1:
                        bucket.add(left)
                        found = True
                        break
                if not found:
                    bucket_list.append(set([left, right]))

            merging = True
            while merging:
                merging=False
                for i,group in enumerate(bucket_list):
                    merged = next((g for g in bucket_list[i+1:] if g.intersection(group)),None)
                    if not merged: continue
                    group.update(merged)
                    bucket_list.remove(merged)
                    merging = True

            cluster_id_amount = len(bucket_list)
            
            cluster_id_dict = {}
            for i, id_set in enumerate(bucket_list):
                for v in id_set:
                    cluster_id_dict[v] = i
            data = data.set_index('id', drop=False)
            data['cluster_id'] = data['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
            #data = data[data['cluster_id'] != cluster_id_amount]

            single_entities = data[data['cluster_id'] == cluster_id_amount].copy()

            index = single_entities.index

            if dataset == 'abt-buy':
                left_index = [x for x in index if 'abt' in x]
                right_index = [x for x in index if 'buy' in x]
            elif dataset == 'amazon-google':
                left_index = [x for x in index if 'amazon' in x]
                right_index = [x for x in index if 'google' in x]
            elif dataset == 'walmart-amazon':
                left_index = [x for x in index if 'walmart' in x]
                right_index = [x for x in index if 'amazon' in x]
            elif dataset == 'dblp-scholar':
                left_index = [x for x in index if 'dblp' in x]
                right_index = [x for x in index if 'scholar' in x]
            elif dataset == 'beeradvo-ratebeer':
                left_index = [x for x in index if 'beeradvo' in x]
                right_index = [x for x in index if 'ratebeer' in x]
            elif dataset == 'company':
                left_index = [x for x in index if 'companyA' in x]
                right_index = [x for x in index if 'companyB' in x]
            
            single_entities = single_entities.reset_index(drop=True)
            single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index
            single_entities = single_entities.set_index('id', drop=False)
            single_entities_left = single_entities.loc[left_index]
            single_entities_right = single_entities.loc[right_index]
            
            if split:
                data1 = data.copy().drop(single_entities['id'])
                data1 = data1.append(single_entities_left)

                data2 = data.copy().drop(single_entities['id'])
                data2 = data2.append(single_entities_right)

            else:
                data1 = data.copy().drop(single_entities['id'])
                data1 = data1.append(single_entities_left)
                data1 = data1.append(single_entities_right)

                data2 = data.copy().drop(single_entities['id'])
                data2 = data2.append(single_entities_left)
                data2 = data2.append(single_entities_right)

            if intermediate_set is not None:
                interm_data = pd.read_pickle(intermediate_set)
                if dataset != 'lspc':
                    cols = data.columns
                    if 'name' in cols:
                        interm_data = interm_data.rename(columns={'title':'name'})
                    if 'manufacturer' in cols:
                        interm_data = interm_data.rename(columns={'brand':'manufacturer'})
                    interm_data['cluster_id'] = interm_data['cluster_id']+10000
                    if dataset == 'abt-buy' or dataset == 'amazon-google' or dataset == 'walmart-amazon':
                        interm_data['price'] = ''
                    if dataset == 'walmart-amazon':
                        interm_data['modelno'] = ''
                        interm_data['category'] = ''

                
                if only_interm:
                    data1 = interm_data
                    data2 = interm_data
                else:
                    data1 = data1.append(interm_data)
                    data2 = data2.append(interm_data)

            data1 = data1.reset_index(drop=True)
            data2 = data2.reset_index(drop=True)

            label_enc = LabelEncoder()
            cluster_id_set = set()
            cluster_id_set.update(data1['cluster_id'])
            cluster_id_set.update(data2['cluster_id'])
            label_enc.fit(list(cluster_id_set))
            data1['labels'] = label_enc.transform(data1['cluster_id'])
            data2['labels'] = label_enc.transform(data2['cluster_id'])

            self.label_encoder = label_enc
                
        data1 = data1.reset_index(drop=True)

        data1 = data1.fillna('')
        data1 = self._prepare_data(data1)

        data2 = data2.reset_index(drop=True)

        data2 = data2.fillna('')
        data2 = self._prepare_data(data2)

        diff = abs(len(data1)-len(data2))

        if len(data1) > len(data2):
            if len(data2) < diff:
                sample = data2.sample(diff, replace=True)
            else:
                sample = data2.sample(diff)
            data2 = data2.append(sample)
            data2 = data2.reset_index(drop=True)

        elif len(data2) > len(data1):
            if len(data1) < diff:
                sample = data1.sample(diff, replace=True)
            else:
                sample = data1.sample(diff)
            data1 = data1.append(sample)
            data1 = data1.reset_index(drop=True)

        self.data1 = data1
        self.data2 = data2
        # data1.to_json('data1.json.gz', compression='gzip', lines=True,
        #           orient='records')
        # data2.to_json('data2.json.gz', compression='gzip', lines=True,
        #           orient='records')

    def __getitem__(self, idx):
        example1 = self.data1.loc[idx].copy()
        selection1 = self.data1[self.data1['labels'] == example1['labels']]
        # if len(selection1) > 1:
        #     selection1 = selection1.drop(idx)
        pos1 = selection1.sample(1).iloc[0].copy()

        example2 = self.data2.loc[idx].copy()
        selection2 = self.data2[self.data2['labels'] == example2['labels']]
        # if len(selection2) > 1:
        #     selection2 = selection2.drop(idx)
        pos2 = selection2.sample(1).iloc[0].copy()

        if self.aug:
            example1['features'] = self.augmenter.apply_aug(example1['features'])
            pos1['features'] = self.augmenter.apply_aug(pos1['features'])
            example2['features'] = self.augmenter.apply_aug(example2['features'])
            pos2['features'] = self.augmenter.apply_aug(pos2['features'])

        return ((example1, pos1), (example2, pos2))

    def __len__(self):
        return len(self.data1)
    
    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'abt-buy':
            data['features'] = data.apply(serialize_sample_abtbuy_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'walmart-amazon':
            data['features'] = data.apply(serialize_sample_walmartamazon_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'dblp-scholar':
            data['features'] = data.apply(serialize_sample_dblpscholar_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'beeradvo-ratebeer':
            data['features'] = data.apply(serialize_sample_beeradvoratebeer_contrastive, args=(self.only_title,), axis=1)

        elif self.dataset == 'company':
            data['features'] = data.apply(serialize_sample_company_contrastive, args=(self.only_title,), axis=1)

        data = data[['features', 'labels']]

        return data

class ContrastiveClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, dataset='lspc', aug=False, additional_data=None, only_additional=False, only_title=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.only_title = only_title

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path)
            filename = Path(path).name
            category = filename.split('_')[1]
        else:
            data = pd.read_json(path, lines=True)

        if dataset == 'abt-buy':
            data['brand_left'] = ''
            data['brand_right'] = ''

        if dataset == 'amazon-google' or dataset == 'walmart-amazon':
            data['description_left'] = ''
            data['description_right'] = ''

        if self.dataset_type != 'test':
            if dataset == 'lspc':
                if 'wdcproducts' not in path:
                    validation_ids = pd.read_csv(f'../../data/raw/wdc-lspc/validation-sets/{category}_valid_{size}.csv')
                else:
                    validation_set = pd.read_pickle((f'../../data/interim/wdc-lspc/validation-sets/preprocessed_{category}_valid_{size}.pkl.gz'))
            elif dataset == 'abt-buy':
                validation_ids = pd.read_csv(f'../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                validation_ids = pd.read_csv(f'../../data/interim/amazon-google/amazon-google-valid.csv')
            elif dataset == 'walmart-amazon':
                validation_ids = pd.read_csv(f'../../data/interim/walmart-amazon/walmart-amazon-valid.csv')
            elif dataset == 'dblp-scholar':
                validation_ids = pd.read_csv(f'../../data/interim/dblp-scholar/dblp-scholar-valid.csv')
            elif dataset == 'beeradvo-ratebeer':
                validation_ids = pd.read_csv(f'../../data/interim/beeradvo-ratebeer/beeradvo-ratebeer-valid.csv')
            elif dataset == 'company':
                validation_ids = pd.read_csv(f'../../data/interim/company/company-valid.csv')
            if self.dataset_type == 'train':
                if 'wdcproducts' not in path:
                    data = data[~data['pair_id'].isin(validation_ids['pair_id'])]
                else:
                    pass
            else:
                if 'wdcproducts' not in path:
                    data = data[data['pair_id'].isin(validation_ids['pair_id'])]
                else:
                    data = validation_set

        if additional_data is not None and self.dataset_type != 'test':
            add_data = pd.read_pickle(additional_data)
            val_ids_add = pd.read_csv(f'{additional_data.replace("training-sets", "validation-sets").replace("_train.pkl.gz", "_valid.csv")}')

            if dataset != 'lspc':
                cols = data.columns
                if dataset == 'abt-buy':
                    add_data = add_data.rename(columns={'title_left':'name_left', 'title_right':'name_right'})
                if dataset == 'amazon-google':
                    add_data = add_data.rename(columns={'brand_left':'manufacturer_left', 'brand_right':'manufacturer_right'})
                if dataset == 'abt-buy' or dataset == 'walmart-amazon' or dataset == 'amazon-google':
                    add_data['price_left'] = ''
                    add_data['price_right'] = ''
                if dataset == 'walmart-amazon':
                    add_data['model_no_left'] = ''
                    add_data['model_no_right'] = ''
                    add_data['category_left'] = ''
                    add_data['category_right'] = ''
                add_data['cluster_id_left'] = add_data['cluster_id_left']+10000
                add_data['cluster_id_right'] = add_data['cluster_id_right']+10000

            if only_additional:
                if self.dataset_type == 'train':
                    data = add_data[~add_data['pair_id'].isin(val_ids_add['pair_id'])]
                else:
                    data = add_data[add_data['pair_id'].isin(val_ids_add['pair_id'])]
            else:
                if self.dataset_type == 'train':
                    data = data.append(add_data[~add_data['pair_id'].isin(val_ids_add['pair_id'])])
                else:
                    data = data.append(add_data[add_data['pair_id'].isin(val_ids_add['pair_id'])])

        data = data.fillna('')
        data = data.reset_index(drop=True)

        if 'wdcproducts' in path:
            data = self._prepare_data(data, True)
        else:
            data = self._prepare_data(data, False)
        
        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        if self.aug:
            example['features_left'] = self.augmenter.apply_aug(example['features_left'])
            example['features_right'] = self.augmenter.apply_aug(example['features_right'])

        return example

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data, use_price=False):

        if self.dataset == 'lspc':
            data['features_left'] = data.apply(serialize_sample_lspc_pairwise, args=('left',self.only_title,use_price), axis=1)
            data['features_right'] = data.apply(serialize_sample_lspc_pairwise, args=('right',self.only_title,use_price), axis=1)
        elif self.dataset == 'abt-buy':
            data['features_left'] = data.apply(serialize_sample_abtbuy_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_abtbuy_pairwise, args=('right',self.only_title), axis=1)
        elif self.dataset == 'amazon-google':
            data['features_left'] = data.apply(serialize_sample_amazongoogle_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_amazongoogle_pairwise, args=('right',self.only_title), axis=1)
        elif self.dataset == 'walmart-amazon':
            data['features_left'] = data.apply(serialize_sample_walmartamazon_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_walmartamazon_pairwise, args=('right',self.only_title), axis=1)
        elif self.dataset == 'dblp-scholar':
            data['features_left'] = data.apply(serialize_sample_dblpscholar_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_dblpscholar_pairwise, args=('right',self.only_title), axis=1)
        elif self.dataset == 'beeradvo-ratebeer':
            data['features_left'] = data.apply(serialize_sample_beeradvoratebeer_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_beeradvoratebeer_pairwise, args=('right',self.only_title), axis=1)
        elif self.dataset == 'company':
            data['features_left'] = data.apply(serialize_sample_company_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_company_pairwise, args=('right',self.only_title), axis=1)

        data = data[['features_left', 'features_right', 'label']]
        data = data.rename(columns={'label': 'labels'})

        return data

    

class BaselineClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=256, dataset='lspc', aug=False, additional_data=None, only_additional=False, only_title=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.only_title = only_title

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path)
            filename = Path(path).name
            category = filename.split('_')[1]
        else:
            data = pd.read_json(path, lines=True)

        if dataset == 'abt-buy':
            data['brand_left'] = ''
            data['brand_right'] = ''

        if dataset == 'amazon-google' or dataset == 'walmart-amazon':
            data['description_left'] = ''
            data['description_right'] = ''

        if self.dataset_type != 'test':
            if dataset == 'lspc':
                if 'wdcproducts' not in path:
                    validation_ids = pd.read_csv(f'../../data/raw/wdc-lspc/validation-sets/{category}_valid_{size}.csv')
                else:
                    validation_set = pd.read_pickle((f'../../data/interim/wdc-lspc/validation-sets/preprocessed_{category}_valid_{size}.pkl.gz'))
            elif dataset == 'abt-buy':
                validation_ids = pd.read_csv(f'../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                validation_ids = pd.read_csv(f'../../data/interim/amazon-google/amazon-google-valid.csv')
            elif dataset == 'walmart-amazon':
                validation_ids = pd.read_csv(f'../../data/interim/walmart-amazon/walmart-amazon-valid.csv')
            elif dataset == 'beeradvo-ratebeer':
                validation_ids = pd.read_csv(f'../../data/interim/beeradvo-ratebeer/beeradvo-ratebeer-valid.csv')
            elif dataset == 'dblp-scholar':
                validation_ids = pd.read_csv(f'../../data/interim/dblp-scholar/dblp-scholar-valid.csv')
            elif dataset == 'company':
                validation_ids = pd.read_csv(f'../../data/interim/company/company-valid.csv')

            if self.dataset_type == 'train':
                if 'wdcproducts' not in path:
                    data = data[~data['pair_id'].isin(validation_ids['pair_id'])]
                else:
                    pass
            else:
                if 'wdcproducts' not in path:
                    data = data[data['pair_id'].isin(validation_ids['pair_id'])]
                else:
                    data = validation_set

        if additional_data is not None and self.dataset_type != 'test':
            add_data = pd.read_pickle(additional_data)
            val_ids_add = pd.read_csv(f'{additional_data.replace("training-sets", "validation-sets").replace("_train.pkl.gz", "_valid.csv")}')

            if dataset != 'lspc':
                cols = data.columns
                if dataset == 'abt-buy':
                    add_data = add_data.rename(columns={'title_left':'name_left', 'title_right':'name_right'})
                if dataset == 'amazon-google':
                    add_data = add_data.rename(columns={'brand_left':'manufacturer_left', 'brand_right':'manufacturer_right'})
                if dataset == 'abt-buy' or dataset == 'walmart-amazon' or dataset == 'amazon-google':
                    add_data['price_left'] = ''
                    add_data['price_right'] = ''
                if dataset == 'walmart-amazon':
                    add_data['model_no_left'] = ''
                    add_data['model_no_right'] = ''
                    add_data['category_left'] = ''
                    add_data['category_right'] = ''
                add_data['cluster_id_left'] = add_data['cluster_id_left']+10000
                add_data['cluster_id_right'] = add_data['cluster_id_right']+10000

            if only_additional:
                if self.dataset_type == 'train':
                    data = add_data[~add_data['pair_id'].isin(val_ids_add['pair_id'])]
                else:
                    data = add_data[add_data['pair_id'].isin(val_ids_add['pair_id'])]
            else:
                if self.dataset_type == 'train':
                    data = data.append(add_data[~add_data['pair_id'].isin(val_ids_add['pair_id'])])
                else:
                    data = data.append(add_data[add_data['pair_id'].isin(val_ids_add['pair_id'])])

        data = data.fillna('')
        data = data.reset_index(drop=True)
        
        if 'wdcproducts' in path:
            data = self._prepare_data(data, True)
        else:
            data = self._prepare_data(data, False)

        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        if self.aug:
            example['features_left'] = self.augmenter.apply_aug(example['features_left'])
            example['features_right'] = self.augmenter.apply_aug(example['features_right'])

        example_tokenized = self.tokenizer(example['features_left'], example['features_right'], padding=False, truncation='longest_first', max_length=self.max_length)
        example_tokenized['label'] = example['label']

        return example_tokenized

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data, use_price=False):

        if self.dataset == 'lspc':
            data['features_left'] = data.apply(serialize_sample_lspc_pairwise, args=('left',self.only_title, use_price), axis=1)
            data['features_right'] = data.apply(serialize_sample_lspc_pairwise, args=('right',self.only_title, use_price), axis=1)
        elif self.dataset == 'abt-buy':
            data['features_left'] = data.apply(serialize_sample_abtbuy_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_abtbuy_pairwise, args=('right',self.only_title), axis=1)
        elif self.dataset == 'amazon-google':
            data['features_left'] = data.apply(serialize_sample_amazongoogle_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_amazongoogle_pairwise, args=('right',self.only_title), axis=1)
        elif self.dataset == 'walmart-amazon':
            data['features_left'] = data.apply(serialize_sample_walmartamazon_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_walmartamazon_pairwise, args=('right',self.only_title), axis=1)
        elif self.dataset == 'beeradvo-ratebeer':
            data['features_left'] = data.apply(self.serialize_sample_beeradvoratebeer, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_beeradvoratebeer, args=('right',self.only_title), axis=1)
        elif self.dataset == 'dblp-scholar':
            data['features_left'] = data.apply(self.serialize_sample_dblpscholar, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_dblpscholar, args=('right',self.only_title), axis=1)
        elif self.dataset == 'company':
            data['features_left'] = data.apply(serialize_sample_company_pairwise, args=('left',self.only_title), axis=1)
            data['features_right'] = data.apply(serialize_sample_company_pairwise, args=('right',self.only_title), axis=1)

        data = data[['features_left', 'features_right', 'label']]

        return data

class BaselineMultiClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=256, dataset='lspc', aug=False, additional_data=None, only_additional=False, only_title=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.only_title = only_title

        if self.aug:
            self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path)
            filename = Path(path).name
            category = filename.split('_')[1]
        else:
            data = pd.read_json(path, lines=True)

        data = data.fillna('')
        data = data.reset_index(drop=True)

        if 'wdcproducts' in path:
            data = self._prepare_data(data, True)
        else:
            data = self._prepare_data(data, False)

        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        if self.aug:
            example['features'] = self.augmenter.apply_aug(example['features'])

        example_tokenized = self.tokenizer(example['features'], padding=False, truncation='longest_first', max_length=self.max_length)
        example_tokenized['label'] = example['label']

        return example_tokenized

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data, use_price=False):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc_contrastive, args=(self.only_title,use_price), axis=1)

        label_enc = LabelEncoder()
        data['label'] = label_enc.fit_transform(data['label'])

        self.label_encoder = label_enc

        data = data[['features', 'label']]

        return data