import numpy as np
np.random.seed(42)
import random
random.seed(42)

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from pdb import set_trace

@dataclass
class DataCollatorContrastivePretrainSelfSupervised:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input):
        
        features_left = [x[0]['features'] for x in input]
        labels = [x[0]['labels'] for x in input]
        
        batch = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)

        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']

        batch['labels'] = torch.LongTensor(labels)
        
        return batch

@dataclass
class DataCollatorContrastivePretrain:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input):

        features_left = [x[0]['features'] for x in input]
        features_right = [x[1]['features'] for x in input]
        labels = [x[0]['labels'] for x in input]

        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        
        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)

        return batch

@dataclass
class DataCollatorContrastivePretrainDeepmatcher:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input_both):

        rnd = random.choice(range(len(input_both[0])))
        input = [x[rnd] for x in input_both]

        features_left = [x[0]['features'] for x in input]
        features_right = [x[1]['features'] for x in input]

        labels = [x[0]['labels'] for x in input]

        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        
        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)

        return batch
@dataclass
class DataCollatorContrastiveClassification:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input):

        features_left = [x['features_left'] for x in input]
        features_right = [x['features_right'] for x in input]
        labels = [x['labels'] for x in input]
        
        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        
        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)

        return batch

@dataclass
class DataCollatorContrastiveCrossClassification:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 256
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input):

        features_left = [x['features_left'] for x in input]
        features_right = [x['features_right'] for x in input]
        labels = [x['labels'] for x in input]
        
        batch = self.tokenizer(features_left, features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)

        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']

        batch['labels'] = torch.LongTensor(labels)

        return batch

@dataclass
class DataCollatorMatrix:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input):

        features_left = [x[0]['features'] for x in input]
        features_right = [x[1]['features'] for x in input]
        labels = [x[0]['labels'] for x in input]

        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        
        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch_size = len(input)
        anchor_count = len(input[0])

        labels = torch.LongTensor(labels)
        labels_contrastive = labels.clone()
        labels = labels.repeat(2)

        pairwise_diff = (labels.unsqueeze(1) - labels) + 1
        pairwise_diff[pairwise_diff != 1] = 0

        idx = torch.triu_indices(pairwise_diff.shape[0], pairwise_diff.shape[0], 1)
        pairwise_diff = pairwise_diff[idx[0],idx[1]]
        
        batch['labels'] = pairwise_diff.reshape(len(batch['input_ids']), -1)
        batch['contrastive'] = labels_contrastive

        return batch

@dataclass
class DataCollatorMatrixDeepmatcher:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input_both):

        rnd = random.choice([0,1])
        input = [x[rnd] for x in input_both]

        features_left = [x[0]['features'] for x in input]
        features_right = [x[1]['features'] for x in input]
        labels = [x[0]['labels'] for x in input]

        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        
        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch_size = len(input)
        anchor_count = len(input[0])

        labels = torch.LongTensor(labels)
        labels_contrastive = labels.clone()
        labels = labels.repeat(2)

        pairwise_diff = (labels.unsqueeze(1) - labels) + 1
        pairwise_diff[pairwise_diff != 1] = 0

        idx = torch.triu_indices(pairwise_diff.shape[0], pairwise_diff.shape[0], 1)
        pairwise_diff = pairwise_diff[idx[0],idx[1]]
        
        batch['labels'] = pairwise_diff.reshape(len(batch['input_ids']), -1)
        batch['contrastive'] = labels_contrastive

        return batch
    
@dataclass
class DataCollatorMatrixNew:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input):

        features_left = [x[0]['features'] for x in input]
        features_right = [x[1]['features'] for x in input]
        labels = [x[0]['labels'] for x in input]

        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        
        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch_size = len(input)
        anchor_count = len(input[0])

        labels = torch.LongTensor(labels)
        labels = labels.repeat(2)

        pairwise_diff = (labels.unsqueeze(1) - labels) + 1
        pairwise_diff[pairwise_diff != 1] = 0

        idx = torch.triu_indices(pairwise_diff.shape[0], pairwise_diff.shape[0], 1)
        pairwise_diff = pairwise_diff[idx[0],idx[1]]
        
        batch['labels'] = pairwise_diff.reshape(len(batch['input_ids']), -1)

        return batch