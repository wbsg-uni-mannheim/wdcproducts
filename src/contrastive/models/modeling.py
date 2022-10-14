import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from transformers import AutoModel, AutoConfig
from src.contrastive.models.loss import SupConLoss

from pdb import set_trace

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BaseEncoder(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model)
        self.transformer.resize_token_embeddings(len_tokenizer)

    def forward(self, input_ids, attention_mask):
        
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        return output

class ContrastiveClassifierBaseModel(nn.Module):

    def __init__(self, len_tokenizer, checkpoint_path, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, frozen=True):
        super().__init__()

        self.pool = pool
        self.frozen = frozen
        self.checkpoint_path = checkpoint_path

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)
            # self.load_state_dict(checkpoint_path, strict=False)

        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right):
        
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']

        cosine = F.cosine_similarity(output_left, output_right, -1)

        return (torch.Tensor([0.0]), cosine)

class ContrastivePretrainBaseModel(nn.Module):

    def __init__(self, len_tokenizer, checkpoint_path, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, proj=32, temperature=0.07):
        super().__init__()

        self.pool = pool
        self.proj = proj
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)
            # self.load_state_dict(checkpoint_path, strict=False)

        
    def forward(self, input_ids, attention_mask):
        
        if self.pool:
            output = self.encoder(input_ids, attention_mask)
            output = mean_pooling(output, attention_mask)
        else:
            output = self.encoder(input_ids, attention_mask)['pooler_output']

        output = F.normalize(output, dim=-1)

        return ((0.0, output))

class ContrastiveSelfSupervisedPretrainModel(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', ssv=True, pool=False, proj='mlp', temperature=0.07, num_augments=2):
        super().__init__()

        self.ssv = ssv
        self.pool = pool
        self.proj = proj
        self.temperature = temperature
        self.num_augments = num_augments
        self.criterion = SupConLoss(self.temperature)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        self.contrastive_head = ContrastivePretrainHead(self.config.hidden_size, self.proj)

        
    def forward(self, input_ids, attention_mask, labels):
        
        additional_outputs = []
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            for num in range(self.num_augments-1):
                output_right = self.encoder(input_ids, attention_mask)
                output_right = mean_pooling(output_right, attention).unsqueeze(1)
                additional_outputs.append(output_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output'].unsqueeze(1)
            for num in range(self.num_augments-1):
                additional_outputs.append(self.encoder(input_ids, attention_mask)['pooler_output'].unsqueeze(1))
        
        output = torch.cat((output_left, *additional_outputs), 1)

        output = F.normalize(output, dim=-1)

        proj_output = self.contrastive_head(output)

        proj_output = F.normalize(proj_output, dim=-1)

        if self.ssv:
            loss = self.criterion(proj_output)
        else:
            loss = self.criterion(proj_output, labels)

        return ((loss,))

class ContrastivePretrainModel(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, proj=32, temperature=0.07):
        super().__init__()

        self.pool = pool
        self.proj = proj
        self.temperature = temperature
        self.criterion = SupConLoss(self.temperature)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        #self.transform = nn.Linear(self.config.hidden_size, self.proj)

        #self.contrastive_head = ContrastivePretrainHead(self.config.hidden_size, self.proj)

        
    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right):
        
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']
        
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        #output = torch.tanh(self.transform(output))

        output = F.normalize(output, dim=-1)

        #proj_output = self.contrastive_head(output)

        #proj_output = F.normalize(proj_output, dim=-1)

        loss = self.criterion(output, labels)

        return ((loss,))

class ContrastivePretrainHead(nn.Module):

    def __init__(self, hidden_size, proj='mlp'):
        super().__init__()
        if proj == 'linear':
            self.proj = nn.Linear(hidden_size, hidden_size)
        elif proj == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, hidden_states):
        x = self.proj(hidden_states)
        return x

class ContrastiveClassifierModel(nn.Module):

    def __init__(self, len_tokenizer, checkpoint_path, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, comb_fct='concat-abs-diff-mult', frozen=True, pos_neg=False, proj=16):
        super().__init__()

        self.pool = pool
        self.frozen = frozen
        self.checkpoint_path = checkpoint_path
        self.comb_fct = comb_fct
        self.pos_neg = pos_neg
        self.proj = proj

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        #self.transform = nn.Linear(self.config.hidden_size, self.proj)

        if self.pos_neg:
            self.criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_neg]))
        else:
            self.criterion = BCEWithLogitsLoss()
        self.classification_head = ClassificationHead(self.config, self.comb_fct)

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)
            # self.load_state_dict(checkpoint_path, strict=False)

        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right):
        
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']


        #output_left = torch.tanh(self.transform(output_left))
        #output_right = torch.tanh(self.transform(output_right))

        if self.comb_fct == 'concat-abs-diff':
            output = torch.cat((output_left, output_right, torch.abs(output_left - output_right)), -1)
        elif self.comb_fct == 'concat-mult':
            output = torch.cat((output_left, output_right, output_left * output_right), -1)
        elif self.comb_fct == 'concat':
            output = torch.cat((output_left, output_right), -1)
        elif self.comb_fct == 'abs-diff':
            output = torch.abs(output_left - output_right)
        elif self.comb_fct == 'mult':
            output = output_left * output_right
        elif self.comb_fct == 'abs-diff-mult':
            output = torch.cat((torch.abs(output_left - output_right), output_left * output_right), -1)
        elif self.comb_fct == 'concat-abs-diff-mult':
            output = torch.cat((output_left, output_right, torch.abs(output_left - output_right), output_left * output_right), -1)

        proj_output = self.classification_head(output)

        loss = self.criterion(proj_output.view(-1), labels.float())

        proj_output = torch.sigmoid(proj_output)

        return (loss, proj_output)

class ClassificationHead(nn.Module):

    def __init__(self, config, comb_fct):
        super().__init__()

        if comb_fct in ['concat-abs-diff', 'concat-mult']:
            self.hidden_size = 3 * config.hidden_size
        elif comb_fct in ['concat', 'abs-diff-mult']:
            self.hidden_size = 2 * config.hidden_size
        elif comb_fct in ['abs-diff', 'mult']:
            self.hidden_size = config.hidden_size
        elif comb_fct in ['concat-abs-diff-mult']:
            self.hidden_size = 4 * config.hidden_size

        # self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, features):
        # x = self.dropout(features)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.dropout(features)
        x = self.out_proj(x)
        return x
        
class ContrastiveClassifierModelMulti(nn.Module):

    def __init__(self, len_tokenizer, checkpoint_path, num_labels, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, frozen=True, pos_neg=False, proj=16):
        super().__init__()

        self.pool = pool
        self.frozen = frozen
        self.checkpoint_path = checkpoint_path
        self.pos_neg = pos_neg
        self.proj = proj
        self.num_labels = num_labels

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        #self.transform = nn.Linear(self.config.hidden_size, self.proj)

        if any(self.pos_neg):
            self.criterion = CrossEntropyLoss(weight=torch.Tensor(self.pos_neg))
        else:
            self.criterion = CrossEntropyLoss()
        self.classification_head = ClassificationHeadMulti(self.config, self.num_labels)

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)
            # self.load_state_dict(checkpoint_path, strict=False)

        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, labels):
        
        if self.pool:
            output= self.encoder(input_ids, attention_mask)
            output= mean_pooling(output, attention_mask)
        else:
            output = self.encoder(input_ids, attention_mask)['pooler_output']

        #output = torch.tanh(self.transform(output))

        proj_output = self.classification_head(output)

        loss = self.criterion(proj_output, labels)

        proj_output = torch.nn.functional.softmax(proj_output, -1)

        return (loss, proj_output)

class ClassificationHeadMulti(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_labels = num_labels

        # self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features):
        # x = self.dropout(features)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.dropout(features)
        x = self.out_proj(x)
        return x

class ContrastiveCrossClassifierModel(nn.Module):

    def __init__(self, len_tokenizer, checkpoint_path, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, frozen=True, pos_neg=False):
        super().__init__()

        self.pool = pool
        self.frozen = frozen
        self.checkpoint_path = checkpoint_path
        self.pos_neg = pos_neg

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config
        if self.pos_neg:
            self.criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_neg]))
        else:
            self.criterion = BCEWithLogitsLoss()
        self.classification_head = CrossClassificationHead(self.config)

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)
            # self.load_state_dict(checkpoint_path, strict=False)
        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, labels):
        
        if self.pool:
            output = self.encoder(input_ids, attention_mask)
            output = mean_pooling(output, attention_mask)
        else:
            output = self.encoder(input_ids, attention_mask)['pooler_output']

        proj_output = self.classification_head(output)

        loss = self.criterion(proj_output.view(-1), labels.float())

        proj_output = torch.sigmoid(proj_output)

        return (loss, proj_output)

class CrossClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()

        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features):
        # x = self.dropout(features)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.dropout(features)
        x = self.out_proj(x)
        return x

class MatrixModel(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, comb_fct='concat-abs-diff', pos_neg=False, temperature=0.07, proj=32):
        super().__init__()

        self.pool = pool
        self.comb_fct = comb_fct
        self.pos_neg = pos_neg
        self.temperature = temperature
        self.proj = proj

        self.criterion = SupConLoss(self.temperature)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        self.transform = nn.Linear(self.config.hidden_size, self.proj)

        self.classification_head = MatrixModelHead(self.config, self.comb_fct)

        
    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right, contrastive):
        
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']
        
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        output = torch.tanh(self.transform(output))

        output_contrastive = F.normalize(output.clone(), dim=-1)

        loss_contrastive = self.criterion(output_contrastive, contrastive)

        #calculate posneg in batch
        value_counts = labels.unique(return_counts=True)
        pos_neg = int(value_counts[1][0].item() / value_counts[1][1].item())

        logits = self.classification_head(output)

        logits = logits.reshape(input_ids.shape[0], -1).unsqueeze(-1)
        labels = labels.unsqueeze(-1).float()

        #loss = self.criterion(logits, labels)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.Tensor([pos_neg]).to(logits.device))

        combined_loss = loss_contrastive + loss

        logits = torch.nan_to_num(logits, nan=-10.0)
        proj_output = torch.sigmoid(logits)

        return ((combined_loss, proj_output))

class MatrixModelNew(nn.Module):

    def __init__(self, len_tokenizer, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, comb_fct='concat-abs-diff', pos_neg=False):
        super().__init__()

        self.pool = pool
        self.comb_fct = comb_fct
        self.pos_neg = pos_neg

        if self.pos_neg:
            self.criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_neg]))
        else:
            self.criterion = BCEWithLogitsLoss()

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

        self.classification_head = MatrixModelHead(self.config, self.comb_fct)

        
    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right):
        
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']
        
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        #output = F.normalize(output, dim=-1)

        #calculate posneg in batch
        value_counts = labels.unique(return_counts=True)
        pos_neg = int(value_counts[1][0].item() / value_counts[1][1].item())

        logits = self.classification_head(output)

        logits = logits.reshape(input_ids.shape[0], -1).unsqueeze(-1)
        labels = labels.unsqueeze(-1).float()

        #loss = self.criterion(logits, labels)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.Tensor([pos_neg]).to(logits.device))

        logits = torch.nan_to_num(logits, nan=-10.0)
        proj_output = torch.sigmoid(logits)

        return ((loss, proj_output))

class MatrixModelHead(nn.Module):

    def __init__(self, config, comb_fct):
        super().__init__()

        if comb_fct in ['concat-abs-diff', 'concat-mult']:
            #self.hidden_size = 3 * config.hidden_size
            self.hidden_size = 3 * 32
        elif comb_fct in ['concat', 'abs-diff-mult']:
            self.hidden_size = 2 * config.hidden_size
        elif comb_fct in ['abs-diff', 'mult']:
            self.hidden_size = 1 * config.hidden_size
        elif comb_fct in ['concat-abs-diff-mult']:
            self.hidden_size = 4 * config.hidden_size

        # self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, features):

        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        pairwise_diff = torch.abs(features.unsqueeze(1) - features)
        idx = torch.triu_indices(pairwise_diff.shape[0], pairwise_diff.shape[0], 1)
        pairwise_diff = pairwise_diff[idx[0],idx[1]]

        only_left = features.unsqueeze(1) - torch.abs(features*0)
        idx = torch.triu_indices(only_left.shape[0], only_left.shape[0], 1)
        only_left = only_left[idx[0],idx[1]]

        only_right = (torch.abs(features.unsqueeze(1)*0) - features)*-1
        idx = torch.triu_indices(only_right.shape[0], only_right.shape[0], 1)
        only_right = only_right[idx[0],idx[1]]

        logits = torch.cat((only_left, only_right, pairwise_diff), -1)
        
        # x = self.dropout(logits)
        # x = self.dense(x)
        # x = torch.tanh(x)
        x = self.dropout(logits)
        x = self.out_proj(x)

        return x