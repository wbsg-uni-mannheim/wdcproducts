import json
import random
import numpy as np

class Augmenter(object):
    """Data augmentation operator.
    Support both span and attribute level augmentation operators.
    """
    def __init__(self):
        pass

    def augment(self, tokens, labels, op='del'):
        """ Performs data augmentation on a sequence of tokens
        The supported ops:
           ['del',
            'swap',
            'all']
        Args:
            tokens (list of strings): the input tokens
            labels (list of strings): the labels of the tokens
            op (str, optional): a string encoding of the operator to be applied
        Returns:
            list of strings: the augmented tokens
            list of strings: the augmented labels
        """
        if 'del' in op:
            # insert padding to keep the length consistent
            # span_len = random.randint(1, 3)
            span_len = random.randint(1, 2)
            pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            new_tokens = tokens[:pos1] + tokens[pos2+1:]
            new_labels = tokens[:pos1] + labels[pos2+1:]
        elif 'swap' in op:
            span_len = random.randint(2, 4)
            pos1, pos2 = self.sample_span(tokens, labels, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            sub_arr = tokens[pos1:pos2+1]
            random.shuffle(sub_arr)
            new_tokens = tokens[:pos1] + sub_arr + tokens[pos2+1:]
            new_labels = tokens[:pos1] + ['O'] * (pos2 - pos1 + 1) + labels[pos2+1:]
        else:
            new_tokens, new_labels = tokens, labels

        return new_tokens, new_labels


    def augment_sent(self, text, op='all'):
        """ Performs data augmentation on a classification example.
        Similar to augment(tokens, labels) but works for sentences
        or sentence-pairs.
        Args:
            text (str): the input sentence
            op (str, optional): a string encoding of the operator to be applied
        Returns:
            str: the augmented sentence
        """
        # 50% of chance of flipping
        if ' [SEP] ' in text and random.randint(0, 1) == 0:
            left, right = text.split(' [SEP] ')
            text = right + ' [SEP] ' + left

        # tokenize the sentence
        current = ''
        tokens = text.split(' ')

        # avoid the special tokens
        labels = []
        for token in tokens:
            if token in ['COL', 'VAL']:
                labels.append('HD')
            elif token in ['[CLS]', '[SEP]']:
                labels.append('<SEP>')
            else:
                labels.append('O')

        if op == 'all':
            # RandAugment: https://arxiv.org/pdf/1909.13719.pdf
            N = 2
            ops = ['del', 'swap']
            for op in random.choices(ops, k=N):
                tokens, labels = self.augment(tokens, labels, op=op)
        else:
            tokens, labels = self.augment(tokens, labels, op=op)
        results = ' '.join(tokens)
        return results

    def sample_span(self, tokens, labels, span_len=3):
        candidates = []
        for idx, token in enumerate(tokens):
            if idx + span_len - 1 < len(labels) and ''.join(labels[idx:idx+span_len]) == 'O'*span_len:
                candidates.append((idx, idx+span_len-1))
        if len(candidates) <= 0:
            return -1, -1
        return random.choice(candidates)

    def sample_position(self, tokens, labels, tfidf=False):
        candidates = []
        for idx, token in enumerate(tokens):
            if labels[idx] == 'O':
                candidates.append(idx)
        if len(candidates) <= 0:
            return -1
        return random.choice(candidates)


if __name__ == '__main__':
    ag = Augmenter()
    text = 'COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020 [SEP] COL content VAL sigmod conference 2010 papers 2019-12-31 COL year VAL 2019'
    for op in ['del',
               'swap',
               'all']:
        print(op)
        print(ag.augment_sent(text, op=op))
