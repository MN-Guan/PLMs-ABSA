# -*- coding: utf-8 -*-
# file: apcte.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
from random import *
from random import randint
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer, GPT2Tokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    # 实现填充和阶段
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    '''
    参数： max_seq_len 序列最大长度
          pretrained_bert_name 加载预训练模型的名称
    加载预训练模型的Tokenizer 且包含将文本转换为ids并填充和阶段、翻转功能的函数
    '''

    def __init__(self, max_seq_len, pretrained_bert_name, pretrained_model):
        if pretrained_model == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_bert_name)
            self.mask_idx = 50264
            self.cls = '<s>'
            self.sep = '</s>'
        elif pretrained_model == 'bart-base':
            self.tokenizer = BartTokenizer.from_pretrained(pretrained_bert_name)
            self.mask_idx = 50264
            self.cls = '<s>'
            self.sep = '</s>'
        elif pretrained_model == 'bert-base':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
            self.mask_idx = 103
            self.cls = '[CLS]'
            self.sep = '[SEP]'
        elif pretrained_model == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_bert_name)
            self.mask_idx = 9335
            self.cls = '<|endoftext|>'
            self.sep = '<|endoftext|>'
        else:
            print('not support your pre-trained model, please check!')
            exit()
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text1, text2=None, reverse=False, padding='post', truncating='post'):
        # 将text分词并转换为id
        if text2 is None:
            sequence = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(self.cls + ' ' + text1 + ' ' + self.sep))
        else:
            sequence = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(self.cls + ' ' + text1 + ' ' + self.sep + ' ' + text2 + ' ' + self.sep))
        # 如果sequence为空
        if len(sequence) == 0:
            sequence = [0]
        if self.mask_idx in sequence:
            mask_idx = sequence.index(self.mask_idx)
        else:
            mask_idx = 0
        if mask_idx >= self.max_seq_len:
            mask_idx = 0
        if reverse:  # 是否需要翻转
            sequence = sequence[::-1]
        # 返回padding和truncating后的数据
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating), mask_idx


polarity_list = ['Negative', 'Neutral', 'Positive']  # -1,0,1 --> 0,1,2


def get_prompt(model_name, A, data_name, prompt_rank):
    prompt = {
        1: 'this is a [Z] [K] for [A]',
        2: 'the [A] gets a [Z] [K]',
        3: 'the [K] of the [A] is [Z]',
        4: '[A] that gets a [Z] [K]',
        5: '[A] whose [K] is [Z]',
        6: '[A] with [Z] [K]'
    }

    keyword_dict_bart = {'restaurant': 'critical review',
                         'laptop': 'polarity',
                         'twitter': 'persuasion',
                         'res15': 'follow-up',
                         'res16': 'polarity'}

    keyword_dict_roberta = {'restaurant': 'intuitive feeling',
                            'laptop': 'intuitive feeling',
                            'twitter': 'purview',
                            'res15': 'intuitive feeling',
                            'res16': 'intuitive feeling'}

    keyword_dict_bert = {
        'restaurant': 'sign',  # template rank: 5 3 4 2 6 7
        'laptop': 'review article',  # t 3 5 4 2 7 6
        'twitter': 'touch sensation',  # 5 3 4 2 7 6
        'res15': 'comment',  # 5 3 4 2 7 6
        'res16': 'comment'  # 5 3 4 2 7 6
    }

    keyword_dict_gpt2 = {
        'restaurant': 'reassessment',
        'laptop': 'purview',
        'twitter': 'humour',
        'res15': 'opinion',
        'res16': 'smell'}

    prompt_map_bert = {
        'restaurant': [4, 2, 3, 1, 5, 6],
        'laptop': [2, 4, 3, 1, 6, 5],
        'twitter': [4, 2, 3, 1, 6, 5],
        'res15': [4, 2, 3, 1, 6, 5],
        'res16': [4, 2, 3, 1, 6, 5]
    }

    prompt_map_roberta = {
        'restaurant': [4, 6, 2, 1, 5, 3],
        'laptop': [1, 6 ,4, 2, 5, 3],
        'twitter': [3, 1, 5 ,4 ,6 ,2],
        'res15': [6, 4, 2, 1, 5, 3],
        'res16': [6, 4, 2, 1, 5, 3]
    }

    prompt_map_bart = {
        'restaurant': [1, 2, 4, 6, 3, 5],
        'laptop': [1, 2, 4, 6, 3, 5],
        'twitter': [3, 5, 4, 6, 2 ,1],
        'res15': [1, 2, 4, 6, 3, 5],
        'res16': [1, 2, 4, 6, 3, 5]
    }

    prompt_map_gpt2 = {
        'restaurant': [5, 6, 2, 3, 4, 1],
        'laptop': [3, 6, 5, 1, 4, 2],
        'twitter': [2, 4, 1, 6, 3, 5],
        'res15': [5, 4, 3, 6, 2, 1],
        'res16': [4, 5, 3, 2, 1, 6]
    }

    if model_name == 'bart-base':
        keyword_dict = keyword_dict_bart
        prompt_map = prompt_map_bart
    elif model_name == 'roberta-base':
        keyword_dict = keyword_dict_roberta
        prompt_map = prompt_map_roberta
    elif model_name == 'gpt2':
        keyword_dict = keyword_dict_gpt2
        prompt_map = prompt_map_gpt2
    else:
        keyword_dict = keyword_dict_bert
        prompt_map = prompt_map_bert

    K = keyword_dict[data_name]
    # 每个数据集中不同prompt的效果不同，所以传过来的prompt_type代表的是prompt的rank顺序
    # if prompt_rank == 0:
    #     return '{} {}'.format(Z, A), 'prefix'
    # prompt_number = key_prompt_map[data_name][prompt_rank-1]
    prompt_number = prompt_map[data_name][prompt_rank]
    if prompt_number >= 4:
        prompt_type = 'cloze'
    else:
        prompt_type = 'prefix'
    prompt_detail = prompt[prompt_number]
    if model_name == 'bert-base':
        mask = '[MASK]'
    elif model_name == 'gpt2':
        mask = 'mask'
    else:
        mask = '<mask>'
    return_prompt = prompt_detail.replace('[A]', A).replace('[Z]', mask).replace('[K]', K)
    return return_prompt, prompt_type


def get_prompt_wok(model_name, A, data_name, prompt_rank):
    prompt = {
        1: 'this is a [Z] for [A]',
        2: 'the [A] gets a [Z]',
        3: 'the of the [A] is [Z]',
        4: '[A] that gets a [Z]',
        5: '[A] whose is [Z]',
        6: '[A] with [Z]'
    }

    # prompt_map_bert = {
    #     'restaurant': [4, 2, 3, 1, 5, 6],
    #     'laptop': [2, 4, 3, 1, 6, 5],
    #     'twitter': [4, 2, 3, 1, 6, 5],
    #     'res15': [4, 2, 3, 1, 6, 5],
    #     'res16': [4, 2, 3, 1, 6, 5]
    # }
    prompt_map_bert = {
        'restaurant': [3, 2, 1, 4, 5, 6],
        'laptop': [2, 3, 1, 4, 6, 5],
        'twitter': [3, 1, 2, 4, 4, 6],
        'res15': [2, 3, 1, 4, 6, 5],
        'res16': [2, 3, 1, 4, 5, 6]
    }

    prompt_map_roberta = {
        'restaurant': [4, 6, 2, 1, 5, 3],
        'laptop': [1, 6 ,4, 2, 5, 3],
        'twitter': [3, 1, 5 ,4 ,6 ,2],
        'res15': [6, 4, 2, 1, 5, 3],
        'res16': [6, 4, 2, 1, 5, 3]
    }

    prompt_map_bart = {
        'restaurant': [1, 2, 4, 6, 3, 5],
        'laptop': [1, 2, 4, 6, 3, 5],
        'twitter': [3, 5, 4, 6, 2 ,1],
        'res15': [1, 2, 4, 6, 3, 5],
        'res16': [1, 2, 4, 6, 3, 5]
    }

    prompt_map_gpt2 = {
        'restaurant': [5, 6, 2, 3, 4, 1],
        'laptop': [3, 6, 5, 1, 4, 2],
        'twitter': [2, 4, 1, 6, 3, 5],
        'res15': [5, 4, 3, 6, 2, 1],
        'res16': [4, 5, 3, 2, 1, 6]
    }

    if model_name == 'bart-base':
        #keyword_dict = keyword_dict_bart
        prompt_map = prompt_map_bart
    elif model_name == 'roberta-base':
        #keyword_dict = keyword_dict_roberta
        prompt_map = prompt_map_roberta
    elif model_name == 'gpt2':
        #keyword_dict = keyword_dict_gpt2
        prompt_map = prompt_map_gpt2
    else:
        #keyword_dict = keyword_dict_bert
        prompt_map = prompt_map_bert

    # 每个数据集中不同prompt的效果不同，所以传过来的prompt_type代表的是prompt的rank顺序
    # if prompt_rank == 0:
    #     return '{} {}'.format(Z, A), 'prefix'
    # prompt_number = key_prompt_map[data_name][prompt_rank-1]
    prompt_number = prompt_map[data_name][prompt_rank]
    if prompt_number >= 4:
        prompt_type = 'cloze'
    else:
        prompt_type = 'prefix'
    prompt_detail = prompt[prompt_number]
    if model_name == 'bert-base':
        mask = '[MASK]'
    elif model_name == 'gpt2':
        mask = 'mask'
    else:
        mask = '<mask>'
    return_prompt = prompt_detail.replace('[A]', A).replace('[Z]', mask)
    return return_prompt, prompt_type


# PROMPT-CSR
class PromptDataset(Dataset):
    def __init__(self, pre_model, f_name, tokenizer, max_seq_len, data_name, twitter_set, wo_keyword, type='train'):
        self.model_name = pre_model
        self.polarity_map = {polarity: i for i, polarity in
                             enumerate(polarity_list, 0)}  # {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.label_map = {value: key for key, value in
                          self.polarity_map.items()}  # {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        self.f_name = f_name
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.twitter_set = twitter_set
        self.data_type = type
        self.wo_keyword = wo_keyword
        self.data = self._get_data()

    def _get_prompt_dict(self):
        '''
        twitter：积极和消极两种模板，中性一种模板；
        laptop:  中性两种模板，积极消极一种模板；
        res14： 消极中性三种模板，积极一种模板；
        res15 res16：中性6种模板 积极消极一种模板
        '''
        prompt_dict = {
            'twitter': {
                'Positive': [0, 1, 2],
                'Negative': [0, 1, 2],
                'Neutral': [0]
            },
            'laptop': {
                'Positive': [0],
                'Negative': [0],
                'Neutral': [0, 1, 2]
            },
            'restaurant': {
                'Positive': [0],
                'Negative': [0],
                'Neutral': [0, 1]
            },
            'res15': {
                'Positive': [0],
                'Negative': [0, 1, 2],
                'Neutral': [0, 1, 2, 3, 4, 5]
            },
            'res16': {
                'Positive': [0],
                'Negative': [0, 1],
                'Neutral': [0, 1, 2, 3, 4, 5]
            },
        }
        if self.twitter_set and random() < 0.5:  # twitter中性一半使用两种模板
            prompt_dict['twitter']['Neutral'] = [0, 1]
        else:
            pass
        return prompt_dict

    def _get_data(self):
        all_data = []
        fin = open(self.f_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines), 3):
            left, _, right = [s.lower().strip() for s in lines[i].partition('$T$')]
            aspect = lines[i + 1].lower().strip()
            re_polarity = lines[i + 2].strip()
            re_polarity = int(re_polarity) + 1   # label从-1~1转换到0~2

            text = left + ' ' + aspect + ' ' + right
            text_indices, _ = self.tokenizer.text_to_sequence(text)
            text_len = np.sum(text_indices != 0)

            # 测试数据随机选择模板进行测试
            prompt_dict = self._get_prompt_dict() # 获取提示排序列表
            prompt_rank_list = prompt_dict[self.data_name][self.label_map[re_polarity]]
            if self.data_type == 'test':
                if len(prompt_rank_list) > 1:
                    prompt_number = randint(0, len(prompt_rank_list)-1)  # 双向闭合
                    tmp = prompt_rank_list[prompt_number]
                else:
                    tmp = prompt_rank_list[0]
                prompt_rank_list = [tmp]

            for prompt_rank in prompt_rank_list:
                if self.wo_keyword == 'wo_keyword':
                    prompt, prompt_type = get_prompt_wok(self.model_name, aspect, self.data_name, prompt_rank)
                else:
                    prompt, prompt_type = get_prompt(self.model_name, aspect, self.data_name, prompt_rank)
                if prompt_type == 'cloze':
                    prompt_text = left + ' ' + prompt + ' ' + right
                    prompt_input_indices, mask_idx = self.tokenizer.text_to_sequence(prompt_text)
                    prompt_text_len = np.sum(prompt_input_indices != 0)
                    prompt_segments_indices = [0] * (prompt_text_len + 2)
                    attention_mask = [1] * (prompt_text_len + 2)
                else:
                    prompt_indices = self.tokenizer.text_to_sequence(prompt)
                    prompt_len = np.sum(prompt_indices != 0)
                    prompt_input_indices, mask_idx = self.tokenizer.text_to_sequence(text, prompt)  # 获取MASK位置以便后续的loss计算
                    prompt_segments_indices = [0] * (text_len + 2) + [1] * (prompt_len + 1)
                    attention_mask = [1] * (text_len + 2) + [1] * (prompt_len + 1)

                prompt_segments_indices = pad_and_truncate(prompt_segments_indices, self.max_seq_len)
                attention_mask = pad_and_truncate(attention_mask, self.max_seq_len)
                attention_mask[mask_idx] = 0

                data = {
                    'label': re_polarity,
                    'mask_idx': mask_idx,
                    'prompt_bert_indices': prompt_input_indices,
                    'prompt_segments_indices': prompt_segments_indices,
                    'attention_mask': attention_mask
                }
                all_data.append(data)

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# w/o prompt
class ABSADataset(Dataset):
    def __init__(self, f_name, tokenizer, max_seq_len):
        fin = open(f_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            left, _, right = [s.lower().strip() for s in lines[i].partition('$T$')]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            re_polarity = int(polarity) + 1

            text = left + ' ' + aspect + ' ' + right
            text_indices, _ = tokenizer.text_to_sequence(text)
            text_len = np.sum(text_indices != 0)
            prompt_segments_indices = [0] * (text_len + 2)

            segments_indices = pad_and_truncate(prompt_segments_indices, tokenizer.max_seq_len)

            data = {
                'label': re_polarity,
                'text_indices': text_indices,
                'segments_indices': segments_indices
            }

            all_data.append(data)
        self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]