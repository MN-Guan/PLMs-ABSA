# -*- coding: gb2312 -*-

import argparse
import logging
import sys
import time
from random import randint
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BartModel, RobertaModel, GPT2Model
import torch.nn as nn
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from nltk.corpus import wordnet as wn
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, pad_and_truncate

polarity_list = ['Negative', 'Neutral', 'Positive']

data_files = {
    'restaurant': {
        'train': 'Dataset/semeval14/Restaurants_Train.xml.seg',
        'test': 'Dataset/semeval14/Restaurants_Test_Gold.xml.seg',
    },
    'laptop': {
        'train': 'Dataset/semeval14/Laptops_Train.xml.seg',
        'test': 'Dataset/semeval14/Laptops_Test_Gold.xml.seg',
    },
    'twitter': {
        'train': 'Dataset/acl-14-short-data/train.raw',
        'test': 'Dataset/acl-14-short-data/test.raw'
    },
    'res15': {
        'train': 'Dataset/res15/res_15_train.txt',
        'test': 'Dataset/res15/res_15_test.txt'
    },
    'res16': {
        'train': 'Dataset/res16/res_16_train.txt',
        'test': 'Dataset/res16/res_16_test.txt'
    }
}


def get_word(word):
    '''
    :param word: �ҳ�word��ͬ��ʼ���
    :return: word_synset��ɵ��б�
    '''
    #print('find synset for {}'.format(word))
    word_synset = set()
    # TODO ��ȡWordNet�е�ͬ��ʼ�
    synsets = wn.synsets(word, pos=wn.NOUN)  # word���ڵĴʼ��б�
    for synset in synsets:
        words = synset.lemma_names()  # ��λ��
        for word in words:
            word = word.replace('_', ' ')
            word_synset.add(word)
    #print(word_synset)
    return list(word_synset)


class BERT_SPC(nn.Module):
    def __init__(self, bert):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768, 3)
        self.sotfmax = nn.Softmax(dim=-1)

    def _getMask(self, input, mask_idx):
        '''
        :param input:  shape:b,n,d
        :param mask_idx: b
        :return: b,d
        '''
        new_input = None
        for idx in range(input.size(0)):
            if new_input == None:
                new_input = input[idx][mask_idx[idx]].unsqueeze(dim=0)
            else:
                new_input = torch.cat((new_input, input[idx][mask_idx[idx]].unsqueeze(dim=0)), dim=0)
        return new_input

    def forward(self, inputs):
        if self.bert == GPT2Model:
            concat_input_ids, concat_token_type_ids, mask_idx, attention_mask = inputs[0], inputs[1], inputs[2], inputs[3]
            bert_out = self.bert(input_ids=concat_input_ids, attention_mask = attention_mask)[0]
        else:
            concat_input_ids, concat_token_type_ids, mask_idx = inputs[0], inputs[1], inputs[2]
            bert_out = self.bert(input_ids=concat_input_ids)[0]
        mask_out = self.dropout(self._getMask(bert_out, mask_idx))
        logits = self.dense(mask_out)
        return self.sotfmax(logits)


class PromptDataset(Dataset):
    def __init__(self,pre_model, f_name, tokenizer, max_seq_len, keywords):
        self.pre_model = pre_model
        self.polarity_map = {polarity: i for i, polarity in enumerate(polarity_list, 0)}
        self.f_name = f_name
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = self._get_data()

    def _get_data(self):
        all_data = []
        fin = open(self.f_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in tqdm(range(0, len(lines), 3)):
            left, _, right = [s.lower().strip() for s in lines[i].partition('$T$')]
            aspect = lines[i + 1].lower().strip()
            re_polarity = lines[i + 2].strip()
            re_polarity = int(re_polarity) + 1
            text = left + ' ' + aspect + ' ' + right

            for keyword in self.keywords:
                if self.pre_model == 'gpt2':
                    prompt = aspect + ' ' + 'mask' + ' ' + keyword
                else:
                    prompt = aspect + ' ' + '<mask>' + ' ' + keyword
                text_indices, _ = tokenizer.text_to_sequence(text)
                text_len = np.sum(text_indices != 0)
                prompt_indices, _ = tokenizer.text_to_sequence(aspect)
                prompt_len = np.sum(prompt_indices != 0)

                prompt_input_indices, mask_idx = tokenizer.text_to_sequence(text, prompt)
                prompt_segments_indices = [0] * (text_len + 2) + [1] * (prompt_len + 1)
                attention_mask = [1] * len(prompt_segments_indices)
                prompt_segments_indices = pad_and_truncate(prompt_segments_indices, tokenizer.max_seq_len)
                attention_mask = pad_and_truncate(attention_mask, tokenizer.max_seq_len)

                data = {
                    'polarity': re_polarity,
                    'keyword': keyword,
                    'prompt_bert_indices': prompt_input_indices,
                    'mask_idx': mask_idx,
                    'prompt_segments_indices': prompt_segments_indices,
                    'attention_mask': attention_mask
                    }
                all_data.append(data)
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def _train(loader):
    acc_dict = {}

    for i, batch in tqdm(enumerate(loader)):
        inputs = [batch[col].to(device) for col in ['prompt_bert_indices', 'prompt_segments_indices', 'mask_idx', 'attention_mask']]
        outputs = model(inputs)
        keywords = batch['keyword']
        targets = batch['polarity']
        preds = torch.argmax(outputs, -1)

        for i in range(len(targets)):# ��target=outputʱ����Ӧ��keyword�÷�+1
            if preds[i] == targets[i]:
                acc_dict[keywords[i]] = acc_dict.get(keywords[i], 0) + 1
    return acc_dict


if __name__ == '__main__':
    pre_train_map = {
        'bart-base': {
            'tokenizer_file': 'model/bart-base/',
            'pre_model_file': 'model/bart-base'
        },
        'roberta-base': {
            'tokenizer_file': 'model/roberta-base/',
            'pre_model_file': 'model/roberta-base'
        },
        'bert-base':
            {
                'tokenizer_file': 'model/uncased_L-12_H-768_A-12/',
                'pre_model_file': 'model/uncased_L-12_H-768_A-12'
            },
        'gpt2':
            {
                'tokenizer_file': 'model/gpt2/',
                'pre_model_file': 'model/gpt2'
            }
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='bert-base', type=str, help='bart-base,roberta-base,gpt2')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    opt = parser.parse_args()

    seed = opt.seed
    max_seq_len = opt.max_seq_len
    batch_size = opt.batch_size
    pretrained_model = opt.pretrained_model

    if seed is not None:
        random.seed(0)
        np.random.seed(seed)
        torch.manual_seed(seed)  # ȷ����ͬ�����������������Ľ��һ�� ��ʹ��cuda
        torch.cuda.manual_seed(seed)  # ȷ����ͬ�����������������Ľ��һ�� ʹ��cuda
        torch.cuda.manual_seed_all(seed)  # ȷ����ͬ�����������������Ľ��һ�� ʹ��cuda
        torch.backends.cudnn.deterministic = True  # Ӧ�ÿ��Ա�֤ÿ�����������ʱ����ͬ���������ǹ̶���
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer4Bert(opt.max_seq_len, pre_train_map[opt.pretrained_model]['tokenizer_file'], pretrained_model)
    if pretrained_model == 'roberta-base':
        pre_model = RobertaModel.from_pretrained(pre_train_map[pretrained_model]['pre_model_file'])
    elif pretrained_model == 'bart-base':
        pre_model = BartModel.from_pretrained(pre_train_map[pretrained_model]['pre_model_file'])
    elif pretrained_model == 'gpt2':
        pre_model = GPT2Model.from_pretrained(pre_train_map[pretrained_model]['pre_model_file'])
    else:
        pre_model = BertModel.from_pretrained(pre_train_map[pretrained_model]['pre_model_file'])
    model = BERT_SPC(pre_model).to(device=device)

    #�ҳ�ͬ��ʴʼ�
    task_word1 = ['sentiment', 'feeling', 'emotion', 'mood', 'polarity']  # �����
    task_word2 = ['review', 'evaluation', 'comment', 'perspective', 'expression', 'view']  # ������
    keywords_list1 = []
    for word in task_word1:
        keywords_list1 += get_word(word)
    keywords_set1 = list(set(keywords_list1))
    print(keywords_set1)
    print(len(keywords_set1))
    keywords_list2 = []
    for word in task_word2:
        keywords_list2 += get_word(word)
    keywords_set2 = list(set(keywords_list2))
    print(keywords_set2)
    print(len(keywords_set2))

    keywords_set = keywords_set1
    for word in keywords_set2:
        if word in keywords_set1:
            continue
        else:
            keywords_set.append(word)

    data_keyword_dict = {}

    #print(len(keywords_set))
    for data_name in data_files.keys():
        if data_name == 'restaurant':
            continue
    #for data_name in ['res15']:
        #keywords_set = ['sentiment', 'polarity']
        logger = logging.getLogger()
        file = os.path.basename(sys.argv[0])
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        if not os.path.exists('log'):  # �洢������־�ļ���
            os.mkdir('log')
        if not os.path.exists('log/find_keyword'):  # �洢������־�ļ���
            os.mkdir('log/find_keyword')
        log_file = 'log/find_keyword/{}_{}_{}_{}.log'.format(file.split('.')[0], pretrained_model, data_name, time.strftime("%y%m%d-%H:%M", time.localtime()))
        logger.addHandler(logging.FileHandler(log_file))
        logger.info(keywords_set)

        dataset = PromptDataset(pretrained_model, data_files[data_name]['train'], tokenizer, max_seq_len, keywords_set)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger.info('>> len(dataset):{}'.format(len(dataset)))
        acc_dict = _train(loader)
        sorted_acc_dict = sorted(acc_dict.items(), key=lambda x: x[1], reverse=True)
        top_acc_word = None
        for key, value in sorted_acc_dict:
            if top_acc_word == None:
                top_acc_word = key
            logger.info('>> score of {} is {}'.format(key, value))
        data_keyword_dict[data_name] = top_acc_word
        logger.handlers = logger.handlers[:1]
    print(data_keyword_dict)