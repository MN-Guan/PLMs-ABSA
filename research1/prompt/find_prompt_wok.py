# -*- coding: gb2312 -*-
'''
空白prompt：[X], [A] [Z] [K]
构造prompt：
1、 [X], [A] that gets a [Z] [K] （删除）
2、 [X], this is a [Z] [K] for [A]
3、 [X], the [A] gets a [Z] [K]
4、 [X], the [K] of the [A] is [Z]
5、[X1] [A] that gets a [Z][K] [X2]
6、 [X1] [A] whose [K] is [Z] [X2]
7、 [X1] [A] with [Z] [K] [X2]
'''
import argparse
import logging
import time
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BartModel, RobertaModel, GPT2Model
import torch.nn as nn
import torch
import random
import numpy as np
from tqdm import tqdm
import os
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, pad_and_truncate
import sys
from random import randint


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


def get_prompt(A, K, Z, prompt_number):
    if prompt_number == 1:
        return 'this is a {} {} for {}'.format(Z, K, A)
    elif prompt_number == 2:
        return 'the {} gets a {} {}'.format(A, Z, K)
    elif prompt_number == 3:
        return 'the {} of the {} is {}'.format(K, A, Z)
    elif prompt_number == 4:
        return '{} that gets a {} {}'.format(A, Z, K)
    elif prompt_number == 5:
        return '{} whose the {} is {}'.format(A, K, Z)
    elif prompt_number == 6:
        return '{} with {} {}'.format(A, Z, K)
    else:
        raise ValueError('value error in prompt_number')


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
    def __init__(self, pre_model, f_name, tokenizer, max_seq_len, data_name):
        if pre_model == 'gpt2':
            self.Z = 'mask'
        else:
            self.Z = '<mask>'
        self.pre_model = pre_model
        self.f_name = f_name
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = self._get_data()

    def _get_data(self):
        all_data = []
        fin = open(self.f_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        id = 0
        keyword = ''
        for i in range(0, len(lines), 3):
            id += 1
            left, _, right = [s.lower().strip() for s in lines[i].partition('$T$')]
            aspect = lines[i + 1].lower().strip()
            re_polarity = lines[i + 2].strip()
            re_polarity = int(re_polarity) + 1
            text = left + ' ' + aspect + ' ' + right

            for i in range(1, 7):
                prompt = get_prompt(aspect, keyword, self.Z, i)
                if i <= 3:
                    text_indices, _ = tokenizer.text_to_sequence(text)
                    text_len = np.sum(text_indices != 0)
                    prompt_indices, _ = tokenizer.text_to_sequence(aspect)
                    prompt_len = np.sum(prompt_indices != 0)
                    prompt_input_indices, mask_idx = tokenizer.text_to_sequence(text, prompt)
                    prompt_segments_indices = [0] * (text_len + 2) + [1] * (prompt_len + 1)
                else:
                    prompt_text = left + ' ' + prompt + ' ' + right
                    prompt_input_indices, mask_idx = self.tokenizer.text_to_sequence(prompt_text)
                    # mask_idx = prompt_input_indices.index(103)  # 获取MASK位置以便后续的loss计算
                    prompt_text_len = np.sum(prompt_input_indices != 0)
                    prompt_segments_indices = [0] * (prompt_text_len + 2)

                attention_mask = [1] * len(prompt_segments_indices)
                prompt_segments_indices = pad_and_truncate(prompt_segments_indices, tokenizer.max_seq_len)
                attention_mask = pad_and_truncate(attention_mask, tokenizer.max_seq_len)
                data = {
                    'polarity': re_polarity,
                    'prompt_number': str(i),  #选最佳prompt
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
        prompt_number = batch['prompt_number']
        targets = batch['polarity']
        preds = torch.argmax(outputs, -1)
        for i in range(len(targets)): # 当target=output时，对应的prompt得分+1
            if preds[i] == targets[i]:
                acc_dict[prompt_number[i]] = acc_dict.get(prompt_number[i], 0) + 1
    return acc_dict

if __name__ == '__main__':
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists('log/find_prompt_wok'):
        os.mkdir('log/find_prompt_wok')
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
    parser.add_argument('--pretrained_model', default='bert-base', type=str, help='bart-base,roberta-base')
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
        torch.manual_seed(seed)  # 确保相同随机种子下随机抽样的结果一致 不使用cuda
        torch.cuda.manual_seed(seed)  # 确保相同随机种子下随机抽样的结果一致 使用cuda
        torch.cuda.manual_seed_all(seed)  # 确保相同随机种子下随机抽样的结果一致 使用cuda
        torch.backends.cudnn.deterministic = True  # 应该可以保证每次运行网络的时候相同输入的输出是固定的
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    # keyword_dict_bart = {'restaurant': 'critical review', 'laptop': 'polarity', 'twitter': 'persuasion', 'res15': 'follow-up', 'res16': 'polarity'}
    # keyword_dict_bert = {'restaurant': 'sign', 'laptop': 'review article', 'twitter': 'touch sensation', 'res15': 'comment', 'res16': 'comment'}
    # keyword_dict_roberta = {'restaurant': 'intuitive feeling', 'laptop': 'intuitive feeling', 'twitter': 'purview', 'res15': 'intuitive feeling', 'res16': 'intuitive feeling'}
    # keyword_dict_gpt2 = {'restaurant': 'reassessment', 'laptop': 'purview', 'twitter': 'humour', 'res15': 'opinion', 'res16': 'smell'}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer4Bert(max_seq_len, pre_train_map[pretrained_model]['tokenizer_file'], pretrained_model)
    if pretrained_model == 'bart-base':
        pre_model = BartModel.from_pretrained(pre_train_map[pretrained_model]['pre_model_file'])
        #keyword_dict = keyword_dict_bart
    elif pretrained_model == 'roberta-base':
        pre_model = RobertaModel.from_pretrained(pre_train_map[pretrained_model]['pre_model_file'])
        #keyword_dict = keyword_dict_roberta
    elif pretrained_model == 'gpt2':
        pre_model = GPT2Model.from_pretrained(pre_train_map[pretrained_model]['pre_model_file'])
        #keyword_dict = keyword_dict_gpt2
    else:
        pre_model = BertModel.from_pretrained(pre_train_map[pretrained_model]['pre_model_file'])
        #keyword_dict = keyword_dict_bert
    model = BERT_SPC(pre_model).to(device=device)


    for data_name in data_files.keys():
    #for data_name in ['laptop']:
        file = os.path.basename(sys.argv[0])  # 当前文件名名称
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        log_file = 'log/find_prompt_wok/{}_{}_{}_{}.log'.format(file.split('.')[0], pretrained_model, data_name, time.strftime("%y%m%d-%H:%M", time.localtime()))
        logger.addHandler(logging.FileHandler(log_file))

        best_score = 0
        logger.info('>> DATASET:{} START FIND PROMPT TEMPLATE'.format(data_name))
        dataset = PromptDataset(pretrained_model, data_files[data_name]['train'], tokenizer, max_seq_len, data_name)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        result_dict = _train(loader)
        sorted_result_dict = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        keyword_scores = {}
        for key, value in sorted_result_dict:
            logger.info('&& the acc of tempalte {} is {}'.format(key, value))
        logger.handlers = logger.handlers[:1]



